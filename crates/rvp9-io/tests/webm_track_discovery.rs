//! Track discovery must retain active ancestors, independent of sibling count.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rvp9_io::container::{ContainerError, PacketReader, read_webm_packets};

thread_local! {
    static MEASURING: Cell<bool> = const { Cell::new(false) };
    static REQUESTED_BYTES: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

fn record_request(bytes: usize) {
    if MEASURING.try_with(Cell::get).unwrap_or(false) {
        let _ = REQUESTED_BYTES.try_with(|total| total.set(total.get().saturating_add(bytes)));
    }
}

// SAFETY: All allocation operations delegate their unchanged arguments to System.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_request(layout.size());
        // SAFETY: The caller supplies the valid allocation layout required by System.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_request(layout.size());
        // SAFETY: The caller supplies the valid allocation layout required by System.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: The caller supplies the original pointer and layout, forwarded unchanged.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_request(new_size);
        // SAFETY: The caller supplies System's original allocation and valid new size.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

struct Measurement {
    previous: bool,
}

impl Measurement {
    fn start() -> Self {
        REQUESTED_BYTES.with(|total| total.set(0));
        Self {
            previous: MEASURING.with(|active| active.replace(true)),
        }
    }
}

impl Drop for Measurement {
    fn drop(&mut self) {
        MEASURING.with(|active| active.set(self.previous));
    }
}

const EBML: &[u8] = &[0x1a, 0x45, 0xdf, 0xa3];
const SEGMENT: &[u8] = &[0x18, 0x53, 0x80, 0x67];
const TRACKS: &[u8] = &[0x16, 0x54, 0xae, 0x6b];

fn element(id: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut bytes = id.to_vec();
    if payload.len() < 127 {
        bytes.push(0x80 | payload.len() as u8);
    } else {
        bytes.extend_from_slice(&(0x0100_0000_0000_0000 | payload.len() as u64).to_be_bytes());
    }
    bytes.extend_from_slice(payload);
    bytes
}

fn webm(payload: &[u8]) -> Vec<u8> {
    let mut bytes = element(EBML, &[]);
    bytes.extend(element(SEGMENT, payload));
    bytes
}

fn wide_payload() -> Vec<u8> {
    element(TRACKS, &[]).repeat(65_536)
}

fn track(number: u8) -> Vec<u8> {
    element(
        TRACKS,
        &element(
            &[0xae],
            &[
                0xd7, 0x81, number, 0x83, 0x81, 1, 0x86, 0x85, b'V', b'_', b'V', b'P', b'9',
            ],
        ),
    )
}

fn cluster() -> Vec<u8> {
    let mut payload = element(&[0xa3], &[0x81, 0, 0, 0x80, 0x11]);
    payload.extend(element(&[0xa3], &[0x82, 0, 0, 0x80, 0x22]));
    element(&[0x1f, 0x43, 0xb6, 0x75], &payload)
}

struct FixtureFile(PathBuf);

impl FixtureFile {
    fn new(bytes: &[u8]) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "rvp9_track_discovery_{}_{}.webm",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).unwrap();
        Self(path)
    }
}

impl Drop for FixtureFile {
    fn drop(&mut self) {
        std::fs::remove_file(&self.0).unwrap();
    }
}

fn assert_bad_ebml(result: Result<impl Sized, ContainerError>, expected: &str) {
    match result {
        Err(ContainerError::BadEbml(message)) => assert_eq!(message, expected),
        Err(error) => panic!("unexpected error: {error:?}"),
        Ok(_) => panic!("expected BadEbml: {expected}"),
    }
}

#[test]
fn wide_empty_tracks_use_bounded_allocation_and_match_streaming_error() {
    let bytes = webm(&wide_payload());
    // Count cumulative requested allocation/reallocation bytes, not peak RSS.
    // Input storage and all assertions, formatting, and file I/O are outside the interval.
    let measurement = Measurement::start();
    let result = read_webm_packets(&bytes);
    drop(measurement);
    let requested = REQUESTED_BYTES.with(Cell::get);
    assert_bad_ebml(result, "VP9 video track not found");
    let file = FixtureFile::new(&bytes);
    assert_bad_ebml(PacketReader::open(&file.0), "VP9 video track not found");
    assert!(
        requested <= 16 * 1024,
        "wide shallow discovery requested {requested} cumulative allocation bytes; budget is 16384"
    );
}

#[test]
fn excessive_depth_before_tracks_matches_streaming_depth_error() {
    let mut payload = Vec::new();
    for _ in 0..33 {
        payload = element(TRACKS, &payload);
    }
    payload.extend(track(1));
    let bytes = webm(&payload);
    let file = FixtureFile::new(&bytes);
    let expected = "recognized container nesting depth exceeds limit";
    assert_bad_ebml(PacketReader::open(&file.0), expected);
    assert_bad_ebml(read_webm_packets(&bytes), expected);
}

fn assert_packet_parity(bytes: &[u8], expected: &[u8]) {
    let ranges = read_webm_packets(bytes).unwrap();
    assert_eq!(ranges.len(), 1);
    assert_eq!(ranges[0].slice(bytes), expected);
    let file = FixtureFile::new(bytes);
    let mut reader = PacketReader::open(&file.0).unwrap();
    assert_eq!(reader.next_packet().unwrap().unwrap(), expected);
    assert!(reader.next_packet().unwrap().is_none());
}

#[test]
fn wide_shallow_tracks_preserve_packet_payload() {
    let mut payload = wide_payload();
    payload.extend(track(1));
    payload.extend(cluster());
    assert_packet_parity(&webm(&payload), &[0x11]);
}

#[test]
fn first_vp9_track_in_document_order_matches_streaming() {
    let mut payload = track(1);
    payload.extend(track(2));
    payload.extend(cluster());
    assert_packet_parity(&webm(&payload), &[0x11]);
}
