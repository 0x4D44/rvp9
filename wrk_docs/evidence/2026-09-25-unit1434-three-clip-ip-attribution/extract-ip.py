#!/usr/bin/env python3
"""Extract exact-image xperf samples and resolve them with the paired PDB.

The parser is intentionally independent of Windows APIs. On Windows, normal
ETL mode first runs xperf, then feeds its dumper rows through the same parser
used by ``--raw-csv`` replay mode. Replay mode accepts an archived dumper CSV
and a known runtime image base, which makes old captures reproducible without
running xperf again.
"""

from __future__ import annotations

import argparse
import collections
import csv
import ctypes as C
import ctypes.wintypes as W
import hashlib
import json
import locale
import ntpath
import os
import re
import struct
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, TextIO


MSF7_MAGIC = b"Microsoft C/C++ MSF 7.00\r\n\x1aDS\0\0\0"
SYMBOL_NAME_CAPACITY = 4096
PID_SUFFIX = re.compile(r"^(.*?)\s*\(\s*(\d+)\s*\)\s*$")
ADDRESS_TOKEN = re.compile(r"0x([0-9a-fA-F]+)\s*$")
REPORT_DIAGNOSTIC = re.compile(r"\b(?:warn(?:ing)?|error|lost|loss|drop(?:ped)?)\b", re.I)


class ExtractError(RuntimeError):
    """An input, parsing, verification, or symbol-resolution failure."""


@dataclass(frozen=True)
class ImageRange:
    pid: int
    base: int
    size: int
    source: str

    @property
    def end(self) -> int:
        return self.base + self.size


@dataclass(frozen=True)
class Sample:
    pid: int
    timestamp: int
    tid: int
    ip: int
    raw_count: int
    module_text: str
    module_ip_text: int
    raw_fields: tuple[str, ...]
    raw_line: str


@dataclass
class ParseResult:
    images: dict[int, ImageRange] = field(default_factory=dict)
    samples: list[Sample] = field(default_factory=list)
    other_module_samples: list[Sample] = field(default_factory=list)
    sampled_profile_rows: int = 0
    target_process_rows: int = 0
    pid_filtered_rows: int = 0
    other_module_rows: int = 0
    diagnostics: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PeInfo:
    preferred_base: int
    image_size: int
    pdb_name: str
    pdb_guid: bytes
    pdb_age: int


@dataclass(frozen=True)
class SymbolInfo:
    symbol: str
    file: str
    line: int
    displacement: int
    symbol_ok: bool
    line_ok: bool


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bounded_read(stream: TextIO, count: int, what: str) -> bytes:
    data = stream.read(count)
    if len(data) != count:
        raise ExtractError(f"truncated {what}")
    return data


def _file_slice(stream: TextIO, offset: int, count: int, what: str) -> bytes:
    stream.seek(offset)
    return _bounded_read(stream, count, what)


def _parse_pdb_identity(path: Path) -> tuple[bytes, int]:
    """Read PDB stream 1's GUID and age from an MSF 7.0 PDB."""
    with path.open("rb") as stream:
        header = _file_slice(stream, 0, 56, "PDB header")
        if header[:32] != MSF7_MAGIC:
            raise ExtractError(f"unsupported PDB format (expected MSF 7.0): {path}")
        block_size, _free_map, block_count, directory_size, _reserved, block_map = struct.unpack_from(
            "<6I", header, 32
        )
        file_size = path.stat().st_size
        if block_size < 512 or block_size > 65536 or block_size & (block_size - 1):
            raise ExtractError(f"invalid PDB block size {block_size}")
        if block_count * block_size > file_size or directory_size <= 0:
            raise ExtractError("invalid PDB block or directory size")
        directory_block_count = (directory_size + block_size - 1) // block_size
        block_map_bytes = directory_block_count * 4
        if block_map * block_size + block_map_bytes > file_size:
            raise ExtractError("PDB directory block map extends past end of file")
        directory_blocks = struct.unpack(
            f"<{directory_block_count}I",
            _file_slice(stream, block_map * block_size, block_map_bytes, "PDB directory block map"),
        )
        directory = b"".join(
            _file_slice(stream, block * block_size, block_size, "PDB directory block")
            for block in directory_blocks
        )[:directory_size]
        if len(directory) < 4:
            raise ExtractError("PDB directory has no stream count")
        stream_count = struct.unpack_from("<I", directory, 0)[0]
        if stream_count < 2 or stream_count > 1_000_000:
            raise ExtractError(f"invalid PDB stream count {stream_count}")
        sizes_end = 4 + 4 * stream_count
        if sizes_end > len(directory):
            raise ExtractError("truncated PDB stream-size table")
        sizes = struct.unpack_from(f"<{stream_count}I", directory, 4)
        cursor = sizes_end
        stream_blocks: list[int] = []
        for stream_size in sizes:
            if stream_size == 0xFFFFFFFF:
                continue
            count = (stream_size + block_size - 1) // block_size
            bytes_needed = count * 4
            if cursor + bytes_needed > len(directory):
                raise ExtractError("truncated PDB stream block list")
            stream_blocks.append(count)
            cursor += bytes_needed
        if len(stream_blocks) <= 1 or sizes[1] == 0xFFFFFFFF or sizes[1] < 28:
            raise ExtractError("PDB has no complete information stream")

        # Walk stream block lists again: absent streams have no block entries.
        cursor = sizes_end
        info_blocks: tuple[int, ...] | None = None
        for index, stream_size in enumerate(sizes):
            if stream_size == 0xFFFFFFFF:
                continue
            count = (stream_size + block_size - 1) // block_size
            bytes_needed = count * 4
            blocks = struct.unpack_from(f"<{count}I", directory, cursor) if count else ()
            cursor += bytes_needed
            if index == 1:
                info_blocks = blocks
                break
        if info_blocks is None:
            raise ExtractError("PDB information stream block list is missing")
        info_size = sizes[1]
        info_data = b"".join(
            _file_slice(stream, block * block_size, block_size, "PDB information stream block")
            for block in info_blocks
        )[:info_size]
        if len(info_data) < 28:
            raise ExtractError("PDB information stream is truncated")
        age = struct.unpack_from("<I", info_data, 8)[0]
        guid = info_data[12:28]
        return guid, age


def _rva_to_offset(rva: int, size: int, sections: list[tuple[int, int, int, int]]) -> int:
    for virtual_address, virtual_size, raw_offset, raw_size in sections:
        extent = max(virtual_size, raw_size)
        if virtual_address <= rva and rva + size <= virtual_address + extent:
            delta = rva - virtual_address
            if delta + size > raw_size:
                break
            return raw_offset + delta
    raise ExtractError(f"PE RVA {rva:#x}+{size:#x} is not backed by section data")


def read_pe_info(exe_path: Path, pdb_path: Path) -> PeInfo:
    """Read PE image size and its RSDS CodeView record; confirm PDB pairing."""
    with exe_path.open("rb") as stream:
        dos = _file_slice(stream, 0, 64, "DOS header")
        if dos[:2] != b"MZ":
            raise ExtractError(f"not a PE executable: {exe_path}")
        pe_offset = struct.unpack_from("<I", dos, 0x3C)[0]
        header = _file_slice(stream, pe_offset, 24, "PE and COFF headers")
        if header[:4] != b"PE\0\0":
            raise ExtractError("invalid PE signature")
        section_count = struct.unpack_from("<H", header, 6)[0]
        optional_size = struct.unpack_from("<H", header, 20)[0]
        if section_count == 0 or section_count > 96:
            raise ExtractError(f"invalid PE section count {section_count}")
        optional_offset = pe_offset + 24
        optional = _file_slice(stream, optional_offset, optional_size, "PE optional header")
        magic = struct.unpack_from("<H", optional, 0)[0]
        if magic == 0x20B:
            preferred_base = struct.unpack_from("<Q", optional, 24)[0]
            directory_count_offset, directory_offset = 108, 112
        elif magic == 0x10B:
            preferred_base = struct.unpack_from("<I", optional, 28)[0]
            directory_count_offset, directory_offset = 92, 96
        else:
            raise ExtractError(f"unsupported PE optional-header magic {magic:#x}")
        image_size = struct.unpack_from("<I", optional, 56)[0]
        directory_count = struct.unpack_from("<I", optional, directory_count_offset)[0]
        if image_size <= 0 or directory_count <= 6:
            raise ExtractError("PE has no usable image size or debug directory")
        debug_rva, debug_size = struct.unpack_from("<II", optional, directory_offset + 6 * 8)
        if debug_rva == 0 or debug_size < 28:
            raise ExtractError("PE debug directory is absent")
        section_offset = optional_offset + optional_size
        section_bytes = _file_slice(stream, section_offset, 40 * section_count, "PE section table")
        sections: list[tuple[int, int, int, int]] = []
        for index in range(section_count):
            start = index * 40
            virtual_size, virtual_address, raw_size, raw_offset = struct.unpack_from("<IIII", section_bytes, start + 8)
            sections.append((virtual_address, virtual_size, raw_offset, raw_size))
        debug_offset = _rva_to_offset(debug_rva, debug_size, sections)
        codeview: list[tuple[bytes, int, str]] = []
        if debug_size % 28 != 0:
            raise ExtractError("PE debug directory size is not a multiple of IMAGE_DEBUG_DIRECTORY")
        for offset in range(debug_offset, debug_offset + debug_size, 28):
            entry = _file_slice(stream, offset, 28, "PE debug directory entry")
            debug_type = struct.unpack_from("<I", entry, 12)[0]
            data_size = struct.unpack_from("<I", entry, 16)[0]
            data_offset = struct.unpack_from("<I", entry, 24)[0]
            if debug_type != 2:
                continue
            if data_size < 25:
                raise ExtractError("PE CodeView record is too small")
            data = _file_slice(stream, data_offset, data_size, "PE CodeView record")
            if data[:4] != b"RSDS":
                continue
            guid = data[4:20]
            age = struct.unpack_from("<I", data, 20)[0]
            raw_name = data[24:].split(b"\0", 1)[0]
            pdb_name = raw_name.decode("utf-8", "replace")
            codeview.append((guid, age, pdb_name))
        if len(codeview) != 1:
            raise ExtractError(f"expected one PE RSDS record, found {len(codeview)}")
        pdb_guid, pdb_age, embedded_pdb_name = codeview[0]

    if ntpath.basename(embedded_pdb_name).casefold() != pdb_path.name.casefold():
        raise ExtractError(
            f"PE names PDB {ntpath.basename(embedded_pdb_name)!r}, supplied {pdb_path.name!r}"
        )
    actual_guid, actual_age = _parse_pdb_identity(pdb_path)
    if actual_guid != pdb_guid or actual_age != pdb_age:
        raise ExtractError(
            "PDB does not match PE CodeView identity "
            f"(PE GUID/age {uuid.UUID(bytes_le=pdb_guid)}/{pdb_age}, "
            f"PDB GUID/age {uuid.UUID(bytes_le=actual_guid)}/{actual_age})"
        )
    return PeInfo(preferred_base, image_size, embedded_pdb_name, pdb_guid, pdb_age)


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1]
    return value


def _process_identity(value: str) -> tuple[str, int | None]:
    text = _unquote(value).strip()
    match = PID_SUFFIX.fullmatch(text)
    if match:
        return ntpath.basename(match.group(1).strip()).casefold(), int(match.group(2))
    return ntpath.basename(text).casefold(), None


def _looks_like_target_process(value: str, exe_name: str) -> bool:
    text = _unquote(value).strip()
    leading_name = re.split(r"\s+\(\s*\d+\s*\)\s*$", text, maxsplit=1)[0]
    return ntpath.basename(leading_name.strip()).casefold() == exe_name.casefold()


def _hex_address(value: str, field_name: str, line_number: int) -> int:
    text = _unquote(value).strip()
    try:
        return int(text, 16)
    except ValueError as error:
        raise ExtractError(f"dumper line {line_number}: invalid {field_name} {text!r}") from error


def _decimal(value: str, field_name: str, line_number: int) -> int:
    text = _unquote(value).strip().replace(",", "")
    try:
        return int(text, 10)
    except ValueError as error:
        raise ExtractError(f"dumper line {line_number}: invalid {field_name} {text!r}") from error


def _sample_module(text: str, line_number: int) -> tuple[str, int]:
    value = _unquote(text).strip()
    module, separator, address = value.partition("!")
    match = ADDRESS_TOKEN.search(address) if separator else None
    if not module.strip() or match is None:
        raise ExtractError(f"dumper line {line_number}: malformed SampledProfile IP text {value!r}")
    return ntpath.basename(module.strip()).casefold(), int(match.group(1), 16)


def _record_image(
    fields: list[str],
    line_number: int,
    exe_name: str,
    pid_filter: int | None,
    result: ParseResult,
) -> None:
    event = fields[0].strip()
    if len(fields) < 3 or not _looks_like_target_process(fields[2], exe_name):
        return
    process_name, pid = _process_identity(fields[2])
    if process_name != exe_name.casefold():
        return
    if pid is None:
        raise ExtractError(f"dumper line {line_number}: target image record has no PID")
    if pid_filter is not None and pid != pid_filter:
        return
    if len(fields) < 5:
        raise ExtractError(f"dumper line {line_number}: target image record has fewer than five fields")
    image_field = fields[8] if event == "I-Start" and len(fields) > 8 else fields[-1]
    image_name = ntpath.basename(_unquote(image_field).strip().strip('"')).casefold()
    if image_name != exe_name.casefold():
        return
    base = _hex_address(fields[3], "image base", line_number)
    extent = _hex_address(fields[4], "image size/end", line_number)
    size = extent - base if event == "I-Start" else extent
    if size <= 0:
        raise ExtractError(f"dumper line {line_number}: invalid target image range")
    discovered = ImageRange(pid, base, size, event)
    previous = result.images.get(pid)
    if previous is not None and (previous.base != base or previous.size != size):
        raise ExtractError(
            f"PID {pid} has conflicting {event} image range {base:#x}..{discovered.end:#x}; "
            f"previous {previous.base:#x}..{previous.end:#x} from {previous.source}"
        )
    if previous is None or event == "I-Start":
        result.images[pid] = discovered


def parse_dumper_rows(
    lines: Iterable[str],
    exe_name: str,
    pid_filter: int | None = None,
    fallback_image: ImageRange | None = None,
) -> ParseResult:
    """Parse xperf dumper text, failing on malformed target-process rows."""
    exe_name = ntpath.basename(exe_name).casefold()
    result = ParseResult()
    for line_number, raw in enumerate(lines, 1):
        raw_line = raw.rstrip("\r\n")
        if not raw_line.strip():
            continue
        try:
            fields = next(csv.reader([raw_line], skipinitialspace=True, strict=True))
        except csv.Error as error:
            if raw_line.lstrip().startswith("SampledProfile"):
                # xperf also prints non-CSV quoting in unrelated process rows
                # (for example, an Unknown process with embedded quotes).
                # Keep target rows strict, but do not let unrelated rows abort
                # exact target-executable attribution.
                result.sampled_profile_rows += 1
                if exe_name.casefold() in raw_line.casefold():
                    raise ExtractError(f"malformed target CSV at dumper line {line_number}: {error}") from error
            continue
        if not fields:
            continue
        event = fields[0].strip().lstrip("\ufeff")
        fields[0] = event
        if REPORT_DIAGNOSTIC.search(raw_line):
            result.diagnostics.append(raw_line.strip())
        if event in ("ImageId", "I-Start"):
            _record_image(fields, line_number, exe_name, pid_filter, result)
            continue
        if event != "SampledProfile":
            continue
        result.sampled_profile_rows += 1
        if len(fields) < 3:
            raise ExtractError(f"dumper line {line_number}: SampledProfile row has no process field")
        if not _looks_like_target_process(fields[2], exe_name):
            continue
        process_name, pid = _process_identity(fields[2])
        if process_name != exe_name:
            continue
        if pid is None:
            raise ExtractError(f"dumper line {line_number}: target SampledProfile row has no PID")
        if pid_filter is not None and pid != pid_filter:
            result.pid_filtered_rows += 1
            continue
        result.target_process_rows += 1
        if len(fields) < 10:
            raise ExtractError(
                f"dumper line {line_number}: target SampledProfile row has {len(fields)} fields, expected at least 10"
            )
        timestamp = _decimal(fields[1], "timestamp", line_number)
        tid = _decimal(fields[3], "thread ID", line_number)
        ip = _hex_address(fields[4], "sample IP", line_number)
        raw_count = _decimal(fields[8], "sample count", line_number)
        if raw_count <= 0:
            raise ExtractError(f"dumper line {line_number}: target sample count must be positive")
        module_name, text_ip = _sample_module(fields[7], line_number)
        if text_ip != ip:
            raise ExtractError(
                f"dumper line {line_number}: IP field {ip:#x} does not match IP text {text_ip:#x}"
            )
        sample = Sample(
            pid=pid,
            timestamp=timestamp,
            tid=tid,
            ip=ip,
            raw_count=raw_count,
            module_text=_unquote(fields[7]).strip(),
            module_ip_text=text_ip,
            raw_fields=tuple(fields),
            raw_line=raw_line,
        )
        if module_name == exe_name:
            result.samples.append(sample)
        else:
            result.other_module_samples.append(sample)
            result.other_module_rows += 1

    if fallback_image is not None:
        if pid_filter is None or fallback_image.pid != pid_filter:
            raise ExtractError("raw replay needs --pid equal to the supplied image PID")
        previous = result.images.get(fallback_image.pid)
        if previous is not None and (previous.base, previous.size) != (fallback_image.base, fallback_image.size):
            raise ExtractError("caller-supplied image range conflicts with dumper-discovered range")
        result.images[fallback_image.pid] = fallback_image

    if not result.samples:
        raise ExtractError("no target-executable SampledProfile rows were retained")
    if sum(sample.raw_count for sample in result.samples) <= 0:
        raise ExtractError("retained target-executable samples have zero direct count")
    for sample in result.samples:
        image = result.images.get(sample.pid)
        if image is None:
            raise ExtractError(f"PID {sample.pid} has target samples but no discovered image range")
        if not image.base <= sample.ip < image.end:
            raise ExtractError(
                f"PID {sample.pid} target IP {sample.ip:#x} is outside image range "
                f"{image.base:#x}..{image.end:#x}"
            )
    return result


def _load_dbghelp(pdb_directory: Path, pid: int):
    if os.name != "nt":
        raise ExtractError("DbgHelp symbol extraction requires Windows")

    class SYMBOL_INFO(C.Structure):
        _fields_ = [
            ("SizeOfStruct", W.ULONG),
            ("TypeIndex", W.ULONG),
            ("Reserved", C.c_ulonglong * 2),
            ("Index", W.ULONG),
            ("Size", W.ULONG),
            ("ModBase", C.c_ulonglong),
            ("Flags", W.ULONG),
            ("Value", C.c_ulonglong),
            ("Address", C.c_ulonglong),
            ("Register", W.ULONG),
            ("Scope", W.ULONG),
            ("Tag", W.ULONG),
            ("NameLen", W.ULONG),
            ("MaxNameLen", W.ULONG),
            ("Name", C.c_char * (SYMBOL_NAME_CAPACITY + 1)),
        ]

    class IMAGEHLP_LINE64(C.Structure):
        _fields_ = [
            ("SizeOfStruct", W.DWORD),
            ("Key", C.c_void_p),
            ("LineNumber", W.DWORD),
            ("FileName", C.c_char_p),
            ("Address", C.c_ulonglong),
        ]

    class GUID(C.Structure):
        _fields_ = [
            ("Data1", W.DWORD),
            ("Data2", W.WORD),
            ("Data3", W.WORD),
            ("Data4", C.c_ubyte * 8),
        ]

    class IMAGEHLP_MODULEW64(C.Structure):
        _fields_ = [
            ("SizeOfStruct", W.DWORD),
            ("BaseOfImage", C.c_ulonglong),
            ("ImageSize", W.DWORD),
            ("TimeDateStamp", W.DWORD),
            ("CheckSum", W.DWORD),
            ("NumSyms", W.DWORD),
            ("SymType", W.DWORD),
            ("ModuleName", W.WCHAR * 32),
            ("ImageName", W.WCHAR * 256),
            ("LoadedImageName", W.WCHAR * 256),
            ("LoadedPdbName", W.WCHAR * 256),
            ("CVSig", W.DWORD),
            ("CVData", W.WCHAR * (260 * 3)),
            ("PdbSig", W.DWORD),
            ("PdbSig70", GUID),
            ("PdbAge", W.DWORD),
            ("PdbUnmatched", W.BOOL),
            ("DbgUnmatched", W.BOOL),
            ("LineNumbers", W.BOOL),
            ("GlobalSymbols", W.BOOL),
            ("TypeInfo", W.BOOL),
            ("SourceIndexed", W.BOOL),
            ("Publics", W.BOOL),
            ("MachineType", W.DWORD),
            ("Reserved", W.DWORD),
        ]

    dh = C.WinDLL("dbghelp.dll", use_last_error=True)
    dh.SymSetOptions.argtypes = [W.DWORD]
    dh.SymSetOptions.restype = W.DWORD
    dh.SymInitializeW.argtypes = [W.HANDLE, W.LPCWSTR, W.BOOL]
    dh.SymInitializeW.restype = W.BOOL
    dh.SymLoadModuleExW.argtypes = [
        W.HANDLE, W.HANDLE, W.LPCWSTR, W.LPCWSTR, C.c_ulonglong,
        W.DWORD, C.c_void_p, W.DWORD,
    ]
    dh.SymLoadModuleExW.restype = C.c_ulonglong
    dh.SymGetModuleInfoW64.argtypes = [W.HANDLE, C.c_ulonglong, C.POINTER(IMAGEHLP_MODULEW64)]
    dh.SymGetModuleInfoW64.restype = W.BOOL
    dh.SymFromAddr.argtypes = [
        W.HANDLE, C.c_ulonglong, C.POINTER(C.c_ulonglong), C.POINTER(SYMBOL_INFO),
    ]
    dh.SymFromAddr.restype = W.BOOL
    dh.SymGetLineFromAddr64.argtypes = [
        W.HANDLE, C.c_ulonglong, C.POINTER(W.DWORD), C.POINTER(IMAGEHLP_LINE64),
    ]
    dh.SymGetLineFromAddr64.restype = W.BOOL
    dh.SymCleanup.argtypes = [W.HANDLE]
    dh.SymCleanup.restype = W.BOOL
    # Keep symbol loading eager so the exact LoadedPdbName is populated at
    # module registration and cannot be mistaken for a later lazy lookup.
    dh.SymSetOptions(0x00000002 | 0x00000010 | 0x00000400 | 0x00080000)
    # DbgHelp keys its loaded-module table by process handle. A distinct
    # synthetic handle per PID keeps identical virtual addresses from
    # different process address spaces from colliding.
    handle = W.HANDLE(0x10000 + pid * 16)
    search_path = os.fspath(pdb_directory)
    if not dh.SymInitializeW(handle, search_path, False):
        raise OSError(C.get_last_error(), "SymInitializeW failed")
    return dh, handle, SYMBOL_INFO, IMAGEHLP_LINE64, IMAGEHLP_MODULEW64


def _normalized_windows_path(value: str | Path) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(os.fspath(value))))


def resolve_symbols(
    exe_path: Path,
    pdb_path: Path,
    images: dict[int, ImageRange],
    samples: list[Sample],
) -> dict[tuple[int, int], SymbolInfo]:
    resolved: dict[tuple[int, int], SymbolInfo] = {}
    for pid in sorted({sample.pid for sample in samples}):
        image = images.get(pid)
        if image is None:
            raise ExtractError(f"PID {pid} has samples but no image range")
        dh, handle, symbol_type, line_type, module_type = _load_dbghelp(pdb_path.parent, pid)
        try:
            loaded = dh.SymLoadModuleExW(
                handle, None, os.fspath(exe_path), None, image.base, image.size, None, 0
            )
            if loaded != image.base:
                raise ExtractError(
                    f"DbgHelp loaded {os.fspath(exe_path)} at {loaded:#x}, expected {image.base:#x} for PID {pid}"
                )
            module = module_type()
            module.SizeOfStruct = C.sizeof(module_type)
            if not dh.SymGetModuleInfoW64(handle, image.base, C.byref(module)):
                raise OSError(C.get_last_error(), f"SymGetModuleInfoW64 failed at {image.base:#x}")
            if int(module.BaseOfImage) != image.base or int(module.ImageSize) != image.size:
                raise ExtractError(
                    f"DbgHelp reports module {int(module.BaseOfImage):#x}+{int(module.ImageSize):#x}, "
                    f"expected {image.base:#x}+{image.size:#x} for PID {pid}"
                )
            loaded_pdb = "".join(module.LoadedPdbName).split("\0", 1)[0]
            if not loaded_pdb or _normalized_windows_path(loaded_pdb) != _normalized_windows_path(pdb_path):
                raise ExtractError(
                    f"DbgHelp loaded PDB {loaded_pdb or '<none>'!r}, expected {os.fspath(pdb_path)!r} for PID {pid}"
                )

            addresses = sorted({sample.ip for sample in samples if sample.pid == pid})
            for ip in addresses:
                info = symbol_type()
                info.SizeOfStruct = 88
                info.MaxNameLen = SYMBOL_NAME_CAPACITY
                displacement = C.c_ulonglong(0)
                symbol_ok = bool(dh.SymFromAddr(handle, ip, C.byref(displacement), C.byref(info)))
                symbol = (
                    bytes(info.Name).split(b"\0", 1)[0].decode("utf-8", "replace")
                    if symbol_ok else ""
                )
                line_info = line_type()
                line_info.SizeOfStruct = C.sizeof(line_type)
                line_displacement = W.DWORD(0)
                line_call_ok = bool(
                    dh.SymGetLineFromAddr64(handle, ip, C.byref(line_displacement), C.byref(line_info))
                )
                filename = ""
                lineno = 0
                if line_call_ok:
                    lineno = int(line_info.LineNumber)
                    if line_info.FileName:
                        filename = line_info.FileName.decode("mbcs", "replace")
                line_ok = bool(line_call_ok and filename and lineno > 0)
                resolved[(pid, ip)] = SymbolInfo(
                    symbol=symbol,
                    file=filename,
                    line=lineno,
                    displacement=int(displacement.value),
                    symbol_ok=symbol_ok,
                    line_ok=line_ok,
                )
        finally:
            dh.SymCleanup(handle)
    return resolved


def _read_xperf_report(xperf: str, etl_path: Path, action: str, timeout: int) -> tuple[int, bytes]:
    command = [xperf, "-i", os.fspath(etl_path), "-a", action]
    try:
        run = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise ExtractError(f"xperf {action} exceeded {timeout}s") from error
    return run.returncode, run.stdout


def _write_xperf_reports(xperf: str, etl_path: Path, outdir: Path, timeout: int) -> dict[str, object]:
    reports: dict[str, object] = {}
    for action, filename in (("tracestats", "xperf-tracestats.txt"), ("profile", "xperf-profile.txt")):
        exit_code, output = _read_xperf_report(xperf, etl_path, action, timeout)
        (outdir / filename).write_bytes(output)
        reports[action] = {
            "exit_code": exit_code,
            "report": filename,
            "diagnostic_lines": [
                line.strip()
                for line in output.decode(locale.getpreferredencoding(False), "replace").splitlines()
                if REPORT_DIAGNOSTIC.search(line)
            ],
        }
        if exit_code != 0:
            raise ExtractError(f"xperf {action} exited {exit_code}; see {outdir / filename}")
    return reports


def _run_xperf_dumper(
    xperf: str, etl_path: Path, exe_name: str, pid_filter: int | None, timeout: int
) -> tuple[ParseResult, list[str], int]:
    if os.name != "nt":
        raise ExtractError("ETL dumper execution requires Windows; use --raw-csv to replay archived rows")
    command = [xperf, "-i", os.fspath(etl_path), "-a", "dumper", "-add_fieldnames"]
    with tempfile.TemporaryFile(mode="w+b") as output:
        try:
            run = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise ExtractError(f"xperf dumper exceeded {timeout}s") from error
        if run.returncode != 0:
            raise ExtractError(f"xperf dumper exited {run.returncode}")
        output.seek(0)
        encoding = "mbcs" if os.name == "nt" else "utf-8"
        with open(output.fileno(), "r", encoding=encoding, errors="replace", newline="", closefd=False) as text_stream:
            parsed = parse_dumper_rows(text_stream, exe_name, pid_filter=pid_filter)
        return parsed, command, run.returncode


def _sample_csv_rows(
    samples: list[Sample], images: dict[int, ImageRange], symbols: dict[tuple[int, int], SymbolInfo]
) -> Iterator[list[object]]:
    yield [
        "sample_index", "timestamp", "pid", "tid", "ip", "raw_count", "module_base",
        "module_end", "rva", "module_ip_text", "symbol", "file", "line", "symbol_ok",
        "line_ok", "raw_fields_json", "raw_dumper_line",
    ]
    for index, sample in enumerate(samples, 1):
        image = images[sample.pid]
        symbol = symbols[(sample.pid, sample.ip)]
        yield [
            index,
            sample.timestamp,
            sample.pid,
            sample.tid,
            f"0x{sample.ip:016x}",
            sample.raw_count,
            f"0x{image.base:016x}",
            f"0x{image.end:016x}",
            f"0x{sample.ip - image.base:x}",
            sample.module_text,
            symbol.symbol,
            symbol.file,
            symbol.line,
            int(symbol.symbol_ok),
            int(symbol.line_ok),
            json.dumps(sample.raw_fields, ensure_ascii=False),
            sample.raw_line,
        ]


def _other_module_csv_rows(samples: list[Sample]) -> Iterator[list[object]]:
    yield ["timestamp", "pid", "tid", "ip", "raw_count", "module_ip_text", "raw_fields_json", "raw_dumper_line"]
    for sample in samples:
        yield [
            sample.timestamp,
            sample.pid,
            sample.tid,
            f"0x{sample.ip:016x}",
            sample.raw_count,
            sample.module_text,
            json.dumps(sample.raw_fields, ensure_ascii=False),
            sample.raw_line,
        ]


def _ip_map_rows(
    samples: list[Sample], images: dict[int, ImageRange], symbols: dict[tuple[int, int], SymbolInfo]
) -> list[list[object]]:
    grouped: dict[tuple[int, int], list[Sample]] = collections.defaultdict(list)
    for sample in samples:
        grouped[(sample.pid, sample.ip)].append(sample)
    rows: list[list[object]] = [[
        "pid", "ip", "module_base", "rva", "raw_count", "sample_rows", "symbol", "file", "line",
        "symbol_ok", "line_ok",
    ]]
    for (pid, ip), grouped_samples in sorted(grouped.items()):
        image = images[pid]
        info = symbols[(pid, ip)]
        rows.append([
            pid, f"0x{ip:016x}", f"0x{image.base:016x}", f"0x{ip - image.base:x}",
            sum(item.raw_count for item in grouped_samples), len(grouped_samples),
            info.symbol, info.file, info.line, int(info.symbol_ok), int(info.line_ok),
        ])
    return rows


def _summary_rows(
    samples: list[Sample], images: dict[int, ImageRange], symbols: dict[tuple[int, int], SymbolInfo]
):
    by_symbol: dict[str, list[Sample]] = collections.defaultdict(list)
    by_line: dict[tuple[str, int, str], list[Sample]] = collections.defaultdict(list)
    for sample in samples:
        info = symbols[(sample.pid, sample.ip)]
        symbol = info.symbol or "<unresolved>"
        by_symbol[symbol].append(sample)
        by_line[(info.file or "<no source line>", info.line, symbol)].append(sample)

    symbol_rows = [["symbol", "sample_rows", "raw_count", "unique_ips"]]
    for symbol, group in sorted(by_symbol.items(), key=lambda item: (-sum(s.raw_count for s in item[1]), item[0])):
        symbol_rows.append([
            symbol, len(group), sum(item.raw_count for item in group), len({item.ip for item in group}),
        ])

    line_rows = [["file", "line", "symbol", "sample_rows", "raw_count", "unique_ips"]]
    for (file, line, symbol), group in sorted(
        by_line.items(), key=lambda item: (-sum(s.raw_count for s in item[1]), item[0])
    ):
        line_rows.append([
            file, line, symbol, len(group), sum(item.raw_count for item in group), len({item.ip for item in group}),
        ])
    return symbol_rows, line_rows


def compare_ip_maps(
    actual_rows: list[dict[str, str]], expected_rows: list[dict[str, str]]
) -> dict[str, object]:
    """Compare stable IP, base/RVA, count, symbol and numeric source mapping."""
    def index(rows: list[dict[str, str]], name: str) -> dict[int, dict[str, str]]:
        mapped: dict[int, dict[str, str]] = {}
        for row_number, row in enumerate(rows, 2):
            try:
                ip = int(row["ip"], 16)
            except (KeyError, ValueError) as error:
                raise ExtractError(f"{name} row {row_number} has invalid IP") from error
            if ip in mapped:
                raise ExtractError(f"{name} contains duplicate IP {ip:#x}")
            mapped[ip] = row
        return mapped

    actual = index(actual_rows, "actual map")
    expected = index(expected_rows, "expected map")
    mismatches: list[dict[str, object]] = []
    for ip in sorted(set(actual) | set(expected)):
        if ip not in actual or ip not in expected:
            mismatches.append({"ip": f"0x{ip:016x}", "field": "presence"})
            continue
        for key in ("module_base", "rva", "raw_count", "symbol", "file", "line"):
            left = actual[ip].get(key, "")
            right = expected[ip].get(key, "")
            if key in ("module_base", "rva", "raw_count", "line"):
                try:
                    equal = int(left, 16) == int(right, 16) if key in ("module_base", "rva") else int(left) == int(right)
                except ValueError:
                    equal = False
            else:
                equal = left == right
            if not equal:
                mismatches.append({
                    "ip": f"0x{ip:016x}", "field": key, "actual": left, "expected": right,
                })
    archived_line_ok_mismatches = 0
    for ip in sorted(set(actual) & set(expected)):
        expected_line_ok = expected[ip].get("line_ok")
        if expected_line_ok not in (None, ""):
            actual_line = int(actual[ip].get("line", "0") or "0")
            actual_file = actual[ip].get("file", "")
            computed_line_ok = bool(actual_line > 0 and actual_file)
            if str(int(computed_line_ok)) != expected_line_ok:
                archived_line_ok_mismatches += 1
    return {
        "actual_unique_ips": len(actual),
        "expected_unique_ips": len(expected),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "ignored_archived_line_ok_mismatch_count": archived_line_ok_mismatches,
        "comparison_fields": ["module_base", "rva", "raw_count", "symbol", "file", "numeric line"],
    }


def _load_map(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "ip" not in reader.fieldnames:
            raise ExtractError(f"comparison map has no ip header: {path}")
        return [dict(row) for row in reader]


def _write_csv(path: Path, rows: Iterable[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows(rows)


def _prepare_outdir(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ExtractError(f"output path is not a directory: {path}")
        existing = list(path.iterdir())
        if existing:
            raise ExtractError(f"refusing to overwrite non-empty output directory: {path}")
    else:
        path.mkdir(parents=True)


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--etl", type=Path, help="ETL trace to dump and extract")
    sources.add_argument("--raw-csv", type=Path, help="archived xperf dumper rows for replay")
    parser.add_argument("--exe", required=True, type=Path, help="exact sampled executable")
    parser.add_argument("--pdb", required=True, type=Path, help="expected paired PDB")
    parser.add_argument("--expected-exe-sha256", required=True, help="required SHA-256 for --exe")
    parser.add_argument("--expected-pdb-sha256", required=True, help="required SHA-256 for --pdb")
    parser.add_argument("--outdir", required=True, type=Path, help="new or empty output directory")
    parser.add_argument("--pid", type=int, help="retain one target process instance")
    parser.add_argument("--xperf", default="xperf.exe", help="xperf executable (default: xperf.exe)")
    parser.add_argument("--timeout-seconds", type=int, default=600, help="per-xperf-command timeout")
    parser.add_argument("--image-base", help="known runtime base required with --raw-csv")
    parser.add_argument("--compare-map", type=Path, help="archived IP map for exact replay comparison")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-fA-F]{64}", args.expected_exe_sha256):
        parser.error("--expected-exe-sha256 must contain exactly 64 hexadecimal characters")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", args.expected_pdb_sha256):
        parser.error("--expected-pdb-sha256 must contain exactly 64 hexadecimal characters")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.pid is not None and args.pid <= 0:
        parser.error("--pid must be positive")
    if args.raw_csv is not None:
        if args.pid is None or args.image_base is None:
            parser.error("--raw-csv replay requires both --pid and --image-base")
        try:
            int(args.image_base, 0)
        except ValueError:
            parser.error("--image-base must be an integer such as 0x00007ff66d4b0000")
    elif args.image_base is not None:
        parser.error("--image-base is only valid with --raw-csv")
    if args.compare_map is not None and args.raw_csv is None:
        parser.error("--compare-map is only valid with --raw-csv replay")
    if args.compare_map is not None and args.pid is None:
        parser.error("--compare-map requires --pid so every expected IP has one process mapping")
    return args


def main() -> int:
    args = _cli()
    if not args.exe.is_file() or not args.pdb.is_file():
        raise ExtractError("--exe and --pdb must name existing files")
    actual_exe_sha = sha256(args.exe)
    actual_pdb_sha = sha256(args.pdb)
    if actual_exe_sha.casefold() != args.expected_exe_sha256.casefold():
        raise ExtractError(f"EXE SHA-256 mismatch: actual {actual_exe_sha}")
    if actual_pdb_sha.casefold() != args.expected_pdb_sha256.casefold():
        raise ExtractError(f"PDB SHA-256 mismatch: actual {actual_pdb_sha}")
    pe = read_pe_info(args.exe, args.pdb)
    if args.raw_csv is not None and not args.raw_csv.is_file():
        raise ExtractError(f"raw dumper CSV does not exist: {args.raw_csv}")
    if args.etl is not None and not args.etl.is_file():
        raise ExtractError(f"ETL does not exist: {args.etl}")
    _prepare_outdir(args.outdir)

    xperf_reports: dict[str, object] = {}
    if args.raw_csv is not None:
        assert args.pid is not None and args.image_base is not None
        fallback = ImageRange(args.pid, int(args.image_base, 0), pe.image_size, "caller-supplied replay base")
        with args.raw_csv.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
            parsed = parse_dumper_rows(stream, args.exe.name, args.pid, fallback)
        source_mode = "raw-csv-replay"
        xperf_dumper = {"exit_code": None, "not_run": "raw CSV replay"}
    else:
        assert args.etl is not None
        parsed, dumper_command, dumper_exit = _run_xperf_dumper(
            args.xperf, args.etl, args.exe.name, args.pid, args.timeout_seconds
        )
        source_mode = "etl"
        xperf_dumper = {"command": dumper_command, "exit_code": dumper_exit}
        xperf_reports = _write_xperf_reports(args.xperf, args.etl, args.outdir, args.timeout_seconds)
        if args.pid is not None:
            if args.pid not in parsed.images or not parsed.samples:
                raise ExtractError(f"no target-executable image or samples for requested PID {args.pid}")

    expected_size = pe.image_size
    for image in parsed.images.values():
        if image.size != expected_size:
            raise ExtractError(
                f"xperf image size {image.size:#x} for PID {image.pid} differs from PE SizeOfImage {expected_size:#x}"
            )
    if not parsed.samples or sum(sample.raw_count for sample in parsed.samples) <= 0:
        raise ExtractError("no retained target rows with nonzero direct count")

    symbols = resolve_symbols(args.exe, args.pdb, parsed.images, parsed.samples)
    samples_path = args.outdir / "samples.csv"
    ip_map_path = args.outdir / "ip-symbols.csv"
    other_path = args.outdir / "other-process-modules.csv"
    symbol_summary_path = args.outdir / "summary-by-symbol.csv"
    line_summary_path = args.outdir / "summary-by-source-line.csv"
    _write_csv(samples_path, _sample_csv_rows(parsed.samples, parsed.images, symbols))
    _write_csv(other_path, _other_module_csv_rows(parsed.other_module_samples))
    map_rows = _ip_map_rows(parsed.samples, parsed.images, symbols)
    _write_csv(ip_map_path, map_rows)
    symbol_rows, line_rows = _summary_rows(parsed.samples, parsed.images, symbols)
    _write_csv(symbol_summary_path, symbol_rows)
    _write_csv(line_summary_path, line_rows)

    comparison: dict[str, object] | None = None
    if args.compare_map is not None:
        comparison = compare_ip_maps(_load_map(ip_map_path), _load_map(args.compare_map))

    unique_symbols = {info.symbol for info in symbols.values() if info.symbol_ok}
    nonzero_line_rows = sum(
        sample.raw_count for sample in parsed.samples if symbols[(sample.pid, sample.ip)].line_ok
    )
    summary: dict[str, object] = {
        "source_mode": source_mode,
        "source_path": os.fspath(args.etl or args.raw_csv),
        "exe_path": os.fspath(args.exe),
        "exe_sha256": actual_exe_sha,
        "pdb_path": os.fspath(args.pdb),
        "pdb_sha256": actual_pdb_sha,
        "pdb_pairing": {
            "method": "PE RSDS GUID/age matched against PDB MSF information stream; DbgHelp loaded path checked",
            "embedded_pdb_name": pe.pdb_name,
            "guid": str(uuid.UUID(bytes_le=pe.pdb_guid)),
            "age": pe.pdb_age,
        },
        "pe": {
            "preferred_base": f"0x{pe.preferred_base:016x}",
            "size_of_image": f"0x{pe.image_size:x}",
        },
        "images": [
            {
                "pid": image.pid,
                "base": f"0x{image.base:016x}",
                "end_exclusive": f"0x{image.end:016x}",
                "size": f"0x{image.size:x}",
                "source": image.source,
            }
            for image in sorted(parsed.images.values(), key=lambda item: item.pid)
        ],
        "counts": {
            "dumper_sampled_profile_rows_all_processes": parsed.sampled_profile_rows,
            "target_process_sampled_profile_rows": parsed.target_process_rows,
            "pid_filtered_rows": parsed.pid_filtered_rows,
            "target_process_other_module_rows": parsed.other_module_rows,
            "retained_target_executable_rows": len(parsed.samples),
            "retained_direct_count": sum(sample.raw_count for sample in parsed.samples),
            "unique_ips": len({sample.ip for sample in parsed.samples}),
            "unique_pid_ip_pairs": len({(sample.pid, sample.ip) for sample in parsed.samples}),
            "resolved_unique_symbols": len(unique_symbols),
            "nonzero_source_line_direct_count": nonzero_line_rows,
            "nonzero_source_line_unique_ips": len({
                sample.ip for sample in parsed.samples if symbols[(sample.pid, sample.ip)].line_ok
            }),
            "nonzero_source_line_pid_ip_pairs": len({
                (sample.pid, sample.ip) for sample in parsed.samples if symbols[(sample.pid, sample.ip)].line_ok
            }),
        },
        "xperf": {
            "dumper": xperf_dumper,
            "reports": xperf_reports,
            "loss_or_warning_lines_from_dumper": parsed.diagnostics,
            "loss_or_warning_lines_from_reports": [
                {"action": action, "line": line}
                for action, report in xperf_reports.items()
                for line in report.get("diagnostic_lines", [])
            ],
            "loss_interpretation": "verbatim matching lines are recorded; an empty list is not a zero-loss assertion",
        },
        "outputs": {
            "samples_csv": os.fspath(samples_path),
            "unique_ip_map_csv": os.fspath(ip_map_path),
            "other_process_modules_csv": os.fspath(other_path),
            "summary_by_symbol_csv": os.fspath(symbol_summary_path),
            "summary_by_source_line_csv": os.fspath(line_summary_path),
        },
        "comparison": comparison,
    }
    summary_path = args.outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if comparison is not None and comparison["mismatch_count"]:
        raise ExtractError(
            f"archived map comparison found {comparison['mismatch_count']} mismatches; "
            f"see {summary_path}"
        )
    print(f"source_mode={source_mode}")
    print(f"exe_sha256={actual_exe_sha}")
    print(f"pdb_sha256={actual_pdb_sha}")
    print(f"sample_rows={len(parsed.samples)} direct_count={sum(sample.raw_count for sample in parsed.samples)}")
    print(f"unique_ips={summary['counts']['unique_ips']} nonzero_source_line_direct_count={nonzero_line_rows}")
    print(f"samples_csv={samples_path}")
    print(f"ip_symbols_csv={ip_map_path}")
    print(f"summary_json={summary_path}")
    if comparison is not None:
        print(
            "comparison="
            f"{comparison['actual_unique_ips']} actual IPs, "
            f"{comparison['mismatch_count']} mismatches, "
            f"{comparison['ignored_archived_line_ok_mismatch_count']} archived line_ok discrepancies ignored"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ExtractError, OSError, ValueError, struct.error) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
