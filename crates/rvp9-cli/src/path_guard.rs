//! Protect active CLI path roles before opening and before truncating outputs.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

#[cfg(unix)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FileId(u64, u64);

#[cfg(windows)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FileId(u64, [u8; 16]);

#[cfg(not(any(unix, windows)))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FileId;

#[derive(Default)]
struct Identity {
    id: Option<FileId>,
    nonregular: bool,
}

struct Role {
    name: &'static str,
    path: PathBuf,
    identity: Identity,
}

pub(crate) struct PathGuard {
    roles: Vec<Role>,
}

impl PathGuard {
    /// Capture input identity and reject aliases that already resolve.
    pub(crate) fn new(paths: &[(&'static str, &Path)]) -> Result<Self, String> {
        let mut roles = Vec::with_capacity(paths.len());
        for &(name, path) in paths {
            let identity = path_identity(path)
                .map_err(|error| format!("checking {name} {}: {error}", path.display()))?;
            roles.push(Role {
                name,
                path: path.to_owned(),
                identity,
            });
        }
        for (index, role) in roles.iter().enumerate() {
            for other in &roles[..index] {
                if same_id(role.identity.id, other.identity.id)
                    || (role.path == other.path
                        && !role.identity.nonregular
                        && !other.identity.nonregular)
                {
                    return Err(alias_error(role, other));
                }
            }
        }
        Ok(Self { roles })
    }

    /// Open without truncation, check the actual handle against every other
    /// role, and only then truncate that handle. Re-probing paths catches
    /// future case/Unicode aliases and dangling symlinks after creation.
    pub(crate) fn create(&mut self, name: &str) -> Result<File, String> {
        let index = self
            .roles
            .iter()
            .position(|role| role.name == name)
            .ok_or_else(|| format!("inactive output role {name}"))?;
        let role = &self.roles[index];
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(&role.path)
            .map_err(|error| format!("creating {}: {error}", role.path.display()))?;
        let identity = file_identity(&file)
            .map_err(|error| format!("checking {name} {}: {error}", role.path.display()))?;
        for (other_index, other) in self.roles.iter().enumerate() {
            if other_index == index {
                continue;
            }
            let current = path_identity(&other.path).map_err(|error| {
                format!("checking {} {}: {error}", other.name, other.path.display())
            })?;
            if same_id(identity.id, other.identity.id) || same_id(identity.id, current.id) {
                return Err(alias_error(role, other));
            }
        }
        if !identity.nonregular {
            file.set_len(0)
                .map_err(|error| format!("truncating {}: {error}", role.path.display()))?;
        }
        self.roles[index].identity = identity;
        Ok(file)
    }
}

fn same_id(left: Option<FileId>, right: Option<FileId>) -> bool {
    left.is_some() && left == right
}

fn alias_error(left: &Role, right: &Role) -> String {
    format!(
        "{} {} and {} {} alias the same file; refusing to overwrite",
        left.name,
        left.path.display(),
        right.name,
        right.path.display()
    )
}

#[cfg(not(windows))]
fn path_identity(path: &Path) -> io::Result<Identity> {
    let metadata = match std::fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Identity::default()),
        Err(error) => return Err(error),
    };
    metadata_identity(&metadata)
}

#[cfg(windows)]
fn path_identity(path: &Path) -> io::Result<Identity> {
    use std::os::windows::fs::OpenOptionsExt;
    // Attributes-only access follows reparse points without read-data access.
    let file = match OpenOptions::new()
        .access_mode(0)
        .custom_flags(0x0200_0000)
        .open(path)
    {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Identity::default()),
        Err(error) => return Err(error),
    };
    file_identity(&file)
}

#[cfg(not(windows))]
fn metadata_identity(metadata: &std::fs::Metadata) -> io::Result<Identity> {
    if !metadata.is_file() {
        return Ok(Identity {
            id: None,
            nonregular: true,
        });
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok(Identity {
            id: Some(FileId(metadata.dev(), metadata.ino())),
            nonregular: false,
        })
    }
    #[cfg(not(unix))]
    {
        Err(io::Error::other(
            "regular-file identity is unavailable on this platform",
        ))
    }
}

fn file_identity(file: &File) -> io::Result<Identity> {
    #[cfg(windows)]
    {
        windows_identity(file)
    }
    #[cfg(not(windows))]
    {
        metadata_identity(&file.metadata()?)
    }
}

#[cfg(windows)]
fn windows_identity(file: &File) -> io::Result<Identity> {
    use std::ffi::c_void;
    use std::os::windows::io::AsRawHandle;

    #[repr(C)]
    struct FileIdInfo {
        volume_serial_number: u64,
        file_id: [u8; 16],
    }
    #[link(name = "kernel32")]
    // SAFETY: These declarations match kernel32's documented system-ABI
    // signatures. HANDLE is pointer-sized, DWORD is u32, and FileIdInfo uses
    // the C layout of Windows FILE_ID_INFO.
    unsafe extern "system" {
        fn GetFileType(handle: *mut c_void) -> u32;
        fn GetFileInformationByHandleEx(
            handle: *mut c_void,
            class: i32,
            information: *mut c_void,
            size: u32,
        ) -> i32;
    }
    // SAFETY: File owns a live handle for the duration of this query.
    let kind = unsafe { GetFileType(file.as_raw_handle()) };
    if kind == 0 {
        return Err(io::Error::last_os_error());
    }
    // FILE_TYPE_DISK = 1. NUL, consoles and pipes have no regular-file ID.
    if kind != 1 || !file.metadata()?.is_file() {
        return Ok(Identity {
            id: None,
            nonregular: true,
        });
    }
    let mut information = FileIdInfo {
        volume_serial_number: 0,
        file_id: [0; 16],
    };
    // SAFETY: the live File owns the handle; class 18 is FileIdInfo, whose
    // initialized repr(C) buffer and exact size are passed for this call only.
    let success = unsafe {
        GetFileInformationByHandleEx(
            file.as_raw_handle(),
            18,
            (&mut information as *mut FileIdInfo).cast(),
            std::mem::size_of::<FileIdInfo>() as u32,
        )
    };
    if success == 0 {
        return Err(io::Error::last_os_error());
    }
    if information.file_id == [0; 16] {
        return Err(io::Error::other(
            "regular file has no reliable file identity",
        ));
    }
    Ok(Identity {
        id: Some(FileId(
            information.volume_serial_number,
            information.file_id,
        )),
        nonregular: false,
    })
}
