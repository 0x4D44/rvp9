# Windows file identity FFI reference check

Read-only source/reference check on 2026-10-17 for candidate `ebca64b0`, source file `crates/rvp9-cli/src/path_guard.rs`. No native ABI/race execution or product modification is claimed.

Microsoft documents GetFileInformationByHandleEx as accepting a live file handle, an information-class enum, an output buffer and its byte length; zero indicates failure. The class/structure table gives FileIdInfo as `0x12` (decimal18). The current Rust declaration and call use those parameter shapes and pass the initialized buffer size. [GetFileInformationByHandleEx](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-getfileinformationbyhandleex).

The documented FILE_ID_INFO fields are a 64-bit volume serial followed by FILE_ID_128, which contains16 identifier bytes. This matches the Rust repr(C) u64-plus-byte-array declaration. Microsoft describes comparing the serial and identifier together to identify the same file; the current equality key contains both. This is a source-level correspondence, not a measured layout or filesystem-race proof. [FILE_ID_INFO](https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-file_id_info), [FILE_ID_128](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-file_id_128).

The File borrow retains handle ownership through both synchronous calls, and truncation occurs on the opened handle after identity checks. Existing tests and actual Windows ABI/error/alias behavior still need candidate runtime evidence. No additional confirmed defect follows from this bounded check.
