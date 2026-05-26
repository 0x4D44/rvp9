#!/usr/bin/env python3
"""Resolve module-relative RVAs to function names via the Windows DbgHelp
API (reads the native Rust PDB, which samply's own parser fails to name).

Used by profile_hotspots.py. Standalone test:
  python symbolize_pdb.py <exe_path> 0x7d0d0 0xb6820 ...
"""
import ctypes as C
from ctypes import wintypes

MAX_SYM_NAME = 2000
SYMOPT_UNDNAME = 0x00000002
SYMOPT_DEFERRED_LOADS = 0x00000004
SYMOPT_LOAD_LINES = 0x00000010
SYMOPT_NO_PROMPTS = 0x00080000

dbghelp = C.WinDLL("dbghelp.dll")
kernel32 = C.WinDLL("kernel32.dll")


class SYMBOL_INFO(C.Structure):
    _fields_ = [
        ("SizeOfStruct", wintypes.ULONG),
        ("TypeIndex", wintypes.ULONG),
        ("Reserved", C.c_ulonglong * 2),
        ("Index", wintypes.ULONG),
        ("Size", wintypes.ULONG),
        ("ModBase", C.c_ulonglong),
        ("Flags", wintypes.ULONG),
        ("Value", C.c_ulonglong),
        ("Address", C.c_ulonglong),
        ("Register", wintypes.ULONG),
        ("Scope", wintypes.ULONG),
        ("Tag", wintypes.ULONG),
        ("NameLen", wintypes.ULONG),
        ("MaxNameLen", wintypes.ULONG),
        ("Name", C.c_char * (MAX_SYM_NAME + 1)),
    ]


# Fixed part size = offsetof(Name)
_NAME_OFFSET = SYMBOL_INFO.Name.offset

dbghelp.SymInitialize.argtypes = [wintypes.HANDLE, wintypes.LPCSTR, wintypes.BOOL]
dbghelp.SymInitialize.restype = wintypes.BOOL
dbghelp.SymSetOptions.argtypes = [wintypes.DWORD]
dbghelp.SymSetOptions.restype = wintypes.DWORD
dbghelp.SymLoadModuleExW.argtypes = [
    wintypes.HANDLE, wintypes.HANDLE, wintypes.LPCWSTR, wintypes.LPCWSTR,
    C.c_ulonglong, wintypes.DWORD, C.c_void_p, wintypes.DWORD,
]
dbghelp.SymLoadModuleExW.restype = C.c_ulonglong
dbghelp.SymFromAddr.argtypes = [
    wintypes.HANDLE, C.c_ulonglong, C.POINTER(C.c_ulonglong), C.POINTER(SYMBOL_INFO)
]
dbghelp.SymFromAddr.restype = wintypes.BOOL


# Windows requires SizeOfStruct = sizeof(SYMBOL_INFO) = 88 (fixed part,
# 8-byte aligned). The ctypes field offset (84) is NOT accepted.
SYMBOL_INFO_SIZE = 88


class Resolver:
    """Resolve module-relative RVAs to names. Caches per-RVA lookups."""

    def __init__(self, exe_path, base=0x140000000):
        import os

        self.h = wintypes.HANDLE(0xABCD)  # arbitrary pseudo process handle
        self.base = base
        self._cache = {}
        dbghelp.SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_NO_PROMPTS)
        # Search path: the exe's dir plus its deps/ sibling (Rust writes the
        # PDB into target/<profile>/deps/ next to the hashed artifact).
        d = os.path.dirname(os.path.abspath(exe_path))
        search = ";".join([d, os.path.join(d, "deps"), os.path.dirname(d)])
        if not dbghelp.SymInitialize(self.h, search.encode(), False):
            raise OSError(f"SymInitialize failed: {kernel32.GetLastError()}")
        got = dbghelp.SymLoadModuleExW(self.h, None, exe_path, None, base, 0, None, 0)
        self.base = got or base

    def name(self, rva):
        if rva in self._cache:
            return self._cache[rva]
        sym = SYMBOL_INFO()
        sym.SizeOfStruct = SYMBOL_INFO_SIZE
        sym.MaxNameLen = MAX_SYM_NAME
        disp = C.c_ulonglong(0)
        ok = dbghelp.SymFromAddr(self.h, self.base + rva, C.byref(disp), C.byref(sym))
        out = sym.Name.decode("utf-8", "replace") if ok else None
        self._cache[rva] = out
        return out


if __name__ == "__main__":
    import sys
    exe = sys.argv[1]
    r = Resolver(exe)
    for a in sys.argv[2:]:
        rva = int(a, 16) if a.startswith("0x") else int(a)
        print(f"  0x{rva:x} -> {r.name(rva)}")
