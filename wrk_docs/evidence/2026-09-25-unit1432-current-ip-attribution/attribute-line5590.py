import csv
import ctypes as C
import ctypes.wintypes as W
import hashlib
import os
import re
import subprocess

SCRATCH = r'C:\Users\marti\AppData\Local\Temp\deltic-scratch\unit1432-line5590-disasm-e8351621'
MAP = os.path.join(SCRATCH, 'ip-lines.csv')
OUT_ROWS = os.path.join(SCRATCH, 'line5590-ip-rvas.csv')
OUT_MAP = os.path.join(SCRATCH, 'line5590-instruction-attribution.csv')
OUT_DISASM = os.path.join(SCRATCH, 'line5590-symbol-disassembly.txt')
EXE = r'D:\worktrees\rvp9-2\20260925-TSK-CDX@CRUCIBLE-unit1421-cpu-profile\.scratch\target\release\rvp9enc.exe'
PDB = r'D:\worktrees\rvp9-2\20260925-TSK-CDX@CRUCIBLE-unit1421-cpu-profile\.scratch\target\release\rvp9enc.pdb'
OBJDUMP = r'C:\Strawberry\c\bin\objdump.exe'
EXPECTED_MAP_SHA = 'af8396c8e0bed426b76c0d45baa4fc54d7f16fa143901359478aa1b355bf7eae'
EXPECTED_EXE_SHA = 'c371d472690baba94dc1374b9ce4b65dddd3f3ac2381feca362efa25e55f0d98'
EXPECTED_PDB_SHA = 'a3995992a6eb0ce343fed14f01830737ce577b708dde8379d6913902c7ae67fc'
TARGET_SYMBOL = 'rvp9_core::mode_decision::inter::reconstruct_via_round_trip'
TARGET_SOURCE_SUFFIX = r'crates\rvp9-core\src\mode_decision\inter.rs'
TARGET_LINE = '5590'
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
HANDLE = W.HANDLE(0xABCD)
MAX_SYM_NAME = 2000


class SYMBOL_INFO(C.Structure):
    _fields_ = [
        ('SizeOfStruct', W.ULONG), ('TypeIndex', W.ULONG),
        ('Reserved', C.c_ulonglong * 2), ('Index', W.ULONG), ('Size', W.ULONG),
        ('ModBase', C.c_ulonglong), ('Flags', W.ULONG), ('Value', C.c_ulonglong),
        ('Address', C.c_ulonglong), ('Register', W.ULONG), ('Scope', W.ULONG),
        ('Tag', W.ULONG), ('NameLen', W.ULONG), ('MaxNameLen', W.ULONG),
        ('Name', C.c_char * (MAX_SYM_NAME + 1)),
    ]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


if sha256(MAP) != EXPECTED_MAP_SHA:
    raise RuntimeError('archived IP map hash mismatch')
exe_sha, pdb_sha = sha256(EXE), sha256(PDB)
if exe_sha != EXPECTED_EXE_SHA or pdb_sha != EXPECTED_PDB_SHA:
    raise RuntimeError(f'EXE/PDB hash mismatch: {exe_sha} {pdb_sha}')
for path in (OUT_ROWS, OUT_MAP, OUT_DISASM):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite {path}')

selected = []
with open(MAP, 'r', encoding='utf-8', newline='') as stream:
    for row in csv.DictReader(stream):
        source = row['file'].replace('/', '\\')
        if not source.lower().endswith(TARGET_SOURCE_SUFFIX.lower()) or row['line'] != TARGET_LINE:
            continue
        if TARGET_SYMBOL not in row['symbol']:
            raise RuntimeError('line 5590 row has unexpected PDB symbol: ' + row['symbol'])
        selected.append({
            'ip': int(row['ip'], 16),
            'base': int(row['module_base'], 16),
            'rva': int(row['rva'], 16),
            'raw_count': int(row['raw_count']),
            'symbol': row['symbol'],
        })
if not selected or sum(row['raw_count'] for row in selected) != 142:
    raise RuntimeError(f'expected 142 mapped counts at line 5590; got {sum(row["raw_count"] for row in selected)}')
bases = {row['base'] for row in selected}
if len(bases) != 1:
    raise RuntimeError(f'multiple image bases in selected rows: {sorted(hex(x) for x in bases)}')
module_base = next(iter(bases))
if any(row['ip'] - module_base != row['rva'] for row in selected):
    raise RuntimeError('stored RVA does not equal IP - image base')

dh = C.WinDLL('dbghelp.dll', use_last_error=True)
dh.SymSetOptions.argtypes = [W.DWORD]
dh.SymSetOptions.restype = W.DWORD
dh.SymInitialize.argtypes = [W.HANDLE, W.LPCSTR, W.BOOL]
dh.SymInitialize.restype = W.BOOL
dh.SymLoadModuleExW.argtypes = [W.HANDLE, W.HANDLE, W.LPCWSTR, W.LPCWSTR, C.c_ulonglong, W.DWORD, C.c_void_p, W.DWORD]
dh.SymLoadModuleExW.restype = C.c_ulonglong
dh.SymFromAddr.argtypes = [W.HANDLE, C.c_ulonglong, C.POINTER(C.c_ulonglong), C.POINTER(SYMBOL_INFO)]
dh.SymFromAddr.restype = W.BOOL
dh.SymCleanup.argtypes = [W.HANDLE]
dh.SymCleanup.restype = W.BOOL
dh.SymSetOptions(0x2 | 0x10 | 0x80000)
if not dh.SymInitialize(HANDLE, os.path.dirname(EXE).encode('mbcs'), False):
    raise OSError('SymInitialize failed: ' + str(C.get_last_error()))

try:
    image_size = 0x001DA000
    loaded_base = dh.SymLoadModuleExW(HANDLE, None, EXE, None, module_base, image_size, None, 0)
    if loaded_base != module_base:
        raise RuntimeError(f'DbgHelp loaded base {loaded_base:#x}, expected {module_base:#x}')
    funcs = {}
    for row in selected:
        info = SYMBOL_INFO()
        info.SizeOfStruct = 88
        info.MaxNameLen = MAX_SYM_NAME
        displacement = C.c_ulonglong(0)
        if not dh.SymFromAddr(HANDLE, row['ip'], C.byref(displacement), C.byref(info)):
            raise RuntimeError(f'SymFromAddr failed at {row["ip"]:#x}: {C.get_last_error()}')
        name = bytes(info.Name).split(b'\0', 1)[0].decode('utf-8', 'replace')
        if name != TARGET_SYMBOL:
            raise RuntimeError(f'IP {row["ip"]:#x} resolved outside target symbol: {name}')
        start, size = int(info.Address), int(info.Size)
        if size <= 0 or not (start <= row['ip'] < start + size):
            raise RuntimeError(f'IP {row["ip"]:#x} lacks validated function range: start={start:#x} size={size:#x}')
        funcs[(name, start, size)] = funcs.get((name, start, size), 0) + row['raw_count']
finally:
    dh.SymCleanup(HANDLE)

if len(funcs) != 1:
    raise RuntimeError(f'line 5590 samples resolve to multiple symbol ranges: {funcs!r}')
(function_name, function_start, function_size), validated_count = next(iter(funcs.items()))
if validated_count != 142:
    raise RuntimeError(f'function-range validation covered {validated_count}, not 142 counts')
function_rva = function_start - module_base
function_end_rva = function_rva + function_size
if function_rva < 0 or function_end_rva > image_size:
    raise RuntimeError(f'function symbol extends outside executable image: {function_rva:#x}..{function_end_rva:#x}')

info_run = subprocess.run([OBJDUMP, '-p', EXE], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, encoding='mbcs', errors='replace', timeout=10)
if info_run.returncode != 0:
    raise RuntimeError('objdump -p failed: ' + info_run.stderr[-1000:])
image_base_match = re.search(r'(?im)^\s*ImageBase\s+([0-9a-f]+)', info_run.stdout)
if not image_base_match:
    raise RuntimeError('objdump -p did not report PE ImageBase')
preferred_base = int(image_base_match.group(1), 16)
dis_start = preferred_base + function_rva
dis_end = dis_start + function_size
dis_cmd = [OBJDUMP, '-d', '-Mintel', f'--start-address=0x{dis_start:x}', f'--stop-address=0x{dis_end:x}', EXE]
dis_run = subprocess.run(dis_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, encoding='mbcs', errors='replace', timeout=30)
if dis_run.returncode != 0:
    raise RuntimeError('objdump function-range disassembly failed: ' + dis_run.stderr[-1000:])
if len(dis_run.stdout.encode('mbcs', errors='replace')) > MAX_OUTPUT_BYTES:
    raise RuntimeError('bounded symbol disassembly exceeded 16 MiB output cap')
with open(OUT_DISASM, 'w', encoding='utf-8', newline='') as stream:
    stream.write('command=' + subprocess.list2cmdline(dis_cmd) + '\n')
    stream.write(f'function={function_name} start={function_start:#x} size={function_size:#x} preferred_range={dis_start:#x}..{dis_end:#x}\n')
    stream.write(dis_run.stdout)

inst_re = re.compile(r'^\s*([0-9a-fA-F]+):\s+((?:[0-9a-fA-F]{2}\s+)+)\s*([a-zA-Z][a-zA-Z0-9_.]*)\s*(.*?)\s*$')
instructions = []
for line in dis_run.stdout.splitlines():
    match = inst_re.match(line)
    if not match:
        continue
    address = int(match.group(1), 16)
    byte_values = match.group(2).split()
    instructions.append((address, len(byte_values), ' '.join(byte_values), match.group(3).lower(), match.group(4).strip()))
if not instructions:
    raise RuntimeError('objdump produced no parseable instructions in validated symbol range')

def category(mnemonic, operands):
    m = mnemonic.lower()
    o = operands.lower()
    if m.startswith('call'):
        return 'call'
    if m.startswith(('j', 'loop')) or m in ('ret', 'retf', 'iret', 'iretd', 'iretq'):
        return 'branch'
    if m.startswith(('phadd', 'hadd', 'psadbw', 'vpsadbw', 'pmadd', 'vpmadd')):
        return 'reduction'
    if m.startswith(('mov', 'vmov')) and '[' in o:
        left, _, right = o.partition(',')
        return 'store' if '[' in left else 'load'
    if '[' in o and m.startswith(('set', 'cmp', 'test')):
        return 'compare/load'
    if m.startswith(('add', 'sub', 'mul', 'imul', 'div', 'idiv', 'inc', 'dec', 'neg', 'and', 'or', 'xor', 'shl', 'shr', 'sar', 'sal', 'padd', 'psub', 'pmul', 'psll', 'psrl', 'psra', 'pand', 'por', 'pxor', 'punpck', 'pack', 'pshuf', 'unpck', 'vadd', 'vsub', 'vmul', 'vpmul', 'vpadd', 'vpsub', 'vpsll', 'vpsrl', 'vpsra', 'vpand', 'vpor', 'vpxor', 'vpunpck', 'vpack', 'vpshuf')):
        return 'arithmetic'
    if '[' in o:
        return 'memory-other'
    if m.startswith(('cmp', 'test', 'set', 'cmov')):
        return 'compare/control'
    return 'other'

weighted = []
not_decoded = []
for row in selected:
    target = preferred_base + row['rva']
    found = next((ins for ins in instructions if ins[0] <= target < ins[0] + ins[1]), None)
    if found is None:
        not_decoded.append(row)
        continue
    address, size, opcode, mnemonic, operands = found
    weighted.append((row['rva'], row['raw_count'], target, address, size, opcode, mnemonic, operands, category(mnemonic, operands), target == address))
if not_decoded:
    raise RuntimeError('could not locate target IPs in exact function disassembly: ' + ','.join(hex(r['ip']) for r in not_decoded[:10]))
if sum(row[1] for row in weighted) != 142:
    raise RuntimeError(f'disassembly mapped {sum(row[1] for row in weighted)} counts, expected 142')

with open(OUT_ROWS, 'w', encoding='utf-8', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['ip', 'rva', 'raw_count', 'symbol', 'function_start', 'function_size'])
    for row in selected:
        writer.writerow([f'0x{row["ip"]:016x}', f'0x{row["rva"]:x}', row['raw_count'], function_name, f'0x{function_start:016x}', f'0x{function_size:x}'])
with open(OUT_MAP, 'w', encoding='utf-8', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['rva', 'raw_count', 'preferred_vma', 'instruction_address', 'instruction_bytes', 'mnemonic', 'operands', 'category', 'exact_ip'])
    for row in sorted(weighted):
        writer.writerow([f'0x{row[0]:x}', row[1], f'0x{row[2]:x}', f'0x{row[3]:x}', row[5], row[6], row[7], row[8], int(row[9])])

groups = {}
instruction_groups = {}
for row in weighted:
    groups[row[8]] = groups.get(row[8], 0) + row[1]
    key = (row[6], row[7])
    instruction_groups[key] = instruction_groups.get(key, 0) + row[1]
top_instructions = sorted(instruction_groups.items(), key=lambda item: (-item[1], item[0]))[:20]
print('input_map_sha256=' + sha256(MAP))
print('exe_sha256=' + sha256(EXE) + ' pdb_sha256=' + sha256(PDB))
print('objdump=' + OBJDUMP)
print('objdump_pe_image_base=' + hex(preferred_base))
print('target_symbol=' + function_name + ' runtime_start=' + hex(function_start) + ' size=' + hex(function_size) + ' rva_range=' + hex(function_rva) + '..' + hex(function_end_rva))
print('dbghelp_load_base=' + hex(loaded_base))
print('selected_unique_ips=' + str(len(selected)) + ' raw_counts=' + str(sum(row['raw_count'] for row in selected)) + ' function_range_validated_ips=' + str(len(weighted)))
print('mapped_instruction_ips=' + str(len(weighted)) + ' exact_instruction_start=' + str(sum(1 for row in weighted if row[9])) + ' interior_or_nearest=' + str(sum(1 for row in weighted if not row[9])))
print('instruction_category_counts=' + repr(dict(sorted(groups.items()))))
print('disasm_command=' + subprocess.list2cmdline(dis_cmd))
print('disassembly_stdout_bytes=' + str(len(dis_run.stdout.encode('mbcs', errors='replace'))))
print('instruction_csv=' + OUT_MAP + ' sha256=' + sha256(OUT_MAP))
print('top_instructions_begin')
for (mnemonic, operands), count in top_instructions:
    print(f'  count={count} {mnemonic} {operands}')
print('top_instructions_end')
print('instruction_rows_begin')
for row in sorted(weighted, key=lambda item: (-item[1], item[0]))[:30]:
    print(f'  count={row[1]} rva={row[0]:#x} {row[5]} {row[6]} {row[7]} category={row[8]} exact={row[9]}')
print('instruction_rows_end')
