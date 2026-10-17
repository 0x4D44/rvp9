import subprocess,pathlib,shlex,time,sys
p=pathlib.Path(__file__).parent
host='marti@crucible'
root='/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6'
native='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6'
work='/d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi'
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15',host,'"C:/Program Files/Git/bin/bash.exe" --noprofile --norc -s']
def copy(remote,name):
 subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15',host+':'+remote,str(p/name)],check=True,stdin=subprocess.DEVNULL,timeout=60)
def run(name,bound,cmd,artifacts=()):
 script=f'''set -uo pipefail
cd {work}
export PATH="/c/Program Files/Git/bin:/c/Users/marti/.cargo/bin:$PATH"
export RUSTUP_TOOLCHAIN=1.95.0 CARGO_BUILD_JOBS=4
export CARGO_TARGET_DIR={root}/cargo-target
root={root}
[ "$(git rev-parse HEAD)" = 474dfd99a673df85f750136f7552ac156b7a24e9 ] || exit 91
[ -z "$(git status --porcelain)" ] || exit 92
printf '%s\\n' {shlex.quote(cmd)} > "$root/logs/{name}.command.txt"
printf 'START {name} bash_pid=%s utc=%s\\n' "$$" "$(date -u +%FT%TZ)"
deltic timeout {bound} {cmd} </dev/null > "$root/logs/{name}.log" 2>&1
s=$?
printf '%s\\n' "$s" > "$root/logs/{name}.exit.txt"
printf 'END {name} exit=%s utc=%s\\n' "$s" "$(date -u +%FT%TZ)"
exit "$s"
'''
 p.joinpath(name+'.runner.sh').write_text(script)
 with p.joinpath(name+'.transport.log').open('wb') as log:
  proc=subprocess.Popen(ssh,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  print(f'TRACKED phase={name} local_ssh_pid={proc.pid}',flush=True)
  proc.stdin.write(script.encode());proc.stdin.close()
  for line in proc.stdout:
   log.write(line);log.flush();print(line.decode(errors='replace').rstrip(),flush=True)
  code=proc.wait(timeout=bound+90)
 for ext in ['log','exit.txt','command.txt']:
  copy(native+'/logs/'+name+'.'+ext,name+'.'+ext)
 for remote,name2 in artifacts:copy(remote,name2)
 print(f'PRESERVED phase={name} ssh_exit={code}',flush=True)
 if code:sys.exit(code)
run('pgo-plan',60,'bash scripts/build-encoder-pgo.sh --print-training-plan')
run('pgo-input-identity',300,'bash -c '+shlex.quote('rustc -vV; cargo -vV; sha256sum scripts/build-encoder-pgo.sh tests/corpus/in_to_tree_420_720p50.y4m tests/corpus/ducks_take_off_420_720p50.y4m tests/corpus/old_town_cross_420_720p50.y4m; "$(rustc --print sysroot)/lib/rustlib/x86_64-pc-windows-msvc/bin/llvm-profdata.exe" --version'))
command='[ ! -e '+root+'/pgo ] || exit 93; shasum() { shift 2; command sha256sum "$@"; }; source scripts/build-encoder-pgo.sh --training-dir '+work+'/tests/corpus --build-root '+root+'/pgo'
run('pgo-build',14400,'bash -c '+shlex.quote(command))
import re
log=p.joinpath('pgo-build.log').read_text()
manifest=re.search(r'^Training manifest: (.+)$',log,re.M).group(1)
manifest_native='C:/'+manifest[3:] if manifest.startswith('/c/') else manifest
copy(manifest_native,'pgo-manifest.txt')
p.joinpath('pgo-run-path.txt').write_text(manifest_native+'\n')
print('PGO_COMPLETE_AND_MANIFEST_PRESERVED '+manifest_native,flush=True)
