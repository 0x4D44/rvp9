import pathlib,shlex,subprocess
p=pathlib.Path(__file__).parent
r='/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6'
n='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6'
w='/d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi'
adapters='shasum() { shift 2; command sha256sum "$@"; }; stat() { if [ "$1" = -f ] && [ "$2" = %z ]; then command stat -c %s "$3"; else command stat "$@"; fi; }; '
args=['source','scripts/qualify-production-encoder.sh','--resume','--phase','speed','--work-dir',r+'/production-speed','--pgo-manifest',r+'/pgo/runs/20261017T052417Z-1120/manifest.txt','--test-encoder',r+'/pgo/optimized/release/rvp9enc.exe','--source-dir',w+'/tests/corpus','--libvpx-source',w+'/third_party/libvpx']
for flag,name in [('--bdrate','bdrate'),('--bdrate-svc','bdrate-svc'),('--encode-headtohead','encode-headtohead'),('--realtime','realtime'),('--rvp9dec','rvp9dec')]:args += [flag,r+'/cargo-target/release/'+name+'.exe']
for flag,name in [('--vpxenc','vpxenc'),('--vpxdec','vpxdec'),('--temporal-anchor','vpx_temporal_svc_reference')]:args += [flag,w+'/build/libvpx/'+name+'.exe']
command='deltic timeout 86400 bash -c '+shlex.quote(adapters+' '.join(shlex.quote(a) for a in args))
s=f'''set -uo pipefail
exec </dev/null
cd {w}
export PATH="/c/Program Files/Git/bin:/c/Users/marti/.cargo/bin:$PATH"
export RUSTUP_TOOLCHAIN=1.95.0 CARGO_BUILD_JOBS=4
root={r}
logs="$root/performance-resume-logs"
owner=rvp9-resume-speed474-20261017
trap 'rc=$?; deltic quiet end --owner "$owner" > "$logs/quiet-end.log" 2>&1; exit "$rc"' EXIT
[ "$(git rev-parse HEAD)" = 474dfd99a673df85f750136f7552ac156b7a24e9 ] || exit 91
[ -z "$(git status --porcelain)" ] || exit 92
[ -f "$root/production-speed/manifest.txt" ] || exit 93
printf '%s\\n' {shlex.quote(command)} > "$logs/campaign.command.txt"
date -u +%FT%TZ > "$logs/started.txt"
deltic quiet start --reason rvp9-resume-speed474 --for 20 --wait --max 900 --owner "$owner" > "$logs/quiet-start.log" 2>&1
rc=$?; printf '%s\\n' "$rc" > "$logs/quiet-start.exit.txt"
[ "$rc" = 0 ] || exit "$rc"
deltic quiet status --json > "$logs/quiet-grant.json"
grep -q '"heavy_processes": 0' "$logs/quiet-grant.json" || {{ printf 'refusing: heavy processes still running after grant\\n' > "$logs/campaign.log"; exit 94; }}
{command} > "$logs/campaign.log" 2>&1
rc=$?
printf '%s\\n' "$rc" > "$logs/campaign.exit.txt"
date -u +%FT%TZ > "$logs/ended.txt"
{{ git rev-parse HEAD; git status --porcelain; }} > "$logs/final-source.txt"
printf 'canonical_speed_terminal_exit=%s\\n' "$rc"
exit "$rc"
'''
p.joinpath('campaign.sh').write_text(s)
subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15',str(p/'campaign.sh'),'marti@crucible:'+n+'/performance-resume-logs/campaign.sh'],check=True,stdin=subprocess.DEVNULL,timeout=60)
launch=f'''$ErrorActionPreference='Stop'
$env:PATH='C:/Program Files/Git/bin;C:/Users/marti/.cargo/bin;'+$env:PATH
$r='{n}'
$p=Start-Process 'C:/Program Files/Git/bin/bash.exe' -ArgumentList @('--noprofile','--norc',"$r/performance-resume-logs/campaign.sh") -PassThru -NoNewWindow
$p.Id | Set-Content "$r/performance-resume-logs/launcher.pid"
"native_root_pid=$($p.Id)"
if(!$p.WaitForExit(88000000)) {{ taskkill /PID $p.Id /T /F; deltic quiet end --owner rvp9-resume-speed474-20261017; throw 'bounded campaign launcher timeout' }}
$p.Refresh(); "launcher_completed=$($p.HasExited)"
'''
p.joinpath('launch.ps1').write_text(launch)
