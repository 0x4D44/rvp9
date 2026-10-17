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
run('identity',60,'bash -c '+shlex.quote('date -u; uname -a; git rev-parse HEAD; git status --porcelain; rustc -vV; cargo -vV; echo CARGO_BUILD_JOBS=$CARGO_BUILD_JOBS; echo CARGO_TARGET_DIR=$CARGO_TARGET_DIR; git submodule status; sha256sum build/libvpx/vpxenc.exe build/libvpx/vpxdec.exe build/libvpx/vpx_temporal_svc_reference.exe; find tests/conformance/vectors -maxdepth 1 -type f | wc -l'))
run('build',1800,'cargo build --release --locked -p rvp9-harness --bin conformance -p rvp9-cli --bin rvp9enc --bin rvp9dec -p rvp9-bench --bin bdrate --bin bdrate-svc --bin encode-headtohead --bin realtime')
for scalar in [False,True]:
 for threads in [1,2]:
  name='conformance-'+('scalar' if scalar else 'native')+'-'+str(threads)
  prefix='env RVP9_FORCE_SCALAR=1 ' if scalar else ''
  run(name,1800,prefix+root+f'/cargo-target/release/conformance.exe --vectors-dir tests/conformance/vectors --official --strict --threads {threads}')
run('scalar-core',1800,'cargo test --locked -p rvp9-core --lib force_scalar_subprocess -- --nocapture')
run('scalar-laplacian',1800,'cargo test --locked -p rvp9-encoder --lib force_scalar_laplacian_dispatch -- --nocapture')
run('avx2-subpel',1800,'cargo test --locked -p rvp9-core --lib dsp::x86_64::subpel_variance_avx2::tests -- --nocapture')
run('avx2-variance',1800,'cargo test --locked -p rvp9-core --lib dsp::x86_64::variance_avx2::tests -- --nocapture')
run('laplacian-rows',1800,'cargo test --locked -p rvp9-encoder --lib hf_laplacian_rows_match_scalar_for_steps_spans_and_tight_tails -- --nocapture')
probe='''cat > "$root/native-features.rs" <<'RS'
fn main() { println!("avx2={} sse2={} ssse3={} sse41={} bmi2={} fma={}", std::is_x86_feature_detected!("avx2"), std::is_x86_feature_detected!("sse2"), std::is_x86_feature_detected!("ssse3"), std::is_x86_feature_detected!("sse4.1"), std::is_x86_feature_detected!("bmi2"), std::is_x86_feature_detected!("fma")); }
RS
rustc --crate-name replay_features "$root/native-features.rs" -o "$root/native-features.exe" && "$root/native-features.exe"'''
run('native-features',60,'bash -c '+shlex.quote('root='+root+'\n'+probe))
adapter='shasum() { shift 2; command sha256sum "$@"; }; stat() { if [ "$1" = -f ] && [ "$2" = %z ]; then command stat -c %s "$3"; else command stat "$@"; fi; }; '
alpha=f'export RVP9_ALPHA_WORK_DIR={root}/alpha32 RVP9_ALPHA_VPXDEC={work}/build/libvpx/vpxdec.exe; '+adapter+'source scripts/qualify-alpha-encoder.sh'
run('alpha32',14400,'bash -c '+shlex.quote(alpha),[(native+'/alpha32/qualification.txt','alpha32-qualification.txt')])
reverse=f'export RVP9_REFERENCE_DECODER_WORK_DIR={root}/reverse24 RVP9_REFERENCE_DECODER_VPXENC={work}/build/libvpx/vpxenc.exe RVP9_REFERENCE_DECODER_VPXDEC={work}/build/libvpx/vpxdec.exe; '+adapter+'source scripts/qualify-reference-decoder.sh'
run('reverse24',3600,'bash -c '+shlex.quote(reverse),[(native+'/reverse24/qualification.txt','reverse24-qualification.txt')])
print('CORRECTNESS_COMPLETE; PGO not started',flush=True)
