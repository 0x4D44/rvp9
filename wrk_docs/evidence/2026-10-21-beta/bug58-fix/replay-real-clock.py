"""Replay retained genuine beta encodes through bdrate; no encoder rerun or input mutation."""
import argparse,csv,hashlib,json,pathlib,shlex,subprocess,time
parser=argparse.ArgumentParser()
parser.add_argument('--output-root',type=pathlib.Path)
parser.add_argument('--bdrate',type=pathlib.Path)
args=parser.parse_args()
task_root=pathlib.Path(__file__).resolve().parent.parent
root=(args.output_root or task_root/'.scratch').resolve()
root.mkdir(parents=True,exist_ok=True)
old=pathlib.Path('/private/var/folders/bl/xncqcmfj4cj3b8z7j0sy78rr0000gn/T/deltic-scratch/rvp9-beta-final-macos-81301c61/quality')
binary=(args.bdrate or task_root/'target/debug/bdrate').resolve()
decoder=pathlib.Path('/Users/md/worktrees/rvp9/20261015-TSK-HUM-prepare-final-release-readiness-evidence/build/libvpx/vpxdec')
results=[]
for scene in ['in_to_tree','ducks_take_off']:
 out=root/('real-replay-'+scene);out.mkdir()
 original=old/'quality/rows'/('aq3-cbr-720p30-'+scene+'-cpu6')/'streams'
 for role in ['anchor','test']:
  wrapper=out/('replay-'+role)
  wrapper.write_text('#!/bin/sh\nset -eu\nbitrate=\noutput=\nwhile [ "$#" -gt 0 ]; do\ncase "$1" in --target-bitrate=*) bitrate=${1#--target-bitrate=} ;; -o) shift; output=$1 ;; esac\nshift\ndone\ncp '+shlex.quote(str(original/role))+'_'+'"$bitrate"'+'k.ivf "$output"\n')
  wrapper.chmod(0o700)
 cmd=['deltic','timeout','600',str(binary),'--anchor-encoder',str(out/'replay-anchor'),'--test-encoder',str(out/'replay-test'),'--decoder',str(decoder),'--input',str(old/'inputs'/(scene+'_720p30_clean.y4m')),'--fps','30/1','--bitrates','420,630,840,1260,2100,3360','--work-dir',str(out/'artifacts'),'--csv',str(out/'bdrate.csv')]
 start=time.monotonic()
 with (out/'bdrate.raw.txt').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
 assert r.returncode==0,(scene,(out/'bdrate.raw.txt').read_text())
 checks=[]
 for row in csv.reader((out/'bdrate.csv').read_text().splitlines()[1:]):
  if row[0]=='bdrate_pct':continue
  role,target,actual=row[:3];ivf=out/'artifacts'/(role+'_'+target+'k.ivf');data=ivf.read_bytes();original_data=(original/ivf.name).read_bytes();assert data==original_data
  expected=len(data)*8/1000/2;assert abs(float(actual)-expected)<=0.0005,(row,expected)
  checks.append({'stream':ivf.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'measured_kbps':float(actual),'independent_expected_kbps':expected,'clock_rate':int.from_bytes(data[16:20],'little')})
 # Keep compressed streams and small evidence, release only our generated decode files.
 for file in (out/'artifacts').glob('*.dec.y4m'):file.unlink()
 results.append({'scene':scene,'command':cmd,'elapsed_seconds':time.monotonic()-start,'exit_code':r.returncode,'rates':checks,'csv':(out/'bdrate.csv').read_text()})
 (root/'real-clock-replay-results.json').write_text(json.dumps(results,indent=2)+'\n')
 print(scene,'PASS all12 actual-role rates',flush=True)
