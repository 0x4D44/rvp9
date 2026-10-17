import subprocess,pathlib,json,re
p=pathlib.Path(__file__).parent
s='''set -euo pipefail
cd /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi
[ "$(git rev-parse HEAD)" = 474dfd99a673df85f750136f7552ac156b7a24e9 ]
[ -z "$(git status --porcelain)" ]
deltic scratch new --label rvp9-native-packagegraph474 --json
'''
p.joinpath('register.sh').write_text(s)
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','marti@crucible','"C:/Program Files/Git/bin/bash.exe" --noprofile --norc -s'],input=s.encode(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
p.joinpath('register.raw.txt').write_bytes(r.stdout)
if r.returncode:print(r.stdout.decode());raise SystemExit(r.returncode)
data=json.loads(r.stdout);native=re.sub(r'\\+', '/',data['path']);posix='/c/'+native[3:]
p.joinpath('root.json').write_text(json.dumps({'id':data['id'],'native':native,'posix':posix,'task':data['task']},indent=2)+'\n')
print(native)
