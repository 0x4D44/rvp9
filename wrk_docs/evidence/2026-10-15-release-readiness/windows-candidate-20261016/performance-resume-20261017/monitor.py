import pathlib,subprocess,base64,json,time,hashlib,threading,sys
p=pathlib.Path(__file__).parent
host='marti@crucible';native='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6'
def ssh_ps(script,timeout=30):
 encoded=base64.b64encode(script.encode('utf-16le')).decode()
 return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15',host,'powershell -NoProfile -EncodedCommand '+encoded],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
def copy(remote,local,required=True):
 local.parent.mkdir(parents=True,exist_ok=True)
 r=subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15',host+':'+remote.replace('\\','/'),str(local)],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
 if r.returncode and required:raise RuntimeError(r.stdout.decode(errors='replace'))
 return r.returncode==0
def snapshot_globals(tag):
 for remote,name in [('performance-resume-logs/campaign.log','campaign.raw.txt'),('performance-resume-logs/campaign.command.txt','campaign.command.txt'),('production-speed/manifest.txt','manifest.raw.txt')]:
  copy(native+'/'+remote,p/name,False)
 for name in ['quiet-start.log','quiet-start.exit.txt','quiet-grant.json','quiet-end.log','started.txt','ended.txt','campaign.exit.txt','final-source.txt']:
  copy(native+'/performance-resume-logs/'+name,p/name,False)
launch=p.joinpath('launch.ps1').read_text();encoded=base64.b64encode(launch.encode('utf-16le')).decode()
log=p.joinpath('launch.transport.raw.txt').open('wb')
proc=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15',host,'powershell -NoProfile -EncodedCommand '+encoded],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
print(f'TRACKED local_ssh_pid={proc.pid} campaign_bound_seconds=86400',flush=True)
p.joinpath('local-handle.json').write_text(json.dumps({'ssh_pid':proc.pid,'bound_seconds':86400,'outer_native_wait_ms':88000000})+'\n')
started=time.monotonic();exported=set();iteration=0;native_pid=None;last_tail=None;terminal_seen=False
inventory=p.joinpath('inventory.ps1').read_text()
while True:
 iteration+=1
 if time.monotonic()-started>88500:
  raise RuntimeError('Local monitor bound reached; native outer handle remains bounded88000s')
 result=ssh_ps(inventory)
 p.joinpath('latest-inventory.raw.txt').write_bytes(result.stdout)
 lines=[s for s in result.stdout.decode(errors='replace').splitlines() if s.startswith('{')]
 if result.returncode or not lines:
  print('INVENTORY_REFUSAL '+result.stdout.decode(errors='replace')[-800:],flush=True)
  if proc.poll() is not None:break
  time.sleep(30);continue
 data=json.loads(lines[-1])
 if native_pid is None and data.get('native_root_pid'):
  native_pid=data['native_root_pid'];p.joinpath('native-handle.json').write_text(json.dumps({'native_root_pid':native_pid,'observed_utc':data['observed_utc']})+'\n')
  print(f'NATIVE_ROOT_PID={native_pid} observed_utc={data["observed_utc"]}',flush=True)
  snapshot_globals('initial')
 for group in data['groups']:
  gid=group['id']
  if gid in exported:continue
  if not gid or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in gid):raise RuntimeError('invalid group id')
  ok=True
  for f in group['files']:
   if f['name'] not in ['headtohead.log','run/raw.csv','run/summary.csv','run/manifest.txt']:raise RuntimeError('unexpected group artifact')
   target=p/'groups'/gid/f['name']
   if not copy(f['path'],target,False):ok=False;break
   b=target.read_bytes()
   if len(b)!=f['bytes'] or hashlib.sha256(b).hexdigest()!=f['sha256']:raise RuntimeError('group artifact remote/download mismatch '+gid+' '+f['name'])
  if not ok:
   print('EXPORT_PENDING possible final-directory rename: '+gid,flush=True);continue
  p.joinpath('groups',gid,'remote-hashes.json').write_text(json.dumps(group,indent=2)+'\n')
  exported.add(gid)
  print(f'EXPORTED completed={len(exported)}/75 group={gid} status={group["status"]} separate={group["separate_ratio"]} paired={group["paired_ratio"]}',flush=True)
  snapshot_globals(gid)
 tail=' | '.join(data.get('tail',[]))
 if tail!=last_tail:
  print('STATUS '+data['observed_utc']+' '+tail,flush=True);last_tail=tail
 p.joinpath('progress.json').write_text(json.dumps({'observed_utc':data['observed_utc'],'native_root_pid':native_pid,'exported_groups':sorted(exported),'quiet':data['quiet'],'terminal_exit':data.get('terminal_exit')},indent=2)+'\n')
 if proc.poll() is not None:
  snapshot_globals('terminal')
  p.joinpath('terminal-inventory.raw.txt').write_bytes(result.stdout)
  print(f'TERMINAL ssh_exit={proc.returncode} campaign_exit={data.get("terminal_exit")} exported_groups={len(exported)}',flush=True)
  break
 time.sleep(1 if native_pid is None else 30)
log.close()
print('MONITOR_COMPLETE',flush=True)
