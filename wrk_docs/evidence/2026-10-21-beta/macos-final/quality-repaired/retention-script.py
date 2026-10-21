"""Retain terminal Mac beta quality evidence; refuse prefixes and failed phases.
Run only after observing the qualification process exit. No codec/source writes.
"""
import argparse,csv,datetime,hashlib,io,json,math,pathlib,shutil,struct,sys
from decimal import Decimal
from fractions import Fraction
PRIMARY=pathlib.Path('/Users/md/worktrees/rvp9/20261021-DEV-CDX-qualify-and-release-rvp9-beta')
SCRATCH=pathlib.Path('/private/var/folders/bl/xncqcmfj4cj3b8z7j0sy78rr0000gn/T/deltic-scratch/rvp9-beta-final-macos-81301c61')
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--phase-exit-code',required=True,type=int,help='Actual terminal exit observed from the harness; must be0')
p.add_argument('--source',type=pathlib.Path,default=SCRATCH/'quality-repaired')
p.add_argument('--output',type=pathlib.Path,default=PRIMARY/'wrk_docs/evidence/2026-10-21-beta/macos-final/quality-repaired')
a=p.parse_args()
def require(condition,message):
    if not condition:raise RuntimeError(message)
def sha(data):return hashlib.sha256(data).hexdigest()
def read_fields(data, allow_resume_history=False):
    result={}
    for line in data.decode().splitlines():
        if '=' not in line:continue
        key,value=line.split('=',1)
        require(key not in result or (allow_resume_history and key=='resumed_utc'),'ambiguous duplicate manifest/receipt key: '+key)
        result[key]=value
    return result
def csv_dicts(path):return list(csv.DictReader(io.StringIO(path.read_text())))
require(a.phase_exit_code==0,'qualification process did not exit0; no PASS retention')
source=a.source.resolve();output=a.output.resolve()
require(not (source/'.qualify-run.lock').exists(),'qualification invocation lock is still present')
manifest_bytes=(source/'manifest.txt').read_bytes()
require(manifest_bytes.decode().splitlines().count('run_end status=pass')==1,'missing/ambiguous terminal run_end status=pass')
m=read_fields(manifest_bytes,allow_resume_history=True)
require(m.get('matrix')=='beta-cbr' and m.get('phase')=='quality','wrong campaign contract')
require(m.get('quality_verdict')=='pass' and m.get('aq_oracle_status')=='PASS' and m.get('aq_oracle_exit_code')=='0','quality/oracle verdict is not terminalPASS')
require(m.get('aq_oracle_activation')=='present','AQ activation was not observed')
for key in ['aq_oracle_gain_rvp9_pct','aq_oracle_gain_libvpx_pct','aq_oracle_delta_pp']:
    require(math.isfinite(float(m[key])),'nonfinite oracle metric '+key)
require(abs(float(m['aq_oracle_delta_pp']))<=0.3,'AQ gain parity exceeds0.3percentage points')
for key,count in {'quality_curves':27,'quality_clip_layer_checks':27,'quality_aggregate_groups':9,'quality_aggregate_layer_gates':9,'candidate_streams':162,'decoder_runs':486,'required_oracles':1,'quality_clip_layer_failures':0,'quality_aggregate_failures':0}.items():
    require(m.get(key)==str(count),f'wrong census {key}: {m.get(key)} != {count}')
completed=source/'quality/completed';require(completed.is_dir(),'no published completed phase')
for name,key in [('quality-summary.csv','quality_summary_sha256'),('quality-aggregate.csv','quality_aggregate_sha256'),('cross-decoder.csv','cross_decoder_summary_sha256'),('aq-oracle.log','aq_oracle_log_sha256'),('aq-oracle-summary.csv','aq_oracle_summary_sha256')]:
    require(sha((completed/name).read_bytes())==m.get(key),'terminal manifest hash mismatch: '+name)
summary=csv_dicts(completed/'quality-summary.csv');aggregate=csv_dicts(completed/'quality-aggregate.csv');cross=csv_dicts(completed/'cross-decoder.csv')
require((len(summary),len(aggregate),len(cross))==(27,9,162),'published CSV census mismatch')
scenes=['in_to_tree','ducks_take_off','old_town_cross'];targets={'720p30':(1280,720,30,[420,630,840,1260,2100,3360]),'1080p30':(1920,1080,30,[950,1425,1890,2835,4725,7560]),'1080p60':(1920,1080,60,[1900,2850,3780,5670,9450,15120])}
expected={f'aq3-cbr-{target}-{scene}-cpu{cpu}' for target in targets for scene in scenes for cpu in [6,7,8]}
require({row['row_id'] for row in summary}==expected,'missing/duplicate matrix row identity')
require({d.name for d in (source/'quality/rows').iterdir() if d.is_dir() and not d.name.startswith('.')}==expected,'completed row directory census differs')
stream_audits=[];rows_by_id={r['row_id']:r for r in summary};stream_by_key={};rate_errors=[]
for row_id in sorted(expected):
    row_dir=source/'quality/rows'/row_id;row=rows_by_id[row_id];target=row['target'];width,height,fps,grid=targets[target]
    require(row['feature']=='aq3' and row['mode']=='cbr' and row['aq_mode']=='3' and row['noise_sensitivity']=='0' and row['temporal_layers']=='1','row feature mismatch '+row_id)
    require(row['status']=='PASS' and float(row['gate_pct'])==2 and math.isfinite(float(row['bdrate_pct'])) and float(row['bdrate_pct'])<=2,'clip guard failed '+row_id)
    data=list(csv.reader(io.StringIO((row_dir/'bdrate.csv').read_text())))
    require(len(data)==14 and data[-1][0]=='bdrate_pct','bad RD CSV '+row_id)
    require(float(data[-1][1])==float(row['bdrate_pct']),'summary/row BD-rate mismatch '+row_id)
    measures={(r[0],int(r[1])):(Decimal(r[2]),float(r[3])) for r in data[1:-1]}
    require(len(measures)==12,'wrong RD point count '+row_id)
    expected_streams={f'{role}_{br}k.ivf' for role in ['anchor','test'] for br in grid}
    require({f.name for f in (row_dir/'streams').iterdir()}==expected_streams,'unexpected/missing stream '+row_id)
    for role in ['anchor','test']:
        for br in grid:
            name=f'{role}_{br}k.ivf';file=row_dir/'streams'/name;raw=file.read_bytes()
            require(len(raw)>=32,'short IVF '+str(file))
            magic,version,header,codec,w,h,rate,scale,declared,reserved=struct.unpack_from('<4sHH4sHHIIII',raw)
            require((magic,version,header,codec,w,h)==(b'DKIF',0,32,b'VP90',width,height),'bad IVF header '+str(file))
            require(rate>0 and scale>0,'zero clock '+str(file))
            pos=header;pts=[];payload_bytes=0
            while pos<len(raw):
                require(pos+12<=len(raw),'truncated packet header '+str(file));size,timestamp=struct.unpack_from('<IQ',raw,pos);pos+=12
                require(size>0 and pos+size<=len(raw),'truncated/empty payload '+str(file));pos+=size;payload_bytes+=size;pts.append(timestamp)
            require(pos==len(raw) and len(pts)==declared==60,'frame/byte count mismatch '+str(file))
            for index,timestamp in enumerate(pts):
                ticks=Fraction(index*rate,scale*fps)
                require(timestamp>=pts[0] and (index==0 or timestamp>=pts[index-1]) and math.floor(ticks)<=timestamp-pts[0]<=math.ceil(ticks),'inconsistent PTS '+str(file))
            duration=Fraction(len(pts),fps);expected_rate=Decimal(len(raw))*8/1000*Decimal(duration.denominator)/Decimal(duration.numerator)
            actual,psnr=measures[(role,br)];require(actual>0 and actual.is_finite() and math.isfinite(psnr),'nonfinite RD point '+str(file))
            error=actual-expected_rate;require(abs(error)<=Decimal('0.0005'),'CSV measured rate differs from independent CFR arithmetic '+str(file));rate_errors.append(abs(error))
            audit={'row_id':row_id,'file':str(file),'role':role,'target_kbps':br,'bytes':len(raw),'sha256':sha(raw),'clock_rate':rate,'clock_scale':scale,'packet_count':len(pts),'header_frames':declared,'payload_bytes':payload_bytes,'timestamps':pts,'fps_contract':f'{fps}/1','duration_seconds':str(duration),'csv_rate_kbps':str(actual),'independent_rate_kbps':str(expected_rate),'rate_error_kbps':str(error),'psnr_db':psnr}
            stream_audits.append(audit);stream_by_key[(row_id,name)]=audit
    # Recompute the exact ordered row receipt payload, without using the Rust helper.
    fingerprint=''
    for name in ['bdrate.csv','runner.log','quality-fragment.csv','cross-decoder-fragment.csv','cross-decoder.log']:
        raw=(row_dir/name).read_bytes();fingerprint+=f'{name}|{len(raw)}|{sha(raw)}\n'
    for br in grid:
        audit=stream_by_key[(row_id,f'test_{br}k.ivf')];fingerprint+=f'streams/test_{br}k.ivf|{audit["bytes"]}|{audit["sha256"]}\n'
    receipt=read_fields((row_dir/'receipt').read_bytes())
    require(receipt.get('row_id')==row_id and receipt.get('run_fingerprint_sha256')==m.get('run_fingerprint_sha256') and receipt.get('artifact_sha256')==sha(fingerprint.encode()),'row receipt mismatch '+row_id)
require(len(stream_audits)==324,'expected324 both-role stream audits')
seen=set();empty_hash=sha(b'')
for record in cross:
    key=(record['row_id'],record['stream']);require(key not in seen,'duplicate cross-decoder record');seen.add(key)
    require(key in stream_by_key and key[1].startswith('test_'),'cross-decoder stream is not a candidate');audit=stream_by_key[key]
    require(int(record['bytes'])==audit['bytes'] and record['stream_sha256']==audit['sha256'],'cross-decoder compressed identity mismatch')
    require(record['status']=='PASS' and record['decoders_identical']=='yes' and record['ffmpeg_decode']=='ok' and record['rvp9dec_sha256']==record['vpxdec_sha256'] and record['rvp9dec_sha256']!=empty_hash,'cross-decoder verdict mismatch')
require(len(seen)==162,'candidate cross-decoder census mismatch')
seen_groups=set();groups=[]
for record in aggregate:
    key=(record['target'],record['cpu_used']);require(key not in seen_groups,'duplicate aggregate group');seen_groups.add(key)
    members=[r for r in summary if (r['target'],r['cpu_used'])==key];require(len(members)==3 and {r['clip'] for r in members}==set(scenes),'aggregate clip census mismatch')
    mean=sum(Decimal(r['bdrate_pct']) for r in members)/3
    require(abs(Decimal(record['mean_bdrate_pct'])-mean)<=Decimal('0.000001'),'aggregate arithmetic mean mismatch')
    require(record['status']=='PASS' and Decimal(record['gate_pct'])==Decimal('0.5') and mean<=Decimal('0.5'),'aggregate gate failed')
    groups.append({'target':key[0],'cpu_used':int(key[1]),'independent_mean_bdrate_pct':str(mean),'reported_mean_bdrate_pct':record['mean_bdrate_pct'],'status':'PASS'})
require(len(seen_groups)==9,'aggregate group census mismatch')
# Retain every declared identity; rehash tools/small metadata. Existing manifest
# media hashes are retained without rereading gigabytes of prepared/source Y4M.
provenance=[]
for key,value in m.items():
    if not key.endswith('_path'):continue
    stem=key[:-5];hash_key=stem+'_sha256'
    if hash_key not in m:continue
    file=pathlib.Path(value);require(file.is_file(),'missing provenance artifact '+str(file))
    if file.suffix.lower() in {'.y4m','.yuv','.ivf','.webm'}:
        provenance.append({'key':stem,'path':value,'bytes':file.stat().st_size,'sha256':m[hash_key],'verification':'existing run-manifest media identity retained; no fresh media reread'})
        continue
    digest=hashlib.sha256()
    with file.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    require(digest.hexdigest()==m[hash_key],'provenance artifact hash mismatch '+str(file))
    provenance.append({'key':stem,'path':value,'bytes':file.stat().st_size,'sha256':digest.hexdigest(),'verification':'independently rehashed'})
require(manifest_bytes==(source/'manifest.txt').read_bytes(),'manifest changed during terminal audit')
require(output!=source and source not in output.parents,'retention output must be outside source campaign')
require(not output.exists(),'output exists; retain separately, do not overwrite prior evidence')
output.mkdir(parents=True)
copy_map=[]
def retain(file,relative):
    raw=file.read_bytes();relative=pathlib.Path(relative)
    if relative.suffix=='.log':relative=relative.with_suffix('.raw.txt')
    destination=output/relative;require(not destination.exists(),'retention path collision '+str(relative));destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw)
    copy_map.append({'source':str(file),'retained':str(relative),'bytes':len(raw),'sha256':sha(raw),'transformation':'filename .log -> .raw.txt only' if file.suffix=='.log' else 'none'})
exclude={'.y4m','.yuv','.ivf','.webm','.raw','.bin','.crate','.gz','.zip','.tar','.profraw','.profdata'}
for file in sorted(source.rglob('*')):
    if not file.is_file() or file.suffix.lower() in exclude:continue
    require(not file.is_symlink(),'refuse symlink '+str(file));require(file.stat().st_size<=16*1024*1024,'unexpectedly large text evidence '+str(file));retain(file,pathlib.Path('campaign')/file.relative_to(source))
for file in [source.parent/'quality-repaired.raw.txt',source.parent/'repaired-bdrate-build.raw.txt']:
    if file.is_file():retain(file,pathlib.Path('outer')/file.name)
retain(pathlib.Path(__file__),pathlib.Path('retention-script.py'))
report={'retained_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase_process_exit_code_observed_by_lead':a.phase_exit_code,'source':str(source),'terminal_quality_verdict':'pass','run_fingerprint_sha256':m['run_fingerprint_sha256'],'git_commit':m['git_commit'],'artifact_source_commit':m['artifact_source_commit'],'quality_curves':27,'candidate_streams':162,'decoder_runs':486,'both_role_ivfs_independently_audited':324,'packet_count':sum(r['packet_count'] for r in stream_audits),'max_rate_error_kbps':str(max(rate_errors)),'aggregate_groups':groups,'provenance':provenance,'large_media_copied':False,'candidate_decode_checks':'Validated retained162 cross-decoder observations and stream identities; no fresh decoding performed','existing_independent_audit':'Sibling repaired-independent-audit untouched','copy_map':copy_map,'stream_audits':stream_audits,'defects':[]}
(output/'retention-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
lines=[sha(f.read_bytes())+'  '+str(f.relative_to(output)) for f in sorted(output.rglob('*')) if f.is_file()]
(output/'SHA256SUMS.raw.txt').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['copy_map','stream_audits','provenance']},indent=2))
