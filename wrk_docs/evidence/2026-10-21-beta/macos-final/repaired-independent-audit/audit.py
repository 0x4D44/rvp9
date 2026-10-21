"""Independent, read-only audit of the first completed repaired Mac beta group.
No Rust helpers, subprocess codec invocations, or unfinished row directories.
"""
import csv, datetime, hashlib, io, json, math, pathlib, struct
from decimal import Decimal
from fractions import Fraction
ROOT=pathlib.Path('/private/var/folders/bl/xncqcmfj4cj3b8z7j0sy78rr0000gn/T/deltic-scratch/rvp9-beta-final-macos-81301c61/quality-repaired')
OUT=pathlib.Path(__file__).resolve().parent
SCENES=['in_to_tree','ducks_take_off','old_town_cross']
TARGETS=[420,630,840,1260,2100,3360]
def sha(data): return hashlib.sha256(data).hexdigest()
def fields(text):
    out={}
    for line in text.splitlines():
        if '=' in line:
            k,v=line.split('=',1)
            if k in out: continue  # live manifest may append unrelated observations
            out[k]=v
    return out
# Independent ordinary least-squares cubic, solved by pivoted elimination.
def cubic_mean(curve,lo,hi):
    ps=[p[1] for p in curve]; center=(min(ps)+max(ps))/2; scale=(max(ps)-min(ps))/2
    assert scale>0 and len(set(ps))>=4
    xy=[((p-center)/scale,math.log(rate)) for rate,p in curve]
    matrix=[[sum(x**(i+j) for x,y in xy) for j in range(4)]+[sum(y*x**i for x,y in xy)] for i in range(4)]
    for col in range(4):
        pivot=max(range(col,4),key=lambda row:abs(matrix[row][col]));matrix[col],matrix[pivot]=matrix[pivot],matrix[col]
        divisor=matrix[col][col];assert abs(divisor)>1e-12
        matrix[col]=[v/divisor for v in matrix[col]]
        for row in range(4):
            if row==col:continue
            factor=matrix[row][col];matrix[row]=[a-factor*b for a,b in zip(matrix[row],matrix[col])]
    a,b=(lo-center)/scale,(hi-center)/scale
    return sum(matrix[i][4]*(b**(i+1)-a**(i+1))/(i+1)/(b-a) for i in range(4))
def bd_rate(a,b):
    lo=max(min(p[1] for p in a),min(p[1] for p in b));hi=min(max(p[1] for p in a),max(p[1] for p in b));assert hi>lo
    return math.expm1(cubic_mean(b,lo,hi)-cubic_mean(a,lo,hi))*100
manifest=fields((ROOT/'manifest.txt').read_text())
assert manifest['matrix']=='beta-cbr' and manifest['phase']=='quality'
assert manifest['git_commit']=='a37aba2b523a12dba8783c0bba16f4b88f68738c'
assert manifest['artifact_source_commit']=='15bf7162c9fb92a5165fab652460bd42da03a6e6'
assert manifest['libvpx_commit']=='12f3a2ac603e8f10742105519e0cd03c3b8f71dd'
provenance={k:manifest[k] for k in ['git_commit','artifact_source_commit','libvpx_commit','run_fingerprint_sha256','artifact_manifest_sha256','test_encoder_sha256','rvp9dec_sha256','bdrate_sha256','vpxenc_sha256','vpxdec_sha256']}
for name in ['test_encoder','rvp9dec','bdrate','vpxenc','vpxdec']:
    path=pathlib.Path(manifest[name+'_path']);assert sha(path.read_bytes())==manifest[name+'_sha256'],name
rows=[]
for scene in SCENES:
    row_id='aq3-cbr-720p30-'+scene+'-cpu6';row=ROOT/'quality/rows'/row_id
    assert row.is_dir() and not row.name.startswith('.')
    receipt_bytes=(row/'receipt').read_bytes();receipt=fields(receipt_bytes.decode())
    assert receipt['row_id']==row_id and receipt['run_fingerprint_sha256']==manifest['run_fingerprint_sha256']
    csv_bytes=(row/'bdrate.csv').read_bytes();parsed=list(csv.reader(io.StringIO(csv_bytes.decode())))
    assert parsed[0]==['role','target_kbps','measured_kbps','psnr_db']
    assert len(parsed)==14 and parsed[-1][0]=='bdrate_pct'
    reported_bd=float(parsed[-1][1]);assert math.isfinite(reported_bd)
    measurements={(r[0],int(r[1])):(Decimal(r[2]),float(r[3])) for r in parsed[1:-1]}
    assert len(measurements)==12
    expected_names={f'{role}_{target}k.ivf' for role in ['anchor','test'] for target in TARGETS}
    assert {p.name for p in (row/'streams').iterdir()}==expected_names
    streams=[];curves={'anchor':[],'test':[]}
    for role in ['anchor','test']:
        for target in TARGETS:
            path=row/'streams'/f'{role}_{target}k.ivf';before=path.stat();data=path.read_bytes()
            magic,version,header_size,codec,width,height,rate,scale,declared,reserved=struct.unpack_from('<4sHH4sHHIIII',data)
            assert (magic,version,header_size,codec,width,height)==(b'DKIF',0,32,b'VP90',1280,720)
            assert rate>0 and scale>0
            offset=header_size;packets=[]
            while offset<len(data):
                assert offset+12<=len(data),'truncated frame header'
                size,pts=struct.unpack_from('<IQ',data,offset);offset+=12
                assert size>0 and offset+size<=len(data),'truncated payload'
                packets.append({'bytes':size,'pts':pts});offset+=size
            assert offset==len(data) and len(packets)==declared==60
            first=packets[0]['pts'];previous=first
            for index,packet in enumerate(packets):
                pts=packet['pts'];assert pts>=previous
                ticks=Fraction(index*rate,scale*30)
                assert math.floor(ticks)<=pts-first<=math.ceil(ticks),(path,index,pts,ticks)
                previous=pts
            duration=Fraction(len(packets),30)
            independent=Decimal(len(data))*8/1000*Decimal(duration.denominator)/Decimal(duration.numerator)
            reported,psnr=measurements[(role,target)]
            assert reported==independent,(path,reported,independent)
            assert reported>0 and math.isfinite(psnr)
            curves[role].append((float(independent),psnr))
            after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
            streams.append({'file':str(path),'role':role,'target_kbps':target,'bytes':len(data),'sha256':sha(data),'header_clock_rate':rate,'header_clock_scale':scale,'declared_frames':declared,'packet_count':len(packets),'payload_bytes':sum(p['bytes'] for p in packets),'packets':packets,'fps_contract':'30/1','duration_seconds':str(duration),'independent_measured_kbps':str(independent),'csv_measured_kbps':str(reported),'rate_error_kbps':str(reported-independent),'csv_psnr_db':psnr})
    # Cross-decoder evidence is checked against independent stream hashes; no fresh decode is claimed.
    cross=list(csv.reader(io.StringIO((row/'cross-decoder-fragment.csv').read_text())))
    assert len(cross)==6
    by_name={pathlib.Path(s['file']).name:s for s in streams}
    for r in cross:
        s=by_name[r[1]];assert r[0]==row_id and int(r[3])==s['bytes'] and r[4]==s['sha256']
        assert r[5]==r[6] and r[7:] == ['yes','ok','PASS']
    recomputed=bd_rate(curves['anchor'],curves['test']);assert math.isfinite(recomputed)
    # Published PSNR is rounded to4decimals, unlike the original in-memory fit.
    assert abs(recomputed-reported_bd)<0.02,(scene,recomputed,reported_bd)
    fragment_bytes=(row/'quality-fragment.csv').read_bytes();fragment=next(csv.reader(io.StringIO(fragment_bytes.decode())))
    assert fragment[0]==row_id and float(fragment[17])==reported_bd and float(fragment[18])==2.0 and fragment[19]=='PASS'
    assert reported_bd<=2.0 and recomputed<=2.0
    assert receipt_bytes==(row/'receipt').read_bytes() and csv_bytes==(row/'bdrate.csv').read_bytes() and fragment_bytes==(row/'quality-fragment.csv').read_bytes()
    rows.append({'scene':scene,'row_id':row_id,'receipt':receipt,'receipt_sha256':sha(receipt_bytes),'csv_sha256':sha(csv_bytes),'quality_fragment_sha256':sha(fragment_bytes),'reported_bdrate_pct':reported_bd,'independently_recomputed_bdrate_pct_from_rounded_csv':recomputed,'bdrate_rounding_difference_pp':recomputed-reported_bd,'important_clip_gate_pct':2.0,'important_clip_status':'PASS','candidate_cross_decoder_receipt_rows_matched_to_stream_hashes':6,'curves':curves,'streams':streams})
mean=sum(r['reported_bdrate_pct'] for r in rows)/3
independent_mean=sum(r['independently_recomputed_bdrate_pct_from_rounded_csv'] for r in rows)/3
assert mean<=0.5 and independent_mean<=0.5
report={'audited_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':sha(pathlib.Path(__file__).read_bytes()),'scope':'Completed CPU6/AQ3/CBR/denoiser0/single-stream720p30 group only; no unfinished rows or full-campaign verdict','source_root':str(ROOT),'provenance':provenance,'stream_count':sum(len(r['streams']) for r in rows),'csv_rate_comparisons':36,'all_rate_errors_exactly_zero':True,'all_packet_counts_60':True,'all_cfr_durations_seconds':2,'reported_arithmetic_mean_bdrate_pct':mean,'independent_cubic_mean_from_rounded_csv_pct':independent_mean,'aggregate_gate_pct':0.5,'aggregate_status':'PASS','important_clip_gate_pct':2.0,'important_clip_status':'PASS','rows':rows,'defects':[]}
(OUT/'audit-results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
