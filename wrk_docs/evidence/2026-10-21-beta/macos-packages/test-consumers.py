import pathlib,subprocess,sys,json
s=pathlib.Path("/private/var/folders/bl/xncqcmfj4cj3b8z7j0sy78rr0000gn/T/deltic-scratch/rvp9-beta-packages-a073aeeb");e=pathlib.Path("/Users/md/worktrees/rvp9/20261021-DEV-CDX-qualify-and-release-rvp9-beta/wrk_docs/evidence/2026-10-21-beta/macos-packages")
config=['--config','source.crates-io.replace-with="beta-staging"','--config','source.beta-staging.local-registry="'+str(s/'target/package/tmp-registry')+'"']
results=[]
for name in ['rvp9-core','rvp9-io','rvp9-decoder','rvp9-encoder','rvp9-cli']:
 cwd=s/'consumers'/(name+'-0.2.0-beta.1')
 cmd=['python3',str(s/'run.py'),str(e/(name+'-unit')), 'deltic','timeout','1800','cargo','test','--offline','--locked']+config+['--lib' if name!='rvp9-cli' else '--bins']
 result=subprocess.run(cmd,cwd=cwd);results.append({'crate':name,'exit_code':result.returncode});print(name,result.returncode,flush=True)
(e/'consumer-results.json').write_text(json.dumps(results,indent=2)+'\n')
sys.exit(int(any(r['exit_code'] for r in results)))
