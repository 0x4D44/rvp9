import datetime,json,os,pathlib,subprocess,sys,time
prefix=pathlib.Path(sys.argv[1]);cmd=sys.argv[2:]
env=dict(os.environ,RUSTFLAGS="",CARGO_ENCODED_RUSTFLAGS="",CARGO_TARGET_DIR="/private/var/folders/bl/xncqcmfj4cj3b8z7j0sy78rr0000gn/T/deltic-scratch/rvp9-beta-packages-a073aeeb/target")
start=time.monotonic();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
with open(str(prefix)+".raw.txt","w") as out:
    result=subprocess.run(cmd,env=env,stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
record={"command":cmd,"cwd":os.getcwd(),"env":{k:env[k] for k in ("RUSTFLAGS","CARGO_ENCODED_RUSTFLAGS","CARGO_TARGET_DIR")},"started_utc":utc,"elapsed_seconds":time.monotonic()-start,"exit_code":result.returncode}
pathlib.Path(str(prefix)+".json").write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record));sys.exit(result.returncode)
