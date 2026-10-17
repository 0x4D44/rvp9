import subprocess,base64,pathlib,sys
p=pathlib.Path(__file__).parent
s=p.joinpath(sys.argv[1]).read_text()
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','marti@crucible','powershell -NoProfile -EncodedCommand '+base64.b64encode(s.encode('utf-16le')).decode()],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=int(sys.argv[2]) if len(sys.argv)>2 else 90)
p.joinpath(sys.argv[1]+'.log').write_bytes(r.stdout)
print(r.stdout.decode(errors='replace'));sys.exit(r.returncode)
