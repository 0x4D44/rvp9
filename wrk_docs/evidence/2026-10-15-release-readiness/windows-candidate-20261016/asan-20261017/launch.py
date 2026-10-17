import pathlib,subprocess,base64
p=pathlib.Path(__file__).parent;s=p.joinpath('launch.ps1').read_text()
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','marti@crucible','powershell -NoProfile -EncodedCommand '+base64.b64encode(s.encode('utf-16le')).decode()],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=210)
p.joinpath('launch.raw.txt').write_bytes(r.stdout);print(r.stdout.decode(errors='replace'));raise SystemExit(r.returncode)
