"""Optional resumable CUDA wheel download for constrained Windows connections."""
from pathlib import Path
import time, urllib.request
root=Path(__file__).resolve().parents[1]
dest=root/'.tmp'/'torch-2.9.1+cu128-cp313-cp313-win_amd64.whl'
dest.parent.mkdir(parents=True,exist_ok=True)
url='https://download-r2.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp313-cp313-win_amd64.whl'
for attempt in range(20):
    offset=dest.stat().st_size if dest.exists() else 0
    if offset==2862036321:
        print('CUDA wheel already downloaded',flush=True)
        break
    print(f'CUDA wheel resume attempt={attempt+1} bytes={offset}',flush=True)
    try:
        req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-','User-Agent':'sentiment-project/1.0'})
        with urllib.request.urlopen(req,timeout=90) as response:
            mode='ab' if response.status==206 else 'wb'
            with dest.open(mode) as file:
                while True:
                    chunk=response.read(1024*1024)
                    if not chunk: break
                    file.write(chunk)
        print(f'CUDA wheel complete bytes={dest.stat().st_size}',flush=True); break
    except Exception as exc:
        print(type(exc).__name__,str(exc),flush=True); time.sleep(2)
else: raise RuntimeError('CUDA wheel download incomplete; rerun to resume')
