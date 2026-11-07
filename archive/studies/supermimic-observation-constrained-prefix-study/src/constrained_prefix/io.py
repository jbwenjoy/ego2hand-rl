"""Same-payload bounded atomic commits for JSON, NPZ and text evidence."""
from io import BytesIO
from pathlib import Path
import hashlib,json,sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'vendor'))
from runtime_io import write_json,replace_with_bounded_retry

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def atomic_bytes(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='wb',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
        temporary=Path(stream.name);stream.write(payload)
    replace_with_bounded_retry(temporary,path,max_seconds=.75)
def atomic_npz(path,**arrays):
    buffer=BytesIO();np.savez_compressed(buffer,**arrays);atomic_bytes(path,buffer.getvalue())
def atomic_jsonl(path,rows):atomic_bytes(path,(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows)).encode())
