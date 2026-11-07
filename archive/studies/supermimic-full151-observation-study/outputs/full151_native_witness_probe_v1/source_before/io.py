"""Small, framework-free provenance helpers for the full151 input stage."""
from pathlib import Path
import hashlib,json
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pin(path):
    p=Path(path).resolve(strict=True)
    return {'path':str(p),'sha256':sha(p)}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def checked(value):
    if type(value) is not dict or set(value)!={'path','sha256'}:raise ValueError('Exact pin required')
    p=Path(value['path']).resolve(strict=True)
    if sha(p)!=value['sha256']:raise ValueError('Pinned bytes changed: '+str(p))
    return p
def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [jsonable(v) for v in value]
    return value
def write(path,value):
    Path(path).write_text(json.dumps(jsonable(value),indent=2,allow_nan=False)+'\n',encoding='utf-8')
def load_selected(value,keys):
    with np.load(checked(value),allow_pickle=False) as z:
        return {key:z[key].copy() for key in keys}
def all_pins(value):
    if isinstance(value,dict):
        if set(value)=={'path','sha256'}:yield value
        else:
            for v in value.values():yield from all_pins(v)
    elif isinstance(value,list):
        for v in value:yield from all_pins(v)
