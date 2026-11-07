"""Strict finite JSON and content-addressed local artifacts; no GPU imports."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import numpy as np
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pin(path):
    path=Path(path).resolve(strict=True)
    return {'path':str(path),'sha256':sha(path)}
def checked(value):
    if set(value)!={'path','sha256'} or not Path(value['path']).is_absolute() or sha(value['path'])!=value['sha256']:raise ValueError('Missing/changed absolute pinned file')
    return Path(value['path'])
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def jsonable(value):
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [jsonable(v) for v in value]
    if isinstance(value,np.ndarray):return jsonable(value.tolist())
    if isinstance(value,np.generic):return jsonable(value.item())
    if isinstance(value,Path):return str(value)
    if hasattr(value,'detach'):return jsonable(value.detach().cpu().tolist())
    if value is None or isinstance(value,(str,bool,int)):return value
    if isinstance(value,float) and math.isfinite(value):return value
    raise TypeError('Unsupported or nonfinite metadata: '+type(value).__name__)
def canonical(value):return json.dumps(jsonable(value),sort_keys=True,separators=(',',':'),allow_nan=False)
def write(path,value):Path(path).write_text(json.dumps(jsonable(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def loadnp(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
