"""The previously qualified same-payload bounded atomic JSON commit."""
from pathlib import Path
import hashlib,sys
from .io import jsonable
VENDOR=Path(__file__).resolve().parents[2]/'vendor'
if hashlib.sha256((VENDOR/'runtime_io.py').read_bytes()).hexdigest()!='159c0d11e50017a4b5fab511e6594efb290955c5ee4a543f67374b7ed19675d1':raise ValueError('Qualified metadata writer changed')
sys.path.insert(0,str(VENDOR))
from runtime_io import write_json
def write(path,value):write_json(path,jsonable(value))
