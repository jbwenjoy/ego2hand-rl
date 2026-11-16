"""Strict JSON records with explicit NumPy scalar support;never array coercion."""
import json
from pathlib import Path
import numpy as np


def scalar(value):
    if isinstance(value,np.bool_):return bool(value)
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,(np.float16,np.float32,np.float64)):return float(value)
    if isinstance(value,np.str_):return str(value)
    raise TypeError('Unsupported research JSON value: '+type(value).__name__)


def write_json(path,value):
    text=json.dumps(value,indent=2,allow_nan=False,default=scalar)+'\n'
    Path(path).write_text(text,encoding='utf-8')
