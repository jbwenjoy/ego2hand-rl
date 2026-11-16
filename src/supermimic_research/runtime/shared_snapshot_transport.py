"""Exact native4 reconstruction with shared,verified immutable fields only."""
import copy
import hashlib
import io
import struct
from pathlib import PurePath
from enum import Enum
import numpy as np
import torch
from .snapshot_transport import pack_bytes,unpack_bytes
from ..learning.world_model import raw_tree_equal

SCHEMA='owned_native4_shared_immutable_transport:1'


def tree_digest(value):
    h=hashlib.sha256()
    def walk(v):
        h.update((type(v).__module__+'.'+type(v).__qualname__).encode()+b'\0')
        if isinstance(v,torch.Tensor):
            h.update(str((v.dtype,tuple(v.shape),v.stride())).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v,np.ndarray):h.update(str((v.dtype.str,v.shape)).encode());h.update(v.tobytes())
        elif isinstance(v,np.generic):h.update(v.dtype.str.encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            h.update(str(len(v)).encode())
            for k in sorted(v,key=lambda k:repr(k)):walk(k);walk(v[k])
        elif isinstance(v,(list,tuple)):
            h.update(str(len(v)).encode())
            for item in v:walk(item)
        elif isinstance(v,(set,frozenset)):
            h.update(str(len(v)).encode())
            for item in sorted(v,key=repr):walk(item)
        elif isinstance(v,slice):walk((v.start,v.stop,v.step))
        elif isinstance(v,Enum):walk(v.name);walk(v.value)
        elif isinstance(v,str):h.update(v.encode())
        elif isinstance(v,bytes):h.update(v)
        elif isinstance(v,PurePath):h.update(str(v).encode())
        elif isinstance(v,bool):h.update(b'1' if v else b'0')
        elif isinstance(v,int):h.update(str(v).encode())
        elif isinstance(v,float):h.update(struct.pack('!d',v))
        elif v is not None:raise TypeError('Unsupported snapshot digest leaf '+type(v).__name__)
        h.update(b'\xff')
    walk(value);return h.hexdigest()


def validate(state):
    if not isinstance(state,dict) or state.get('schema')!='controlled_pen_owned_state:4' or type(state.get('worlds')) is not int or state['worlds']!=1 or state['metadata'].get('fixed_frame_features')!='LIVE' or set(state['native'])!={'model','data'}:
        raise ValueError('Exact owned native4 LIVE W1 state required')


def static_fields(state):
    return dict(model=state['native']['model'],tensors={p:state['python']['tensors'][p] for p in state['immutable']},
        arrays={p:state['python']['arrays'][p] for p in state['immutable_arrays']},scalars={p:state['python']['scalars'][p] for p in state['immutable_scalars']})


def serialized(value):
    stream=io.BytesIO();torch.save(value,stream);return stream.getvalue()


class SharedImmutableTransport:
    def __init__(self,path):self.path=path;self.shared=None;self.record=None

    def save(self,path,state):
        validate(state);shared=static_fields(state)
        if self.shared is None:
            self.shared=copy.deepcopy(shared);self.record=pack_bytes(self.path,serialized(dict(schema='owned_native4_immutable_fields:1',fields=self.shared)))
        elif not raw_tree_equal(self.shared,shared):raise ValueError('Shared immutable snapshot fields changed')
        partial=dict(state);partial['native']=dict(data=state['native']['data']);partial['python']=dict(state['python'])
        for category in ['tensors','arrays','scalars']:partial['python'][category]={p:v for p,v in state['python'][category].items() if p not in self.shared[category]}
        payload=pack_bytes(path,serialized(dict(schema='owned_native4_delta_payload:1',partial=partial)))
        return dict(schema=SCHEMA,shared=self.record,payload=payload,full_tree_sha256=tree_digest(state))


def load_shared(record):
    if record.get('schema')!=SCHEMA:raise ValueError('Explicit shared snapshot transport required')
    # Both files and uncompressed payloads are verified before either load.
    a=unpack_bytes(record['shared']);b=unpack_bytes(record['payload'])
    shared=torch.load(io.BytesIO(a),map_location='cpu',weights_only=False);delta=torch.load(io.BytesIO(b),map_location='cpu',weights_only=False)
    if shared.get('schema')!='owned_native4_immutable_fields:1' or delta.get('schema')!='owned_native4_delta_payload:1':raise ValueError('Snapshot partition schemas differ')
    fields=shared['fields'];state=delta['partial'];state['native']=dict(model=fields['model'],data=state['native']['data'])
    for category in ['tensors','arrays','scalars']:
        if state['python'][category].keys()&fields[category].keys():raise ValueError('Overlapping snapshot partition fields')
        state['python'][category].update(fields[category])
    validate(state)
    if set(fields['tensors'])!=set(state['immutable']) or set(fields['arrays'])!=set(state['immutable_arrays']) or set(fields['scalars'])!=set(state['immutable_scalars']):raise ValueError('Shared fields do not match owned immutable manifest')
    if tree_digest(state)!=record['full_tree_sha256']:raise ValueError('Reconstructed full snapshot differs')
    return state
