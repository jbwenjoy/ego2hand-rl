"""Exclusive-create, fsynced native raw sidecars before replay/label use.

No native simulator or Torch import. Raw NaNs remain in NPZ; diagnostic JSON
represents nonfinite values as null rather than fabricating a finite target.
"""
from pathlib import Path
import hashlib,json,os
import numpy as np

def need(v,msg):
    if not v:raise ValueError(msg)

def pin(path):
    path=Path(path).resolve(strict=True);b=path.read_bytes();return dict(path=str(path),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def same(a,b):
    a,b=np.asarray(a),np.asarray(b);return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def clean(v):
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,dict):return {k:clean(a) for k,a in v.items()}
    if isinstance(v,(list,tuple)):return [clean(a) for a in v]
    if isinstance(v,(bool,np.bool_)):return bool(v)
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    return v

def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(clean(value),f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())

def save(path,arrays):
    with Path(path).open('xb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())

def tensor_arrays(value,prefix=''):
    if isinstance(value,dict):
        result={}
        for k,v in value.items():result.update(tensor_arrays(v,prefix+str(k)+'__'))
        return result
    if hasattr(value,'detach'):return {prefix[:-2]:value.detach().cpu().numpy()}
    if isinstance(value,np.ndarray):return {prefix[:-2]:value}
    return {}

def receipt_meta(v):return {k:a for k,a in v.items() if k!='integration_state'}|dict(dtype=v['integration_state'].dtype.str,shape=list(v['integration_state'].shape))

class PersistentRawSink:
    def __init__(self,folder,episode_id):self.folder=folder;self.episode_id=episode_id;self.k=None;self.initial=None;self.intervals={};self.calls=0
    def __call__(self,v):
        self.calls+=1
        if v['schema']=='phone_actor_native_reset_receipt:1':
            need(self.initial is None,'One raw reset sink per episode');r=v['reset'];path=self.folder/'initial.npz'
            arrays={k:np.asarray(a) for k,a in r.items() if isinstance(a,np.ndarray)};arrays['integration_state']=r['receipt']['integration_state']
            save(path,arrays);meta=dict(schema=v['schema'],episode_id=self.episode_id,source_initial=v['source_initial'],receipt300=receipt_meta(r['receipt']),contact_phase=r['contact_phase'],raw_arrays=pin(path))
            write(self.folder/'initial.json',meta);self.initial=dict(arrays=pin(path),metadata=pin(self.folder/'initial.json'),receipt=meta['receipt300']);return
        need(v['schema']=='phone_actor_raw_transition_sidecar:1' and self.k is not None and self.k not in self.intervals,'Known step raw sink exactly once')
        r=v['interval'];contacts=v['tripod_contacts'];path=self.folder/f'interval_{self.k:02d}.npz'
        arrays=dict(r['trace'])|dict(source_state58=r['source_state58'],observed_prefix_target58=v['observed_prefix_state58'],target_at_H_state58=v['target_H58'],physical_world_delta52=r['physical_world_delta52'],actual_control20=r['actual_control20'],source_integration_state=r['source_receipt']['integration_state'],source_after_action_integration_state=r['source_after_action_receipt']['integration_state'],target_integration_state=r['target_receipt']['integration_state'],memory_before83_F64=v['memory_before'],action_normalized20_F64=v['action_normalized20'],residual20_F64=v['residual20'],tripod_pair_positive_Fn22_N=contacts['per_tick_pair_positive_Fn22_N'],tripod_pair_generated22=contacts['per_tick_pair_generated22'],tripod_pair_coverage22=contacts['pair_coverage22'],tripod_pair_mean_positive_Fn22_N=contacts['pair_mean_positive_Fn22_N'])
        save(path,arrays)
        meta=dict(schema=v['schema'],episode_id=self.episode_id,step_k=self.k,raw=pin(path),source_receipt=receipt_meta(v['source_receipt']),source_after_action_receipt=receipt_meta(v['after_action_receipt']),target_receipt=receipt_meta(v['target_receipt']),complete_H=v['complete_H'],valid=v['valid'],termination=r['termination'],force_phase=r['force_phase'],original_two_pad_summary=contacts['original_two_pad_summary'],triple_simultaneous_coverage=contacts['triple_simultaneous_coverage'],actual_contact_known=True,proxy_is_observed_contact=False,warning_numbers=r['warning_numbers'])
        write(self.folder/f'interval_{self.k:02d}.json',meta);self.intervals[self.k]=meta
