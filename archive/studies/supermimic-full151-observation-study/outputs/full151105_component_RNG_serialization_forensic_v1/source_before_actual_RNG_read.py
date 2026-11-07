"""Only own trusted initial+two100 component RNG serialized readback; no model."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/full151105_WM_common_pre_calibration_CPU_v1'
OUT=ROOT/'outputs/full151105_component_RNG_serialization_forensic_v1'
ARMS=('pointwise','pointwise_plus_common_pre_response')


def pin(p):return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def raw_recursive_differences(a,b,path='root'):
    out=[]
    if type(a) is not type(b):return [{'path':path,'kind':'TYPE','a':str(type(a)),'b':str(type(b))}]
    if isinstance(a,np.ndarray):
        if a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes()!=b.tobytes():out.append({'path':path,'kind':'NDARRAY_BYTES_DTYPE_OR_SHAPE'})
    elif isinstance(a,(tuple,list)):
        if len(a)!=len(b):out.append({'path':path,'kind':'LENGTH'})
        else:
            for i,(x,y) in enumerate(zip(a,b)):out.extend(raw_recursive_differences(x,y,path+'/'+str(i)))
    elif isinstance(a,dict):
        if set(a)!=set(b):out.append({'path':path,'kind':'KEYS'})
        else:
            for k in a:out.extend(raw_recursive_differences(a[k],b[k],path+'/'+str(k)))
    elif isinstance(a,(float,np.floating)):
        if np.asarray(a).dtype!=np.asarray(b).dtype or np.asarray(a).tobytes()!=np.asarray(b).tobytes():out.append({'path':path,'kind':'SCALAR_FLOAT_BITS'})
    elif a!=b:out.append({'path':path,'kind':'VALUE','a':repr(a),'b':repr(b)})
    return out


def main():
    OUT.mkdir(exist_ok=False);source=pin(Path(__file__));(OUT/'source_before_actual_RNG_read.py').write_bytes(Path(__file__).read_bytes())
    (OUT/'source_freeze.json').write_text(json.dumps({'source':source,'actual_RNG_read':False,'raw_comparison':'recursive exact Python types/values, NumPy dtype/shape/header/bytes, TorchCPU dtype/shape/bytes; pickle hashes only diagnostic'},indent=2))
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch
    names={'initial':'initial_component_state.pt',**{a:a+'_update_0100.pt' for a in ARMS}};states={};pins={}
    for k,name in names.items():
        path=RUN/name;pins[k]=pin(path);payload=torch.load(path,map_location='cpu',weights_only=False)
        if payload['schema']!='full151105_WM_common_pre_component_calibration:1':raise ValueError('Own component schema only')
        states[k]=payload['RNG'];del payload
        if pin(path)!=pins[k]:raise ValueError('Source payload changed')
    validation=json.loads((RUN/'initial_validation.json').read_text());vectors={};details={};base=states['initial'];checks={}
    for k,v in states.items():
        p=v['python'];n=v['numpy'];t=v['torch_CPU']
        host={key:value for key,value in v.items() if key!='torch_CPU'}
        basehost={key:value for key,value in base.items() if key!='torch_CPU'}
        diff=raw_recursive_differences(basehost,host)
        tgood=isinstance(t,torch.Tensor) and t.device.type=='cpu' and t.dtype==torch.uint8 and t.shape==base['torch_CPU'].shape and t.contiguous().numpy().tobytes()==base['torch_CPU'].contiguous().numpy().tobytes()
        legacy=dict(v);legacy['torch_CPU']=t.contiguous().numpy().tobytes();blob=pickle.dumps(legacy,protocol=5)
        vectors[k+'__Torch_CPU']=t.contiguous().numpy().copy();vectors[k+'__NumPy_MT']=n[1].copy();vectors[k+'__Python_words']=np.asarray(p[1],dtype=np.uint64)
        details[k]={'host_recursive_exact':not diff,'host_recursive_differences':diff,'Torch_CPU_dtype':str(t.dtype),'Torch_CPU_shape':list(t.shape),'Torch_CPU_bytes_equal_initial':tgood,
           'Python_version_type':str(type(p[0])),'Python_state_container':str(type(p[1])),'Python_word_count':len(p[1]),'Python_cached_gauss_type':str(type(p[2])),
           'NumPy_algorithm':n[0],'NumPy_MT_dtype':n[1].dtype.str,'NumPy_MT_shape':list(n[1].shape),'NumPy_position':int(n[2]),'NumPy_has_gauss':int(n[3]),'NumPy_cached_gauss':float(n[4]),
           'legacy_repickled_digest':hashlib.sha256(blob).hexdigest(),'legacy_repickled_length':len(blob),'matches_original_inmemory_digest':hashlib.sha256(blob).hexdigest()==validation['RNG_sha256']}
        checks[k+'_raw_state_equal_initial']=not diff and tgood and v['CUDA_initialized'] is False
    np.savez_compressed(OUT/'RNG_raw_vectors.npz',**vectors)
    report={'schema':'full151105_component_RNG_serialization_forensic:1','source':source,'own_payload_pins':pins,'initial_validation':pin(RUN/'initial_validation.json'),
       'original_producer_inmemory_pickle_digest':validation['RNG_sha256'],'checks':checks,'passed':sum(checks.values()),'total':len(checks),'details':details,
       'new_model_FW_AD_native_Env_physics_GPU':0,'own_trusted_Torch_CPU_loads':3,'scope':'Raw serialized RNG equality, not a claim that re-pickle digests preserve original object-sharing/memo topology or that RNG never changed between samples.'}
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps({'report':pin(OUT/'report.json'),**report},indent=2));return 0 if all(checks.values()) else 2


if __name__=='__main__':raise SystemExit(main())
