"""Real Warp CPU empty-array ABI only; no Env, physics, WM or CUDA arrays."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rough_prefix.probe_transaction_v3 import _meta,_bytes,_clone,_copy_into,TransactionError


def pin(path):
    p=Path(path).resolve(strict=True)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    source=ROOT/'src/rough_prefix/probe_transaction_v3.py';v2=ROOT/'src/rough_prefix/probe_transaction_v2.py';v1=ROOT/'src/rough_prefix/probe_transaction.py'
    before={p.name:pin(p) for p in (source,v2,v1)}
    for p in (source,Path(__file__)):(args.output/p.name).write_bytes(p.read_bytes())
    checks={};records=[]
    def check(k,v):checks[k]=bool(v)
    try:
        import warp as wp
        wp.init()
        devices=wp.get_devices()
        check('CPU_device_only_before_allocations',os.environ['CUDA_VISIBLE_DEVICES']=='-1' and bool(devices)
              and all(d.is_cpu for d in devices) and len(wp.get_cuda_devices())==0)
        if not checks['CPU_device_only_before_allocations']:raise RuntimeError('CPU-only Warp device boundary failed')
        specifications=[('float32_0',(0,),wp.float32),('float32_0x3',(0,3),wp.float32),
            ('float32_2x0',(2,0),wp.float32),('float32_0x0',(0,0),wp.float32),
            ('int32_0',(0,),wp.int32),('bool_0',(0,),wp.bool),
            ('vec3_0',(0,),wp.vec3),('mat33_0',(0,),wp.mat33),('vec3_0x2',(0,2),wp.vec3)]
        pairs=[]
        for name,shape,dtype in specifications:
            target=wp.empty(shape=shape,dtype=dtype,device='cpu');saved=_clone(target)
            metadata=_meta(target);raw=_bytes(target);view=target.numpy()
            _copy_into(target,saved)
            check('real_Warp_CPU_empty_'+name,metadata[0]=='warp' and metadata[2]==shape and metadata[-1] is None
                  and target.ptr is None and raw==b'' and _bytes(saved)==b'' and _bytes(target)==b''
                  and _meta(target)==metadata and view.size==0 and target.device.is_cpu)
            pairs.append((target,saved))
            records.append({'name':name,'array_shape':list(shape),'Warp_dtype':str(target.dtype),'numpy_shape':list(view.shape),
                            'numpy_dtype':str(view.dtype),'ptr':target.ptr,'metadata':list(metadata),'raw_byte_count':len(raw),
                            'clone_ptr':saved.ptr,'restore_copy_executed':True,'device':str(target.device)})
        check('empty_shapes_and_dtypes_not_collapsed_to_one_metadata',len({_meta(a) for a,b in pairs})==len(pairs))
        full=wp.array(np.array([1.,-0.],np.float32),dtype=wp.float32,device='cpu')
        m=_meta(full)
        check('real_nonempty_CPU_pointer_and_bytes_unchanged',isinstance(m[-1],int) and m[-1]>0 and m[-1]==int(full.ptr)
              and _bytes(full)==np.array([1.,-0.],np.float32).tobytes())
        class NullArray:
            __module__='warp._src.types'
            ptr=None;dtype=wp.float32;device='cpu'
            def __init__(self,shape):self.shape=shape
            def numpy(self):raise AssertionError('Null nonempty array must reject in metadata, before reading memory')
        for shape in ((1,),(2,3),()):
            try:_meta(NullArray(shape))
            except TransactionError:checks['nonempty_null_pointer_rejected_'+repr(shape)]=True
            else:checks['nonempty_null_pointer_rejected_'+repr(shape)]=False
        # The production constructor consumes this exact pair-list metadata
        # comprehension. This is a list of real CPU buffers, not a native Env
        # or a claim that WarpTaskSnapshot has been constructed/qualified.
        native_list=SimpleNamespace(arrays=pairs+[(full,wp.clone(full))],tensors=[])
        parsed=[(_meta(target),_bytes(saved)) for target,saved in (*native_list.arrays,*native_list.tensors)]
        check('native_pair_list_metadata_parse_empty_and_nonempty',len(parsed)==10 and all(meta[-1] is None and raw==b'' for meta,raw in parsed[:-1])
              and parsed[-1][0][-1] is not None and len(parsed[-1][1])==8)
        old=ast.parse(v2.read_text());new=ast.parse(source.read_text())
        def rest(tree):return [ast.dump(n,include_attributes=False) for n in tree.body
                              if not (isinstance(n,ast.Import) and len(n.names)==1 and n.names[0].name=='math')
                              and not (isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='_meta')]
        check('only_meta_and_math_import_differ_from_frozen_v2',rest(old)==rest(new))
        def nonwarp(tree):
            f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_meta')
            return [ast.dump(n,include_attributes=False) for n in f.body if not (isinstance(n,ast.If) and ast.unparse(n.test)=="kind == 'warp'")]
        check('non_Warp_metadata_branches_unchanged',nonwarp(old)==nonwarp(new))
        check('old_v1_and_v2_original_source_bytes_unchanged',before[v2.name]==pin(v2) and before[v1.name]==pin(v1)
              and pin(v2)['sha256']=='aef300dae9bf2a7f6fd4e64f424c6d4b321afcfc3d60d823a917b9b62bb41d6f'
              and pin(v1)['sha256']=='2743180a350c953bb98dd059ba93254a906c265c843d2c433cd1edc6b7f103b8')
        check('new_source_frozen_before_and_after_real_CPU_checks',before[source.name]==pin(source))
        check('no_Env_MuJoCo_Torch_or_CUDA_device',not any(k in sys.modules for k in ('torch','mujoco','mujoco_warp')) and len(wp.get_cuda_devices())==0)
        result={'schema':'probe_transaction_v3_real_Warp_CPU_empty_array_contract:1',
            'status':'PASS_EMPTY_ARRAY_ABI_ONLY_NOT_NATIVE_PHYSICS_QUALIFICATION' if all(checks.values()) else 'FAIL_RETAINED',
            'checks':checks,'passed':sum(checks.values()),'total':len(checks),'cases':records,
            'source':pin(source),'checker':pin(__file__),'retained_v2':pin(v2),'retained_v1':pin(v1),
            'Warp_version':wp.__version__,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'Warp_devices':[str(d) for d in devices],
            'actual_cuda_device_count':len(wp.get_cuda_devices()),'real_Warp_CPU_allocation_clone_copy':True,
            'native_snapshot_constructed':False,'native_snapshot_list_metadata_only':True,'original40_or33_fake_contracts_repeated':False,
            'Env_constructed':False,'physics_steps':0,'WM_calls':0,'GPU_arrays_or_kernels':0,
            'scope':'Only null-pointer empty-array metadata/bytes/clone-copy/pair-list ABI. No retry of the retained failed physical attempt or proof of native full restore.'}
    except BaseException as exc:
        result={'schema':'probe_transaction_v3_real_Warp_CPU_empty_array_contract:1','status':'HARNESS_OR_ABI_FAILURE_RETAINED','checks':checks,
                'passed':sum(checks.values()),'total':len(checks),'error':repr(exc),'traceback':traceback.format_exc(),'cases':records,
                'source':pin(source),'checker':pin(__file__),'Env_constructed':False,'physics_steps':0,'WM_calls':0,'GPU_arrays_or_kernels':0}
    target=args.output/'report.json';target.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'passed':result['passed'],'total':result['total'],'failed':[k for k,v in checks.items() if not v],'report':pin(target)},indent=2))
    raise SystemExit(0 if result['status']=='PASS_EMPTY_ARRAY_ABI_ONLY_NOT_NATIVE_PHYSICS_QUALIFICATION' else 2)


if __name__=='__main__':main()
