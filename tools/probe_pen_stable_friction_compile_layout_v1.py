"""Separate original/candidate helper kernels to diagnose mixed-kernel CSE."""
import importlib.util
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.runtime.pen_stable_friction_factory_v1 import load_generated
from supermimic_research.strict_json import write_json


def main():
    start=time.perf_counter(); out=ROOT/'runs/pen_stable_friction_compile_layout_v1'
    if out.exists(): raise ValueError('Fresh diagnostic output required')
    qpin=file_pin(ROOT/'runs/pen_stable_friction_backend_qualification_v2/report.json'); q=json.loads(verify_file_pin(qpin).read_text())
    manifest=json.loads(verify_file_pin(q['factory_manifest']).read_text()); module=load_generated(manifest)
    raw=dict(np.load(verify_file_pin(q['inputs']['fixed_inputs']))); batch=raw['inputs']; n=len(batch)
    source='import warp as wp\nfrom '+module.__name__+' import _original_single,_original_triple,_compute_efc_eval_pt_pyramidal as _candidate_single,_compute_efc_eval_pt_3alphas_pyramidal as _candidate_triple\n'
    for name in ['original_single','candidate_single','original_triple','candidate_triple']:
        source+='\n@wp.kernel\ndef '+name+'(j:wp.array[float],v:wp.array[float],d:wp.array[float],f:wp.array[float],a:wp.array[float],out:wp.array2d[float]):\n    i=wp.tid(); n=j.shape[0]\n'
        if name.endswith('single'): source+='    p=_'+name+'(i,a[i],0,n,d[i],f,j[i],v[i])\n'
        else: source+='    p,q,r=_'+name+'(i,a[i],a[i],a[i],0,n,d[i],f,j[i],v[i])\n'
        source+='    for k in range(3):\n        out[i,k]=p[k]\n'
    out.mkdir(); path=out/'isolated_comparison.py'; path.write_text(source,encoding='utf-8',newline='\n')
    spec=importlib.util.spec_from_file_location('pen_stable_compile_layout_v1',path); kernels=importlib.util.module_from_spec(spec); sys.modules[spec.name]=kernels; spec.loader.exec_module(kernels)
    import warp as wp
    import mujoco
    import mujoco_warp as mw
    wp.init(); device=wp.get_device('cuda:0'); outputs={}
    write_json(out/'intent.json',dict(rows=n,kernel_launches=4,actual_native_integrations=0,optimizer_steps=0,qualification=qpin,source=file_pin(__file__),generated=file_pin(path)))
    with wp.ScopedDevice(device),patch.object(mujoco,'mj_step',side_effect=AssertionError('No dynamics')),patch.object(mujoco,'mj_forward',side_effect=AssertionError('No forward')),patch.object(mw,'step',side_effect=AssertionError('No dynamics')):
        arrays=[wp.array(batch[:,i].copy(),dtype=float,device=device) for i in range(5)]
        for name in ['original_single','candidate_single','original_triple','candidate_triple']:
            result=wp.zeros((n,3),device=device); wp.launch(getattr(kernels,name),dim=n,inputs=arrays,outputs=[result],device=device); outputs[name]=result.numpy()
        input_equal=all(a.numpy().tobytes()==batch[:,i].tobytes() for i,a in enumerate(arrays))
    mixed=dict(np.load(verify_file_pin(q['outputs']))); h=mixed['handled']; comparisons={}
    for suffix in ['single','triple']:
        old,new=outputs['original_'+suffix],outputs['candidate_'+suffix]
        comparisons[suffix]=dict(derivative_hessian_bytes_equal=old[:,1:].tobytes()==new[:,1:].tobytes(),fallback_bytes_equal=old[~h].tobytes()==new[~h].tobytes(),
            fallback_cost_differences=int(np.sum(old[~h,0].view(np.uint32)!=new[~h,0].view(np.uint32))),
            derivative_differences=int(np.sum(old[:,1:].view(np.uint32)!=new[:,1:].view(np.uint32))))
    np.savez_compressed(out/'outputs.npz',**outputs)
    write_json(out/'report.json',dict(qualification=qpin,source=file_pin(__file__),generated=file_pin(path),fixed_inputs=q['inputs']['fixed_inputs'],outputs=file_pin(out/'outputs.npz'),
        comparisons=comparisons,inputs_unchanged=input_equal,rows=n,kernel_launches=4,actual_native_integrations=0,optimizer_steps=0,wall_s=time.perf_counter()-start,
        scope='Separate kernels vs mixed-kernel observations. Compiler-context diagnostic only;no native dynamics/backend admission.'))
    print(json.dumps(dict(report=file_pin(out/'report.json'),comparisons=comparisons)),flush=True)


if __name__=='__main__': main()
