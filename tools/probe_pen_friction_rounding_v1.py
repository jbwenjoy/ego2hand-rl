"""Original/identity/stable/guarded controls plus explicit GPU rounding fields."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_stable_friction_factory_v1 import load_generated
from supermimic_research.strict_json import write_json


def main():
    start=time.perf_counter(); out=ROOT/'runs/pen_friction_rounding_probe_v1'
    if out.exists(): raise ValueError('Fresh rounding diagnostic required')
    qp=file_pin(ROOT/'runs/pen_stable_friction_backend_qualification_v3/report.json'); q=json.loads(verify_file_pin(qp).read_text())
    manifest=json.loads(verify_file_pin(q['factory_manifest']).read_text()); candidate=load_generated(manifest)
    raw=dict(np.load(verify_file_pin(q['inputs']['fixed_inputs']))); batch=raw['inputs']; n=len(batch)
    if batch.shape!=(139160,5) or batch.dtype!=np.float32 or not np.isfinite(batch).all() or np.any(batch[:,2]<=0): raise ValueError('Exact fixed positive-D inputs required')
    source='import warp as wp\nfrom supermimic_research.runtime.pen_friction_rounding_probe_v1 import original_single,original_triple,identity_single,identity_triple,guarded_single,guarded_triple,explicit_rounding\n'
    source+='from '+candidate.__name__+' import _compute_efc_eval_pt_pyramidal as stable_single,_compute_efc_eval_pt_3alphas_pyramidal as stable_triple\n'
    names=[prefix+'_'+kind for prefix in ['original','identity','stable','guarded'] for kind in ['single','triple']]
    signature='j:wp.array[float],v:wp.array[float],d:wp.array[float],f:wp.array[float],a:wp.array[float],out:wp.array2d[float]'
    for name in names:
        source+='\n@wp.kernel(enable_backward=False)\ndef run_'+name+'('+signature+'):\n    i=wp.tid(); n=j.shape[0]\n'
        source+=('    p='+name+'(i,a[i],0,n,d[i],f,j[i],v[i])\n' if name.endswith('single') else '    p,q,r='+name+'(i,a[i],a[i],a[i],0,n,d[i],f,j[i],v[i])\n')
        source+='    for k in range(3):\n        out[i,k]=p[k]\n'
    source+='\n@wp.kernel(enable_backward=False)\ndef intermediate('+signature+'):\n    i=wp.tid()\n    p=explicit_rounding(j[i],v[i],d[i],f[i],a[i])\n    for k in range(12):\n        out[i,k]=p[k]\n'
    out.mkdir(); path=out/'kernels.py'; path.write_text(source,encoding='utf-8',newline='\n')
    receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),path],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write_json(out/'source_receipt.json',receipt)
    write_json(out/'intent.json',dict(qualification=qp,rows=n,kernel_launches=9,native_cap=0,optimizer_budget=0,normalizer_fit_budget=0,source_receipt=file_pin(out/'source_receipt.json')))
    spec=importlib.util.spec_from_file_location('pen_friction_rounding_probe_v1_kernels',path); module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
    import warp as wp
    import mujoco
    import mujoco_warp as mw
    wp.init(); device=wp.get_device('cuda:0')
    if not device.is_cuda: raise ValueError('Actual GPU required')
    outputs={}
    with wp.ScopedDevice(device),patch.object(mujoco,'mj_step',side_effect=AssertionError('No dynamics')),patch.object(mujoco,'mj_forward',side_effect=AssertionError('No forward')),patch.object(mw,'step',side_effect=AssertionError('No dynamics')):
        arrays=[wp.array(batch[:,i].copy(),dtype=float,device=device) for i in range(5)]
        for name in names+['intermediate']:
            result=wp.zeros((n,12 if name=='intermediate' else 3),device=device)
            kernel=getattr(module,'run_'+name if name!='intermediate' else name)
            wp.launch(kernel,dim=n,inputs=arrays,outputs=[result],device=device); outputs[name]=result.numpy()
        input_equal=all(a.numpy().tobytes()==batch[:,i].tobytes() for i,a in enumerate(arrays))
    old=dict(np.load(verify_file_pin(q['outputs']))); handled=old['handled']; fallback=~handled
    comparisons={}
    for prefix in ['identity','stable','guarded']:
        comparisons[prefix]={}
        for kind in ['single','triple']:
            original,current=outputs['original_'+kind],outputs[prefix+'_'+kind]
            differences=original.view(np.uint32)!=current.view(np.uint32)
            comparisons[prefix][kind]=dict(gradient_changed=int(differences[:,1].sum()),hessian_changed=int(differences[:,2].sum()),fallback_cost_changed=int(differences[fallback,0].sum()),all_cost_changed=int(differences[:,0].sum()))
    fields=outputs['intermediate']; observations={}
    for name in names:
        observations[name]={label:int(np.sum(outputs[name][fallback,0].view(np.uint32)==fields[fallback,k].view(np.uint32))) for label,k in [('sub_rounded_costs',9),('fma_negative_cost0_product',10),('fma_positive_cost1_product',11)]}
    cache=Path(wp.config.kernel_cache_dir); cuda=[]
    for p in sorted(cache.glob('wp_pen_friction_rounding_probe_v1_kernels_*/*')):
        if p.suffix in ['.cu','.ptx','.meta']:
            target=out/'compiled'/p.name; target.parent.mkdir(exist_ok=True); shutil.copyfile(p,target); cuda.append(dict(origin=file_pin(p),saved=file_pin(target)))
    np.savez_compressed(out/'outputs.npz',**outputs,handled=handled)
    checks=dict(all139160=n==139160,finite_outputs=all(np.isfinite(a).all() for a in outputs.values()),inputs_unchanged=input_equal,compiled_CUDA_and_PTX=any(p['saved']['path'].endswith('.cu') for p in cuda) and any(p['saved']['path'].endswith('.ptx') for p in cuda))
    result=dict(schema='pen_friction_rounding_diagnostic:1',qualification=qp,fixed_inputs=q['inputs']['fixed_inputs'],source_receipt=file_pin(out/'source_receipt.json'),generated=file_pin(path),
        outputs=file_pin(out/'outputs.npz'),compiled=cuda,comparisons=comparisons,explicit_rounding_matches_on_fallback=observations,
        intermediate_fields=['x_fma','x_unfused','radius','prefix0','term0','rounded_cost0','prefix1','term1','rounded_cost1','sub_rounded_costs','fma_negative_cost0_product','fma_positive_cost1_product'],
        checks={k:bool(v) for k,v in checks.items()},passed=sum(bool(v) for v in checks.values()),total=len(checks),rows=n,fallback_rows=int(fallback.sum()),kernel_launches=9,
        actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,wall_s=time.perf_counter()-start,
        scope='Fixed saved-input compiler/rounding controls only. Guarded wrapper is an experimental arithmetic control,not a registered backend. No native/factory/graph/restore/Actor/task/ego admission. Source CUDA/PTX are saved;PTX does not itself prove final driver-generated SASS.')
    verify_source_snapshot(receipt,live=True); write_json(out/'report.json',result)
    print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],comparisons=comparisons,rounding_matches=observations)),flush=True)
    if not all(checks.values()): raise ValueError('Rounding diagnostic integrity mismatch')


if __name__=='__main__': main()
