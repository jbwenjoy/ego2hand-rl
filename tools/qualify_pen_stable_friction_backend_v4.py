"""Actual CUDA qualification of the generated cost backend, without dynamics."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_stable_friction_factory_v4 import generate,load_generated,NEW_FACTORY,verify_kernel_owner
from supermimic_research.runtime.pen_stable_friction_binding_v2 import FactoryBinding
from supermimic_research.runtime.pen_stable_friction_profile_v2 import execution_identity,ALGORITHMS
from supermimic_research.strict_json import write_json


def stats(x):
    return dict(n=len(x),median=float(np.median(x)),p95=float(np.quantile(x,.95)),max=float(x.max()))


def main():
    start=time.perf_counter(); out=ROOT/'runs/pen_stable_friction_backend_qualification_v4'
    if out.exists(): raise ValueError('Fresh qualification output required;retain failures')
    prior_pin=file_pin(ROOT/'runs/pen_solve_exit_replay_registration_v2/registration.json')
    prior=json.loads(verify_file_pin(prior_pin).read_text()); installed=prior['inputs']['installed_solver']
    backend_pin=prior['inputs']['backend_manifest']; backend=json.loads(verify_file_pin(backend_pin).read_text())
    analysis_pin=file_pin(ROOT/'runs/pen_paired_ray_friction_analysis_v1/report.json')
    analysis=json.loads(verify_file_pin(analysis_pin).read_text())
    if analysis['passed']!=analysis['total'] or not all(analysis['checks'].values()) or analysis['primitive_rows']!=139160: raise ValueError('Closed complete recorded input analysis required')
    tests=ROOT/'runs/pen_guarded_cost_tests638_v1.log'; text=tests.read_text(encoding='utf-8-sig')
    if 'Ran 638 tests' not in text or '\nOK (skipped=1)\n' not in text: raise ValueError('Full638 CPU tests required')
    data=dict(np.load(verify_file_pin(analysis['fixed_inputs']))); batch=data['inputs']; target=data['FP64_delta']
    if batch.shape!=(139160,5) or batch.dtype!=np.float32 or not np.isfinite(batch).all() or np.any(batch[:,2]<=0) or np.any(batch[:,3]<0): raise ValueError('Exact finite positive-D fixed input population required')
    out.mkdir(); manifest=generate(installed,out/'isolated_factory.py'); write_json(out/'factory_manifest.json',manifest)
    sources=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'tests/test_pen_friction_rounding_probe_v2.py']+sorted((ROOT/'tests').glob('test_pen_stable_friction*.py'))
    receipt=freeze_sources(sources,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write_json(out/'source_receipt.json',receipt)
    inputs=dict(prior_registration=prior_pin,analysis=analysis_pin,fixed_inputs=analysis['fixed_inputs'],prior_primitive_outputs=analysis['primitive_outputs'],cpu_tests=file_pin(tests),backend_manifest=backend_pin,retained_v1_failure=file_pin(ROOT/'runs/pen_stable_friction_backend_qualification_v1/report.json'),retained_v2_failure=file_pin(ROOT/'runs/pen_stable_friction_backend_qualification_v2/report.json'),compile_layout_diagnostic=file_pin(ROOT/'runs/pen_stable_friction_compile_layout_v1/report.json'),retained_v3_failure=file_pin(ROOT/'runs/pen_stable_friction_backend_qualification_v3/report.json'),rounding_probe=file_pin(ROOT/'runs/pen_friction_rounding_probe_v2/report.json'))
    write_json(out/'intent.json',dict(inputs=inputs,rows=len(batch),GPU_kernel_launches=6,native_cap=0,optimizer_budget=0,normalizer_fit_budget=0,scope='Generated production helpers and factory compile/binding only;no Model/Data/step/forward/native graph.'))
    import warp as wp
    import mujoco
    import mujoco_warp as mw
    from mujoco_warp._src import solver
    wp.init(); device=wp.get_device('cuda:0')
    if not device.is_cuda: raise ValueError('Actual CUDA device required')
    before_factory=solver._linesearch_iterative_kernel; before_stopping=solver._solve_done; module=load_generated(manifest)
    gpu_start=time.perf_counter()
    with wp.ScopedDevice(device),patch.object(mujoco,'mj_step',side_effect=AssertionError('No dynamics in qualification')),patch.object(mujoco,'mj_forward',side_effect=AssertionError('No forward in qualification')),patch.object(mw,'step',side_effect=AssertionError('No step in qualification')):
        arrays=[wp.array(batch[:,i].copy(),dtype=float,device=device) for i in range(5)]; n=len(batch)
        outputs=[wp.zeros((n,3),device=device),wp.zeros((n,3),device=device),wp.zeros((n,3,3),device=device),wp.zeros((n,3,3),device=device)]
        wp.launch(module.evaluate_cost_helpers,dim=n,inputs=arrays,outputs=outputs,device=device); wp.synchronize()
        old,new,old3,new3=[x.numpy() for x in outputs]
        alphas=(batch[:,4,None]*np.asarray([1.,.5,1.5],np.float32)).astype(np.float32)
        extra=[wp.array(np.arange(n,dtype=np.int32),dtype=int,device=device),wp.zeros(n,dtype=int,device=device),wp.array(np.full(n,n,np.int32),dtype=int,device=device)]
        alpha_array=wp.array(alphas,dtype=float,device=device)
        distinct_outputs=[wp.zeros((n,3),device=device),wp.zeros((n,3),device=device),wp.zeros((n,3,3),device=device),wp.zeros((n,3,3),device=device)]
        wp.launch(module.evaluate_declared_rows,dim=n,inputs=extra+arrays[:4]+[alpha_array],outputs=distinct_outputs,device=device)
        distinct=[x.numpy() for x in distinct_outputs]
        code='import warp as wp\nfrom '+module.__name__+' import _original_single,_original_triple,_compute_efc_eval_pt_pyramidal as _candidate_single,_compute_efc_eval_pt_3alphas_pyramidal as _candidate_triple\n'
        for label in ['original_single','candidate_single','original_triple','candidate_triple']:
            code+='\n@wp.kernel(enable_backward=False)\ndef '+label+'(j:wp.array[float],v:wp.array[float],d:wp.array[float],f:wp.array[float],a:wp.array[float],out:wp.array2d[float]):\n    i=wp.tid(); n=j.shape[0]\n'
            code+=('    p=_'+label+'(i,a[i],0,n,d[i],f,j[i],v[i])\n' if label.endswith('single') else '    p,q,r=_'+label+'(i,a[i],a[i],a[i],0,n,d[i],f,j[i],v[i])\n')
            code+='    for k in range(3):\n        out[i,k]=p[k]\n'
        solo_path=out/'separate_kernels.py'; solo_path.write_text(code,encoding='utf-8',newline='\n')
        spec=importlib.util.spec_from_file_location('pen_guarded_cost_qualification_v4_separate',solo_path)
        solo=importlib.util.module_from_spec(spec); sys.modules[spec.name]=solo; spec.loader.exec_module(solo)
        separate={}
        for label in ['original_single','candidate_single','original_triple','candidate_triple']:
            result=wp.zeros((n,3),device=device); wp.launch(getattr(solo,label),dim=n,inputs=arrays,outputs=[result],device=device); separate[label]=result.numpy()
        input_equal=all(a.numpy().tobytes()==batch[:,i].tobytes() for i,a in enumerate(arrays)) and alpha_array.numpy().tobytes()==alphas.tobytes()
        input_equal=input_equal and all(a.numpy().tobytes()==b.tobytes() for a,b in zip(extra,[np.arange(n,dtype=np.int32),np.zeros(n,np.int32),np.full(n,n,np.int32)]))

        with FactoryBinding(manifest,synchronize=wp.synchronize) as binding:
            kernel=solver._linesearch_iterative_kernel(50,solver.types.ConeType.PYRAMIDAL,True,False,True)
            owner=verify_kernel_owner(module,kernel)
            wp.load_module(module=kernel.module,device=device,block_dim=32)
            binding.verify()
        stopped=solver._solve_done is before_stopping; restored=solver._linesearch_iterative_kernel is before_factory
    gpu_wall=time.perf_counter()-gpu_start
    jar,jv,d,loss,alpha=batch.T; advanced=alpha*jv; x=jar+advanced; radius=loss/d
    handled=((-radius<jar)&(jar<radius)&(-radius<x)&(x<radius))|((jar<=-radius)&(x<=-radius))|((jar>=radius)&(x>=radius))
    previous=dict(np.load(verify_file_pin(analysis['primitive_outputs'])))
    checks=dict(full638=True,all139160=n==139160,all_finite=bool(np.isfinite(old).all() and np.isfinite(new).all() and np.isfinite(old3).all() and np.isfinite(new3).all()),
        input_bytes_unchanged=input_equal,single_derivative_hessian_bytes=old[:,1:].tobytes()==new[:,1:].tobytes(),
        triple_derivative_hessian_bytes=old3[:,:,1:].tobytes()==new3[:,:,1:].tobytes(),
        single_fallback_bytes=old[~handled].tobytes()==new[~handled].tobytes(),triple_fallback_bytes=old3[~handled].tobytes()==new3[~handled].tobytes(),
        same_handled_population=np.array_equal(handled,previous['handled'].astype(bool)),
        unchanged_original_stopping=stopped,factory_restored=restored and binding.audit['bindings_restored'],
        factory_actual_CUDA_compiled=owner and len(binding.audit['calls'])==1,source_proof=all(manifest['source_proof'].values()))

    checks['distinct_derivative_hessian_bytes']=distinct[0][:,1:].tobytes()==distinct[1][:,1:].tobytes() and distinct[2][:,:,1:].tobytes()==distinct[3][:,:,1:].tobytes()
    distinct_fallback_counts=[]
    j64,v64,d64,f64=batch[:,:4].astype(float).T; r64=f64/d64
    for k in range(3):
        x64=j64+alphas[:,k].astype(float)*v64
        h=((-r64<j64)&(j64<r64)&(-r64<x64)&(x64<r64))|((j64<=-r64)&(x64<=-r64))|((j64>=r64)&(x64>=r64))
        checks['distinct_point'+str(k)+'_fallback_bytes']=distinct[2][~h,k].tobytes()==distinct[3][~h,k].tobytes()
        distinct_fallback_counts.append(int((~h).sum()))
    for label in ['single','triple']:
        x,y=separate['original_'+label],separate['candidate_'+label]
        checks['separate_'+label+'_derivative_hessian_bytes']=x[:,1:].tobytes()==y[:,1:].tobytes()
        checks['separate_'+label+'_fallback_bytes']=x[~handled].tobytes()==y[~handled].tobytes()
    checks['all_distinct_and_separate_finite']=all(np.isfinite(x).all() for x in distinct+list(separate.values()))
    np.savez_compressed(out/'distinct_outputs.npz',alphas=alphas,original_single=distinct[0],candidate_single=distinct[1],original_three=distinct[2],candidate_three=distinct[3])
    np.savez_compressed(out/'separate_outputs.npz',**separate)
    np.savez_compressed(out/'outputs.npz',original_single=old,candidate_single=new,original_three=old3,candidate_three=new3,handled=handled)
    errors={}
    for name,values in [('original_single',old),('candidate_single',new),('original_three',old3[:,0]),('candidate_three',new3[:,0])]: errors[name]=stats(abs(values[:,0].astype(float)-target))
    identities={alg:execution_identity(alg,backend['backend_id'],manifest) for alg in ALGORITHMS}
    result=dict(schema='pen_stable_friction_backend_arithmetic_qualification:4',inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),
        factory_manifest=file_pin(out/'factory_manifest.json'),outputs=file_pin(out/'outputs.npz'),identities=identities,
        distinct_outputs=file_pin(out/'distinct_outputs.npz'),separate_outputs=file_pin(out/'separate_outputs.npz'),separate_kernels=file_pin(solo_path),distinct_fallback_counts=distinct_fallback_counts,GPU_kernel_launches=6,
        checks={k:bool(v) for k,v in checks.items()},passed=sum(bool(v) for v in checks.values()),total=len(checks),
        prior_program_derivative_bytes_equal=old[:,1:].tobytes()==previous['original'][:,1:].tobytes(),
        prior_program_comparison_scope='Observation across distinct kernel layouts,not the paired original/candidate preservation gate. V1 already showed this original-program comparison can differ.',
        rows=n,handled=int(handled.sum()),fallback=int((~handled).sum()),errors=errors,factory_binding=binding.audit,
        same_alpha_single_triple_cost_bytes_equal=dict(original=old[:,0].tobytes()==old3[:,0,0].tobytes(),candidate=new[:,0].tobytes()==new3[:,0,0].tobytes()),
        actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,wall_s=time.perf_counter()-start,gpu_pipeline_wall_s=gpu_wall,
        qualification='GUARDED_ARITHMETIC_AND_FACTORY_BINDING_ONLY_NOT_NATIVE_DYNAMICS',
        scope='Production generated single/three helpers on all139160 saved friction rows. Actual isolated CUDA factory compiled at block32,not launched on native state. '
              'Original OR stopping and metadata-only initialization preserved;W1 graph/restore/physical comparison and W32 remain unqualified. '
              'FMA/layout may change absolute arithmetic;record errors without using error improvement as a passing gate. No Actor,task,ego or acceleration admission.')
    for pin in inputs.values(): verify_file_pin(pin)
    verify_source_snapshot(receipt,live=True); write_json(out/'report.json',result)
    print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],rows=n,errors=errors,native=0)),flush=True)
    if not all(checks.values()): raise ValueError('Stable cost qualification discrepancy')


if __name__=='__main__': main()
