"""Actual CUDA qualification of the generated cost backend, without dynamics."""
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
from supermimic_research.runtime.pen_stable_friction_factory_v1 import generate,load_generated,NEW_FACTORY,verify_kernel_owner
from supermimic_research.runtime.pen_stable_friction_binding_v1 import FactoryBinding
from supermimic_research.runtime.pen_stable_friction_profile_v1 import execution_identity,ALGORITHMS
from supermimic_research.strict_json import write_json


def stats(x):
    return dict(n=len(x),median=float(np.median(x)),p95=float(np.quantile(x,.95)),max=float(x.max()))


def main():
    start=time.perf_counter(); out=ROOT/'runs/pen_stable_friction_backend_qualification_v1'
    if out.exists(): raise ValueError('Fresh qualification output required;retain failures')
    prior_pin=file_pin(ROOT/'runs/pen_solve_exit_replay_registration_v2/registration.json')
    prior=json.loads(verify_file_pin(prior_pin).read_text()); installed=prior['inputs']['installed_solver']
    backend_pin=prior['inputs']['backend_manifest']; backend=json.loads(verify_file_pin(backend_pin).read_text())
    analysis_pin=file_pin(ROOT/'runs/pen_paired_ray_friction_analysis_v1/report.json')
    analysis=json.loads(verify_file_pin(analysis_pin).read_text())
    if analysis['passed']!=analysis['total'] or not all(analysis['checks'].values()) or analysis['primitive_rows']!=139160: raise ValueError('Closed complete recorded input analysis required')
    tests=ROOT/'runs/pen_stable_friction_tests627_v1.log'; text=tests.read_text(encoding='utf-8-sig')
    if 'Ran 627 tests' not in text or '\nOK\n' not in text: raise ValueError('Full627 CPU tests required')
    data=dict(np.load(verify_file_pin(analysis['fixed_inputs']))); batch=data['inputs']; target=data['FP64_delta']
    if batch.shape!=(139160,5) or batch.dtype!=np.float32 or not np.isfinite(batch).all() or np.any(batch[:,2]<=0) or np.any(batch[:,3]<0): raise ValueError('Exact finite positive-D fixed input population required')
    out.mkdir(); manifest=generate(installed,out/'isolated_factory.py'); write_json(out/'factory_manifest.json',manifest)
    sources=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__)]+sorted((ROOT/'tests').glob('test_pen_stable_friction*.py'))
    receipt=freeze_sources(sources,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write_json(out/'source_receipt.json',receipt)
    inputs=dict(prior_registration=prior_pin,analysis=analysis_pin,fixed_inputs=analysis['fixed_inputs'],prior_primitive_outputs=analysis['primitive_outputs'],cpu_tests=file_pin(tests),backend_manifest=backend_pin)
    write_json(out/'intent.json',dict(inputs=inputs,rows=len(batch),native_cap=0,optimizer_budget=0,normalizer_fit_budget=0,scope='Generated production helpers and factory compile/binding only;no Model/Data/step/forward/native graph.'))
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
        input_equal=all(a.numpy().tobytes()==batch[:,i].tobytes() for i,a in enumerate(arrays))
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
    checks=dict(full627=True,all139160=n==139160,all_finite=bool(np.isfinite(old).all() and np.isfinite(new).all() and np.isfinite(old3).all() and np.isfinite(new3).all()),
        input_bytes_unchanged=input_equal,single_derivative_hessian_bytes=old[:,1:].tobytes()==new[:,1:].tobytes(),
        triple_derivative_hessian_bytes=old3[:,:,1:].tobytes()==new3[:,:,1:].tobytes(),
        single_fallback_bytes=old[~handled].tobytes()==new[~handled].tobytes(),triple_fallback_bytes=old3[~handled].tobytes()==new3[~handled].tobytes(),
        same_handled_population=np.array_equal(handled,previous['handled'].astype(bool)),
        unchanged_original_stopping=stopped,factory_restored=restored and binding.audit['bindings_restored'],
        factory_actual_CUDA_compiled=owner and len(binding.audit['calls'])==1,source_proof=all(manifest['source_proof'].values()),
        derivative_matches_prior=old[:,1:].tobytes()==previous['original'][:,1:].tobytes())
    np.savez_compressed(out/'outputs.npz',original_single=old,candidate_single=new,original_three=old3,candidate_three=new3,handled=handled)
    errors={}
    for name,values in [('original_single',old),('candidate_single',new),('original_three',old3[:,0]),('candidate_three',new3[:,0])]: errors[name]=stats(abs(values[:,0].astype(float)-target))
    identities={alg:execution_identity(alg,backend['backend_id'],manifest) for alg in ALGORITHMS}
    result=dict(schema='pen_stable_friction_backend_arithmetic_qualification:1',inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),
        factory_manifest=file_pin(out/'factory_manifest.json'),outputs=file_pin(out/'outputs.npz'),identities=identities,
        checks={k:bool(v) for k,v in checks.items()},passed=sum(bool(v) for v in checks.values()),total=len(checks),
        rows=n,handled=int(handled.sum()),fallback=int((~handled).sum()),errors=errors,factory_binding=binding.audit,
        same_alpha_single_triple_cost_bytes_equal=dict(original=old[:,0].tobytes()==old3[:,0,0].tobytes(),candidate=new[:,0].tobytes()==new3[:,0,0].tobytes()),
        actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,wall_s=time.perf_counter()-start,gpu_pipeline_wall_s=gpu_wall,
        qualification='ARITHMETIC_AND_FACTORY_BINDING_ONLY_NOT_NATIVE_DYNAMICS',
        scope='Production generated single/three helpers on all139160 saved friction rows. Actual isolated CUDA factory compiled at block32,not launched on native state. '
              'Original OR stopping and metadata-only initialization preserved;W1 graph/restore/physical comparison and W32 remain unqualified. '
              'FMA/layout may change absolute arithmetic;record errors without using error improvement as a passing gate. No Actor,task,ego or acceleration admission.')
    for pin in inputs.values(): verify_file_pin(pin)
    verify_source_snapshot(receipt,live=True); write_json(out/'report.json',result)
    print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],rows=n,errors=errors,native=0)),flush=True)
    if not all(checks.values()): raise ValueError('Stable cost qualification discrepancy')


if __name__=='__main__': main()
