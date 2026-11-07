"""One bounded four-forward interface contract; no integration/optimization."""
import hashlib
import json
from pathlib import Path
import time
import traceback

ROOT=Path(__file__).resolve().parent
P=ROOT.parent


def pin(path):
    path=Path(path).resolve();b=path.read_bytes()
    return {'path':str(path),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def plain(value):
    import numpy as np
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value


def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(plain(value),f,indent=2,allow_nan=False)


def main():
    source=ROOT/'evaluator_native_v6.py'
    sources={'evaluator':pin(source),'checker':pin(__file__),'geometry':pin(P/'retarget_gravity_v3/evaluator_v3.py')}
    inputs={'seed0':pin(P/'retarget_gravity_v3/support_refinement_v5_run1/seed0_result.json'),
            'seed1':pin(P/'retarget_gravity_v3/support_refinement_v5_run1/seed1_result.json'),
            'v5report':pin(P/'retarget_gravity_v3/support_refinement_v5_run1/report.json'),
            'binding':pin(P/'hand2_holding_v1/compiled_nominal_v2/binding.json')}
    protocol={'schema':'native_equilibrium_interface_four_forward_contract:6',
        'source_pins':sources,'input_pins':inputs,'native_forward_authorized':True,
        'integration_optimization_training_GPU_authorized':False,
        'probes':['seed0_proposed_ctrl','seed0_exact_repeat','seed1_ctrl_equals_q20','seed1_exact_repeat'],
        'fresh_MjData_each_call':True,'qvel_qacc_warmstart_applied_forces':0,
        'maximum_native_forward_calls':4,'digest_check_period':1,
        'PD_formula_absolute_roundoff_bound':1e-12,
        'geometry_calls':0,'geometry_scope':'Both exact source poses already have pinned v5 final22/189; this contract is not a new geometry enrollment.',
        'no_accuracy_or_equilibrium_threshold':True,'automatic_retry':False}
    pp=ROOT/'four_forward_contract_protocol_v1.json'
    write(pp,protocol)
    out=ROOT/'four_forward_contract_v1';out.mkdir()
    started=time.perf_counter();phase='construct';e=None;results=[]
    try:
        import numpy as np
        from evaluator_native_v6 import NativeEquilibriumEvaluator
        for rec in [*sources.values(),*inputs.values()]:assert pin(rec['path'])==rec
        e=NativeEquilibriumEvaluator.from_existing_pilot(digest_check_period=1)
        for seed in range(2):
            saved=json.loads(Path(inputs['seed'+str(seed)]['path']).read_text(encoding='utf8'))
            x=np.asarray(saved['selected_evaluation']['x26'],dtype=np.float64)
            ctrl=np.asarray(saved['proposed_ctrl20'],dtype=np.float64) if seed==0 else x[:20].copy()
            for repetition in range(2):
                phase=f'seed{seed}_repeat{repetition}'
                r=e.evaluate(x[:20],x[20:23],x[23:],ctrl)
                assert np.array_equal(r['raw']['qpos27'],np.asarray(saved['verification_state']['qpos27']))
                np.savez_compressed(out/(phase+'_raw.npz'),**r['raw'])
                write(out/(phase+'.json'),r)
                results.append(r)
        checks={'four_native_forward_returned':e.counts['mj_forward_returned']==4,
            'four_fresh_native_data':e.counts['native_MjData']==4,
            'one_COM_native_jac_per_forward':e.counts['COM_mj_jac']==4,
            'no_integration_training_GPU':e.counts['integration_steps']==0 and e.counts['training_GPU']==0,
            'all_finite_no_warning_or_semantic_unknown':all(not r['unknown'] for r in results),
            'zero_raw_warmstart_before_every_probe':all(not np.any(r['raw']['qacc_warmstart_before26']) for r in results),
            'zero_velocity_and_applied_every_probe':all(not np.any(r['raw']['qvel26']) and not np.any(r['raw']['qfrc_applied26']) and not np.any(r['raw']['xfrc_applied']) for r in results),
            'PD_formula_roundoff':all(max(np.max(np.abs(r['PD']['actuator_formula_difference20'])),np.max(np.abs(r['PD']['joint_formula_difference20'])))<=1e-12 for r in results),
            'all_per_probe_model_hashes_checked':all(r['model_hash_evidence']['before_probe_checked'] and r['model_hash_evidence']['after_probe_checked'] for r in results)}
        for seed in range(2):
            a,b=results[2*seed:2*seed+2]
            for name in a['raw']:
                aa,bb=a['raw'][name],b['raw'][name]
                if name=='contact_rows24':aa,bb=aa[:,1:],bb[:,1:]
                checks[f'seed{seed}_repeat_'+name+'_bits']=(aa.dtype==bb.dtype and aa.shape==bb.shape and np.ascontiguousarray(aa).tobytes()==np.ascontiguousarray(bb).tobytes())
        assert all(checks.values()),{k:v for k,v in checks.items() if not v}
        report={'schema':'native_equilibrium_four_forward_contract_result:6','status':'PASS_COLD_START_INTERFACE_NOT_EQUILIBRIUM',
            'protocol':pin(pp),'source_pins':sources,'checks':checks,'passed':sum(checks.values()),'total':len(checks),
            'counts':dict(e.counts),'model_audit':e.audit_model(),'elapsed_seconds':time.perf_counter()-started,
            'all_four_descriptions':[{'probe':r['probe_index'],'a_COM_world_m_s2':r['raw']['a_COM_world_m_s2'],'alpha_world_rad_s2':r['raw']['alpha_world_rad_s2'],'joint_qacc20':r['raw']['qacc26'][:20],'native_aggregates':r['native_contact_aggregates'],'contact_count':r['native_contact_count'],'efm_active':r['efm_active']} for r in results],
            'equilibrium_or_holding_qualified':False,'outputs':{p.name:pin(p) for p in out.iterdir() if p.is_file()}}
        write(out/'report.json',report)
        print(json.dumps({'report':pin(out/'report.json'),'passed':report['passed'],'total':report['total'],'counts':e.counts}))
    except BaseException as error:
        write(out/'failure.json',{'phase':phase,'error':repr(error),'traceback':traceback.format_exc(),
            'completed_returned_probes':len(results),'native_counts':None if e is None else dict(e.counts),
            'constructor_attempted':True,'no_integration_or_optimization':True,'automatic_retry':False})
        raise


if __name__=='__main__':main()
