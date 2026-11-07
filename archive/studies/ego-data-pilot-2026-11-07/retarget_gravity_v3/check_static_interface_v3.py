"""Four predeclared static interface states only; no retargeting optimization."""
from pathlib import Path
import hashlib,json,os,time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parent


def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def plain(x):
    import numpy as np
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x


def main():
    import numpy as np
    from scipy.spatial.transform import Rotation
    from evaluator_v3 import StaticRetargetEvaluator
    c=json.loads((ROOT/'static_contract_protocol_v1.json').read_text(encoding='utf8'))
    out=ROOT/'static_contract_v1';out.mkdir(exist_ok=False)
    for p in [*c['inputs'].values(),*c['source_pins'].values()]:assert pin(p['path'])==p
    started=time.perf_counter();E=None;records=[]
    try:
        E=StaticRetargetEvaluator.from_existing_pilot(max_heavy_queries=200,deadline_monotonic=started+30.)
        with np.load(c['inputs']['old_operands']['path'],allow_pickle=False) as a:gold={k:a[k].copy() for k in a.files}
        with np.load(c['inputs']['aligned_state']['path'],allow_pickle=False) as a:aligned=a['qpos27'].copy()
        seed=json.loads(Path(c['inputs']['seed2_result']['path']).read_text(encoding='utf8'))
        x=np.asarray(seed['selected_evaluation']['x26']);ar=Rotation.from_quat(aligned[23:27][[1,2,3,0]]).as_matrix();at=E.Rp.T@(aligned[20:23]-E.pp);av=Rotation.from_matrix(E.Rp.T@ar).as_rotvec()
        cases=[('qualified_seed2',x[:20],x[20:23],x[23:]),('gravity_aligned_seed',aligned[:20],at,av),('seed2_palmY_plus10um',x[:20],x[20:23]+np.array([0.,1e-5,0.]),x[23:]),('aligned_palmY_plus10um',aligned[:20],at+np.array([0.,1e-5,0.]),av)]
        for name,q,t,r in cases:
            before=dict(E.counts);s=E.evaluate(q,t,r);cv=E.constraint_vector(s,pad_gap_window_m=(50e-6,150e-6),nonpad_buffer_m=50e-6,self_buffer_m=50e-6)
            row={'name':name,'state':s,'search_constraints':cv,'counter_delta':{k:E.counts[k]-before[k] for k in E.counts}}
            records.append(row)
            with (out/f'{name}.json').open('x',encoding='utf8') as f:json.dump(plain(row),f,indent=2,allow_nan=False)
        baseline=records[0]['state'];bias_error=float(np.max(abs(baseline['qfrc_bias26_Nm'][:20]-gold['qfrc_bias20'])));passive_error=float(np.max(abs(baseline['qfrc_passive26_Nm']-gold['qfrc_passive26'])));bounds_error=max(float(np.max(abs(baseline['reachable_torque_lower20_Nm']-gold['torque_lower']))),float(np.max(abs(baseline['reachable_torque_upper20_Nm']-gold['torque_upper']))))
        checks={'exactly4_static_evaluations':E.counts['evaluations']==4,'all22_phone_and189_self_each_eval':all(len(r['state']['phone_geometry_rows'])==22 and len(r['state']['self_geometry_rows'])==189 for r in records),'all_states_geometry_unknown_empty':all(not r['state']['unknown'] for r in records),'bias_matches_prior_native_forward_within_1e12Nm':bias_error<=1e-12,'passive_matches_prior_native_forward_within_1e12Nm':passive_error<=1e-12,'true_PD_torque_bounds_match_prior_within_1e12Nm':bounds_error<=1e-12,'baseline_both_actual_contacts_unique':baseline['support_operands']['available'],'no_zero_actual_J_for_unavailable_contact':all(c['actual_hand_point_J3x26'] is None for r in records for c in r['state']['contacts'] if c['pairing']['status']!='UNIQUE_CLOSEST_PAIR'),'all_actual_and_anchor_hand_J_phonecolumns_zero':all(np.count_nonzero(r['state']['anchor_J2x3x26'][:,:,20:])==0 and all(c['actual_hand_point_J3x26'] is None or np.count_nonzero(c['actual_hand_point_J3x26'][:,20:])==0 for c in r['state']['contacts']) for r in records),'q20_all_four_identical':all(np.array_equal(r['state']['q20_rad'],baseline['q20_rad']) for r in records)}
        geometry_inputs={}
        for row in records:
            s=row['state'];prefix=row['name'];geometry_inputs[prefix+'__qpos27']=s['qpos27'];geometry_inputs[prefix+'__bias26']=s['qfrc_bias26_Nm'];geometry_inputs[prefix+'__passive26']=s['qfrc_passive26_Nm'];geometry_inputs[prefix+'__anchorJ']=s['anchor_J2x3x26']
        np.savez_compressed(out/'raw_static_operands.npz',**geometry_inputs)
        report={'schema':'gravity_retarget_single_state_static_contract:1','status':'PASS_STATIC_INTERFACE_NOT_OPTIMIZATION_OR_HOLDING' if all(checks.values()) else 'FAILED_PRESERVED','source':pin(__file__),'interface':pin(ROOT/'evaluator_v3.py'),'protocol':pin(ROOT/'static_contract_protocol_v1.json'),'checks':checks,'passed':sum(checks.values()),'total':len(checks),'native_forward_reference_comparison':{'hand_bias_max_abs_error_Nm':bias_error,'passive_max_abs_error_Nm':passive_error,'reachable_torque_bounds_max_abs_error_Nm':bounds_error,'comparison_allowance_Nm':1e-12,'not_a_scientific_accuracy_gate':True},'four_state_summary':[{'name':r['name'],'contact_pair_statuses':[x['pairing']['status'] for x in r['state']['contacts']],'support_available':r['state']['support_operands']['available'],'fixed_material_line_torque_Nm':r['state']['fixed_material_line_gravity_torque_Nm'],'all_new_search_constraints_satisfied':r['search_constraints']['all_satisfied'],'counter_delta':r['counter_delta']} for r in records],'counts':E.counts,'no_mj_forward_nativeDistance_LP_pose_calls_integration_optimization_training_GPU':True,'elapsed_seconds':time.perf_counter()-started}
        report['outputs']={p.name:pin(p) for p in out.iterdir() if p.is_file()}
        with (out/'report.json').open('x',encoding='utf8') as f:json.dump(plain(report),f,indent=2,allow_nan=False)
        print(json.dumps({'report':pin(out/'report.json'),'status':report['status'],'passed':report['passed'],'total':report['total'],'counts':E.counts}))
        if not all(checks.values()):raise ValueError('Staticinterfacecheckfailurepreserved')
    except BaseException as error:
        with (out/'failure.json').open('x',encoding='utf8') as f:json.dump({'error':repr(error),'completed_states':len(records),'counts':None if E is None else E.counts,'optimization_integration_training_GPU':0},f,indent=2)
        raise


if __name__=='__main__':main()
