"""Two frozen-rank0 fresh-forward contact export probes; no step or LP."""
from pathlib import Path
import hashlib
import json
import time
import traceback

ROOT=Path(__file__).resolve().parent
P=ROOT.parent


def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def plain(v):
    import numpy as np
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    return v


def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(plain(v),f,indent=2,allow_nan=False)


def main():
    rp=P/'native_equilibrium_v6/completed_v1/report.json'
    protocol={'schema':'same_forward_native_contact_design_contract:7',
        'CPU_forward_authorized':True,'step_LP_learning_GPU_authorized':False,
        'source_pins':{'wrapper':pin(ROOT/'evaluator_contact_design_v7.py'),
            'base':pin(P/'native_equilibrium_v6/evaluator_native_v6.py'),'checker':pin(__file__)},
        'inputs':{'old_native_report':pin(rp),
            'old_rank0_state':pin(P/'native_equilibrium_v6/completed_v1/seed0_rank0_state.npz')},
        'rule':'Exactly same frozen seed0rank0 x46 twice, fresh native MjData each; base forward once then only hand-contact jacobians at native con.pos/hand geom body.',
        'maximum_native_forwards':2,'positive_Fn_filter':False,'no_static_acceleration_gate':True,
        'contact_points_not_arbitrary_witness_or_centroid':True,'model_digest_period':1,
        'expected_same_geometry_contacts':3,'all_PD_limit_flags_and_unitgear_saved':True,
        'repeat_rule':'Raw contact/J/PD/context numerical fields byte-equal, excluding diagnostic probe ID and cumulative counters.',
        'automatic_retry':False}
    pp=ROOT/'contract_protocol_v1.json';write(pp,protocol)
    out=ROOT/'contract_v1';out.mkdir();started=time.perf_counter();phase='construct';e=None;results=[]
    try:
        import numpy as np
        from evaluator_contact_design_v7 import NativeContactDesignEvaluator
        for rec in [*protocol['source_pins'].values(),*protocol['inputs'].values()]:assert pin(rec['path'])==rec
        parent=json.loads(rp.read_text(encoding='utf8'));old=next(s for s in parent['seed_results'] if s['seed']==0)['shortlist'][0]
        assert old['rank']==0 and old['state']==protocol['inputs']['old_rank0_state']
        assert parent['model_audit']['passed'] is True and old['unknown']==[]
        x=np.asarray(old['x46'],dtype=np.float64)
        e=NativeContactDesignEvaluator.from_existing_pilot(digest_check_period=1)
        for i in range(2):
            phase=f'probe{i}'
            result=e.evaluate(x[:20],x[20:23],x[23:26],x[26:])
            write(out/f'probe{i}.json',result)
            np.savez_compressed(out/f'probe{i}_raw.npz',**result['raw'])
            results.append(result)
        a,b=results;ca,cb=a['contact_design'],b['contact_design']
        def bits(x,y):
            aa,bb=np.asarray(x),np.asarray(y)
            return aa.dtype==bb.dtype and aa.shape==bb.shape and np.ascontiguousarray(aa).tobytes()==np.ascontiguousarray(bb).tobytes()
        checks={'two_native_forwards_only':e.counts['mj_forward_attempted']==e.counts['mj_forward_returned']==2,
            'two_fresh_native_MjData':e.counts['native_MjData']==2,
            'three_phone_hand_contacts_each':ca['available'] is True and cb['available'] is True and ca['phone_hand_contact_count']==cb['phone_hand_contact_count']==3,
            'six_added_same_state_contact_jacobians':e.counts['phone_hand_contact_jac_attempted']==e.counts['phone_hand_contact_jac_returned']==6,
            'zero_steps_LP_learning_GPU':e.counts['integration_steps']==e.counts['training_GPU']==e.counts['final_geometry_evaluations']==0,
            'known_native_evidence_both':not a['unknown'] and not b['unknown'],
            'same_native_qpos_as_old_rank0':bits(a['raw']['qpos27'],np.asarray(old['native']['raw']['qpos27'])),
            'same_native_ctrl_as_old_rank0':bits(a['raw']['ctrl20'],np.asarray(old['native']['raw']['ctrl20']))}
        for key in a['raw']:
            aa,bb=a['raw'][key],b['raw'][key]
            if key=='contact_rows24':aa,bb=aa[:,1:],bb[:,1:]
            checks['repeat_raw_'+key]=bits(aa,bb)
        for i,(aa,bb) in enumerate(zip(ca['phone_hand_contacts'],cb['phone_hand_contacts'])):
            for key in ('position_world_m','frame_rows_normal_t1_t2_world','native_friction5',
                        'hand_Jp3x20_world','hand_Jp3x26_world'):
                checks[f'repeat_contact{i}_'+key]=bits(aa[key],bb[key])
            checks[f'contact{i}_actual_pos_body_sign_scope']=(aa['native_contact_index']==i
                and aa['force_on_object_sign']==a['native_contact_details'][i]['force_on_object_sign']
                and bits(aa['position_world_m'],a['raw']['contact_rows24'][i,6:9])
                and aa['native_condim']==3 and aa['finger_group'] in ('finger1','finger2')
                and not np.any(aa['hand_Jp3x26_world'][:,20:]))
        for key,value in ca['PD_raw'].items():
            if isinstance(value,np.ndarray):checks['repeat_PD_'+key]=bits(value,cb['PD_raw'][key])
        pd=ca['PD_raw']
        checks['all_actual_limit_flags_true']=all(np.all(pd[k]) for k in ('actuator_ctrllimited20','actuator_forcelimited20','joint_actfrclimited20'))
        checks['actual_direct_unitgear20']=np.array_equal(pd['actuator_trnid20x2'][:,0],np.arange(20)) and np.array_equal(pd['actuator_gear20x6'],np.c_[np.ones(20),np.zeros((20,5))])
        audit=e.audit_model();checks['final_full_model_audit_returned_without_exception']=True
        assert all(checks.values()),{k:v for k,v in checks.items() if not v}
        report={'schema':'native_contact_design_two_forward_contract_result:7',
            'status':'PASS_SAME_FORWARD_CONTACT_JACOBIAN_AND_PD_EXPORT_NOT_HOLDING',
            'protocol':pin(pp),'source_pins':protocol['source_pins'],'checks':checks,
            'passed':int(sum(checks.values())),'total':len(checks),'counts':dict(e.counts),
            'model_audit':audit,'elapsed_seconds':time.perf_counter()-started,
            'probe_files':[pin(out/f'probe{i}.json') for i in range(2)],
            'contact_summary':[{'native_contact_index':c['native_contact_index'],'hand_geom_name':c['hand_geom_name'],
                'hand_body_name':c['hand_body_name'],'finger_group':c['finger_group'],'native_friction5':c['native_friction5'],
                'native_condim':c['native_condim'],'hand_J_norm':float(np.linalg.norm(c['hand_Jp3x20_world']))} for c in ca['phone_hand_contacts']],
            'equilibrium_or_torque_or_holding_qualified':False,
            'outputs':{p.name:pin(p) for p in out.iterdir() if p.is_file()}}
        write(out/'report.json',report)
        print(json.dumps({'report':pin(out/'report.json'),'passed':report['passed'],'total':report['total'],'counts':e.counts}))
    except BaseException as error:
        write(out/'failure.json',{'phase':phase,'error':repr(error),'traceback':traceback.format_exc(),
            'counts':None if e is None else dict(e.counts),'completed_probes':len(results),
            'constructor_attempted':True,'automatic_retry':False})
        raise


if __name__=='__main__':main()
