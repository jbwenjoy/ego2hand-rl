"""Two identical H2 planning branches, one fixed source, no chosen rollout."""
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
    if isinstance(v,(tuple,list)):return [plain(x) for x in v]
    return v


def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(plain(v),f,indent=2,allow_nan=False)


def main():
    rp=P/'native_equilibrium_v6/completed_v1/report.json'
    report=json.loads(rp.read_text(encoding='utf8'))
    candidate=next(s for s in report['seed_results'] if s['seed']==0)['shortlist'][0]
    allowed_false={'COM_acceleration_infinity_m_s2','angular_acceleration_infinity_rad_s2','joint_acceleration_infinity_rad_s2'}
    native_keys=allowed_false|{'native_other_penetration_m','native_other_normal_force_N',
        'native_finite_no_warnings','native_two_pads_loaded','native_pad_budget',
        'native_pad_compression','requested_ctrl_unclamped'}
    repeat_keys={'qpos27','ctrl20','qacc26','qfrc_actuator26','qfrc_constraint26',
                 'a_COM_world_m_s2','alpha_world_rad_s2','contact_rows24'}
    def acceptable_native_gate_map(value):
        return (isinstance(value,dict) and set(value)==native_keys
                and all(type(v) is bool for v in value.values())
                and all(v is True for k,v in value.items() if k not in allowed_false))
    repeat=candidate.get('fresh_repeat_bit_equal')
    if (report['model_audit']['passed'] is not True or candidate['rank']!=0
            or candidate['unknown']!=[] or not isinstance(repeat,dict)
            or set(repeat)!=repeat_keys or not all(v is True for v in repeat.values())
            or not acceptable_native_gate_map(candidate.get('native_entry_gates'))
            or not acceptable_native_gate_map(candidate.get('fresh_repeat_entry_gates'))
            or candidate['geometry']['geometry_gate_pass'] is not True
            or candidate['geometry']['complete22_189'] is not True):
        raise ValueError('Fixed geometry/loaded/ctrl/audit seed0rank0, only acceleration gates descriptive')
    protocol={'schema':'native_mpc_two_branch_H2_contract_protocol:1',
        'planning_steps_authorized':True,'actual_chosen_rollout_authorized':False,
        'source_pins':{'evaluator':pin(ROOT/'rollout_probe_v1.py'),'checker':pin(__file__)},
        'inputs':{'native_search_report':pin(rp),'initial_state':candidate['state'],
                  'model_binding':pin(P/'hand2_holding_v1/compiled_nominal_v2/binding.json')},
        'source_state_rule':'Same exact qpos27/zeroqvel26/ctrl20 seed0rank0; freshData/applied0/warmstart0/time0; one initial forward counted separately.',
        'controls':'Same original ctrl20 held for both20ms intervals in each of two identical complete-copy branches.',
        'dt_s':.00015625,'control_interval_s':.02,'steps_per_control':128,
        'horizon_controls':2,'branches':2,'maximum_planning_steps':512,
        'actual_chosen_steps':0,'initial_forward_count':1,
        'repeat_rule':'Raw equality for all trace/contact24/finalintegration and endpoint physical observations; excludes counters/IDs/walltime.',
        'no_extra_branch_forward':True,'source_integration_and_physical_derived_digest_unchanged':True,
        'neural_training_GPU_or_parameter_change':False,'automatic_retry':False,
        'scope':'Privileged true-model branch interface only, not a holding/controller/policy success test.'}
    pp=ROOT/'two_branch_contract_protocol_v1.json';write(pp,protocol)
    out=ROOT/'two_branch_contract_v1';out.mkdir();start=time.perf_counter();probe=None;phase='construct';initial_forward=0
    try:
        import numpy as np
        import mujoco
        from rollout_probe_v1 import RolloutProbe
        for rec in [*protocol['inputs'].values(),*protocol['source_pins'].values()]:assert pin(rec['path'])==rec
        probe=RolloutProbe.from_existing_pilot(model_digest_period=128)
        source=mujoco.MjData(probe.m)
        with np.load(candidate['state']['path'],allow_pickle=False) as z:
            qpos=z['qpos27'].copy();qvel=z['qvel26'].copy();ctrl=z['ctrl20'].copy()
        assert qpos.shape==(27,) and qvel.shape==(26,) and ctrl.shape==(20,) and not np.any(qvel)
        assert np.array_equal(qpos,np.asarray(candidate['native']['raw']['qpos27']))
        assert np.array_equal(ctrl,np.asarray(candidate['native']['raw']['ctrl20']))
        source.qpos[:]=qpos;source.qvel[:]=qvel;source.ctrl[:]=ctrl
        source.qacc_warmstart[:]=0;source.qfrc_applied[:]=0;source.xfrc_applied[:]=0;source.time=0
        phase='one_initial_source_forward';mujoco.mj_forward(probe.m,source);initial_forward+=1
        source_before=probe.source_digest(source)
        initial_integration=probe.integration_state(source)
        np.savez_compressed(out/'source_boundary_raw.npz',qpos27=source.qpos.copy(),qvel26=source.qvel.copy(),
            ctrl20=source.ctrl.copy(),qacc26=source.qacc.copy(),qacc_warmstart26=source.qacc_warmstart.copy(),
            integration_state=initial_integration)
        controls=np.tile(ctrl,(2,1));branches=[]
        for index in range(2):
            phase=f'planning_branch{index}'
            result=probe.evaluate(source,controls)
            arrays={**{'trace__'+k:v for k,v in result['trace'].items()},
                'contact_rows24':result['contact_rows24'],'final_integration_state':result['final_integration_state']}
            np.savez_compressed(out/f'branch{index}_raw.npz',**arrays)
            write(out/f'branch{index}.json',result)
            branches.append(result)
            if result['unknown']:raise ValueError('Unknown planning branch, raw evidence preserved')
        a,b=branches
        def bits(x,y):
            aa,bb=np.asarray(x),np.asarray(y)
            return aa.dtype==bb.dtype and aa.shape==bb.shape and np.ascontiguousarray(aa).tobytes()==np.ascontiguousarray(bb).tobytes()
        checks={'complete_H2_both':all(v['status']=='COMPLETE_H2_PLANNING_ONLY' and len(v['snapshots'])==2 for v in branches),
            '512_planning_steps':sum(v['actual_planning_steps'] for v in branches)==512,
            'no_chosen_steps_or_extra_forward':probe.counts['actual_chosen_steps']==0 and probe.counts['extra_forward_calls']==0,
            'two_complete_independent_clones':probe.counts['mj_copyData']==2 and probe.counts['branch_MjData']==2 and all(v['complete_clone_equal'] for v in branches),
            'source_before_after_unchanged':source_before==probe.source_digest(source) and all(v['source_state_before']==v['source_state_after']==source_before for v in branches),
            'repeated_contact_rows_bits':bits(a['contact_rows24'],b['contact_rows24']),
            'repeated_final_integration_bits':bits(a['final_integration_state'],b['final_integration_state'])}
        for k in a['trace']:checks['repeat_trace_'+k]=bits(a['trace'][k],b['trace'][k])
        for i in range(2):
            for k in a['snapshots'][i]['state']:checks[f'repeat_POST{i+1}_'+k]=bits(a['snapshots'][i]['state'][k],b['snapshots'][i]['state'][k])
            checks[f'repeat_POST{i+1}_full_integration_bits']=bits(a['snapshots'][i]['integration_state'],b['snapshots'][i]['integration_state'])
            checks[f'repeat_POST{i+1}_integration_sha']=(a['snapshots'][i]['integration_state_sha256']==b['snapshots'][i]['integration_state_sha256'])
        assert all(checks.values()),{k:v for k,v in checks.items() if not v}
        audit=probe.audit_model()
        receipt={'schema':'native_mpc_two_branch_H2_contract_result:1','status':'PASS_REPEATABLE_PLANNING_BRANCH_NOT_CHOSEN_ROLLOUT',
            'protocol':pin(pp),'checks':checks,'passed':sum(checks.values()),'total':len(checks),
            'counts':dict(probe.counts),'initial_source_MjData':1,'initial_forward_count':initial_forward,
            'model_audit':audit,'source_state_before':source_before,'source_state_after':probe.source_digest(source),
            'elapsed_seconds':time.perf_counter()-start,'actual_chosen_steps':0,
            'planning_steps':sum(v['actual_planning_steps'] for v in branches),'holding_or_policy_qualified':False,
            'outputs':{p.name:pin(p) for p in out.iterdir() if p.is_file()}}
        write(out/'report.json',receipt)
        print(json.dumps({'report':pin(out/'report.json'),'passed':receipt['passed'],'total':receipt['total'],'counts':probe.counts,'initial_forward':initial_forward}))
    except BaseException as error:
        write(out/'failure.json',{'phase':phase,'error':repr(error),'traceback':traceback.format_exc(),
            'counts':None if probe is None else dict(probe.counts),'initial_forward_count':initial_forward,
            'actual_chosen_steps':0,'automatic_retry':False})
        raise


if __name__=='__main__':main()
