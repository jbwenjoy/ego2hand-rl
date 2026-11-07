"""Independent retained-model allowlist, native EFC primals and trace checks."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.runtime.native_evaluation import model_digest_except_dt
from review_mug_support_trials import check_case


def main(args):
    import mujoco
    run,reference,out=args.run.resolve(),args.reference.resolve(),args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    report=json.loads((run/'report.json').read_text());intent=json.loads((run/'intent.json').read_text());bind=json.loads((reference/'binding.json').read_text())
    for pin in report['sources']:verify_file_pin(pin)
    baseline=mujoco.MjModel.from_binary_path(str(verify_file_pin(bind['model'])))
    groups={baseline.geom(f'right_finger{i}_{s}').id:i-1 for i in [1,2,3] for s in ['link3_col','link4_col','tip_sensor_col']}
    parts=set(bind['mug_part_geom_ids']);indices=[i for i,(a,b) in enumerate(zip(baseline.pair_geom1,baseline.pair_geom2)) if (a in parts and b in groups) or (b in parts and a in groups)]
    model_checks=[]
    for spec in report['models']:
        model=mujoco.MjModel.from_binary_path(str(verify_file_pin(spec['model'])));expected_dim=baseline.pair_dim.copy();expected_mu=baseline.pair_friction.copy()
        if spec['dim']==4:expected_dim[indices]=4;expected_mu[indices,2]=spec['spin_m']
        exact=np.array_equal(model.pair_dim,expected_dim) and np.array_equal(model.pair_friction,expected_mu)
        model.pair_dim[:]=baseline.pair_dim;model.pair_friction[:]=baseline.pair_friction
        remainder_equal=model_digest_except_dt(model,np)==model_digest_except_dt(baseline,np) and model.opt.timestep==baseline.opt.timestep
        model_checks.append(dict(mode=spec['name'],declared_pair_fields_exact=bool(exact),other_arrays_and_options_equal=bool(remainder_equal)))
    traces=[];statics=[];primal_count=0;efc_count=0
    for case in report['cases']:
        model=mujoco.MjModel.from_binary_path(str(verify_file_pin(case['model'])));model.opt.timestep=case['dt_s']
        seed=next(s for s in intent['seeds'] if s['name']==case['seed']);state=np.load(verify_file_pin(seed['pose']));q,u=state['qpos27'],state['ctrl20']
        traces.append(check_case(case,q,u,model,bind,groups))
        data=mujoco.MjData(model);data.qpos[:]=q;data.ctrl[:]=u;mujoco.mj_forward(model,data)
        static=json.loads(verify_file_pin(case['static']).read_text());raw=np.load(verify_file_pin(case['basis']))
        R=data.xmat[bind['mug_body_id']].reshape(3,3);arm=data.xipos[bind['mug_body_id']]-data.xpos[bind['mug_body_id']]
        W=[];L=[]
        for saved in static['contacts']:
            con=data.contact[saved['contact_index']]
            if list(map(int,con.geom))!=saved['geom_ids'] or con.dim!=saved['dim']:raise ValueError('Changed native contact identity/dimension')
            for j in range(2*(con.dim-1)):
                unit=np.zeros(data.nefc);unit[con.efc_address+j]=1.;generalized=np.zeros(model.nv)
                mujoco.mj_mulJacTVec(model,data,generalized,unit);efc_count+=1
                F=generalized[20:23];T=R@generalized[23:26]-np.cross(arm,F)
                W.append(np.r_[F,T]);L.append(-generalized[:20])
        W=np.array(W).reshape(-1,6).T;L=np.array(L).reshape(-1,20).T
        werr=float(np.max(np.abs(W-raw['wrench6xN']))) if W.size else 0.;lerr=float(np.max(np.abs(L-raw['joint_load20xN']))) if L.size else 0.
        witnesses=[]
        for name in ['cone6D','fixed_command','full_command','maximum_support']:
            answer=static[name]
            if not answer['success']:continue
            lam=np.array(answer['witness']['ray_coefficients_N']);target=raw['target6'].copy()
            if name=='maximum_support':target=np.array([0,0,answer['support_weight_N'],0,0,0.])
            error=W@lam-target;valid=np.min(lam)>=-1e-8 and np.abs(error[:3]).max()<1e-8 and np.abs(error[3:]).max()<1e-9
            if name!='cone6D':
                tau=raw['offset20']+L@lam
                lo=raw['fixed_torque_lower20'] if name=='fixed_command' else raw['torque_lower20']
                hi=raw['fixed_torque_upper20'] if name=='fixed_command' else raw['torque_upper20']
                margin=np.minimum(tau-(lo-raw['dry_friction20']),(hi+raw['dry_friction20'])-tau).min();valid&=margin>=-1e-8
            witnesses.append(dict(name=name,passed=bool(valid)));primal_count+=1
        statics.append(dict(case=case['static'],wrench_error=werr,joint_load_error=lerr,witnesses=witnesses,
                            passed=bool(werr<1e-10 and lerr<1e-10 and all(x['passed'] for x in witnesses))))
    checks=dict(exact12_partial_cases=len(traces)==12 and all(all(t['checks'].values()) for t in traces),
        model_change_allowlist_and_remainder=all(x['declared_pair_fields_exact'] and x['other_arrays_and_options_equal'] for x in model_checks),
        exactly1152_pad_object_pairs=len(indices)==1152,
        baseline_byte_parity_barrier=all(x['passed'] and all(x['arrays'].values()) for x in report['baseline_barrier']) and len(report['baseline_barrier'])==2,
        native_static_EFC_and_primal_checks=all(x['passed'] for x in statics),
        actual_step_total=sum(c['summary']['completed_steps'] for c in report['cases'])==report['integration_steps']==23485,
        no_partial_endpoint_success=all(not x['qualified'] and not x['dt_pair_consistency_pass'] for x in report['dt_pairs']))
    out.mkdir(parents=True);result=dict(schema='mug_contact_modes_saved_review:1',source=file_pin(__file__),trace_helper=file_pin(ROOT/'tools/review_mug_support_trials.py'),run=file_pin(run/'report.json'),
        checks=checks,passed=sum(checks.values()),total=len(checks),models=model_checks,traces=traces,static=statics,
        trace_checks_passed=sum(v for t in traces for v in t['checks'].values()),trace_checks_total=sum(len(t['checks']) for t in traces),
        native_EFC_columns=efc_count,feasible_primals=primal_count,native_static_forward=12,native_endpoint_velocity_checks=12,integration_steps=0,neural_updates=0,
        scope='Declared model-array changes, saved failed trace continuity/COM/limits/endpoint velocity,10ms contact sums and native static EFC/primal readback. Not independent infeasibility/optimality, calibrated material, unrecorded aid arrays, continuous geometry or dynamic success proof.')
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(report=file_pin(out/'report.json'),checks=checks,trace_checks=[result['trace_checks_passed'],result['trace_checks_total']],EFC_columns=efc_count,primals=primal_count)))
    if not all(checks.values()):raise ValueError('Retain failed contact-mode review')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
