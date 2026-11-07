"""Read back cone/gravity primals, source triplets and real failed hold traces."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.contact_envelopes import unit_normal_envelope,rotate_wrenches
from supermimic_research.mechanics.compound_collision import CompoundCollisionAudit
from supermimic_research.mechanics.compound_contacts import CompoundContacts
from review_mug_support_trials import check_case


def main(args):
    import mujoco
    names=['reference','patches','envelopes','selection','fit','output'];paths={n:getattr(args,n).resolve() for n in names};out=paths['output']
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    reports={n:json.loads((paths[n]/'report.json').read_text()) for n in ['envelopes','selection','fit']}
    for report in reports.values():
        for pin in report['sources']:verify_file_pin(pin)
    bind=json.loads((paths['reference']/'binding.json').read_text());model=mujoco.MjModel.from_binary_path(str(verify_file_pin(bind['model'])));data=mujoco.MjData(model)
    p=np.load(verify_file_pin(json.loads((paths['patches']/'report.json').read_text())['output']));z=np.load(paths['reference']/'reference61.npz');COM=z['object_COM_in_BODY_m']
    R=np.array(reports['envelopes']['object_rotation_robot_from_BODY']);sourceR=z['T_source_world_from_object_BODY_F64'][0,:3,:3]
    conditional=R@sourceR.T@np.array([0,0,-9.81]);gravity=np.array(reports['envelopes']['gravity_robot_world'])
    gravity_valid=np.max(abs(conditional-reports['envelopes']['gravity_conditional_source_Z_up_in_robot_world']))<1e-12 and np.array_equal(gravity,model.opt.gravity)
    envelope_primals=0;separators=0
    for envelope in reports['envelopes']['results']:
        raw=np.load(verify_file_pin(envelope['basis']));W=raw['wrench_world'] if 'wrench_world' in raw else raw['wrench6xN']
        if envelope['kind']=='UNCERTAIN_CANDIDATE_ENVELOPE':
            B=raw['wrench_BODY'];ids=raw['source_candidate_index'];normals=-p['outward_source_normals_BODY'][ids];points=p['source_points_BODY_m'][ids]-COM
            if np.max(abs(np.sum(B[:3].T*normals,axis=1)-1))>1e-10 or np.max(abs(B[3:].T-np.cross(points,B[:3].T)))>1e-12 or np.max(abs(W-np.r_[R@B[:3],R@B[3:]]))>1e-12:raise ValueError('Candidate force/COM/phase units changed')
        for target_name,answers in envelope['targets'].items():
            target=np.array(reports['envelopes']['targets'][target_name])
            for name in ['force_only','full6D']:
                answer=answers[name]
                if not answer['success']:continue
                lam=np.array(answer['witness']['ray_coefficients_N']);e=W@lam-target
                if np.min(lam)<-1e-8 or np.max(abs(e[:3]))>1e-8 or (name=='full6D' and np.max(abs(e[3:]))>1e-9):raise ValueError('Envelope primal changed')
                envelope_primals+=1
            sep=answers['separation']
            if sep is not None and sep['verified']:
                y=np.array(sep['witness_dual_to_F_N_T_Nm'])
                if np.min(W.T@y)<-1e-9 or abs(target@y+1)>1e-9:raise ValueError('Counterfactual separator changed')
                separators+=1
    triplet_primals=0;triplet_counts=[]
    for name in ['all30','nonnegative21']:
        saved=np.load(paths['selection']/(name+'.npz'));triples=saved['triples']
        if len(set(map(tuple,triples)))!=len(triples):raise ValueError('Repeated source triplets')
        for edges in [4,8]:
            for gravity_name,target in reports['envelopes']['targets'].items():
                prefix=f'edges{edges}_{gravity_name}';valid=saved[prefix+'_feasible'];coefs=saved[prefix+'_coefficients']
                if not np.isnan(saved[prefix+'_normal_N'][~valid]).all() or np.any(coefs[~valid]):raise ValueError('Missing solver entries were changed into data')
                for i in np.flatnonzero(valid):
                    ids=triples[i];BODY,local=unit_normal_envelope(p['source_points_BODY_m'][ids]-COM,-p['outward_source_normals_BODY'][ids],edges=edges);W=rotate_wrenches(BODY,R)
                    lam=coefs[i];error=W@lam-target;group_N=np.array([lam[local==j].sum() for j in range(3)])
                    if np.min(lam)<-1e-8 or np.max(abs(error[:3]))>1e-8 or np.max(abs(error[3:]))>1e-9 or np.min(group_N)<.1-1e-8 or abs(lam.sum()-saved[prefix+'_normal_N'][i])>1e-8 or np.max(abs(group_N-saved[prefix+'_group_N'][i]))>1e-8:raise ValueError('Saved triplet primal invalid')
                    triplet_primals+=1
                triplet_counts.append(int(valid.sum()))
    groups={i:[model.geom(f'right_finger{i}_{s}').id for s in ['link3_col','link4_col','tip_sensor_col']] for i in [1,2,3]};gid0={g:i-1 for i,gs in groups.items() for g in gs}
    hand=[model.geom(n).id for n in bind['hand_collision_names']];parts=set(bind['mug_part_geom_ids']);audit=CompoundCollisionAudit(model,sorted(parts),hand,model.geom('mug_task_table').id,groups)
    reader=CompoundContacts(model,sorted(parts),bind['mug_body_id'],groups,model.geom('mug_task_table').id)
    geometry=[];seen=set();efc_count=0;native_primals=0;traces=[];reset_diagnostics=[]
    for row in reports['fit']['results']:
        for key,state_key in [('fitted_geometry','fitted'),('geometry','candidate')]:
            pin=row[key]
            if pin['sha256'] in seen:continue
            seen.add(pin['sha256']);saved=json.loads(verify_file_pin(pin).read_text());state=np.load(verify_file_pin(row[state_key]));data.qpos[:]=state['qpos27'];mujoco.mj_kinematics(model,data);fresh=audit.evaluate(data)
            good=fresh['passed']==saved['passed'] and len(fresh['rows'])==len(saved['rows'])==3025
            for a,b in zip(fresh['rows'],saved['rows']):
                good &= a['geom_ids']==b['geom_ids'] and a['passed']==b['passed'] and a['status']==b['status']
                if a.get('gap_m') is not None:good &= abs(a['gap_m']-b['gap_m'])<1e-10
            geometry.append(dict(index=row['index'],kind=key,passed=bool(good),geometry_pass=fresh['passed']))
            print(json.dumps(geometry[-1]),flush=True)
        state=np.load(verify_file_pin(row['candidate']));q=state['qpos27']
        if row['static'] is not None:
            saved=np.load(verify_file_pin(row['basis']));model.opt.timestep=float(mujoco.MjModel.from_binary_path(str(bind['model']['path'])).opt.timestep)
            mujoco.mj_resetData(model,data);data.qpos[:]=q;data.ctrl[:]=q[:20];mujoco.mj_forward(model,data);rotation=data.xmat[bind['mug_body_id']].reshape(3,3);arm=data.xipos[bind['mug_body_id']]-data.xpos[bind['mug_body_id']];W=[];L=[]
            for ci in range(data.ncon):
                con=data.contact[ci];a,b=map(int,con.geom)
                if ((a in parts)==(b in parts)) or con.efc_address<0:continue
                for j in range(2*(con.dim-1)):
                    unit=np.zeros(data.nefc);unit[con.efc_address+j]=1.;force=np.zeros(model.nv);mujoco.mj_mulJacTVec(model,data,force,unit);efc_count+=1
                    W.append(np.r_[force[20:23],rotation@force[23:26]-np.cross(arm,force[20:23])]);L.append(-force[:20])
            W,L=np.array(W).T,np.array(L).T
            if np.max(abs(W-saved['wrench6xN']))>1e-10 or np.max(abs(L-saved['joint_load20xN']))>1e-10:raise ValueError('Actual native EFC readback mismatch')
            for name in ['cone6D','strict_full_command']:
                answer=row['static'][name]
                if not answer['success']:continue
                lam=np.array(answer['witness']['ray_coefficients_N']);e=W@lam-saved['target6']
                if np.min(lam)<-1e-8 or np.max(abs(e[:3]))>1e-8 or np.max(abs(e[3:]))>1e-9:raise ValueError('Native primal invalid')
                if name=='strict_full_command':
                    tau=saved['offset20']+L@lam
                    if np.min(np.minimum(tau-saved['lower20'],saved['upper20']-tau)) < -1e-8:raise ValueError('Strict torque bound invalid')
                native_primals+=1
        for case in row['native_cases']:
            trace=np.load(verify_file_pin(case['trace']));u=trace['ctrl20'];model.opt.timestep=case['dt_s'];traces.append(check_case(case,q,u,model,bind,gid0))
            if case['dt_s']==.0003125:
                mujoco.mj_resetData(model,data);data.qpos[:]=q;data.ctrl[:]=u;mujoco.mj_forward(model,data);snapshot=reader.read(data)
                reset_diagnostics.append(dict(control=case['control'],actual_group_Fn_N=snapshot['normal_by_finger_N'][:3],net_world_force_N=snapshot['force_world_N']+model.body_mass[bind['mug_body_id']]*model.opt.gravity,
                    actual_COM_torque_world_Nm=snapshot['torque_about_COM_world_Nm'],max_joint_acceleration_rad_s2=float(abs(data.qacc[:20]).max()),max_target_change_rad=float(abs(u-q[:20]).max()),
                    initial_trace_Fn_max_error=float(abs(snapshot['normal_by_finger_N'][:3]-trace['pre_group_normal_N'][0]).max())))
    checks=dict(gravity_and_envelope_primals=bool(gravity_valid and envelope_primals==27 and separators==6),triplet_primals=triplet_primals==710,
        six_distinct_geometry_reruns=len(geometry)==6 and all(g['passed'] for g in geometry),actual_native_EFC_and_primals=efc_count==52 and native_primals==2,
        failed_trace_readback=len(traces)==4 and all(all(t['checks'].values()) for t in traces),actual_reset_differs_from_LP=all(d['initial_trace_Fn_max_error']<1e-7 for d in reset_diagnostics) and len(reset_diagnostics)==2)
    checks={k:bool(v) for k,v in checks.items()};out.mkdir(parents=True)
    result=dict(schema='mug_support_selection_saved_review:1',source=file_pin(__file__),trace_helper=file_pin(ROOT/'tools/review_mug_support_trials.py'),inputs={k:file_pin(paths[k]/'report.json') for k in reports},
        checks=checks,passed=sum(checks.values()),total=len(checks),envelope_primals=envelope_primals,checked_separators=separators,triplet_primals=triplet_primals,triplet_counts=triplet_counts,
        geometry=geometry,native_EFC_columns=efc_count,native_primals=native_primals,traces=traces,trace_checks=[sum(v for t in traces for v in t['checks'].values()),sum(len(t['checks']) for t in traces)],
        reset_diagnostics=reset_diagnostics,integration_steps=0,neural_updates=0,
        scope='Saved primal/separating inequality and native EFC/failed-trace readback; geometry reruns reuse shared convex implementation. No independent global optimality/infeasibility, source force/gravity truth, calibrated dynamics or dynamic success proof.')
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False,default=lambda x:x.tolist())+'\n',encoding='utf8');print(json.dumps(dict(report=file_pin(out/'report.json'),checks=checks,trace_checks=result['trace_checks'])))
    if not all(checks.values()):raise ValueError('Retain failed support-selection review')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['reference','patches','envelopes','selection','fit','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
