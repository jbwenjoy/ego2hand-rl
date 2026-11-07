"""Saved material-point, all-piece geometry and frozen-cone witness readback."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.compound_collision import CompoundCollisionAudit
from supermimic_research.mechanics.compound_contacts import CompoundContacts


def main(args):
    import mujoco
    paths={k:getattr(args,k).resolve() for k in ['proposal','coupled','engagement','support','reference','patches','output']};out=paths['output']
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    reports={k:json.loads((paths[k]/'report.json').read_text()) for k in ['proposal','coupled','engagement','support']}
    for report in reports.values():
        for pin in report['sources']:verify_file_pin(pin)
    bind=json.loads((paths['reference']/'binding.json').read_text());model=mujoco.MjModel.from_binary_path(str(verify_file_pin(bind['model'])));data=mujoco.MjData(model)
    hand=[model.geom(n).id for n in bind['hand_collision_names']];parts=set(bind['mug_part_geom_ids'])
    groups={i:[model.geom(f'right_finger{i}_{s}').id for s in ['link3_col','link4_col','tip_sensor_col']] for i in [1,2,3]}
    audit=CompoundCollisionAudit(model,sorted(parts),hand,model.geom('mug_task_table').id,groups)
    reader=CompoundContacts(model,sorted(parts),bind['mug_body_id'],groups,model.geom('mug_task_table').id)
    anchors={r['geom']:r for r in json.loads(verify_file_pin(reports['proposal']['anchors']).read_text())}
    raw=np.load(verify_file_pin(json.loads((paths['patches']/'report.json').read_text())['output']))
    p,n=raw['source_points_BODY_m'],-raw['outward_source_normals_BODY']
    solutions=json.loads(verify_file_pin(reports['proposal']['reach_solutions']).read_text());metric_errors=[]
    seed=np.load(json.loads((paths['proposal']/'intent.json').read_text())['parent']['path'])['qpos27']
    for solution in solutions:
        q=np.array(solution['qpos']);data.qpos[:]=q;mujoco.mj_kinematics(model,data)
        a=anchors[solution['geom']];R=data.geom_xmat[a['geom']].reshape(3,3)
        point=data.geom_xpos[a['geom']]+R@a['point_local'];normal=R@a['normal_local'];bodyR=data.xmat[bind['mug_body_id']].reshape(3,3);origin=data.xpos[bind['mug_body_id']]
        target=p[solution['source_id']]@bodyR.T+origin;target_normal=n[solution['source_id']]@bodyR.T
        metric_errors.append(max(float(np.max(abs(point-solution['point_world']))),abs(np.linalg.norm(point-target)-solution['source_distance_m']),abs(normal@target_normal-solution['normal_dot'])))
        if not np.array_equal(q[20:],seed[20:]):raise ValueError('Material reach changed object')
    geometry_records=[]
    for name,key in [('proposal','candidates'),('coupled','results'),('engagement','results')]:
        for i,row in enumerate(reports[name][key]):
            state=np.load(verify_file_pin(row['candidate']));q=state['qpos27'];data.qpos[:]=q;data.qvel[:]=0;data.ctrl[:]=state['ctrl20'];mujoco.mj_forward(model,data)
            old=json.loads(verify_file_pin(row['geometry']).read_text());new=audit.evaluate(data)
            expected=len(parts)*(len(hand)+1)+len(audit.self)+len(audit.hand_table)
            if len(old['rows'])!=expected or len(new['rows'])!=expected:raise ValueError('Incomplete full geometry')
            errors=[];identity=True
            for a,b in zip(old['rows'],new['rows']):
                identity&=a['geom_ids']==b['geom_ids'] and a['kind']==b['kind'] and a['passed']==b['passed'] and a['status']==b['status']
                for field in ['lower_bound_m','upper_bound_m','gap_m']:
                    if (a.get(field) is None)!=(b.get(field) is None):identity=False
                    elif a.get(field) is not None:errors.append(abs(a[field]-b[field]))
            error=max(errors,default=0.);legal=bool(np.all(q[:20]>=model.jnt_range[:20,0]) and np.all(q[:20]<=model.jnt_range[:20,1]) and np.all(state['ctrl20']>=model.actuator_ctrlrange[:,0]) and np.all(state['ctrl20']<=model.actuator_ctrlrange[:,1]))
            unchanged=np.array_equal(q[20:],seed[20:]);geometry_records.append(dict(stage=name,index=i,passed=bool(identity and error<1e-10 and legal and unchanged and new['passed']==row['geometry_pass']),max_margin_error_m=error,geometry_pass=new['passed']))
            if name=='engagement':
                snapshot=reader.read(data)
                if np.max(abs(snapshot['normal_by_finger_N'][:3]-row['native_finger_normal_N']))>1e-10:raise ValueError('Saved normal force mismatch')
            print(json.dumps(geometry_records[-1]),flush=True)
    separations=[];efc_count=0;primal_count=0
    for row,certificate in zip(reports['engagement']['results'],reports['support']['records']):
        state=np.load(verify_file_pin(row['candidate']));saved=np.load(verify_file_pin(row['basis']));data.qpos[:]=state['qpos27'];data.qvel[:]=0;data.ctrl[:]=state['ctrl20'];mujoco.mj_forward(model,data)
        R=data.xmat[bind['mug_body_id']].reshape(3,3);arm=data.xipos[bind['mug_body_id']]-data.xpos[bind['mug_body_id']];W=[];L=[]
        for ci in range(data.ncon):
            con=data.contact[ci];a,b=map(int,con.geom)
            if ((a in parts)==(b in parts)) or con.efc_address<0:continue
            for j in range(2*(con.dim-1)):
                unit=np.zeros(data.nefc);unit[con.efc_address+j]=1.;force=np.zeros(model.nv);mujoco.mj_mulJacTVec(model,data,force,unit);efc_count+=1
                W.append(np.r_[force[20:23],R@force[23:26]-np.cross(arm,force[20:23])]);L.append(-force[:20])
        W=np.array(W).T;L=np.array(L).T;y=np.array(certificate['separation']['witness_dual_to_F_N_T_Nm'])
        valid=np.max(abs(W-saved['wrench6xN']))<1e-10 and np.max(abs(L-saved['joint_load20xN']))<1e-10 and np.min(W.T@y)>=-1e-9 and abs(saved['target6']@y+1)<1e-9
        for name in ['force_only','cone6D','full_command','maximum_support']:
            answer=row['static'][name]
            if not answer['success']:continue
            lam=np.array(answer['witness']['ray_coefficients_N']);target=saved['target6'].copy()
            if name=='maximum_support':target=np.array([0,0,answer['support_weight_N'],0,0,0])
            error=W@lam-target;valid&=np.min(lam)>=-1e-8 and np.max(abs(error[:3]))<1e-8
            if name!='force_only':valid&=np.max(abs(error[3:]))<1e-9
            if name in ['full_command','maximum_support']:
                tau=saved['offset20']+L@lam;valid&=np.min(np.minimum(tau-(saved['lower20']-saved['friction20']),(saved['upper20']+saved['friction20'])-tau))>=-1e-8
            primal_count+=1
        separations.append(bool(valid))
    checks=dict(material_metrics_36=len(metric_errors)==36 and max(metric_errors)<1e-10,full_geometry_11=len(geometry_records)==11 and all(r['passed'] for r in geometry_records),
        native_EFC_and_separations=len(separations)==3 and all(separations) and efc_count==44,
        no_unqualified_integration=all(r['integration_steps']==0 for r in reports.values()) and reports['engagement']['native_cases']==0,
        known_result_scope=[sum(r['geometry_pass'] for r in geometry_records if r['stage']==s) for s in ['proposal','coupled','engagement']]==[0,1,3])
    checks={k:bool(v) for k,v in checks.items()}
    out.mkdir(parents=True);result=dict(schema='mug_patch_proposals_saved_review:1',source=file_pin(__file__),inputs={k:file_pin(paths[k]/'report.json') for k in reports},
        checks=checks,passed=sum(checks.values()),total=len(checks),geometry=geometry_records,full_pair_queries=11*expected,
        pairs_per_full_audit=expected,object_pairs=len(parts)*(len(hand)+1),self_pairs=len(audit.self),hand_table_pairs=len(audit.hand_table),
        material_metric_max_error=max(metric_errors),native_EFC_columns=efc_count,feasible_primals=primal_count,separations=separations,integration_steps=0,neural_updates=0,
        scope='Fresh full ideal-convex reruns reuse the maintained geometry implementation; material metrics and EFC/separating inequalities are reconstructed independently. Does not certify global collision predicates, source contact truth, real forces, gravity, global infeasibility or dynamic holding.')
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps(dict(report=file_pin(out/'report.json'),checks=checks)))
    if not all(checks.values()):raise ValueError('Retain failed proposal review')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['proposal','coupled','engagement','support','reference','patches','output']:parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
