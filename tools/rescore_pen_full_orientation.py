"""Offline full-orientation counterfactual; never a newly executed trajectory."""
import argparse
import json
import sys
from pathlib import Path
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.runtime.controlled_evaluation import write
from supermimic_research.runtime.pen_native_planning_math import quaternion_distance_wxyz


def choose(score,feasible,violation):
    ids=np.flatnonzero(feasible) if feasible.any() else np.flatnonzero(violation==violation.min())
    return int(ids[np.argmin(score[ids])])


def main(a):
    if a.output.exists():raise ValueError('Fresh offline diagnostic')
    rows=[];cases=[];fulls=[];shafts=[];old_scores=[];new_scores=[]
    for path in a.reports:
        r=json.loads(path.read_text())
        if r.get('experiment')!='STATE_SUPPORT' or r.get('orientation_objective','SHAFT')!='SHAFT':raise ValueError('Original state/support SHAFT cohort')
        for c in r['cases']:
            model=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])))
            data=mujoco.MjData(model);body=c['model_layout']['object_body'];qa=c['model_layout']['object_qposadr']
            ref=np.load(verify_file_pin(c['initialization']['nominal_mean_wrench']));actual=np.load(verify_file_pin(c['trace']))
            jids=np.array([model.joint(f'robot/right_finger{i}_joint{j}').id for i in range(1,6) for j in range(1,5)])
            qids=model.jnt_qposadr[jids]
            topology=dict(free_object=int(model.jnt_type[model.body_jntadr[body]])==int(mujoco.mjtJoint.mjJNT_FREE),
                object_parent_world=int(model.body_parentid[body])==0,fixed_wrist=int(model.body('robot/right_palm_link').jntnum.item())==0,
                COM_origin=not np.any(c['model_layout']['COM_BODY']),reference_world_exact=np.array_equal(ref['quaternion_wxyz'],actual['quaternion_wxyz']))
            first=len(rows)
            for decision in c['planning']:
                k=decision['k'];d=np.load(verify_file_pin(decision['prediction']));source=np.load(verify_file_pin(decision['source']))
                q=d['Q'][:,-1].astype(float);quat=q[:,qa+3:qa+7];rq=ref['quaternion_wxyz'][k+1].astype(float)
                R=Rotation.from_quat(quat[:,[1,2,3,0]]);target=Rotation.from_quat(rq[[1,2,3,0]])
                full=(R*target.inv()).magnitude()
                # Match the historical shaft algebra, preserving its lack of
                # explicit normalization instead of rewriting previous costs.
                w,x,y,z=quat.T;shaft_vectors=np.c_[1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)]
                shaft=np.arccos(np.clip(shaft_vectors@target.as_matrix()[:,0],-1,1))
                position=(((q[:,qa:qa+3]-ref['COM_world_m'][k+1])/.005)**2).sum(1)
                hand=(((q[:,qids]-ref['reference_hand'][k+1])/.02)**2).mean(1)
                residual=((d['EMA_candidates']/.02)**2).mean(1)
                base=position+.01*hand+.01*residual
                support_weight=.1 if c['name']=='STATE_PLUS_SUPPORT' else 0.
                mean=d['W'].astype(float).mean(1);target_w=ref['wrench_world'][k]
                support=(((mean[:,:3]-target_w[:3])/.1)**2).mean(1)+(((mean[:,3:]-target_w[3:])/.005)**2).mean(1)
                shared=base+support_weight*bool(ref['valid_interval'][k])*support
                old=shared+(shaft/np.deg2rad(5))**2;new=shared+(full/np.deg2rad(5))**2
                old_index=choose(old,d['score_feasible'].astype(bool),d['score_violation'])
                new_index=choose(new,d['score_feasible'].astype(bool),d['score_violation'])
                helper=quaternion_distance_wxyz(torch.tensor(quat,dtype=torch.float32),torch.tensor(rq,dtype=torch.float32)).numpy()
                data.qpos[:]=source['qpos'][0];mujoco.mj_forward(model,data)
                source_q=source['qpos'][0,qa+3:qa+7].astype(float)
                norm_max=float(max(abs(np.linalg.norm(quat,axis=1)-1).max(),abs(np.linalg.norm(rq)-1),abs(np.linalg.norm(source_q)-1)))
                sourceR=Rotation.from_quat(source_q[[1,2,3,0]])
                checks=dict(unit_quaternion_FP32=norm_max<1e-6,helper_vs_scipy=bool(np.allclose(helper,full,atol=2e-6,rtol=1e-6)),
                    world_BODY_quaternion=bool(np.max(abs(sourceR.as_matrix()-data.xmat[body].reshape(3,3)))<2e-7),
                    original_phase=int(d['controller_command_phase'][0])==k+1==decision['phase'],
                    original_cost_within_FP32=bool(np.all(abs(old-d['score_total'])<=1e-3+1e-5*abs(old))),
                    original_argmin=old_index==decision['selected'],full_angles_finite=bool(np.isfinite(full).all()))
                feasible=d['score_feasible'].astype(bool)
                rows.append(dict(run=file_pin(path),mode=c['name'],dt=c['dt'],k=k,prediction=decision['prediction'],source=decision['source'],checks=checks,
                    old_selected=old_index,full_selected=new_index,ranking_changed=old_index!=new_index,max_quaternion_norm_error=norm_max,
                    selected_full_error_deg=float(np.rad2deg(full[old_index])),alternative_full_error_deg=float(np.rad2deg(full[new_index])),
                    minimum_feasible_full_error_deg=float(np.rad2deg(full[feasible].min())) if feasible.any() else None,
                    feasible_count=int(feasible.sum()),source_full_error_deg=float(np.rad2deg((sourceR*Rotation.from_quat(ref['quaternion_wxyz'][k][[1,2,3,0]]).inv()).magnitude()))))
                fulls.append(full);shafts.append(shaft);old_scores.append(old);new_scores.append(new)
            cases.append(dict(run=file_pin(path),mode=c['name'],dt=c['dt'],topology_checks=topology,decision_rows=[first,len(rows)],ranking_changes=sum(e['ranking_changed'] for e in rows[first:])))
    a.output.mkdir();np.savez_compressed(a.output/'all57.npz',full_angle_rad=fulls,shaft_angle_rad=shafts,original_score=old_scores,full_orientation_score=new_scores)
    passed=sum(sum(c['topology_checks'].values()) for c in cases)+sum(sum(r['checks'].values()) for r in rows)
    total=sum(len(c['topology_checks']) for c in cases)+sum(len(r['checks']) for r in rows)
    write(a.output/'report.json',dict(schema='offline_full_orientation_rescore:1',source=file_pin(__file__),math_source=file_pin(ROOT/'src/supermimic_research/runtime/pen_native_planning_math.py'),runs=[file_pin(p) for p in a.reports],
        cases=cases,decisions=rows,all57=file_pin(a.output/'all57.npz'),passed=passed,total=total,ranking_changes=sum(r['ranking_changed'] for r in rows),
        native_integrations=0,neural_updates=0,scope='Every saved candidate endpoint, same old source/control/reference/geometry gates. WORLD free-body quaternion, fixed wrist and COM origin checked; not proof all reference contact motions dynamically feasible. Counterfactual rankings are not a new trajectory or expert. No axial twist decomposition.'))
    print(json.dumps(dict(report=file_pin(a.output/'report.json'),checks=[passed,total],decisions=len(rows),ranking_changes=sum(r['ranking_changed'] for r in rows))))
    if passed!=total:raise ValueError('Keep failed interpretation; no new actual control yet')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reports',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
