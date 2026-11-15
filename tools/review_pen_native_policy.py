"""Independent NumPy/SciPy/FK review of actual native161/188 saved transitions."""
import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main(a):
    rp=a.run.resolve();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh independent review')
    run=json.loads(rp.read_text());cfg=tomllib.loads(verify_file_pin(run['config']).read_text())
    if run['passed']!=run['total'] or run['actual_native_integrations']!=4224:raise ValueError('Completed bounded qualification required')
    # Historical H5 is a never-fit physics/frame reference, not an expert label.
    hp=ROOT/'runs/pen_native_horizon_H5_v1/report.json';old=json.loads(hp.read_text())
    case=next(c for c in old['cases'] if c['name']=='STATE_ONLY' and c['dt']==.0025)
    physical=dict(np.load(verify_file_pin(case['GPU_physical_context'])))
    reference=dict(np.load(verify_file_pin(case['initialization']['nominal_mean_wrench'])))
    tp=ROOT/'runs/pen_task_contract_audit_v2/report.json';task=json.loads(tp.read_text())
    raw=dict(np.load(verify_file_pin(task['clip'])))
    Wraw=Rotation.from_quat(raw['wrist_pose_w'][:,[4,5,6,3]])
    links=np.einsum('nij,nkj->nki',Wraw.as_matrix().transpose(0,2,1),raw['link_pos_w']-raw['wrist_pose_w'][:,None,:3])
    m=mujoco.MjModel.from_binary_path(str(verify_file_pin(case['host_topology'])));d=mujoco.MjData(m)
    palm=m.body('robot/right_palm_link').id;body=m.body('object/cube').id;qa=m.jnt_qposadr[m.body_jntadr[body]];da=m.jnt_dofadr[m.body_jntadr[body]]
    jids=[m.joint(f'robot/right_finger{i}_joint{j}').id for i in range(1,6) for j in range(1,5)];qids=m.jnt_qposadr[jids];dofs=m.jnt_dofadr[jids]
    out.mkdir();receipt=freeze_sources([Path(__file__).resolve()],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    write(out/'source_receipt.json',receipt)
    z=dict(np.load(verify_file_pin(run['transitions'])));checks={};rows=[];reset_pins=[];source_inputs=[];physical_results=[]
    for c in run['cases']:
        worlds=c['worlds'];tag=f'worlds{worlds}__';model=dict(np.load(verify_file_pin(c['model_arrays'])))
        checks[f'worlds{worlds}_metadata']=c['metadata']['policy_input_dim']==161 and c['metadata']['critic_input_dim']==188
        for key,x in physical.items():
            ok=np.array_equal(np.broadcast_to(x,model[key].shape),model[key]);checks[f'worlds{worlds}_physics_'+key]=ok
            physical_results.append(dict(worlds=worlds,field=key,exact_all_worlds=ok))
        reset=rp.parent/f'worlds{worlds}'/'partial_reset.npz';reset_pins.append(file_pin(reset));r=dict(np.load(reset))
        ids=r['reset_ids'];untouched=np.setdiff1d(np.arange(worlds),ids)
        for name in [k[8:] for k in r if k.startswith('before__')]:
            checks[f'worlds{worlds}_partial_untouched_'+name]=np.array_equal(r['before__'+name][untouched],r['after__'+name][untouched])
        checks[f'worlds{worlds}_reset_phase']=np.all(r['after__phase'][ids]==1)
        checks[f'worlds{worlds}_reset_pending']=not r['after__pending'][ids].any()
        checks[f'worlds{worlds}_reset_entity_ctrl']=np.array_equal(r['after__entity_position_targets'][ids],r['after__ctrl'][ids])
        for k in range(16):
            errors=[];critic_errors=[];reward_errors=[]
            for w in range(worlds):
                qpos=z[tag+'qpos'][k,w];qvel=z[tag+'qvel'][k,w];phase=int(z[tag+'phase'][k,w]);used=int(z[tag+'used_phase'][k,w])
                d.qpos[:]=qpos;d.mocap_pos[:]=z[tag+'mocap_pos'][k,w];d.mocap_quat[:]=z[tag+'mocap_quat'][k,w]
                mujoco.mj_kinematics(m,d)
                mount=d.xpos[palm];W=Rotation.from_matrix(d.xmat[palm].reshape(3,3))
                def nine(position,quat):
                    R=(W.inv()*Rotation.from_quat(quat[[1,2,3,0]])).as_matrix()
                    return np.r_[.15*np.tanh(W.inv().apply(position-mount)/.15),R[:,0],R[:,1]]
                frame=min(phase,368)
                p=np.r_[reference['reference_hand'][frame],nine(reference['COM_world_m'][frame],reference['quaternion_wxyz'][frame]),
                    links[frame].ravel(),qpos[qids],qvel[dofs]*.25,nine(qpos[qa:qa+3],qpos[qa+3:qa+7]),z[tag+'raw'][k,w]]
                cexpected=np.r_[p,z[tag+'EMA'][k,w],W.inv().apply(qvel[da:da+3]),
                    W.inv().apply(Rotation.from_quat(qpos[qa+np.array([4,5,6,3])]).apply(qvel[da+3:da+6])),frame/368]
                errors.append(np.max(abs(p-z[tag+'policy'][k,w])));critic_errors.append(np.max(abs(cexpected-z[tag+'critic'][k,w])))
                ref=z[tag+'used_reference_pose'][k,w];rw=cfg['reward']
                theta=(Rotation.from_quat(qpos[qa+np.array([4,5,6,3])])*Rotation.from_quat(ref[[4,5,6,3]]).inv()).magnitude()
                dist=np.linalg.norm(qpos[qa:qa+3]-ref[:3]);hand=np.mean((qpos[qids]-reference['reference_hand'][used])**2)
                reward=rw['COM_weight']*np.exp(-(dist/rw['COM_scale_m'])**2)+rw['angle_weight']*np.exp(-(theta/rw['angle_scale_rad'])**2)
                reward+=rw['hand_weight']*np.exp(-hand/rw['hand_scale_rad']**2)-rw['action_weight']*np.mean(z[tag+'applied'][k,w]**2)
                reward_errors.append(abs(reward-z[tag+'reward'][k,w]))
            flags=dict(policy_independent_FK=max(errors)<=1e-6,critic_independent_FK=max(critic_errors)<=2e-5,
                reward_independent=max(reward_errors)<=1e-4,
                unsquashed_sample_clipped=np.array_equal(np.clip(z[tag+'sample'][k],-1,1),z[tag+'applied'][k]),
                previous_action_is_applied=np.array_equal(z[tag+'policy'][k,:,141:161],z[tag+'applied'][k]),
                entity_targets_transmitted=np.array_equal(z[tag+'entity_position_targets'][k],z[tag+'ctrl'][k]))
            rows.append(dict(worlds=worlds,k=k,checks={key:bool(v) for key,v in flags.items()},
                max_policy_error=float(max(errors)),max_critic_error=float(max(critic_errors)),max_reward_error=float(max(reward_errors))))
            checks.update({f'worlds{worlds}_k{k}_'+key:bool(v) for key,v in flags.items()})
    # All used files are exact pins, including historical boundaries and reset NPZ.
    for key in ['trace','GPU_physical_context','host_topology']:source_inputs.append(case[key])
    source_inputs += [case['initialization']['nominal_mean_wrench'],task['clip'],file_pin(hp),file_pin(tp)]
    verify_source_snapshot(receipt,live=True)
    result=dict(schema='pen_native_full_pose_independent_review:1',run=file_pin(rp),source_receipt=file_pin(out/'source_receipt.json'),
        inputs=source_inputs,partial_resets=reset_pins,physical_comparisons=physical_results,rows=rows,
        checks={key:bool(v) for key,v in checks.items()},passed=int(sum(checks.values())),total=len(checks),
        actual_task_optimizer_updates=0,native_integrations=0,CPU_kinematics_queries=528,
        scope='Independent saved-transition/FK/control/reset review. Historical H5 only verifies physics/reference, no fitting or expert promotion. Not a full-task, contact-wrench, portable-resume or training result.')
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],
        max_policy_error=max(r['max_policy_error'] for r in rows),max_critic_error=max(r['max_critic_error'] for r in rows),max_reward_error=max(r['max_reward_error'] for r in rows))))
    if not all(checks.values()):raise ValueError('Retain independent discrepancy; no qualification by flag alone')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
