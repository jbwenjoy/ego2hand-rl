"""Read-only recorded-control rollout and COM momentum consistency diagnostics."""
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from ..artifacts import verify_file_pin,file_pin
from ..mechanics.transition_records import state_rotation
from ..mechanics.response_decode import decode_response,momentum_defect


def load_rollout_cases(review):
    import mujoco
    collection_path=verify_file_pin(review['collection']);collection=json.loads(collection_path.read_text())
    binding=json.loads(verify_file_pin(collection['binding']).read_text());model=mujoco.MjModel.from_binary_path(str(verify_file_pin(binding['model'])))
    body=binding['mug_body_id'];rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,model.body_iquat[body]);Q=rotation.reshape(3,3)
    if np.any(model.dof_damping[20:]) or np.any(model.body_gravcomp):raise ValueError('Declared free-object momentum diagnostic has no damping/gravity compensation')
    physics=dict(mass=float(model.body_mass[body]),inertia_BODY=Q@np.diag(model.body_inertia[body])@Q.T,gravity=model.opt.gravity.copy(),joint_range=model.jnt_range[:20].copy())
    prefixes=[p for p in collection['prefixes'] if p['seed'] in range(8200,8204)]
    if len(prefixes)!=8:raise ValueError('Exactly four independent CHECK seeds x two dt')
    cases=[]
    for p in prefixes:
        data=dict(np.load(verify_file_pin(p['transitions'])))
        for start,horizons in [(2,[1,2,5,10]),(7,[1,2,5])]:
            for horizon in horizons:cases.append(dict(seed=p['seed'],dt=p['dt'],start=start,horizon=horizon,data=data))
    pins=dict(collection=file_pin(collection_path),model=binding['model'],prefixes=[p['transitions'] for p in prefixes])
    return cases,physics,pins


def rollout_metrics(model,cases,physics,device):
    rows=[];arrays=[];mass,I,g=[physics[k] for k in ['mass','inertia_BODY','gravity']]
    for case in cases:
        data=case['data'];start,horizon=case['start'],case['horizon'];state=data['source55'][start].copy();warm=data['warmstart26_before'][start].copy()
        defects=[];native_defects=[];force_errors=[];torque_errors=[];failed=None;trajectory=[state.copy()]
        for i in range(start,start+horizon):
            source=state.copy();x=np.r_[state,warm,data['actual_control20'][i],case['dt']]
            with torch.no_grad():pred=model.prediction(torch.as_tensor(x[None],dtype=torch.float32,device=device))[0].cpu().numpy().astype(float)
            if not np.isfinite(pred).all():failed='NONFINITE_PREDICTION';break
            state=decode_response(source,pred);warm=pred[52:78].copy();trajectory.append(state.copy())
            defects.append(momentum_defect(source,state,pred,mass,I,g));native_defects.append(momentum_defect(data['source55'][i],data['next55'][i],data['y'][i],mass,I,g))
            actual_R=state_rotation(data['source55'][i]);pred_R=state_rotation(source)
            force_errors.append(pred_R@pred[78:81]-actual_R@data['y'][i,78:81]);torque_errors.append(pred_R@pred[81:84]-actual_R@data['y'][i,81:84])
        row={key:case[key] for key in ['seed','dt','start','horizon']};row.update(failure=failed,executed_predictions=len(trajectory)-1)
        if failed is None:
            target=data['next55'][start+horizon-1];R,Rtarget=state_rotation(state),state_rotation(target)
            defect=np.array(defects);native=np.array(native_defects)
            row.update(COM_error_m=float(np.linalg.norm(state[40:43]-target[40:43])),angle_error_rad=float(Rotation.from_matrix(R@Rtarget.T).magnitude()),
                joint_position_RMS_rad=float(np.sqrt(np.mean((state[:20]-target[:20])**2))),COM_velocity_error_m_s=float(np.linalg.norm(state[49:52]-target[49:52])),
                omega_error_rad_s=float(np.linalg.norm(state[52:55]-target[52:55])),world_force_RMS_N=float(np.sqrt(np.mean(np.square(force_errors)))),world_torque_RMS_Nm=float(np.sqrt(np.mean(np.square(torque_errors)))),
                model_linear_momentum_defect_RMS_Ns=float(np.sqrt(np.mean(defect[:,:3]**2))),native_linear_momentum_defect_RMS_Ns=float(np.sqrt(np.mean(native[:,:3]**2))),
                model_angular_momentum_defect_RMS_Nms=float(np.sqrt(np.mean(defect[:,3:]**2))),native_angular_momentum_defect_RMS_Nms=float(np.sqrt(np.mean(native[:,3:]**2))),
                model_joint_limit_excursion_rad=float(max(0.,np.max(physics['joint_range'][:,0]-state[:20]),np.max(state[:20]-physics['joint_range'][:,1]))))
        rows.append(row);arrays.append(np.array(trajectory))
    by_horizon={}
    for horizon in [1,2,5,10]:
        chosen=[r for r in rows if r['horizon']==horizon];valid=[r for r in chosen if r['failure'] is None]
        summary=dict(cases=len(chosen),finite_cases=len(valid),failures=len(chosen)-len(valid))
        if valid:
            for key in valid[0]:
                if key.endswith(('_m','_rad','_m_s','_rad_s','_N','_Nm','_Ns','_Nms')):summary[key+'_mean']=float(np.mean([r[key] for r in valid]))
        by_horizon[str(horizon)]=summary
    lengths=[len(a) for a in arrays]
    return dict(cases=len(rows),by_horizon=by_horizon,rows=rows,
        contract='No native integration or aid; playback of actual recorded CHECK controls, recursively predicted state55/warm26. State rotation uses world-left exponential, no joint clipping/teacher forcing. H1/H2/H5/H10 correspond20/40/100/200ms.',
        momentum_scope='Predicted momentum change versus predicted mean contact wrench plus native gravity; native finite-solver/discrete-step defect reported as reference, not forced to zero. No native damping/gravcomp.',
        limitation='Fixed recorded controls are not closed-loop reference tracking, policy evaluation, geometric contact qualification,2s holding or independently reconstructed ego validation.'),dict(predicted_states=np.concatenate(arrays),trajectory_offsets=np.cumsum([0]+lengths))
