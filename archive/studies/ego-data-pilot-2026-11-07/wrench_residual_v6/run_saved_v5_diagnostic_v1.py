"""Two preregistered six-D LP evaluations on unchanged saved v5 poses."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from scaled_wrench_residual_lp_v6 import scaled_wrench_residual_lp

HERE=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def need(x,label):
    if not x:raise ValueError(label)


def main():
    cfg=read(HERE/'saved_v5_preregistration_v1.json')
    need(pin(__file__)==cfg['source'],'frozen caller')
    for v in cfg['input_pins'].values():need(pin(v['path'])==v,'input bytes '+v['path'])
    r=read(cfg['input_pins']['v5_report']['path']);proof=read(cfg['input_pins']['fixture_report']['path'])
    need(proof['passed']==proof['total']==4,'four analytic fixtures')
    with np.load(cfg['input_pins']['raw_PD_context']['path'],allow_pickle=False) as z:ctrlrange=z['ctrlrange'].copy()
    names={int(k):v['name'] for k,v in read(cfg['input_pins']['geometry_ID_metadata']['path'])['geometry_metadata'].items()}
    keys=['contact_points_world','normals_force_on_object','object_COM','mass','mu_actual','J_robot_contact_3x20','robot_gravity_bias_tau','torque_lower','torque_upper','gravity_world']
    outputs=[];calls=0
    need([s['seed'] for s in r['seed_results']]==[0,1],'both original v5 seeds')
    for s in r['seed_results']:
        st=s['verification_state'];op=st['support_operands'];old=s['support_LP']
        need(op['available'] and [names[x['geom_id']] for x in st['contacts']]==['right_finger1_tip_sensor_col','right_finger2_tip_sensor_col'],'unchanged paired finger order')
        result=scaled_wrench_residual_lp(**{k:op[k] for k in keys},finger_group_ids=['thumb','index'],characteristic_length_m=.077,
           normal_cap_per_finger_N=3.,tangent1_world=old['tangent1_world'],tangent2_world=old['tangent2_world'])
        calls+=result['LP_calls']
        need(np.allclose(result['rays_force_world_per_normal_N'],old['rays_force_world_per_normal_N'],rtol=0,atol=1e-14),'same frozen inner-cone directions')
        ctrl=None;reachable=False;violation=None
        if result['hard_tau_group_cone_feasible']:
            tau=np.array(result['tau_act_required_Nm']);ctrl=(tau-np.array(st['PD_offset20']))/np.array(st['PD_gain20'])
            reachable=bool(np.isfinite(ctrl).all() and np.all(ctrl>=ctrlrange[:,0]) and np.all(ctrl<=ctrlrange[:,1]))
            violation=float(np.max(np.maximum(np.maximum(ctrlrange[:,0]-ctrl,ctrl-ctrlrange[:,1]),0)))
        outputs.append({'seed':s['seed'],'source_selected_state':s['selected_state'],'source_qpos27':st['qpos27'],
          'original_force_hard_feasible':old['feasible_force_balance'],'original_moment_infinity_Nm':old['moment_infinity_Nm'],
          'new_scaled_LP':result,'diagnostic_inverse_ctrl20':ctrl.tolist() if ctrl is not None else None,
          'unclamped_inverse_ctrl_reachable':reachable,'raw_ctrl_range_violation_rad':violation,
          'controller_or_holding_enrolled':False,'source_pose_changed':False})
    need(calls==2,'exactly two new LP solves')
    need(not any(k in sys.modules for k in ['mujoco','torch','warp','mujoco_warp']),'no model/FK/GPU imports')
    report={'schema':'saved_v5_scaled_six_D_wrench_diagnostic:6','status':'COMPLETED_TWO_SAVED_POSE_PROPOSAL_DIAGNOSTICS_NOT_HOLDING',
       'preregistration':pin(HERE/'saved_v5_preregistration_v1.json'),'source':pin(__file__),'input_pins':cfg['input_pins'],
       'counts':{'new_LP_calls':2,'model_FK_optimization_pose_physics_training_GPU_calls':0},'results':outputs,
       'scope':'Same saved paired contacts/J/bias/true PD and native caps, no pose or parameter change. Grouped per-finger3N and original frozen frame retained. Scaled objective is proposal guidance; raw final force/moment gates remain separate. Inverse controls are diagnostic values only, not current-run enrollment or holding qualification.'}
    dest=HERE/'saved_v5_diagnostic_v1.json'
    with dest.open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({'report':pin(dest),'seeds':[{'seed':x['seed'],'scaled_loss':x['new_scaled_LP']['scaled_loss'],'force_inf_N':x['new_scaled_LP'].get('force_infinity_N'),'moment_inf_Nm':x['new_scaled_LP'].get('moment_infinity_Nm'),'ctrl_reachable':x['unclamped_inverse_ctrl_reachable']} for x in outputs]}))


if __name__=='__main__':main()
