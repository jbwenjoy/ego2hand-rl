"""Four physical analytic cases; exactly4 pure LP calls, no native backend."""
from pathlib import Path
import hashlib,json
import numpy as np
from native_wrench_lp_v7 import solve_native_wrench

R=Path(__file__).resolve().parent


def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if hasattr(x,'tolist'):return clean(x.tolist())
    return x


def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(clean(x),f,indent=2,allow_nan=False)


def main():
    out=R/'analytic_contract_v1';out.mkdir(exist_ok=False)
    protocol={'schema':'native4ray_PD_wrench_analytic_protocol:7',
      'source':pin(R/'native_wrench_lp_v7.py'),'checker':pin(__file__),'LP_calls':4,
      'mass_kg':.2,'mu':.7,'gravity':[0,0,-9.81],
      'cases':['centered_opposition_supports_gravity','real_tau_limit_blocks_support',
               'geom_order_flip_preserves_cone','duplicate_samples_share_finger_budget'],
      'no_model_forward_step_training_GPU':True}
    write(out/'protocol.json',protocol)
    C=np.array([[[1,0,0],[0,0,1],[0,-1,0]],
                [[-1,0,0],[0,0,1],[0,1,0]]],dtype=float)
    J=np.zeros((2,3,20));J[:,2,0]=1
    params=dict(points_world=np.array([[-.01,0,0],[.01,0,0]]),frames_rows=C,
      force_on_object_sign=np.ones(2),contact_dims=np.full(2,3),friction5=np.tile([.7,.7,0,0,0],(2,1)),
      J_hand_3x20=J,finger_groups=['thumb','index'],COM_world=np.zeros(3),mass_kg=.2,
      robot_bias_minus_passive20=np.zeros(20),pd_gain20=np.full(20,10.),pd_offset20=np.zeros(20),
      ctrl_range20x2=np.tile([-1.,1.],(20,1)),actuator_force_range20x2=np.tile([-10.,10.],(20,1)),
      joint_force_range20x2=np.tile([-10.,10.],(20,1)),normal_cap_per_finger_N=3.)
    results=[];checks={}
    center=solve_native_wrench(**params);results.append(center)
    checks['centered_gravity_wrench_and_required_actuator']=bool(center['strict_force_moment_pass']
      and abs(center['tau_required20'][0]-1.962)<1e-8 and abs(center['proposed_ctrl20'][0]-.1962)<1e-9)
    blocked={**params,'actuator_force_range20x2':params['actuator_force_range20x2'].copy()}
    blocked['actuator_force_range20x2'][0]=[-.5,.5]
    limited=solve_native_wrench(**blocked);results.append(limited)
    checks['real_tau_limit_blocks_gravity_without_breaking_hard_bounds']=bool(limited['hard_bounds_pass']
      and not limited['strict_force_moment_pass'] and limited['tau_required20'][0]<=.5+1e-9
      and abs(limited['force_residual_world_N'][2])>1.)
    flipped={**params,'frames_rows':np.einsum('ij,njk->nik',np.diag([-1.,1.,-1.]),C),
      'force_on_object_sign':-np.ones(2)}
    equiv=solve_native_wrench(**flipped);results.append(equiv)
    checks['geom_order_flip_preserves_object_force_and_torque']=bool(equiv['strict_force_moment_pass']
      and np.max(abs(equiv['tau_required20']-center['tau_required20']))<1e-8)
    idx=[0,0,1,1]
    shared={**params,'points_world':params['points_world'][idx], 'frames_rows':C[idx],
      'force_on_object_sign':np.ones(4),'contact_dims':np.full(4,3),
      'friction5':params['friction5'][idx],'J_hand_3x20':J[idx],
      'finger_groups':['thumb','thumb','index','index'],'normal_cap_per_finger_N':1.2}
    grouped=solve_native_wrench(**shared);results.append(grouped)
    checks['multiple_points_do_not_multiply_finger_budget']=bool(grouped['hard_bounds_pass']
      and not grouped['strict_force_moment_pass'] and np.max(grouped['group_normal_N'])<=1.2+1e-9
      and abs(grouped['force_residual_world_N'][2])>.25)
    write(out/'report.json',{'schema':'native4ray_PD_wrench_analytic_result:7','protocol':pin(out/'protocol.json'),
      'checks':checks,'passed':sum(checks.values()),'total':len(checks),'results':results,
      'LP_calls':4,'new_native_model_forward_step_training_GPU_calls':0,
      'native_holding_or_ego_qualified':False})
    print(json.dumps({'report':pin(out/'report.json'),'checks':checks}))
    if not all(checks.values()):raise ValueError('Analytic physical contract failed; no automatic retry')


if __name__=='__main__':main()
