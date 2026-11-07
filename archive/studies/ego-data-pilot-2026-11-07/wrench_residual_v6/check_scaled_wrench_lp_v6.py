"""Four closed-form physical LP fixtures; no actual robot/native execution."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scaled_wrench_residual_lp_v6 import scaled_wrench_residual_lp

ROOT=Path(__file__).resolve().parent
def pin(p):
    p=p.resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def run(points,mass=.2,com=(0,0,0),J=None,bias=None):
    J=np.zeros((2,3,20)) if J is None else J
    bias=np.zeros(20) if bias is None else bias
    return scaled_wrench_residual_lp(points,[[0,0,1],[0,0,1]],com,mass,.7,J,bias,
      np.full(20,-.1),np.full(20,.1),finger_group_ids=['same_finger','same_finger'])
p=np.array([[-.001,0,0],[.001,0,0]])
J=np.zeros((2,3,20));J[:,2,0]=.02
a=run(p,J=J)
assert a['hard_tau_group_cone_feasible'] and a['both_residual_gates_pass'] and a['scaled_loss']<1e-9
assert abs(sum(a['normal_force_per_group_N'])-.2*9.81)<1e-9 and abs(a['tau_act_required_Nm'][0]-.02*.2*9.81)<1e-9
b=run([[.03,-.001,0],[.03,.001,0]])
expected=.03/(.077+.03)
assert b['hard_tau_group_cone_feasible'] and abs(b['scaled_loss']-expected)<1e-9
assert not b['force_residual_gate'] and not b['moment_residual_gate']
assert abs(b['force_residual_world_N'][2]+expected*.2*9.81)<1e-9
c=run(p,mass=.4)
expected_deficit=.4*9.81-3
assert c['hard_tau_group_cone_feasible'] and abs(c['force_residual_world_N'][2]+expected_deficit)<1e-9
assert abs(c['scaled_loss']-expected_deficit/(.4*9.81))<1e-9
assert abs(c['normal_force_per_group_N'][0]-3)<1e-9 and not c['force_residual_gate']
bias=np.zeros(20);bias[0]=.2;d=run(p,bias=bias)
assert d['status']=='UNKNOWN_NO_TORQUE_FEASIBLE_CONE' and d['scaled_loss'] is None and not d['both_residual_gates_pass']
assert not any(k in sys.modules for k in ['mujoco','torch','warp','mujoco_warp'])
out=ROOT/'preparation_v1';out.mkdir(exist_ok=False)
report={'schema':'scaled_wrench_LP_synthetic_contract:6','source':pin(ROOT/'scaled_wrench_residual_lp_v6.py'),'checker':pin(Path(__file__)),
  'checks':{'force_moment_and_nonzero_JT_can_all_balance':True,'force_can_balance_but_eccentric_moment_requires_finite_tradeoff':True,
    'shared_finger3N_budget_cannot_be_duplicated_per_point':True,'infeasible_torque_cone_returns_unknown_without_fake_residual':True},
  'passed':4,'total':4,'LP_calls':4,'model_FK_physics_training_GPU_calls':0,
  'analytic_expected':{'eccentric_scaled_optimum':expected,'force_deficit_N':expected_deficit},
  'results':{'balanced':a,'eccentric_moment':b,'force_budget_limited':c,'torque_cone_unknown':d},
  'scope':'Synthetic two-point16lambda cases with explicit grouped budget and synthetic20-coordinate Jacobians only; no actual Hand2 or candidate qualification.'}
with (out/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
print(json.dumps({'passed':4,'total':4,'report':pin(out/'report.json')}))
