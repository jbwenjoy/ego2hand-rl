"""Three analytic statics cases plus one actuator bound rejection; no model."""
import hashlib,json,math,sys
from pathlib import Path
import numpy as np
from support_residual_lp_v3 import support_residual_lp

ROOT=Path(__file__).resolve().parent
def pin(p):
    p=p.resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def run(p,n,com,J=None,bound=.1):
    if J is None:J=np.zeros((len(p),3,20))
    return support_residual_lp(p,n,com,.2,.7,J,np.zeros(20),np.full(20,-bound),np.full(20,bound))
p=np.array([[-.03,0,0],[.03,0,0]]);n=np.array([[1.,0,0],[-1.,0,0]])
J=np.zeros((2,3,20));J[0,2,0]=.02;J[1,2,1]=.02
a=run(p,n,[0,0,0],J)
assert a['feasible_force_balance'] and a['proposal_gate'] and a['moment_infinity_Nm']<1e-9
assert max(a['normal_force_N'])<=3+1e-9 and not a['strict_static_support_qualified']
single=run([[.03,0,0]],[[0.,0,1]],[0,0,0])
assert single['feasible_force_balance'] and not single['proposal_gate']
assert abs(single['moment_infinity_Nm']-.03*.2*9.81)<1e-9
offset=run(p,n,[0,.02,0])
assert offset['feasible_force_balance'] and not offset['proposal_gate']
assert abs(offset['moment_residual_world_Nm'][0]+.02*.2*9.81)<1e-9
assert abs(offset['moment_infinity_Nm']-.02*.2*9.81)<1e-9
limited=run(p,n,[0,0,0],J,bound=.005)
assert not limited['feasible_force_balance'] and limited['solver_status']==2
assert limited['force_recovery']['status']=='FINITE_FORCE_RECOVERY_GUIDANCE_ONLY'
assert limited['force_recovery']['force_infinity_N']>1
assert not limited['force_recovery']['qualified'] and not limited['proposal_gate']
assert not any(k in sys.modules for k in ['mujoco','torch','warp','mujoco_warp'])
out=ROOT/'preparation_v3';out.mkdir(exist_ok=False)
report={'schema':'support_residual_LP_synthetic_contract:3','source':pin(ROOT/'support_residual_lp_v3.py'),'checker':pin(Path(__file__)),
    'checks':{'opposed_points_can_balance_force_and_moment':True,'eccentric_single_point_retains_nonzero_moment':True,
      'two_point_gravity_line_defect_cannot_be_removed':True,'hard_actual_torque_bounds_can_make_force_balance_infeasible':True},
    'passed':4,'total':4,'primary_LP_calls':4,'force_recovery_LP_calls':1,'LP_calls':5,'synthetic_only_no_actual_pose_FK_or_physics':True,
    'scope':'Synthetic20-coordinate Jacobians, zero bias; does not qualify actual Hand2 architecture or any candidate.',
    'results':{'balanced':a,'eccentric_single_point':single,'offset_two_points':offset,'low_torque_infeasible':limited}}
with (out/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
print(json.dumps({'passed':4,'total':4,'report':pin(out/'report.json')}))
