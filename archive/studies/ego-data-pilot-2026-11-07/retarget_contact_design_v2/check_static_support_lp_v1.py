"""Four analytic fixtures, five LP calls; no robot, FK, model or physics."""
import datetime
import hashlib
import json
import pathlib
import sys

import numpy as np

from static_support_lp import static_support_lp


def pin(p):
    p=pathlib.Path(p);b=p.read_bytes();return {'path':str(p.resolve()),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def main():
    root=pathlib.Path(__file__).resolve().parent
    output=root/'preparation_v1';output.mkdir(exist_ok=False)
    m=.2;mu=.7;points=np.array([[-.03,0,0],[.03,0,0]]);normals=np.array([[1.,0,0],[-1.,0,0]])
    J=np.zeros((2,3,20));J[0,2,0]=.02;J[1,2,1]=.02
    bias=np.zeros(20);lower=np.full(20,-.1);upper=np.full(20,.1)
    pair=static_support_lp(points,normals,[0,0,0],m,mu,J,bias,lower,upper)
    assert pair['feasible'] and pair['actuator_bounds_checked']
    assert abs(pair['total_normal_force_N']-m*9.81/mu)<1e-7
    assert np.max(np.abs(pair['COM_wrench_residual_N_Nm']))<1e-9
    eccentric=static_support_lp([[.03,0,0]],[[0,0,1]],[0,0,0],m,mu)
    assert not eccentric['feasible'] and eccentric['solver_status']==2
    low=static_support_lp(points,normals,[0,0,0],m,mu,J,bias,np.full(20,-.005),np.full(20,.005))
    assert not low['feasible'] and low['solver_status']==2
    up=static_support_lp([[0,0,-.02]],[[0,0,1]],[0,0,0],m,mu)
    down=static_support_lp([[0,0,-.02]],[[0,0,-1]],[0,0,0],m,mu)
    assert up['feasible'] and not down['feasible'] and down['solver_status']==2
    checks={'opposing_contacts_support_gravity_and_analytic_minimum_normal_force':True,
            'single_eccentric_point_cannot_supply_required_COM_moment':True,
            'actual_JT_sign_and_low_joint_torque_bounds_reject_support':True,
            'single_under_COM_up_force_normal_works_down_normal_fails':True}
    heavy=[x for x in ['mujoco','mujoco_warp','torch','warp'] if x in sys.modules]
    assert not heavy
    report={'schema':'point_contact_static_support_LP_synthetic_contract:1',
      'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'passed':4,'total':4,
      'pure_LP_calls':5,'producer':pin(root/'static_support_lp.py'),'checker':pin(__file__),
      'fixture_scope':'Synthetic20-coordinate Jacobian/zero bias only, no actual Hand2 model or gravity bias. Independent analytic wrench and lever facts, no mirrored producer implementation as oracle.',
      'results':{'opposing_pair':pair,'single_eccentric':eccentric,'low_torque':low,'normal_up':up,'normal_down':down},
      'heavy_modules_imported':heavy,
      'actual_geometry_FK_retarget_optimization_MuJoCo_GPU_training_calls':0}
    path=output/'report.json'
    with path.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({'report':pin(path),'passed':4,'total':4,'LP_calls':5,'actual_models_or_physics':0},indent=2))


if __name__=='__main__':main()
