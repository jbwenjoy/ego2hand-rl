"""Five closed-form mathematical fixtures; no real calibration/model inputs."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MATH=ROOT/'src/full151_gradient/common_pre_calibration_independent_math_v1.py'
OUT=ROOT/'outputs/full151_common_pre_independent_math_preparation_v1'
TOL=1e-10  # Synthetic F64 analytic comparison only, not an operative data gate.


def pin(p):return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def quat(axis,angle):
    axis=np.asarray(axis,dtype='f8');axis=axis/np.linalg.norm(axis)
    return np.r_[np.cos(angle/2),np.sin(angle/2)*axis]


def mul(a,b):
    return np.r_[a[0]*b[0]-np.dot(a[1:],b[1:]),a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:],b[1:])]


def R(q):
    # Independent unit-quaternion/Rodrigues truth, not native or producer calls.
    w=q[0];v=q[1:];K=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return (w*w-np.dot(v,v))*np.eye(3)+2*np.outer(v,v)+2*w*K


def log_unit(q):
    q=q/np.linalg.norm(q)
    if q[0]<0:q=-q
    sn=np.linalg.norm(q[1:])
    return np.zeros(3) if sn==0 else 2*np.arctan2(sn,q[0])*q[1:]/sn


def state(rotation=None):
    x=np.zeros(58,dtype='f8');x[43:52]=(np.eye(3) if rotation is None else rotation).reshape(9);return x


def main():
    OUT.mkdir(exist_ok=False)
    pins={'math':pin(MATH),'checker':pin(Path(__file__))}
    source_refs={'proposal':ROOT/'docs/full151_WM_common_pre_response_calibration_proposal_v1.md',
      'native_state':Path('D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/state.py'),
      'native_rotations':Path('D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/rotations.py')}
    for key,path in [('math',MATH),('checker',Path(__file__))]:ast.parse(path.read_bytes());(OUT/(key+'_source_before_execution.py')).write_bytes(path.read_bytes())
    (OUT/'predata_freeze.json').write_text(json.dumps({'source_pins':pins,'source_reference_pins':{k:pin(v) for k,v in source_refs.items()},
      'actual_calibration_inputs_or_models_read':False,'synthetic_F64_comparison_tolerance':TOL,'operative_data_tolerances_changed':False},indent=2))
    spec=importlib.util.spec_from_file_location('common_pre_numpy',MATH);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    q0=quat([0,0,1],.6);qa=quat([1,0,0],.5);qb=quat([0,1,0],.4);qc=quat([0,0,1],.7)
    qtm=mul(q0,qa);qtp=mul(qtm,qb);qpm=mul(mul(q0,qc),qa);qpp=mul(qpm,qb)
    pre=state(R(q0))[None];pt=np.stack([state(R(qtm)),state(R(qtp))])[None];pp=np.stack([state(R(qpm)),state(R(qpp))])[None]
    cm=m.difference52(np.broadcast_to(pre[:,None],pp.shape),pp)['delta52'][0]
    cp=m.difference52(np.broadcast_to(pre[:,None],pt.shape),pt)['delta52'][0]
    correct=(cm[1]-cm[0])-(cp[1]-cp[0])
    wrong=m.difference52(pp[:,0],pp[:,1])['delta52'][0]-m.difference52(pt[:,0],pt[:,1])['delta52'][0]
    analytic=(log_unit(mul(mul(qc,qa),qb))-log_unit(mul(qc,qa)))-(log_unit(mul(qa,qb))-log_unit(qa))
    left=quat([1,2,3],.31)
    shifted_pre=state(R(mul(left,q0)))[None]
    shifted_pp=np.stack([state(R(mul(left,qpm))),state(R(mul(left,qpp)))])[None]
    shifted_pt=np.stack([state(R(mul(left,qtm))),state(R(mul(left,qtp)))])[None]
    scm=m.difference52(np.broadcast_to(shifted_pre[:,None],pp.shape),shifted_pp)['delta52'][0]
    scp=m.difference52(np.broadcast_to(shifted_pre[:,None],pt.shape),shifted_pt)['delta52'][0]
    checks={'noncommuting_commonpre_agrees_analytic_wrong_minus_charts_falsely_zero':
       np.max(abs(correct[43:46]-analytic))<TOL and np.linalg.norm(analytic)>.01 and np.max(abs(wrong))<TOL
       and np.max(abs(((scm[1]-scm[0])-(scp[1]-scp[0]))-correct))<TOL}
    proper=m.native_log_reference(R(quat([1,2,-3],.8)));badR=1.2*R(quat([0,0,1],.3));info=m.native_log_reference(badR)
    qw=np.sqrt(1+badR.trace())/2;qz=(badR[1,0]-badR[0,1])/(4*qw);expected_angle=2*np.arctan2(abs(qz),qw)
    identity=m.native_log_reference(np.eye(3))
    checks['actual_native_quaternion_normalization_not_trace_acos_repair']=np.max(abs(proper['rotation_log']-quat([1,2,-3],.8)[1:]/np.sin(.4)*.8))<TOL
    checks['actual_native_quaternion_normalization_not_trace_acos_repair'] &= abs(np.linalg.norm(info['canonical_unit_quaternion'])-1)<TOL and abs(info['chosen_quaternion_norm']-1)>.01
    checks['actual_native_quaternion_normalization_not_trace_acos_repair'] &= abs(info['native_quaternion_angle_rad']-expected_angle)<TOL and bool(info['raw_trace_cosine_outside_unit_interval']) and np.array_equal(identity['candidate_sqrt_clamp_mask'],[False,True,True,True])
    inside=m.native_log_reference(R(quat([0,1,0],np.pi-.011)));outside=m.native_log_reference(R(quat([0,1,0],np.pi-.009)))
    checks['strict_pi_margin_point01_predeclared_no_chart_switch']=bool(inside['strict_pi_margin_passed']) and not bool(outside['strict_pi_margin_passed']) and abs(inside['native_quaternion_angle_rad']-(np.pi-.011))<TOL
    scale=m.fixed_scale52(np.arange(52,dtype='f8')/10,np.full(52,.5));saved=scale.copy()
    uu=np.zeros((2,2,20),dtype='f4');uu[0,:,0]=[np.float32(1-.002),np.float32(1+.002)];uu[1,:,4]=[np.float32(1e-8),np.float32(1)]
    h=uu[:,1].astype('f8')-uu[:,0].astype('f8');actual=h[np.arange(2),[0,4]]
    sp=np.stack([state(),state()]);yp=np.broadcast_to(state(),(2,2,58)).copy();fp=yp.copy();fp[:,1,0]=actual*scale[0]
    point=state()[None];point[0,0]=2*scale[0]
    losses=m.calibration_losses(point,state()[None],sp,fp,yp,uu,np.array([0,4]),scale)
    checks['fixed_mixed_unit_s52_and_actualF32_endpoints_F64_first_loss_closedform']=abs(losses['Lpoint']-4/52)<TOL and abs(losses['Lpair']-1/52)<TOL
    checks['fixed_mixed_unit_s52_and_actualF32_endpoints_F64_first_loss_closedform'] &= abs(losses['method_loss']-5/52)<TOL and np.array_equal(saved,scale) and np.array_equal(losses['s52_frozen'],saved)
    checks['fixed_mixed_unit_s52_and_actualF32_endpoints_F64_first_loss_closedform'] &= losses['actual_h_over_one_rad'][0]!=.004 and losses['actual_h_over_one_rad'][1]!=float(np.float32(uu[1,1,4]-uu[1,0,4])) and losses['finite_native_backward_qualified'] is False
    zero=uu.copy();zero[1,1]=zero[1,0]
    try:m.calibration_losses(point,state()[None],sp,fp,yp,zero,np.array([0,4]),scale);blocked=False
    except ValueError as e:blocked='ZERO_ACTUAL_DENOMINATOR_BLOCKS_ALL' in str(e)
    checks['one_zero_denominator_blocks_all_not_filtered_and_no_backend']=blocked and not any(k in sys.modules for k in ('torch','mujoco','warp','scipy','dex_supermimic','mjlab'))
    checks={k:bool(v) for k,v in checks.items()}
    report={'schema':'full151_common_pre_calibration_independent_math_CPU_proof:1','passed':sum(checks.values()),'total':len(checks),'checks':checks,'source_pins':pins,
      'synthetic_only':True,'actual_calibration_data_or_models_FW_AD_Env_physics_GPU':0,
      'diagnostics':{'noncommuting_correct_common_pre_rotation_residual':correct[43:46].tolist(),'incorrect_separate_minus_rotation_residual':wrong[43:46].tolist(),
       's52_units':{k:v[2] for k,v in m.BLOCKS52.items()},'chart_margin_rule':'native canonical quaternion theta STRICTLY < pi-.01, source-only suggested guard pending formal protocol',
       'non_SO3_fixture':'Deliberate algebraic input tests native quaternion normalization/trace distinction, not an accepted physical rotation or repaired data.'},
      'limitations':['NumPy F64 formula reference, not native FP32 bitwise or backward qualification.','Raw trace/candidate clamps are saved independently; native Log does not use trace-acos.','No actual152/64 response, loss, normalizer, weight or calibration update read/evaluated.']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False));ready={'source_pins':pins,'cpu_report':pin(OUT/'report.json'),'cpu_checks':{'passed':report['passed'],'total':report['total']}}
    (OUT/'READY.json').write_text(json.dumps(ready,indent=2));print(json.dumps({'READY':pin(OUT/'READY.json'),**ready},indent=2));return 0 if all(checks.values()) else 2


if __name__=='__main__':raise SystemExit(main())
