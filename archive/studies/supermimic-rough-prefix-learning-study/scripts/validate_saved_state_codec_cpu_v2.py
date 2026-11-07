"""Coordinate/phase negative cases and actual recorded-data reconstruction only."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse,copy,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rough_prefix.io import pin,checked,loadnp
from rough_prefix.metadata import write
from rough_prefix.bundle_v2 import load_bundle,rotation


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    source=ROOT/'src/rough_prefix/saved_state_codec.py'
    for path in (source,Path(__file__)):(a.output/path.name).write_bytes(path.read_bytes())
    source_pin=pin(source)
    import torch,mujoco
    torch.set_num_threads(2)
    from types import SimpleNamespace
    from rough_prefix.saved_state_codec import native_state58,bind_fixed_palm,executed_intervals,load_saved_transitions
    b=load_bundle(ROOT/'outputs/input_bundle_v1/bundle.json');m=mujoco.MjModel.from_binary_path(str(checked(b['model'])))
    checks={}
    def check(name,v):checks[name]=bool(v)
    def rejects(name,fn):
        try:fn()
        except (ValueError,KeyError):checks[name]=True
        else:checks[name]=False
    joints=np.array([m.joint(f'robot/right_finger{f}_joint{j}').id for f in range(1,6) for j in range(1,5)])
    qa=m.jnt_qposadr[joints];va=m.jnt_dofadr[joints];obj=m.joint('object/cube_freejoint').id;oq=int(m.jnt_qposadr[obj]);ov=int(m.jnt_dofadr[obj])
    q=np.zeros((1,m.nq),np.float32);v=np.zeros((1,m.nv),np.float32);h=np.float32(2**-.5)
    q[:,qa]=np.arange(20,dtype=np.float32)*.01;v[:,va]=np.arange(20,dtype=np.float32)*.1
    q[:,oq:oq+7]=[.1,.2,.3,h,h,0,0];v[:,ov:ov+6]=[1,2,3,4,5,6]
    palm=np.array([-.4,.3,.7,h,0,0,h],np.float32);x=native_state58(q,v,m,palm).numpy()
    Rp=rotation(palm[3:]);Ro=rotation(q[0,oq+3:oq+7]);expected=np.r_[q[0,qa],v[0,va],Rp.T@(q[0,oq:oq+3]-palm[:3]),(Rp.T@Ro).reshape(-1),Rp.T@v[0,ov:ov+3],Rp.T@Ro@v[0,ov+3:ov+6]]
    roundoff_bound=8*np.finfo(np.float32).eps*np.maximum(1.,np.abs(expected))
    check('native58_matches_independent_nontrivial_frame_math',np.all(np.abs(x[0]-expected)<=roundoff_bound))
    check('body_omega_is_not_mistaken_for_world_omega',np.max(np.abs(x[0,-3:]-(Rp.T@v[0,ov+3:ov+6])))>1)
    check('joint_q_qd_order_exact',np.array_equal(x[0,:20],q[0,qa]) and np.array_equal(x[0,20:40],v[0,va]))
    rejects('bad_full_qpos_shape',lambda:native_state58(q[:,:-1],v,m,palm))
    bad=q.copy();bad[0,0]=np.nan;rejects('nonfinite_state',lambda:native_state58(bad,v,m,palm))
    badp=palm.copy();badp[3:]=0;rejects('invalid_palm_quaternion',lambda:native_state58(q,v,m,badp))
    raw={'raw_time_s':np.arange(61)*.005,'ctrl_rad':np.zeros((61,20),np.float32)};raw['ctrl_rad'][0]=42
    events=[{'phase':k,'actualu':(np.ones((1,20),np.float32)*(k+1)).tolist()} for k in range(15)]
    for k,e in enumerate(events):raw['ctrl_rad'][4*k+1:4*k+5]=e['actualu']
    u,knots=executed_intervals(raw,events)
    check('interval0_uses_new_u0_not_initial_old_ctrl',np.array_equal(u[0],np.ones(20)) and raw['ctrl_rad'][0,0]==42)
    check('exact15interval_to_terminal15_knots',np.array_equal(knots,np.arange(0,61,4)) and u.shape==(15,20))
    for index,value,label in ((4,2,'step4_still_interval0'),(5,1,'step5_starts_interval1')):
        z=copy.deepcopy(raw);z['ctrl_rad'][index]=value;rejects(label,lambda z=z:executed_intervals(z,events))
    z=copy.deepcopy(events);z[0]['phase']=True;rejects('boolean_phase_rejected',lambda:executed_intervals(raw,z))
    z=copy.deepcopy(events);z[1]['actualu'][0][0]=float('nan');rejects('nonfinite_u_rejected',lambda:executed_intervals(raw,z))
    check('future_obs15_has_no_extra_u15',len(u)==len(knots)-1==15)
    actual=loadnp(ROOT/'outputs/eval_primary_a1l_seed0_v1/raw_states.npz');pose,pr=bind_fixed_palm(m,b,actual)
    check('actual_identity_child_palm_binding',pr['palm_body_id']==4 and pr['parent_body_id']==3 and pr['parent_mocap_id']==0)
    z=dict(actual);z['mocap_pos_w']=actual['mocap_pos_w'].copy();z['mocap_pos_w'][1,0,0]+=.001
    rejects('moving_or_different_parent_frame_rejected',lambda:bind_fixed_palm(m,b,z))
    positions=m.body_pos.copy();positions[4,0]=.001
    surrogate=SimpleNamespace(body=m.body,body_parentid=m.body_parentid,body_mocapid=m.body_mocapid,body_jntnum=m.body_jntnum,body_pos=positions,body_quat=m.body_quat)
    rejects('nonidentity_palm_child_rejected',lambda:bind_fixed_palm(surrogate,b,actual))
    datasets=[]
    specs=[('A0','native_a0_v2','native_a0_saved_audit_v2'),('A1L','eval_primary_a1l_seed0_v1','policy_saved_audit_primary_a1l_v1'),('P_ONLY','eval_primary_p_only_seed0_v1','policy_saved_audit_primary_p_only_v1'),('A2','eval_primary_a2_seed0_v1','policy_saved_audit_primary_a2_v1')]
    arrays={}
    for method,run,audit in specs:
        d=load_saved_transitions(b,ROOT/'outputs'/run,ROOT/'outputs'/audit/'report.json',method)
        check(method+'_actual15finite_x58_u20',d['x'].shape==(15,58) and d['x_next'].shape==(15,58) and d['u'].shape==(15,20) and torch.isfinite(d['x_next']).all())
        check(method+'_selfcollision_labels_unknown',d['receipt']['selfcollision_labels'].startswith('UNKNOWN'))
        datasets.append(d['receipt']);arrays.update({method+'__'+k:d[k].numpy() for k in ('x','x_next','u')})
    check('no_CUDA_or_Warp_or_mjwarp',not torch.cuda.is_initialized() and 'warp' not in sys.modules and 'mujoco_warp' not in sys.modules)
    check('executed_source_unchanged',pin(source)==source_pin)
    np.savez_compressed(a.output/'reconstructed_native_transitions.npz',**arrays)
    write(a.output/'report.json',{'schema':'rough_saved_state58_codec_CPU:1','status':'PASS_RECONSTRUCTION_ONLY' if all(checks.values()) else 'FAIL','checks':checks,'passed':sum(checks.values()),'total':len(checks),'source':source_pin,'checker':pin(__file__),'datasets':datasets,'physics_steps_added':0,'Env_constructed':False,'WM_quality_assessed':False,'native_features_recorded':False,'arithmetic_comparison':{'maximum_FP32_vs_FP64_error':float(np.max(np.abs(x[0]-expected))),'bound':'8*FP32epsilon*max(1,abs(FP64expected)) percomponent; quaternionnormalization/two rotations roundoff only, not a physics gate'},'prior_fixture_report':pin(ROOT/'outputs/saved_state_codec_cpu_v1/report.json')})
    print({'status':'PASS_RECONSTRUCTION_ONLY' if all(checks.values()) else 'FAIL','passed':sum(checks.values()),'total':len(checks),'failed':[k for k,v in checks.items() if not v]})
    raise SystemExit(int(not all(checks.values())))

if __name__=='__main__':main()
