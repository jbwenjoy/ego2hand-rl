"""Explicit parameter-free CPU arithmetic contract. No NN or native calls.

Default is JSON/hash only. Root must separately authorize --run. FakeActor is
a plain callable, not torch.nn.Module. Replay fixtures are synthetic, not
native integration-state or physical feasibility evidence.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys
import numpy as np
H=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def check(p):assert pin(p['path'])==p;return Path(p['path'])
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def load(name,p):
    sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def metadata(path):
    p=read(path);assert p['schema']=='phone_geometry_actor_parameterfree_CPU_contract:1'
    assert type(p['parameterfree_cpu_run_authorized']) is bool and p['NN_native_training_authorized'] is False
    for v in p['inputs'].values():check(v)
    for v in p['sources'].values():check(v)
    return p
def run(path):
    p=metadata(path);assert p['parameterfree_cpu_run_authorized'] is True
    O=Path(p['output']);O.mkdir(exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch
    torch.set_num_threads(4);rng=torch.get_rng_state().clone()
    T=load('parameterfree_phone_task',p['sources']['task']['path']);R=load('parameterfree_phone_replay',p['sources']['reader_replay']['path'])
    I=load('parameterfree_phone_imagination',p['sources']['imagination']['path']);G=load('parameterfree_phone_directional',p['sources']['directional']['path'])
    Exp=load('parameterfree_world_Exp',p['inputs']['rotation']['path'])
    b=read(p['inputs']['bundle']['path']);x=torch.tensor(b['initial']['native_PRE_state58_F64'],dtype=torch.float64)
    a1=T.PhoneTask(p['inputs']['bundle']['path'],mode='A1_tracking')
    m=a1.initial_memory(x,native_h_s=.00015625,n_substeps=128)
    ref0=a1.reference(m,0);assert torch.max(abs(ref0['COM_world_m']-x[40:43]))<1e-12
    assert torch.max(abs(ref0['R_BODY_world']-x[43:52].reshape(3,3)))<1e-12
    ref1=a1.reference(m,1);y=x.clone();y[40:43]=ref1['COM_world_m'];y[43:52]=ref1['R_BODY_world'].flatten()
    zero_action=torch.zeros(20,dtype=torch.float32);reward=a1.rewards(y,m,zero_action)
    assert int(reward['target_frame'])==1 and abs(float(reward['terms_unscaled']['object_COM'])-2)<1e-12 and abs(float(reward['terms_unscaled']['object_SO3_chordal'])-8)<1e-12
    c,u=a1.control_target(zero_action,m);assert torch.equal(u,torch.tensor(b['initial']['ctrl20_F64'],dtype=torch.float64))
    advanced=a1.advance_memory(m,zero_action,c,u);assert int(a1.reference(advanced,1)['frame'])==2
    scale=torch.tensor(b['action']['symmetric_residual_scale20_rad'],dtype=torch.float64)
    lo=torch.tensor(b['action']['target_lower20'],dtype=torch.float64);hi=torch.tensor(b['action']['target_upper20'],dtype=torch.float64)
    evaluations=0
    for cs in [-1.,1.]:
        for aa in [-1.,1.]:
            mm=dict(m,c1=cs*scale);ac=torch.full((20,),aa,dtype=torch.float32);cc,uu=a1.control_target(ac,mm)
            assert torch.allclose(cc,.5*(aa+cs)*scale,rtol=0,atol=1e-15) and torch.all(uu>=lo) and torch.all(uu<=hi);evaluations+=1
    for j in range(20):
        for sign in [-1.,1.]:
            ac=zero_action.clone();ac[j]=sign;_,uu=a1.control_target(ac,m)
            mask=torch.arange(20)!=j;assert torch.equal(uu[mask],u[mask]);evaluations+=1
    mm=dict(m,step=torch.tensor(7),a1=torch.linspace(-.8,.8,20,dtype=torch.float64),a2=torch.linspace(.7,-.7,20,dtype=torch.float64),c1=scale*.2)
    flat=a1.flatten_memory(mm);back=a1.unflatten_memory(flat)
    assert flat.shape==(83,) and all(torch.equal(mm[k],back[k]) for k in mm)
    rejects=[]
    for name,fn in [('fractional_step',lambda:a1.unflatten_memory(flat+torch.nn.functional.one_hot(torch.tensor(0),83)*.25)),
                    ('fractional_n',lambda:a1.unflatten_memory(flat+torch.nn.functional.one_hot(torch.tensor(82),83)*.25)),
                    ('wrong_H',lambda:a1.initial_memory(x,native_h_s=.0002,n_substeps=128)),
                    ('float_n',lambda:a1.initial_memory(x,native_h_s=.00015625,n_substeps=128.0))]:
        try:fn()
        except ValueError:rejects.append(name)
        else:raise AssertionError(name+' not rejected')
    geometry=G.DirectionalGapProvider(p['inputs']['catalog']['path'])
    obs=a1.observation(x,m,geometry);assert obs.shape==(222,) and obs.dtype==torch.float32
    class FakeActor:
        def __call__(self,o):assert o.shape==(1,222);return torch.zeros((1,20),dtype=torch.float32)
    action=R.online_actor_action(FakeActor(),obs);assert action.shape==(20,)
    a1.control_target(action,m)
    try:I.imagined_actor_objective(actor=None,critic=None,world_model=None,task=a1,geometry=geometry,codec=None,norms=None,physical=None,initial_state58=x,memory=m,horizon=1,wm_step=None)
    except ValueError as e:assert 'B>=1' in str(e)
    else:raise AssertionError('Scalar imagination input accepted')
    # Synthetic raw packet; receipts are fixture tags, not a guessed native layout.
    def packet(count,complete):
        ticks=np.arange(count,dtype=np.int64);contact=[]
        for t in ticks:
            for gid in [15,28,41]:contact.append(np.r_[t,gid,71,3,0,-.000075,[0,0,0],np.eye(3).ravel(),[1,0,0,0,0,0]])
        target=x.numpy().copy();target[40]+=.001
        return dict(state_schema='phone_COM_world_state58:1',delta_schema='phone_physical_world_delta52:2',source_state58=x.numpy(),actual_control20=u.numpy(),
            target_state58=target,complete_H20ms=complete,finite=True,warning_numbers=np.zeros(10,dtype=np.int64),termination='COMPLETE_INTERVAL' if complete else 'ABORT_FIXTURE_PARTIAL',
            source_receipt=dict(fixture=True,role='before_action'),source_after_action_receipt=dict(fixture=True,role='after_action'),target_receipt=dict(fixture=True,role='target'),
            trace=dict(pre_tick=ticks,contact_rows24=np.asarray(contact),pre_qfrc_actuator20=np.zeros((count,20))),summary=dict(original_two_pad_fixture=True))
    sink=[]
    for label,count,complete,known in [('full',128,True,True),('unknown_objective',128,True,False),('partial',13,False,False)]:
        packet_reward=dict(total=torch.tensor(.1 if known else float('nan')),objective_known=torch.tensor(known))
        rr=R.build_replay_transition(interval=packet(count,complete),x=x,memory=m,action=action,residual=c,executed_u=u,observation=obs,reward_packet=packet_reward,task=a1,raw_sink=sink.append,episode_id=0)
        assert bool(rr['valid'][0])==bool(complete and known)
        if label=='partial':assert torch.isnan(rr['x1']).all() and np.isnan(sink[-1]['target_H58']).all() and np.isfinite(sink[-1]['observed_prefix_state58']).all()
        if not known:assert not bool(rr['valid'][0]) and not bool(torch.nan_to_num(rr['valid'].float())[0])
    assert len(sink)==3 and sink[0]['tripod_contacts']['triple_simultaneous_coverage']==1.
    a2=T.PhoneTask(p['inputs']['bundle']['path'],mode='A2_engineered_material_proxy',geometry=geometry)
    xx=x.clone().requires_grad_(True);proxy=a2.proxy(xx)
    assert torch.max(proxy['material_distance_m'])<1e-12 and torch.all(proxy['soft_proximity_weights']==1) and abs(float(proxy['score'].detach())-1)<1e-12
    assert proxy['proxy_known'].all() and not proxy['actual_contact_known'].any() and not proxy['actual_force_known'].any()
    support_grad=torch.autograd.grad(proxy['support_BODY_COM512'].sum(),xx,retain_graph=True)[0]
    chord_grad=torch.autograd.grad(proxy['score'],xx)[0]
    assert torch.isfinite(support_grad[:20]).all() and torch.isfinite(chord_grad[:20]).all()
    theta=torch.zeros(3,dtype=torch.float64,requires_grad=True);R0=x[43:52].reshape(3,3)
    rotated=torch.cat([x[:43],(Exp.so3_exp_torch(theta)@R0).flatten(),x[52:]])
    termination=a1.termination(rotated,m);termination_grad=torch.autograd.grad(termination['soft_loss_probability'],theta)[0]
    assert torch.isfinite(termination_grad).all() and not bool(termination['predicted_loss_of_hold'])
    assert torch.equal(rng,torch.get_rng_state()) and not torch.cuda.is_initialized() and not any(v in sys.modules for v in ['mujoco','warp'])
    arrays=dict(initial_state58_F64=x.numpy(),memory83_F64=flat.numpy(),observation222_FP32=obs.numpy(),proxy_material_distance3_m=proxy['material_distance_m'].detach().numpy(),
        reference_support512=proxy['reference_support_BODY_COM512'].detach().numpy(),current_proxy_support512=proxy['support_BODY_COM512'].detach().numpy(),
        proxy_support_q20_AD=support_grad[:20].numpy(),proxy_CHORD_q20_AD=chord_grad[:20].numpy(),identity_termination_rotvec_AD=termination_grad.numpy())
    with (O/'parameterfree_evidence.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    checks={k:True for k in ['named_initial_reference_and_POST_k_plus1_phase','F32_zero_action_keeps_true_F64_ctrl','EMA_coordinate_corner_ranges_and_axes','strict_memory83_integer_clock_roundtrip','online_singleton_and_batched_imagination_contract','synthetic_complete_unknown_partial_replay_raw_masks','engineered_template_support_and_q_AD_finite','identity_rotation_termination_AD_finite','CPU_rng_unchanged_no_NN_or_native']}
    report=dict(schema='phone_geometry_actor_parameterfree_CPU_result:1',status='PASS_PARAMETERFREE_FUNCTION_CONTRACT_NOT_ACTOR_OR_NATIVE_QUALIFICATION',protocol=pin(path),checks=checks,passed=len(checks),total=len(checks),
        counts=dict(pure_Tensor_FK=geometry.counts['pure_Torch_FK_calls']+sum(a2.proxy.counts[k] for k in ['initial_pure_Tensor_FK','current_pure_Tensor_FK']),input_geometry_AD=3,
            EMA_vector_evaluations=evaluations,synthetic_replay_packets=3,FakeActor_calls=1,Actor_WM_Critic_constructors=0,NN_forward_backward=0,optimizer_updates=0,native_model_FK_forward_steps=0,GPU_calls=0),
        corner_scope='Coordinate-separable four EMA/action endpoint combinations check all20 axes; covers the Cartesian action/EMA corner box algebraically, not an enumerated2^40 performance test',
        proxy_support_q_gradient_norm=float(torch.linalg.vector_norm(support_grad[:20])),proxy_CHORD_q_gradient_norm=float(torch.linalg.vector_norm(chord_grad[:20])),
        zero_angle_subgradient=termination_grad.tolist(),source_rejections=rejects,outputs=dict(evidence=pin(O/'parameterfree_evidence.npz')),
        scope='Parameter-free arithmetic only. Synthetic replay is not native receipt-layout validation; factory CPU/CUDA and real Actor/WM/critic remain unexecuted; no physical gradient, closest normal, contact/force truth, policy training or holding qualification.')
    write(O/'report.json',report);return pin(O/'report.json')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run',action='store_true');a=ap.parse_args()
    if not a.run:
        p=metadata(a.protocol);print(json.dumps(dict(status='METADATA_ONLY_NO_TORCH_ARRAYS_NN_NATIVE',protocol=pin(a.protocol),parameterfree_run_authorized=p['parameterfree_cpu_run_authorized'])))
    else:
        try:print(json.dumps(run(a.protocol)))
        except BaseException as e:
            failure=Path(read(a.protocol)['output']).parent/'parameterfree_failure_v1.json'
            write(failure,dict(error=repr(e),source=pin(Path(__file__)),protocol=pin(a.protocol),automatic_retry=False));raise
