"""Local test of learned contact-model action suggestions against matched native continuations."""
import sys,json,subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation
import collect_guarded_finger_response as b
from supermimic_research.mechanics.pen_load_forecast import LoadForecast
from supermimic_research.mechanics.pen_imagined_load import pen_load_step,pen_rotation
from supermimic_research.mechanics.controlled_pen_records import native_state55
from supermimic_research.mechanics.pen_contact_groups import normal_group
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json
from supermimic_research.runtime.source_validation_epoch import SourceValidationEpoch

PHASES=(160,180,192,200,206,210,214,218,222,226)
H=3
EPS=.02


def tensor(x):
    return torch.as_tensor(x,dtype=torch.float64,device='cpu')


def suggest(rt,net,physics,case,mean):
    phase=int(rt.motion.time_steps[0]);sim=rt.env.sim;d=sim.wp_data
    ml=case['model_layout']
    state=tensor(native_state55(d.qpos.numpy()[0],d.qvel.numpy()[0],case['qids'],case['dofs'],ml['object_qposadr'],ml['object_dofadr']))
    joints=rt.motion._frames['joint_pos'][phase:phase+H].detach().cpu().double()
    axis=tensor(Rotation.from_quat(rt.support_table.pose[phase:phase+H, [4,5,6,3]]).as_matrix()[:,:,0])
    base=mean[0].clamp(-1,1).cpu().double()
    bias=rt.features.robot.data.encoder_bias[0,rt.features.local_joints].cpu().double()
    lower,upper=[v[0].detach().cpu().double() for v in rt.action.target_position_limits]
    ema0=rt.action._processed_actions[0].cpu().double()
    def forecast(action):
        x=state;ema=ema0;loss=[];pd_targets=[]
        for i in range(H):
            u=action if i==0 else base
            ema=.5*ema+.25*u.clamp(-1,1)
            pd=(joints[i]+ema-bias).clamp(lower,upper);pd_targets.append(pd)
            x,_,_=pen_load_step(x,pd,None,net,physics['mass'],physics['inertia_BODY'],physics['gravity'])
            loss.append(1-torch.dot(pen_rotation(x)[:,0],axis[i]))
        return torch.stack(loss).mean(),torch.stack(pd_targets)
    with torch.enable_grad():
        test=base.clone().requires_grad_();cost,_=forecast(test)
        grad=torch.autograd.grad(cost,test)[0].detach()
    # Require some symmetric authority on each chosen axis. No clipping of proposals.
    selected=[j for j,idx in enumerate(case['aids']) if normal_group(sim.mj_model.joint(int(sim.mj_model.actuator_trnid[idx,0])).name) in (1,2)]
    mask=torch.zeros(20,dtype=torch.bool);mask[selected]=True
    intended=joints[0]+.5*ema0+.25*base-bias
    mask&=(base.abs()<1-EPS)&(intended>lower+.25*EPS)&(intended<upper-.25*EPS)
    direction=-grad*mask;norm=direction.norm()
    if not torch.isfinite(norm):raise ValueError('Nonfinite RAW derivative')
    if norm.item()<1e-12:return dict(skip='no finite nonzero admissible direction',phase=phase)
    delta=EPS*direction/norm
    proposals={}
    for name,raw in [('BASE',base),('DESCEND',base+delta),('ASCEND',base-delta)]:
        cost,pd=forecast(raw)
        proposals[name]=dict(raw=raw.tolist(),cost=float(cost.detach()),pd=pd.detach().tolist())
    central=(proposals['DESCEND']['cost']-proposals['ASCEND']['cost'])/(2*EPS)
    return dict(phase=phase,proposals=proposals,active_axes=torch.where(mask)[0].tolist(),raw_grad_norm=float(norm),model_centered_slope=central)


def collect(rt,actor,net,arm,folder,receipt,activate):
    folder.mkdir();sim=rt.env.sim;m=sim.mj_model;b.require_structure(m)
    ph=b.save_physics(rt,folder);model=mujoco.MjModel.from_binary_path(str(folder/'model.mjb'))
    with np.load(folder/'physical_context.npz') as f:physical=dict(f)
    body=rt.features.body;qa=rt.features.qa;da=int(m.jnt_dofadr[m.body_jntadr[body]])
    c=dict(dt=.0025,qids=rt.features.qids.cpu().tolist(),dofs=rt.features.dofs.cpu().tolist(),aids=rt.action._position_ctrl_ids.cpu().tolist(),model_layout=dict(object_body=body,object_qposadr=qa,object_dofadr=da,COM_BODY=m.body_ipos[body].tolist()))
    sources=[]
    for phase in range(1,max(PHASES)+1):
        assert phase==int(rt.motion.time_steps[0])
        mean=actor.distribution(rt.observation()['policy']).mean
        if phase in PHASES:
            src=folder/f'source_{phase}.pt';torch.save(b.native.capture(rt,receipt),src)
            sources.append(dict(source=file_pin(src),**suggest(rt,net,ph,c,mean)))
        if phase==max(PHASES):break
        _,_,term,trunc,_=rt.step(mean)
        if (term|trunc).any():break
    write_json(folder/'proposals.json',dict(declared=PHASES,sources=sources))
    recorder=b.OwnedProbeRecorder(rt);activate(recorder)
    cases=[]
    for si,s in enumerate(sources):
        if 'skip' in s:continue
        phase=s['phase'];snap=torch.load(verify_file_pin(s['source']),map_location='cpu',weights_only=False)
        order=['BASE','DESCEND','ASCEND','ASCEND','DESCEND','BASE']
        if si%2:order=['BASE','ASCEND','DESCEND','DESCEND','ASCEND','BASE']
        with SourceValidationEpoch(snap['sources'],snap['engine_sources']) as epoch:
            for rep,kind in enumerate(order):
                audit=b.native.restore(rt,snap,source_epoch=epoch)
                if not audit['all_array_bytes_equal'] or not audit['all_scalar_values_equal']:raise ValueError('Incomplete native restore')
                item=s['proposals'][kind];steps=[]
                for j in range(H):
                    u=np.array((item if j==0 else s['proposals']['BASE'])['raw'],np.float32)
                    recorder.begin(u)
                    _,reward,term,trunc,info=rt.step(torch.from_numpy(u[None]).to(rt.device))
                    z=recorder.finish()
                    np.testing.assert_allclose(z['post_ctrl'][:,c['aids']],np.broadcast_to(item['pd'][j],(8,20)),atol=2e-7,rtol=0)
                    metrics,decoded=b.decode_saved_case(c,z,model,physical)
                    pkt=b.pack_case(model,c,z,decoded,c['qids'],c['dofs'],c['aids'],z['initial_warmstart'],ph)
                    ok=bool(pkt['valid'][0] and all(metrics['checks'].values()) and np.all(pkt['geometry_flags'][0]<=[.002,.0002,.001]))
                    actual=Rotation.from_quat(z['post_qpos'][-1,qa+np.array([4,5,6,3])]).as_matrix()[:,0]
                    target=Rotation.from_quat(rt.support_table.pose[phase+j,[4,5,6,3]]).as_matrix()[:,0]
                    loss=float(1-actual@target)
                    steps.append(dict(loss=loss,shaft_deg=float(np.degrees(np.arccos(np.clip(1-loss,-1,1)))),reward=float(reward.item()),
                        tracking=float(rt.objective_views['tracking_reward'].item()),support=float(rt.objective_views['support_bonus'].item()),valid=ok,terminal=bool((term|trunc).any()),checks=metrics))
                    path=folder/f'p{phase}_{rep}_{kind}_{j}.npz';np.savez_compressed(path,**z);steps[-1]['trace']=file_pin(path)
                    if not ok or (term|trunc).any():break
                cases.append(dict(phase=phase,kind=kind,rep=rep,steps=steps))
        write_json(folder/'progress.json',dict(phase=phase,cases=len(cases)))
    activate(None)
    write_json(folder/'report.json',dict(arm=arm,proposals=file_pin(folder/'proposals.json'),cases=cases))


def main(out):
    out.mkdir(exist_ok=False);r=b.ROOT/'runs'
    f=file_pin(r/'pen_load_forecast_v1/report.json');pin=json.loads(verify_file_pin(f).read_bytes())['outcomes']['state']['model']
    weights=torch.load(verify_file_pin(pin),map_location='cpu',weights_only=True)
    net=LoadForecast(torch.zeros(2,len(weights['center'])),torch.zeros(2,len(weights['out_center'])));net.load_state_dict(weights);net=net.double().eval().requires_grad_(False)
    actors={'residual':file_pin(r/'pen_batch_updates_v1/W4096_U32/final_actor.pt'),'support':file_pin(r/'pen_scaled_support_matched_v1/actor_02781.pt')}
    write_json(out/'intent.json',dict(actors=actors,model=pin,phases=PHASES,H=H,raw_L2=EPS,arms=['BASE','DESCEND','ASCEND'],repetitions=2,
        scope='M3 prospective action suggestions, no Actor fit. Mean shaft chord loss over three controls. First RAW perturbed, next two fixed to current clipped Actor mean for every branch. Exploratory previously inspected behaviors. Valid complete groups only for paired gains; count exclusions and stops.'))
    receipt=b.freeze_sources(sorted((b.ROOT/'src').rglob('*.py'))+[Path(__file__),b.ROOT/'tools/collect_guarded_finger_response.py',b.ROOT/'tools/benchmark_pen_scaled.py',b.OBJECTIVE,b.COLLISION],out/'snapshot',b.ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=b.ROOT,text=True).strip())
    write_json(out/'source_receipt.json',receipt)
    original=b.Simulation.step;rec=None;count=0
    def activate(v):
        nonlocal rec
        rec=v
    def step(sim):
        nonlocal count
        if rec is not None and rec.enabled:rec.before_step()
        v=original(sim);count+=sim.num_envs
        if rec is not None and rec.enabled:rec.after_step()
        return v
    b.Simulation.step=step
    try:
        manifest=b.factory()
        with torch.no_grad(),torch.jit.optimized_execution(False),b.backend_scope(b.COLLISION) as col:
            ident=b.execution_identity('STABLE_SAME_REGIME_FRICTION_OR',col.contract['backend_id'],manifest)
            for arm,actor_pin in actors.items():
                with b.CostProfile(ident,manifest) as profile:
                    rt=profile.create_runtime(b.OBJECTIVE,1)
                    try:
                        rt.set_fixed_frame_features(False);rt.support_weight=.2 if arm=='support' else 0.
                        st=torch.load(verify_file_pin(actor_pin),map_location='cpu',weights_only=False)
                        actor,_,_=b.model_pair(rt.device,SimpleNamespace(**st['config']));actor.load_state_dict(st['actor']);actor.eval()
                        collect(rt,actor,net,arm,out/arm,receipt,activate)
                    finally:activate(None);rt.close()
        write_json(out/'report.json',dict(native=count,intent=file_pin(out/'intent.json'),arms={a:file_pin(out/a/'report.json') for a in actors}))
    except BaseException as e:
        write_json(out/'failure.json',dict(error=repr(e),native=count));raise
    finally:b.Simulation.step=original
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
