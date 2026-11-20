"""Prospective same-source comparison of capacity corrections selected by two learned heads."""
import json,sys,subprocess
from pathlib import Path
from types import SimpleNamespace
from contextlib import ExitStack
import numpy as np
import torch
import mujoco
import check_pen_native_load_suggestions as old
from supermimic_research.mechanics.pen_capacity_forecast import load_capacity_head,capacity_mismatch
from supermimic_research.mechanics.pen_capacity_trace import packet_capacity
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json

b=old.b
PHASES=[160,180,192,200,206,210,214,218,222,226]
EPS=.02
read=lambda pin:json.loads(verify_file_pin(pin).read_bytes())


def setup(rt):
    sim=rt.env.sim
    body=rt.features.body;qa=rt.features.qa;da=int(sim.mj_model.jnt_dofadr[sim.mj_model.body_jntadr[body]])
    return dict(dt=.0025,qids=rt.features.qids.cpu().tolist(),dofs=rt.features.dofs.cpu().tolist(),aids=rt.action._position_ctrl_ids.cpu().tolist(),
        model_layout=dict(object_body=body,object_qposadr=qa,object_dofadr=da,COM_BODY=sim.mj_model.body_ipos[body].tolist()))


def proposals(rt,nets,mean,c,target):
    s=rt.env.sim;layout=c['model_layout']
    x=old.tensor(old.native_state55(s.wp_data.qpos.numpy()[0],s.wp_data.qvel.numpy()[0],c['qids'],c['dofs'],layout['object_qposadr'],layout['object_dofadr']))
    raw=mean[0].clamp(-1,1).detach().cpu().double()
    intercept=rt.motion.joint_pos[0].cpu().double()+.5*rt.action._processed_actions[0].cpu().double()-rt.features.robot.data.encoder_bias[0,rt.features.local_joints].cpu().double()
    lo,hi=[v[0].cpu().double() for v in rt.action.target_position_limits]
    pd=lambda v:(intercept+.25*v.clamp(-1,1)).clamp(lo,hi)
    active=torch.tensor([old.normal_group(s.mj_model.joint(int(s.mj_model.actuator_trnid[a,0])).name) in (1,2) for a in c['aids']])
    active&=(raw.abs()<1-EPS)&((intercept+.25*raw)>lo+.25*EPS)&((intercept+.25*raw)<hi-.25*EPS)
    def record(u):
        return dict(raw=u.tolist(),pd=pd(u).tolist(),predicted={n:float(capacity_mismatch(net(x,pd(u)),target)) for n,net in nets.items()})
    result={'BASE':record(raw)};norms={}
    for name,head in nets.items():
        with torch.enable_grad():
            u=raw.clone().requires_grad_();cost=capacity_mismatch(head(x,pd(u)),target)
            g=torch.autograd.grad(cost,u)[0].detach()*active
        n=g.norm().item();norms[name]=n
        if not np.isfinite(n):raise ValueError('Nonfinite proposal')
        if n<1e-12:continue
        for sign in (-1,1):
            action=raw+sign*(EPS/n)*g
            result[name+('_down' if sign==-1 else '_up')]=record(action)
    return result,norms


def main(fit,out):
    out.mkdir(exist_ok=False);runs=b.ROOT/'runs'
    report=read(file_pin(fit/'report.json'))
    pins={k:report['variants'][k]['model'] for k in ('absolute','response')}
    nets={k:load_capacity_head(torch.load(verify_file_pin(pin),map_location='cpu',weights_only=True),dtype=torch.float64,device='cpu').requires_grad_(False).eval() for k,pin in pins.items()}
    actors={'new_residual':file_pin(runs/'pen_paired_imagination_v1/A1_7413.pt'),'support':file_pin(runs/'pen_scaled_support_matched_v1/actor_02781.pt')}
    target_pin=read(file_pin(runs/'pen_capacity_targets_v1/report.json'))['refs']['clip_001']['series']
    with np.load(verify_file_pin(target_pin)) as f:eligible=f['eligible'];targets=f['capacity']
    write_json(out/'intent.json',dict(models=pins,actors=actors,targets=target_pin,phases=PHASES,raw_radius=EPS,scope='Previously inspected reference, new residual actor excluded from fitting; support actor exploratory. Both learned heads choose their own signed components on same eight feasible axes; same native source for BASE and both signs, two numerical repeats. No policy updates.'))
    files=sorted((b.ROOT/'src').rglob('*.py'))+[Path(__file__),b.ROOT/'tools/check_pen_native_load_suggestions.py',b.ROOT/'tools/collect_guarded_finger_response.py',b.ROOT/'tools/benchmark_pen_scaled.py',b.OBJECTIVE,b.COLLISION]
    receipt=b.freeze_sources(files,out/'snapshot',b.ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=b.ROOT,text=True).strip())
    write_json(out/'source_receipt.json',receipt)
    original=b.Simulation.step;rec=None;ticks=0;results={}
    def step(sim):
        nonlocal ticks
        if rec is not None and rec.enabled:rec.before_step()
        v=original(sim);ticks+=sim.num_envs
        if rec is not None and rec.enabled:rec.after_step()
        return v
    b.Simulation.step=step
    try:
        with torch.no_grad(),torch.jit.optimized_execution(False),b.backend_scope(b.COLLISION) as col:
            manifest=b.factory();ident=b.execution_identity('STABLE_SAME_REGIME_FRICTION_OR',col.contract['backend_id'],manifest)
            for arm,actor_pin in actors.items():
                folder=out/arm;folder.mkdir()
                with b.CostProfile(ident,manifest) as profile:
                    rt=profile.create_runtime(b.OBJECTIVE,1)
                    try:
                        rt.set_fixed_frame_features(False);rt.support_weight=.2 if arm=='support' else 0.
                        st=torch.load(verify_file_pin(actor_pin),map_location='cpu',weights_only=False)
                        actor,_,_=b.model_pair(rt.device,SimpleNamespace(**st['config']));actor.load_state_dict(st['actor']);actor.eval()
                        c=setup(rt);physics=b.save_physics(rt,folder)
                        model=mujoco.MjModel.from_binary_path(str(folder/'model.mjb'));b.require_structure(model)
                        with np.load(folder/'physical_context.npz') as f:physical=dict(f)
                        bank=[];stopped=None
                        for phase in range(1,max(PHASES)+1):
                            assert int(rt.motion.time_steps[0])==phase
                            raw=actor.distribution(rt.observation()['policy']).mean
                            if phase in PHASES and eligible[phase-1]:
                                src=folder/f'P{phase:03}.pt';torch.save(b.native.capture(rt,receipt),src)
                                actions,norms=proposals(rt,nets,raw,c,old.tensor(targets[phase-1]))
                                bank.append(dict(phase=phase,source=file_pin(src),actions=actions,norms=norms))
                            if phase==max(PHASES):break
                            _,_,term,trunc,_=rt.step(raw)
                            if (term|trunc).any():stopped=phase;break
                        bp=folder/'proposals.json';write_json(bp,dict(prefix_stop=stopped,entries=bank));observations=[]
                        rec=b.OwnedProbeRecorder(rt)
                        for ix,item in enumerate(bank):
                            phase=item['phase'];state=torch.load(verify_file_pin(item['source']),map_location='cpu',weights_only=False)
                            order=list(item['actions'])
                            if ix%2:order=order[:1]+list(reversed(order[1:]))
                            with old.SourceValidationEpoch(state['sources'],state['engine_sources']) as epoch:
                                for slot,name in enumerate(order+list(reversed(order))):
                                    audit=b.native.restore(rt,state,source_epoch=epoch);assert audit['all_array_bytes_equal'] and audit['all_scalar_values_equal']
                                    cmd=np.array(item['actions'][name]['raw'],dtype=np.float32)
                                    rec.begin(cmd);_,reward,term,trunc,_=rt.step(torch.from_numpy(cmd[None]).to(rt.device));z=rec.finish()
                                    np.testing.assert_allclose(z['post_ctrl'][:,c['aids']],np.broadcast_to(item['actions'][name]['pd'],(8,20)),atol=2e-7,rtol=0)
                                    metrics,dec=b.decode_saved_case(c,z,model,physical)
                                    pkt=b.pack_case(model,c,z,dec,c['qids'],c['dofs'],c['aids'],z['initial_warmstart'],physics)
                                    cap=packet_capacity(z,model,c['model_layout'],physical['geom_friction'])
                                    valid=bool(pkt['valid'][0] and all(metrics['checks'].values()) and np.all(pkt['geometry_flags'][0]<=[.002,.0002,.001]) and cap['maximum_containment_excess']<2e-5 and cap['positive_gap_skipped_max_load_N']<=1e-5)
                                    p=folder/f'P{phase:03}_S{slot:02}_{name}.npz';np.savez_compressed(p,**z,capacity=cap['capacity'])
                                    observations.append(dict(phase=phase,name=name,slot=slot,valid=valid,terminal=bool((term|trunc).any()),reward=reward.item(),mismatch=float(np.abs(cap['capacity'].mean(0)-targets[phase-1]).mean()),trace=file_pin(p)))
                        sr={}
                        for name in nets:
                            groups=[]
                            for it in bank:
                                data=[r for r in observations if r['phase']==it['phase']]
                                trials=[[r for r in data if r['name']==key] for key in ('BASE',name+'_down',name+'_up')]
                                if not all(len(rs)==2 and all(r['valid'] and not r['terminal'] for r in rs) for rs in trials):continue
                                base,down,up=[np.array([r['mismatch'] for r in rs]) for rs in trials]
                                groups.append(dict(phase=it['phase'],gain=float(base.mean()-down.mean()),win=bool(down.max()+np.ptp(base)<min(base.min(),up.min()))))
                            sr[name]=dict(groups=groups,complete=len(groups),wins=sum(r['win'] for r in groups),median_gain=float(np.median([r['gain'] for r in groups])) if groups else None)
                        rp=folder/'report.json';write_json(rp,dict(proposals=file_pin(bp),observations=observations,summary=sr));results[arm]=dict(report=file_pin(rp),summary=sr)
                    finally:rec=None;rt.close()
        write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),native_ticks=ticks,results=results))
        for k,v in results.items():print(k,[(m,n['wins'],n['complete'],n['median_gain']) for m,n in v['summary'].items()])
    finally:b.Simulation.step=original
if __name__=='__main__':main(*map(lambda s:Path(s).resolve(),sys.argv[1:]))
