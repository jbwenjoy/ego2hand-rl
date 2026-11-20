"""Prospective action suggestions for phase-conditional finite geometry-capacity mismatch."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys,json,subprocess
import numpy as np
import torch
import check_pen_native_load_suggestions as probe
from supermimic_research.mechanics.pen_capacity_forecast import load_capacity_head,capacity_mismatch
from supermimic_research.mechanics.pen_capacity_trace import packet_capacity
from supermimic_research.artifacts import verify_file_pin,file_pin
from supermimic_research.strict_json import write_json

read=lambda p:json.loads(verify_file_pin(p).read_bytes())

def main(fit,out):
    b=probe.b;root=b.ROOT/'runs';out.mkdir(exist_ok=False)
    model=read(file_pin(fit/'report.json'))['variants']['state']['model']
    net=load_capacity_head(torch.load(verify_file_pin(model),map_location='cpu',weights_only=False),dtype=torch.float64,device='cpu')
    net.eval().requires_grad_(False)
    targets=read(file_pin(root/'pen_capacity_targets_v1/report.json'))['refs']['clip_001']['series']
    with np.load(verify_file_pin(targets)) as z:values=z['capacity'];eligible=z['eligible']
    actors=read(read(file_pin(root/'pen_load_native_suggestions_v1/report.json'))['intent'])['actors']
    write_json(out/'intent.json',dict(actors=actors,model=model,targets=targets,phases=probe.PHASES,raw_L2=probe.EPS,H=1,repeats=2,
        scope='Minimize the mean absolute difference of predicted 8-tick mean directional capacity vs source-frame geometric target. Native directions not used to choose action. Score only eligible frames. No pose loss, actuator optimization or Actor parameter update.'))
    def suggestion(rt,net,ph,c,mean):
        phase=int(rt.motion.time_steps[0]);i=phase-1
        if not eligible[i]:return dict(skip='target masked',phase=phase)
        layout=c['model_layout'];sim=rt.env.sim
        state=probe.tensor(probe.native_state55(sim.wp_data.qpos.numpy()[0],sim.wp_data.qvel.numpy()[0],c['qids'],c['dofs'],layout['object_qposadr'],layout['object_dofadr']))
        base=mean[0].detach().cpu().double().clamp(-1,1)
        ref=rt.motion.joint_pos[0].cpu().double()
        ema=rt.action._processed_actions[0].cpu().double()
        bias=rt.features.robot.data.encoder_bias[0,rt.features.local_joints].cpu().double()
        lo,hi=[v[0].cpu().double() for v in rt.action.target_position_limits]
        target=probe.tensor(values[i])
        def score(raw):
            pd=(ref+.5*ema+.25*raw.clamp(-1,1)-bias).clamp(lo,hi)
            return capacity_mismatch(net(state,pd),target),pd
        with torch.enable_grad():
            u=base.clone().requires_grad_();loss,_=score(u);g=torch.autograd.grad(loss,u)[0].detach()
        active=torch.zeros(20,dtype=torch.bool)
        for j,a in enumerate(c['aids']):
            if probe.normal_group(sim.mj_model.joint(int(sim.mj_model.actuator_trnid[a,0])).name) in (1,2):active[j]=True
        center=ref+.5*ema+.25*base-bias
        active &= (base.abs()<1-probe.EPS)&(center>lo+probe.EPS*.25)&(center<hi-probe.EPS*.25)
        g=g*active;norm=g.norm()
        if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
        if norm<1e-12:return dict(skip='zero feasible gradient',phase=phase)
        step=probe.EPS*g/norm;actions={}
        for k,raw in [('BASE',base),('DESCEND',base-step),('ASCEND',base+step)]:
            v,pd=score(raw);actions[k]=dict(raw=raw.tolist(),cost=float(v),pd=[pd.tolist()])
        return dict(phase=phase,proposals=actions,active_axes=torch.where(active)[0].tolist(),raw_grad_norm=float(norm))
    sources=sorted((b.ROOT/'src').rglob('*.py'))+[Path(__file__),b.ROOT/'tools/check_pen_native_load_suggestions.py',b.ROOT/'tools/collect_guarded_finger_response.py',b.ROOT/'tools/benchmark_pen_scaled.py',b.OBJECTIVE,b.COLLISION]
    receipt=b.freeze_sources(sources,out/'snapshot',b.ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=b.ROOT,text=True).strip())
    write_json(out/'source_receipt.json',receipt)
    original=b.Simulation.step;rec=None;ticks=0
    def activate(v):
        nonlocal rec
        rec=v
    def step(sim):
        nonlocal ticks
        if rec is not None and rec.enabled:rec.before_step()
        ret=original(sim);ticks+=sim.num_envs
        if rec is not None and rec.enabled:rec.after_step()
        return ret
    manifest=b.factory();results={}
    with ExitStack() as stack:
        stack.enter_context(torch.no_grad());stack.enter_context(torch.jit.optimized_execution(False))
        stack.enter_context(patch.object(probe,'H',1));stack.enter_context(patch.object(probe,'suggest',suggestion));stack.enter_context(patch.object(b.Simulation,'step',step))
        col=stack.enter_context(b.backend_scope(b.COLLISION))
        ident=b.execution_identity('STABLE_SAME_REGIME_FRICTION_OR',col.contract['backend_id'],manifest)
        for arm,pin in actors.items():
            with b.CostProfile(ident,manifest) as profile:
                rt=profile.create_runtime(b.OBJECTIVE,1)
                try:
                    rt.set_fixed_frame_features(False);rt.support_weight=.2 if arm=='support' else 0.
                    st=torch.load(verify_file_pin(pin),map_location='cpu',weights_only=False)
                    actor,_,_=b.model_pair(rt.device,SimpleNamespace(**st['config']));actor.load_state_dict(st['actor']);actor.eval()
                    probe.collect(rt,actor,net,arm,out/arm,receipt,activate)
                    layout=dict(object_body=rt.features.body,object_qposadr=rt.features.qa)
                    model=rt.env.sim.mj_model;friction=rt.env.sim.wp_model.geom_friction.numpy()
                    armreport=read(file_pin(out/arm/'report.json'));byphase={}
                    for case in armreport['cases']:
                        phase=case['phase'];s=case['steps'][0]
                        with np.load(verify_file_pin(s['trace'])) as z:packet=packet_capacity(z,model,layout,friction)
                        err=float(np.abs(packet['capacity'].mean(axis=0)-values[phase-1]).mean())
                        byphase.setdefault(phase,{}).setdefault(case['kind'],[]).append(dict(error=err,valid=s['valid'] and not s['terminal'],shaft_deg=s['shaft_deg'],
                            native_reward=s['reward'],containment=packet['maximum_containment_excess']))
                    rows=[]
                    for phase,cases in byphase.items():
                        values_by_kind={k:[v['error'] for v in cs] for k,cs in cases.items()}
                        complete=all(len(cs)==2 and all(v['valid'] and v['containment']<2e-5 for v in cs) for cs in cases.values()) and len(cases)==3
                        vs=values_by_kind
                        win=complete and max(vs['DESCEND'])+np.ptp(vs['BASE'])<min(min(vs['BASE']),min(vs['ASCEND']))
                        rows.append(dict(phase=phase,cases=cases,admitted=bool(complete),strict_win=bool(win),gain=float(np.mean(vs['BASE'])-np.mean(vs['DESCEND']))))
                    results[arm]=dict(declared=len(probe.PHASES),eligible=sum(eligible[np.array(probe.PHASES)-1]),observed=len(rows),rows=rows,strict_wins=sum(r['strict_win'] for r in rows))
                finally:activate(None);rt.close()
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),native=ticks,results=results))
if __name__=='__main__':main(*map(lambda s:Path(s).resolve(),sys.argv[1:]))
