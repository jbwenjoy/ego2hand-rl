"""Prospectively sample one high-risk and one uniform source per temporal stratum."""
import json,sys,subprocess
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
import torch
import collect_guarded_finger_response as b
from supermimic_research.mechanics.pen_switch_risk import PenSwitchRisk
from supermimic_research.mechanics.pen_capacity_trace import packet_capacity
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json

START=96
WIDTH=12
BINS=12
END=START+WIDTH*BINS-1
KEYS=('TRAIN_A','TRAIN_B','TEST_RESIDUAL','TEST_SUPPORT')


def main(out):
    out.mkdir(exist_ok=False)
    root=b.ROOT/'runs'
    actors=dict(TRAIN_A=root/'pen_batch_updates_v1/W512_U32/final_actor.pt',TRAIN_B=root/'pen_batch_updates_v1/W1024_U64/final_actor.pt',
        TEST_RESIDUAL=root/'pen_paired_imagination_v1/A1_7413.pt',TEST_SUPPORT=root/'pen_scaled_support_matched_v1/actor_02781.pt')
    pins={k:file_pin(p) for k,p in actors.items()}
    uniform={k:(START+WIDTH*np.arange(BINS)+np.random.default_rng(7411+j).integers(WIDTH,size=BINS)).tolist() for j,k in enumerate(KEYS)}
    write_json(out/'intent.json',dict(actors=pins,uniform=uniform,first_phase=START,stratum_width=WIDTH,strata=BINS,score='max source-only risk over fingers 2 and 3; first maximum in each consecutive stratum; equal budget uniform phase presampled per stratum',
        perturbation='plus/minus 0.04 RAW L2 (1,-1,-1,1,-1,1,1,-1)/sqrt(8) restricted to D2/D3, around clipped policy mean. Two native sham repeats. No action outcome used for selection.',
        scope='Original clip only. TEST actors excluded from collecting training samples; previously examined behavior families. Natural onsets in nominal prefix censused separately from intervention contrasts.'))
    receipt=b.freeze_sources(sorted((b.ROOT/'src').rglob('*.py'))+[Path(__file__),b.ROOT/'tools/collect_guarded_finger_response.py',b.ROOT/'tools/benchmark_pen_scaled.py',b.OBJECTIVE,b.COLLISION],out/'snapshot',b.ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=b.ROOT,text=True).strip())
    write_json(out/'source_receipt.json',receipt)
    manifest=b.factory();original=b.Simulation.step;rec=None;passive=None;hook=None;nt=0;rows=[];details={}
    def step(sim):
        nonlocal nt
        if rec is not None and rec.enabled:rec.before_step()
        result=original(sim);nt+=1
        if rec is not None and rec.enabled:rec.after_step()
        if passive is not None:passive.append(hook(sim.wp_data))
        return result
    b.Simulation.step=step
    try:
        with torch.no_grad(),torch.jit.optimized_execution(False),b.backend_scope(b.COLLISION) as col:
            ident=b.execution_identity('STABLE_SAME_REGIME_FRICTION_OR',col.contract['backend_id'],manifest)
            for role in KEYS:
                folder=out/role;folder.mkdir()
                with b.CostProfile(ident,manifest) as profile:
                    rt=profile.create_runtime(b.OBJECTIVE,1)
                    try:
                        rt.set_fixed_frame_features(False);rt.support_weight=.2 if role=='TEST_SUPPORT' else 0.
                        st=torch.load(verify_file_pin(pins[role]),map_location='cpu',weights_only=False)
                        actor,_,_=b.model_pair(rt.device,SimpleNamespace(**st['config']));actor.load_state_dict(st['actor']);actor.eval()
                        physics=b.save_physics(rt,folder);model=mujoco.MjModel.from_binary_path(str(folder/'model.mjb'));b.require_structure(model)
                        with np.load(folder/'physical_context.npz') as f:physical=dict(f)
                        body=rt.features.body;f=rt.features;sim=rt.env.sim;da=int(model.jnt_dofadr[model.body_jntadr[body]])
                        layout=dict(object_body=body,object_qposadr=f.qa,object_dofadr=da,COM_BODY=model.body_ipos[body].tolist())
                        case=dict(model_layout=layout,dt=.0025)
                        qids=f.qids.cpu().tolist();dofs=f.dofs.cpu().tolist();aids=rt.action._position_ctrl_ids.cpu().tolist()
                        selected=[i for i,a in enumerate(aids) if b.normal_group(model.joint(int(model.actuator_trnid[a,0])).name) in (1,2)];assert len(selected)==8
                        hands={g for g in range(model.ngeom) if (model.geom(g).name or '').startswith('robot/')}
                        loaded={g for g in hands if b.normal_group(model.geom(g).name) in (1,2)}
                        objects=set(np.flatnonzero(model.geom_bodyid==body))
                        def read_contact(d):
                            n=int(d.nacon.numpy()[0]);gg=d.contact.geom.numpy()[:n];dist=d.contact.dist.numpy()[:n]
                            return sorted([(int(g),int(h)) if g in objects else (int(h),int(g)) for (g,h),v in zip(gg,dist) if v<=0 and ((g in objects and h in hands) or (h in objects and g in hands))])
                        hook=read_contact;risk=PenSwitchRisk(model,body);candidates=[];sources=[];saved={};best=-1
                        for phase in range(1,END+1):
                            assert int(rt.motion.time_steps[0])==phase
                            raw=actor.distribution(rt.observation()['policy']).mean
                            if phase>=START:
                                bi=(phase-START)//WIDTH;d=sim.wp_data
                                risk_row=risk.evaluate(d.qpos.numpy()[0],d.qvel.numpy()[0],d.mocap_pos.numpy()[0],d.mocap_quat.numpy()[0])
                                pairs=[r for r in risk_row['pairs'] if r['hand_geom'] in loaded]
                                # All pairs beyond the registered distance cutoff have zero risk.
                                score=max((r['score'] for r in pairs),default=0.)
                                new_best=score>best
                                if new_best or phase==uniform[role][bi]:
                                    if new_best and saved and winner!=uniform[role][bi]:saved.pop(winner,None)
                                    saved[phase]=(b.native.capture(rt,receipt),raw.clamp(-1,1)[0].cpu().numpy().copy())
                                    if new_best:best=score;winner=phase
                                passive=[]
                            _,_,term,trunc,_=rt.step(raw)
                            done=bool((term|trunc).any())
                            if phase>=START:
                                assert len(passive)==8
                                candidates.append(dict(phase=phase,bin=bi,score=score,pairs=pairs,nominal_ticks=passive))
                                passive=None
                                if (phase-START)%WIDTH==WIDTH-1:
                                    for k in sorted(set((winner,uniform[role][bi]))):
                                        ss,raw0=saved[k];path=folder/f'source_{k}.pt';torch.save(ss,path)
                                        sources.append(dict(phase=k,bin=bi,risk_selected=k==winner,uniform_selected=k==uniform[role][bi],score=next(v['score'] for v in candidates if v['phase']==k),source=file_pin(path),raw=raw0.tolist()))
                                    saved.clear();best=-1
                            if done:break
                        screen=folder/'screen.json';write_json(screen,dict(candidates=candidates,selected=sources,uniform=uniform[role]))
                        details[role]=dict(model=file_pin(folder/'model.mjb'),physical_context=file_pin(folder/'physical_context.npz'),layout=layout,selected=selected,screen=file_pin(screen))
                        rec=b.OwnedProbeRecorder(rt)
                        delta=np.zeros(20);delta[selected]=.04*np.array([1,-1,-1,1,-1,1,1,-1])/np.sqrt(8)
                        for ix,s in enumerate(sources):
                            state=torch.load(verify_file_pin(s['source']),map_location='cpu',weights_only=False)
                            branches=[]
                            with b.SourceValidationEpoch(state['sources'],state['engine_sources']) as epoch:
                                for name,sign in ([('sham',0),('minus',-1),('plus',1),('sham_repeat',0)] if ix%2==0 else [('sham',0),('plus',1),('minus',-1),('sham_repeat',0)]):
                                    restore=b.native.restore(rt,state,source_epoch=epoch);assert restore['all_array_bytes_equal'] and restore['all_scalar_values_equal']
                                    u=(np.asarray(s['raw'])+sign*delta).astype(np.float32)
                                    rec.begin(u);_,_,term,trunc,_=rt.step(torch.from_numpy(u[None]).to(rt.device));z=rec.finish()
                                    metrics,decoded=b.decode_saved_case(case,z,model,physical)
                                    pkt=b.pack_case(model,case,z,decoded,qids,dofs,aids,z['initial_warmstart'],physics)
                                    cap=packet_capacity(z,model,layout,physical['geom_friction'])
                                    valid=bool(all(metrics['checks'].values()) and pkt['valid'][0] and np.all(pkt['geometry_flags'][0]<=[.002,.0002,.001]) and cap['maximum_containment_excess']<2e-5)
                                    path=folder/f'P{s["phase"]}_{name}.npz';np.savez_compressed(path,**z)
                                    branches.append(dict(name=name,trace=file_pin(path),valid=valid,terminal=bool((term|trunc).any()),pd=z['post_ctrl'][0,aids].tolist(),capacity=cap['capacity'].mean(0).tolist()))
                            rows.append(dict(actor=role,**s,branches=branches))
                        write_json(out/'progress.json',dict(rows=len(rows),actors=details,native_ticks=nt))
                    finally:rec=None;passive=None;rt.close()
        write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),actors=details,rows=rows,native_ticks=nt))
    except BaseException as e:
        write_json(out/'failure.json',dict(error=repr(e),native_ticks=nt));raise
    finally:b.Simulation.step=original
if __name__=='__main__':main(Path(sys.argv[1]).resolve())

