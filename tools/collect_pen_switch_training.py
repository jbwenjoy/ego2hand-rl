"""Fresh support-switch training trajectories using only actors from the existing TRAIN partition."""
import sys,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import mujoco
import collect_guarded_finger_response as b
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json

def main(previous,out):
    out.mkdir(exist_ok=False)
    source=json.loads((previous/'intent.json').read_bytes())
    pins={k:source['actors'][k] for k in ['A','B','C','D']}
    schedule=[(driver,phase) for phase in (92,152,198,236) for driver in ('A','B','D')]
    write_json(out/'intent.json',dict(parent=file_pin(previous/'report.json'),inputs=pins,schedule=schedule,
       recipe='No additional raw noise. Follow original training driver, then blend to earlier support Actor C over eight controls and follow C. All sources training-owned. All emitted rows TRAIN.'))
    manifest=b.factory();original=b.Simulation.step;rec=None;ticks=0;episodes=[]
    def step(sim):
        nonlocal ticks
        if rec is not None and rec.enabled:rec.before_step()
        value=original(sim);ticks+=1
        if rec is not None and rec.enabled:rec.after_step()
        return value
    b.Simulation.step=step
    try:
        with torch.no_grad(),torch.jit.optimized_execution(False),b.backend_scope(b.COLLISION) as collision:
            ident=b.execution_identity('STABLE_SAME_REGIME_FRICTION_OR',collision.contract['backend_id'],manifest)
            for driver,phase0 in schedule:
                folder=out/f'{driver}_to_C_{phase0}';folder.mkdir()
                with b.CostProfile(ident,manifest) as profile:
                    runtime=profile.create_runtime(b.OBJECTIVE,1)
                    try:
                        runtime.set_fixed_frame_features(False)
                        actors=[]
                        for label in [driver,'C']:
                            state=torch.load(verify_file_pin(pins[label]),map_location='cpu',weights_only=False)
                            a,_,_=b.model_pair(runtime.device,SimpleNamespace(**state['config']))
                            a.load_state_dict(state['actor']);actors.append(a.eval())
                        m=runtime.env.sim.mj_model;b.require_structure(m)
                        physics=b.save_physics(runtime,folder);model=mujoco.MjModel.from_binary_path(str(folder/'model.mjb'))
                        with np.load(folder/'physical_context.npz') as f:physical=dict(f)
                        body=runtime.features.body;qids=runtime.features.qids.cpu().tolist();dofs=runtime.features.dofs.cpu().tolist();aids=runtime.action._position_ctrl_ids.cpu().tolist()
                        case=dict(dt=.0025,model_layout=dict(object_body=body,object_qposadr=runtime.features.qa,object_dofadr=int(m.jnt_dofadr[m.body_jntadr[body]]),COM_BODY=m.body_ipos[body].tolist()))
                        rec=b.OwnedProbeRecorder(runtime);xs=[];ys=[];valid=[];audit=[]
                        for phase in range(1,369):
                            assert int(runtime.motion.time_steps[0])==phase
                            beta=float(np.clip((phase-phase0+1)/8,0,1))
                            obs=runtime.observation()['policy']
                            u=actors[0].distribution(obs).mean.lerp(actors[1].distribution(obs).mean,beta)
                            rec.begin(u[0].cpu().numpy());_,_,term,trunc,_=runtime.step(u);z=rec.finish()
                            metrics,decoded=b.decode_saved_case(case,z,model,physical)
                            pkt=b.pack_case(model,case,z,decoded,qids,dofs,aids,z['initial_warmstart'],physics)
                            ok=bool(pkt['valid'][0] and all(metrics['checks'].values()) and np.all(pkt['geometry_flags'][0]<=[.002,.0002,.001]))
                            xs.append(pkt['x'][0]);ys.append(pkt['y'][0]);valid.append(ok)
                            audit.append(dict(phase=phase,beta=beta,valid=ok,metrics=metrics,geometry=pkt['geometry_flags'][0].tolist(),terminated=bool(term[0]),truncated=bool(trunc[0])))
                            if term[0] or trunc[0] or not ok:break
                    finally:rec=None;runtime.close()
                path=folder/'trajectory.npz';np.savez_compressed(path,x=xs,y=ys,valid=valid)
                write_json(folder/'audit.json',dict(physics=physics,rows=audit))
                episodes.append(dict(label=driver,phase0=phase0,controls=len(xs),valid=sum(valid),trajectory=file_pin(path),audit=file_pin(folder/'audit.json')))
                write_json(out/'progress.json',dict(episodes=episodes,native=ticks))
        assert ticks==8*sum(e['controls'] for e in episodes)
        write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),episodes=episodes,native=ticks,source=file_pin(__file__)))
    except BaseException as exc:
        write_json(out/'failure.json',dict(error=repr(exc),native=ticks,completed=len(episodes)));raise
    finally:b.Simulation.step=original
if __name__=='__main__':main(*map(lambda x:Path(x).resolve(),sys.argv[1:]))
