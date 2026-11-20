"""Fresh audited consecutive transitions for causal contact-load forecasting."""
from pathlib import Path
import sys,json
from types import SimpleNamespace
import numpy as np
import mujoco
import torch
import collect_guarded_finger_response as b
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json


def main(out):
    out.mkdir(exist_ok=False);root=b.ROOT/'runs'
    actors={'A':root/'pen_batch_updates_v1/W512_U32/final_actor.pt','B':root/'pen_batch_updates_v1/W1024_U64/final_actor.pt',
            'C':root/'pen_scaled_support_matched_v1/actor_01825.pt','D':root/'pen_scaled_train_v1/actor_02781.pt',
            'VAL':root/'pen_batch_updates_v1/W1024_U32/final_actor.pt','TEST1':root/'pen_batch_updates_v1/W2048_U32/final_actor.pt',
            'TEST2':root/'pen_batch_updates_v1/W4096_U32/final_actor.pt','TEST3':root/'pen_scaled_support_matched_v1/actor_02781.pt'}
    schedule=[(label,seed) for label in actors for seed in ((0,23,47) if label in 'ABCD' else (0,))]
    pins={k:file_pin(v) for k,v in actors.items()}
    write_json(out/'intent.json',dict(actors=pins,schedule=schedule,roles='A-D TRAIN; VAL calibration; TEST1/2/3 excluded from fit and selection',
        action='mean policy for seed 0; otherwise independent Gaussian raw perturbation sigma .02 per joint per control',
        labels='CPU EFC-decoded body wrench at source frame, true moving COM; previous window visible only after it ends. One controlled reference.'))
    original=b.Simulation.step;rec=None;ticks=0;completed=[]
    def step(sim):
        nonlocal ticks
        if rec is not None and rec.enabled:rec.before_step()
        v=original(sim);ticks+=1
        if rec is not None and rec.enabled:rec.after_step()
        return v
    b.Simulation.step=step
    try:
        manifest=b.factory()
        with torch.no_grad(),torch.jit.optimized_execution(False),b.backend_scope(b.COLLISION) as collision:
            identity=b.execution_identity('STABLE_SAME_REGIME_FRICTION_OR',collision.contract['backend_id'],manifest)
            for label,seed in schedule:
                folder=out/f'{label}_{seed}';folder.mkdir()
                rng=np.random.default_rng(seed);xs=[];ys=[];valid=[];reasons=[]
                with b.CostProfile(identity,manifest) as profile:
                    runtime=profile.create_runtime(b.OBJECTIVE,1)
                    try:
                        runtime.set_fixed_frame_features(False);runtime.support_weight=.2 if label in ('C','TEST3') else 0.
                        state=torch.load(verify_file_pin(pins[label]),map_location='cpu',weights_only=False)
                        actor,_,_=b.model_pair(runtime.device,SimpleNamespace(**state['config']))
                        actor.load_state_dict(state['actor']);actor.eval()
                        sim=runtime.env.sim;m=sim.mj_model;b.require_structure(m)
                        physics=b.save_physics(runtime,folder)
                        model=mujoco.MjModel.from_binary_path(str(folder/'model.mjb'))
                        with np.load(folder/'physical_context.npz') as f:physical=dict(f)
                        body=runtime.features.body;qids=runtime.features.qids.cpu().tolist();dofs=runtime.features.dofs.cpu().tolist();aids=runtime.action._position_ctrl_ids.cpu().tolist()
                        case=dict(dt=.0025,model_layout=dict(object_body=body,object_qposadr=runtime.features.qa,
                              object_dofadr=int(m.jnt_dofadr[m.body_jntadr[body]]),COM_BODY=m.body_ipos[body].tolist()))
                        rec=b.OwnedProbeRecorder(runtime)
                        for phase in range(1,369):
                            assert int(runtime.motion.time_steps[0])==phase
                            action=actor.distribution(runtime.observation()['policy']).mean
                            if seed:action=action+torch.from_numpy(rng.normal(0,.02,(1,20)).astype(np.float32)).to(runtime.device)
                            rec.begin(action[0].cpu().numpy());_,_,term,trunc,_=runtime.step(action);z=rec.finish()
                            metrics,decoded=b.decode_saved_case(case,z,model,physical)
                            pkt=b.pack_case(model,case,z,decoded,qids,dofs,aids,z['initial_warmstart'],physics)
                            ok=bool(pkt['valid'][0] and all(metrics['checks'].values()) and np.all(pkt['geometry_flags'][0]<=[.002,.0002,.001]))
                            xs.append(pkt['x'][0]);ys.append(pkt['y'][0]);valid.append(ok)
                            reasons.append(dict(phase=phase,invalid_metrics=[k for k,v in metrics['checks'].items() if not v],
                                impulse=metrics['max_linear_impulse_residual_N_s'],geometry=pkt['geometry_flags'][0].tolist(),
                                native_valid=bool(pkt['valid'][0]),terminated=bool(term[0]),truncated=bool(trunc[0])))
                            if term[0] or trunc[0] or not ok:break
                    finally:rec=None;runtime.close()
                path=folder/'trajectory.npz';np.savez_compressed(path,x=xs,y=ys,valid=valid)
                write_json(folder/'audit.json',dict(physics=physics,rows=reasons))
                completed.append(dict(label=label,seed=seed,controls=len(xs),valid=sum(valid),trajectory=file_pin(path),audit=file_pin(folder/'audit.json')))
                write_json(out/'progress.json',dict(episodes=completed,native=ticks))
        if ticks!=8*sum(e['controls'] for e in completed):raise ValueError('Native count mismatch')
        write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),episodes=completed,native=ticks,source=file_pin(__file__)))
    except BaseException as error:
        write_json(out/'failure.json',dict(error=repr(error),native=ticks,completed=len(completed)));raise
    finally:b.Simulation.step=original
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
