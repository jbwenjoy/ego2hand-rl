"""Additional training behaviors plus a separate, preassigned within-family test policy."""
from pathlib import Path
import sys,subprocess
from types import SimpleNamespace
import torch
import collect_guarded_finger_response as base
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json


class AppliedActor:
    def __init__(self,actor):self.actor=actor
    def distribution(self,obs):
        # Runtime clips its sampled action here before the residual scaling and EMA.
        return SimpleNamespace(mean=self.actor.distribution(obs).mean.clamp(-1,1))


def main(out):
    root=base.ROOT
    out.mkdir(exist_ok=False)
    configs={'TRAIN_A':'W512_U32','TRAIN_B':'W1024_U64','TEST':'W4096_U32'}
    pins={k:file_pin(root/'runs/pen_batch_updates_v1'/v/'final_actor.pt') for k,v in configs.items()}
    write_json(out/'intent.json',dict(actors=pins,phases=base.PHASES,axes='same eight D2/D3 RAW coordinates as V1',epsilon=.04,
        probe_origin='clip(mean,-1,1), branch endpoints still pass through original clamp/EMA/PD',train='TRAIN_A and TRAIN_B basis pairs',
        test='Unseen TEST actor. All mixed-direction labels excluded from fitting. Fixed original SUPPORT bank is secondary prior-inspected development comparison.'))
    receipt=base.freeze_sources(sorted((root/'src').rglob('*.py'))+[Path(__file__),root/'tools/collect_guarded_finger_response.py',root/'tools/benchmark_pen_scaled.py',base.OBJECTIVE,base.COLLISION],
        out/'snapshot',root,subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
    write_json(out/'source_receipt.json',receipt)
    original=base.Simulation.step;rec=None;steps=0
    def activate(r):
        nonlocal rec
        rec=r
    def step(sim):
        nonlocal steps
        if rec is not None and rec.enabled:rec.before_step()
        value=original(sim);steps+=sim.num_envs
        if rec is not None and rec.enabled:rec.after_step()
        return value
    base.Simulation.step=step
    try:
        manifest=base.factory()
        with torch.no_grad(),torch.jit.optimized_execution(False),base.backend_scope(base.COLLISION) as collision:
            identity=base.execution_identity('STABLE_SAME_REGIME_FRICTION_OR',collision.contract['backend_id'],manifest)
            for role,pin in pins.items():
                with base.CostProfile(identity,manifest) as profile:
                    runtime=profile.create_runtime(base.OBJECTIVE,1)
                    try:
                        runtime.set_fixed_frame_features(False)
                        state=torch.load(verify_file_pin(pin),map_location='cpu',weights_only=False)
                        assert state['batch_backend']['factory']==manifest['generated'] and state['batch_backend']['collision_backend']==collision.contract['backend_id']
                        actor,_,_=base.model_pair(runtime.device,SimpleNamespace(**state['config']))
                        actor.load_state_dict(state['actor']);actor.eval()
                        base.collect(runtime,AppliedActor(actor),role,out/role,receipt,activate)
                    finally:activate(None);runtime.close()
        assert steps==10200
        write_json(out/'report.json',dict(native=steps,arms={k:file_pin(out/k/'report.json') for k in pins}))
    except BaseException as exc:
        write_json(out/'failure.json',dict(error=repr(exc),native=steps));raise
    finally:base.Simulation.step=original
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
