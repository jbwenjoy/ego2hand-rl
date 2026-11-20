"""Evaluate a scaled-training Actor on the existing strict W1 reference protocol."""
import argparse
import json
from pathlib import Path
import sys
import tomllib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from mjlab.sim.sim import Simulation
from benchmark_pen_scaled import COLLISION, OBJECTIVE, factory
from supermimic_research.learning.native_pen_ppo import model_pair
from supermimic_research.learning.pen_matched_recovery import WorkCounter
from supermimic_research.runtime.explicit_collision_backend import backend_scope
from supermimic_research.runtime.pen_stable_friction_profile_v2 import CostProfile, execution_identity
from supermimic_research.runtime.pen_frozen_cost_rollout_v2 import run_case
from supermimic_research.artifacts import file_pin
from supermimic_research.strict_json import write_json


def main(actor_path, out, repeats):
    out.mkdir(exist_ok=False,parents=True)
    state=torch.load(actor_path,map_location='cpu',weights_only=False)
    receipt=json.loads((actor_path.parent/'source_receipt.json').read_text())
    objective=tomllib.loads(OBJECTIVE.read_text())
    gates=tomllib.loads((ROOT/'configs/controlled_evaluation_v4.toml').read_text())
    reward=tomllib.loads((OBJECTIVE.parent/objective['adapter_config']).read_text())['reward']
    counter=WorkCounter()
    original=Simulation.step
    recorder=None
    def activate(value):
        nonlocal recorder
        recorder=value
    def step(sim):
        if recorder is not None and recorder.enabled: recorder.before_step()
        counter.native_call(original,sim)
        if recorder is not None and recorder.enabled: recorder.after_step()
    Simulation.step=step
    rows=[]
    try:
        manifest=factory()
        with torch.jit.optimized_execution(False),torch.no_grad(),backend_scope(COLLISION) as collision:
            contract=execution_identity('STABLE_SAME_REGIME_FRICTION_OR',collision.contract['backend_id'],manifest)
            for rep in range(repeats):
                with CostProfile(contract,manifest) as profile:
                    runtime=profile.create_runtime(OBJECTIVE,1)
                    try:
                        runtime.set_fixed_frame_features(False)
                        if state['arm']!='SUPPORT' or state['support_weight']!=objective['support_weights']['SUPPORT']:
                            raise ValueError('Expected support-trained Actor')
                        runtime.support_weight=state['support_weight']
                        expected=state['batch_backend']
                        if expected['collision_backend']!=collision.contract['backend_id'] or expected['factory']!=manifest['generated']:
                            raise ValueError('Training/evaluation backend mismatch')
                        actor,_,_=model_pair(runtime.device,__import__('types').SimpleNamespace(**state['config']))
                        actor.load_state_dict(state['actor'],strict=True)
                        actor.eval().requires_grad_(False)
                        name=f'repeat_{rep}'
                        decl=dict(name=name,policy='SUPPORT',repetition=rep,algorithm=contract['identity']['algorithm'])
                        result=run_case(runtime,actor,decl,out/name,receipt,counter,activate,profile,gates,reward)
                        # The legacy three-policy formatter labels every residual as iteration 2000.
                        # Assign the actual iteration before publishing or referencing this new case.
                        result.update(iteration=state['iteration'],weights=file_pin(actor_path),scope='Scaled PPO frozen Actor. Single controlled reference, numerical repetition.')
                        write_json(out/name/'report.json',result)
                        rows.append(dict(report=file_pin(out/name/'report.json'),success=result['full_task_success'],
                                         stop=result['stop_reason'],valid_s=result['valid_prefix_s']))
                    finally:
                        activate(None)
                        runtime.close()
        write_json(out/'report.json',dict(iteration=state['iteration'],cases=rows,successes=sum(r['success'] for r in rows),total=len(rows)))
    finally:
        Simulation.step=original


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('actor',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('repeats',type=int)
    a=p.parse_args()
    main(a.actor,a.out,a.repeats)
