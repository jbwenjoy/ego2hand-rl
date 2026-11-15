"""Actual original-plant finite-amplitude response, first20ms,54 episodes."""
import argparse
import json
import sys
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'src'))
from supermimic_research.artifacts import ArtifactRegistry, file_pin, verify_file_pin
from supermimic_research.runtime.controlled_evaluation import evaluate_case, write
from supermimic_research.runtime.pen_gpu_initialization import initialize_and_query_pen
from supermimic_research.learning.pen_interventions import intervention_profiles, first_raw_action


def main(a):
    import torch
    import warp as wp
    from scipy.spatial.transform import Rotation
    cp = a.config.resolve(); cfg = tomllib.loads(cp.read_text())
    bp = (cp.parent/cfg['base_config']).resolve(); base = tomllib.loads(bp.read_text())
    pp = (cp.parent/cfg['proposal']).resolve(); proposal = json.loads(pp.read_text())
    hp = (cp.parent/cfg['checkpoint_health']).resolve(); health = json.loads(hp.read_text())
    if (cfg['dt'] != [.005,.0025] or cfg['direction_count'] != 2 or cfg['PD_target_eps_rad'] != [.002,.005]
            or cfg['execution_repeats'] != 3 or not cfg['numerical_device_override']
            or not all(health['checks'].values()) or health['update'] != 2000 or health['training_running']):
        raise ValueError('Fixed healthy-model diagnostic protocol')
    checkpoint = verify_file_pin(health['checkpoint'])
    selected = np.load(verify_file_pin(proposal['artifacts']['selected']))
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT/'runs'): raise ValueError('Fresh unified output')
    directions, profiles = intervention_profiles()
    torch.set_num_threads(4); wp.set_device('cuda:0')
    registry = ArtifactRegistry(bp.parent/base['artifact_manifest']); a.output.mkdir()
    np.savez_compressed(a.output/'profiles.npz', directions=directions)
    sources = [file_pin(__file__)] + [file_pin(ROOT/p) for p in [
        'src/supermimic_research/learning/pen_interventions.py',
        'src/supermimic_research/runtime/controlled_evaluation.py',
        'src/supermimic_research/runtime/pen_gpu_initialization.py',
        'src/supermimic_research/runtime/numerical_options.py',
        'src/supermimic_research/mechanics/warp_contact_labels.py']]
    intent = dict(schema='pen_native_first_control_intervention:1', config=file_pin(cp), base_config=file_pin(bp),
        proposal=file_pin(pp), checkpoint=file_pin(checkpoint), checkpoint_health=file_pin(hp), sources=sources,
        profiles=file_pin(a.output/'profiles.npz'), declarations=profiles, neural_updates=0,
        data_role='CHECK_PHYSICAL_RESPONSE_NONEXPERT_NEVER_FIT', environment_seed=base['seed'],
        protocol='2 unit-L2 sine directions, PD amplitudes .002/.005rad, +/- and3 execution repeats at both native dt, plus3 zero repeats per dt. Only first20ms response analyzed. All episodes continue unchanged to first native terminal; none is an expert. No model-guided direction/amplitude/episode selection.',
        interpretation='Same environment seed/initial state/clip. Execution repeats measure implementation variability, not independent samples. Finite-amplitude central response is not a smooth contact derivative; derivative validity requires repeat/amplitude stability.')
    write(a.output/'intent.json', intent); cases = []
    for dt in cfg['dt']:
        for profile in profiles:
            folder = a.output/f'{profile["name"]}_{dt}'
            direction = np.zeros(20) if profile['direction'] < 0 else directions[profile['direction']]
            first = torch.as_tensor(first_raw_action(selected['delta20'], direction, profile['epsilon'], profile['sign']), device='cuda')[None]
            baseline = torch.as_tensor((2*selected['delta20']).astype(np.float32), device='cuda')[None]
            physics = {}
            def initialize(env):
                receipt = initialize_and_query_pen(env, folder, selected['source_qpos'], selected['qpos'],
                    selected['delta20'], selected['target_BODY'], support_all_robot_contacts=True)
                body = env.sim.mj_model.body('object/cube').id; m = env.sim.wp_model
                I, q = m.body_inertia.numpy()[0,body], m.body_iquat.numpy()[0,body]
                R = Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
                physics.update(mass=float(m.body_mass.numpy()[0,body]), inertia_BODY=(R@np.diag(I)@R.T).tolist(), gravity=m.opt.gravity.numpy()[0].tolist())
                return receipt
            case = evaluate_case('hand2-pen-spin','NATIVE_PD_RESPONSE',None,
                {**base,'physics_dt':dt,'numerical_options':cfg['numerical_options'],'numerical_device_override':True},
                bp.parent,registry,folder,initializer=initialize,
                action_provider=lambda env,obs,k:first if k == 0 else baseline)
            case.update(**profile, role='CHECK_PHYSICAL_RESPONSE_NONEXPERT_NEVER_FIT', physics=physics,
                supervision='NATIVE_GRADIENT_DIAGNOSTIC_NONEXPERT_NOT_FIT')
            cases.append(case); write(folder/'report.json',case); write(a.output/'progress.json',dict(cases=cases))
            print(json.dumps(dict(name=profile['name'],dt=dt,intervals=case['actual_intervals'],duration=case['summary']['metrics']['actual_duration_s'])),flush=True)
    for pin in sources: verify_file_pin(pin)
    task_sources = []
    for name,module in list(sys.modules.items()):
        path = getattr(module,'__file__',None)
        if name.startswith('dex_supermimic') and path:
            if not Path(path).resolve().is_relative_to(ROOT/'src'): raise ValueError('Unexpected historical import')
            task_sources.append(file_pin(path))
    write(a.output/'report.json',dict(**intent,cases=cases,task_sources=task_sources))


if __name__ == '__main__':
    import torch
    p = argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    with torch.jit.optimized_execution(False): main(p.parse_args())
