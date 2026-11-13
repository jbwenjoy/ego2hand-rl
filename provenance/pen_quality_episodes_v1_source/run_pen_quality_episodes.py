"""Actual full-case DEFAULT versus preregistered numerical calibration/fallback."""
from pathlib import Path
import argparse
import json
import sys
import tomllib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from supermimic_research.artifacts import ArtifactRegistry, file_pin, verify_file_pin
from supermimic_research.runtime.controlled_evaluation import evaluate_case, write
from supermimic_research.runtime.pen_gpu_initialization import initialize_and_query_pen


def main(args):
    import torch
    import warp as wp
    if args.output.exists():
        raise ValueError('Fresh full numerical-quality episodes')
    cp = args.config.resolve()
    cfg = tomllib.loads(cp.read_text())
    bp = (cp.parent / cfg['base_config']).resolve()
    base = tomllib.loads(bp.read_text())
    fp = (cp.parent / cfg['factorial']).resolve()
    factorial = json.loads(fp.read_text())
    fc = tomllib.loads(verify_file_pin(factorial['config']).read_text())
    selected_variant = None
    for name in cfg['selection_order']:
        rows = [c for c in factorial['cases'] if c['variant'] == name]
        if len(rows) != 6:
            raise ValueError('Two fixed source states times three repetitions')
        if all(c['passed_worlds'] == 128 and all(c['checks'].values()) for c in rows):
            selected_variant = name
            break
    alternative = selected_variant or cfg['fallback']
    pp = (cp.parent / cfg['proposal']).resolve()
    proposal = json.loads(pp.read_text())
    z = np.load(verify_file_pin(proposal['artifacts']['selected']))
    torch.set_num_threads(4)
    wp.set_device('cuda:0')
    registry = ArtifactRegistry(bp.parent / base['artifact_manifest'])
    args.output.mkdir()
    sources = [file_pin(__file__)] + [file_pin(ROOT / p) for p in [
        'src/supermimic_research/runtime/controlled_evaluation.py',
        'src/supermimic_research/runtime/numerical_options.py',
        'src/supermimic_research/runtime/pen_gpu_initialization.py',
        'src/supermimic_research/mechanics/warp_contact_labels.py']]
    intent = dict(schema='pen_numerical_full_episodes:1', config=file_pin(cp), base_config=file_pin(bp),
                  factorial=file_pin(fp), proposal=file_pin(pp), sources=sources,
                  calibrated_variant=selected_variant, actual_alternative=alternative,
                  alternative_status='CALIBRATED_ON_TWO_STATES' if selected_variant else 'NOT_CALIBRATED_DIAGNOSTIC_FALLBACK',
                  neural_updates=0, role='DIAGNOSTIC_ONLY_NOT_EXPERT_NOT_FIT')
    write(args.output / 'intent.json', intent)
    cases = []
    for variant in ['DEFAULT', alternative]:
        for dt in cfg['dt']:
            for profile in cfg['profiles']:
                offset = np.zeros(20)
                if profile != 'BASELINE':
                    direction = int(profile[1])
                    offset = np.sin(np.pi * (np.arange(20) + .5) * direction / 20)
                    offset /= abs(offset).max()
                    offset *= cfg['raw_amplitude'] * (1 if profile.endswith('PLUS') else -1)
                raw = torch.as_tensor(2 * z['delta20'], device='cuda', dtype=torch.float32)[None]
                perturb = torch.as_tensor(offset, device='cuda', dtype=torch.float32)[None]
                folder = args.output / f'{variant}_{profile}_{dt}'

                def provider(env, observation, k):
                    return raw + perturb if k < cfg['intervention_controls'] else raw

                numerical = fc['numerical_options'][variant]
                case = evaluate_case('hand2-pen-spin', 'INTERVENTION', None,
                    {**base, 'physics_dt': dt, 'numerical_options': numerical}, bp.parent, registry, folder,
                    initializer=lambda env: initialize_and_query_pen(env, folder, z['source_qpos'], z['qpos'],
                        z['delta20'], z['target_BODY'], support_all_robot_contacts=True), action_provider=provider)
                case.update(name=f'{variant}_{profile}', profile=profile, numerical_variant=variant)
                cases.append(case)
                write(folder / 'report.json', case)
                write(args.output / 'progress.json', dict(cases=cases))
                print(json.dumps(dict(variant=variant, dt=dt, profile=profile,
                    duration=case['summary']['metrics']['actual_duration_s'], qualified=case['summary']['qualified'])), flush=True)
    for pin in sources:
        verify_file_pin(pin)
    used = []
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name.startswith('dex_supermimic') and path:
            if not Path(path).resolve().is_relative_to(ROOT / 'src'):
                raise ValueError('Unexpected historical source import')
            used.append(file_pin(path))
    write(args.output / 'report.json', dict(**intent, cases=cases, task_sources=used,
        scope='Fresh actual original-clock no-aid/no-reset episodes at fixed action profiles. Only recorded numerical solver options differ. No new Actor, training, full-task success or stochastic/generalization guarantee.'))


if __name__ == '__main__':
    import torch
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    with torch.jit.optimized_execution(False):
        main(p.parse_args())
