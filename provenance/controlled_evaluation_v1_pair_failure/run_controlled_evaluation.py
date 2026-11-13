"""Run fixed original cube/single-pen ZERO/saved-Actor no-reset diagnostic."""
from pathlib import Path
import argparse
import json
import sys
import time
import tomllib
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from supermimic_research.artifacts import ArtifactRegistry, file_pin, verify_file_pin
from supermimic_research.runtime.controlled_evaluation import evaluate_case, write


def main(args):
    import torch
    import warp as wp
    import numpy as np
    config_path = args.config.resolve()
    config = tomllib.loads(config_path.read_text(encoding='utf8'))
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT/'runs'):
        raise ValueError('Fresh unified output required')
    if config['tasks'] != ['hand2-reorient', 'hand2-pen-spin'] or config['controllers'] != ['ZERO', 'SAVED_ACTOR']:
        raise ValueError('Fixed four-case cohort required')
    torch.set_num_threads(4); wp.set_device('cuda:0')
    registry = ArtifactRegistry(config_path.parent/config['artifact_manifest'])
    pins = {task: file_pin((config_path.parent/config[key+'_checkpoint']).resolve())
            for task, key in [('hand2-reorient', 'cube'), ('hand2-pen-spin', 'pen')]}
    for task, key in [('hand2-reorient', 'cube'), ('hand2-pen-spin', 'pen')]:
        if pins[task]['sha256'] != config[key+'_checkpoint_sha256']:
            raise ValueError('Checkpoint bytes changed')
    args.output.mkdir()
    intent = dict(schema='controlled_no_reset_evaluation:1', config=file_pin(config_path),
        recipe=file_pin(config_path.parent/config['pen_recipe']), checkpoints=pins,
        neural_updates=0, learned_policy_success_assumed=False)
    write(args.output/'intent.json', intent)
    start = time.perf_counter(); cases = []
    for task in config['tasks']:
        saved = torch.load(verify_file_pin(pins[task]), map_location='cpu', weights_only=False)
        if saved['config']['task'] != task: raise ValueError('Checkpoint task mismatch')
        for controller in config['controllers']:
            case = evaluate_case(task, controller, saved, config, config_path.parent, registry,
                                 args.output/f'{task}_{controller}')
            cases.append(case)
            write(args.output/'progress.json', dict(cases=cases, wall_s=time.perf_counter()-start))
            print(json.dumps(dict(task=task, controller=controller,
                duration=case['summary']['metrics']['actual_duration_s'], qualified=case['summary']['qualified'])), flush=True)
    initial_pairs = []
    for task in config['tasks']:
        z = [dict(np.load(c['trace']['path'])) for c in cases if c['task']==task]
        equal = all(np.array_equal(z[0][key], z[1][key]) for key in
                    ['initial_qpos', 'initial_qvel', 'time_s', 'COM_world_m', 'quaternion_wxyz'])
        initial_pairs.append(dict(task=task, exact=equal))
    sources = [file_pin(__file__), file_pin(ROOT/'src/supermimic_research/runtime/controlled_evaluation.py'),
               file_pin(ROOT/'src/supermimic_research/runtime/controlled_assets.py')]
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name.startswith('dex_supermimic') and path:
            if not Path(path).resolve().is_relative_to(ROOT/'src'): raise ValueError('Unexpected historical source import')
            sources.append(file_pin(path))
    result = dict(**intent, sources=sources, cases=cases, initial_pairs=initial_pairs,
        wall_s=time.perf_counter()-start, all_pairs_exact=all(p['exact'] for p in initial_pairs),
        GPU=torch.cuda.get_device_name(), peak_Torch_allocated_bytes=torch.cuda.max_memory_allocated(),
        scope='Fixed no-reset diagnostic, native task dt, no learned-model/control fitting or ego success claim. Not the original mug gate.')
    write(args.output/'report.json', result)
    print(json.dumps(dict(report=str(args.output/'report.json'), all_pairs_exact=result['all_pairs_exact'])), flush=True)
    if not result['all_pairs_exact']: raise ValueError('Preserve unequal initialization result')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
