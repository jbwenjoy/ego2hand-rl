"""Two-dt execution of the one independently reviewed bounded support proposal."""
from pathlib import Path
import argparse
import json
import sys
import time
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
        raise ValueError('Fresh support-proposal execution')
    cp = args.config.resolve()
    cfg = tomllib.loads(cp.read_text())
    base_path = (cp.parent / cfg['base_config']).resolve()
    base = tomllib.loads(base_path.read_text())
    pp = (cp.parent / cfg['proposal']).resolve()
    proposal = json.loads(pp.read_text())
    rp = (cp.parent / cfg['review']).resolve()
    review = json.loads(rp.read_text())
    if not proposal['proposal_qualified'] or not all(review['checks'].values()):
        raise ValueError('Full proposal and independent review prerequisite')
    if verify_file_pin(review['run']).resolve() != pp or cfg['dt'] != [.005, .0025]:
        raise ValueError('Exactly the reviewed proposal and both fixed dt')
    for pin in proposal['sources']:
        verify_file_pin(pin)
    verify_file_pin(review['source'])
    z = np.load(verify_file_pin(proposal['artifacts']['selected']))
    sources = [file_pin(__file__)] + [file_pin(ROOT / p) for p in [
        'src/supermimic_research/runtime/controlled_evaluation.py',
        'src/supermimic_research/runtime/pen_gpu_initialization.py',
        'src/supermimic_research/mechanics/warp_contact_labels.py',
        'src/supermimic_research/mechanics/indexed_contact_support.py',
        'src/supermimic_research/mechanics/support.py',
        'src/supermimic_research/mechanics/wrench_separation.py']]
    torch.set_num_threads(4)
    wp.set_device('cuda:0')
    registry = ArtifactRegistry(base_path.parent / base['artifact_manifest'])
    args.output.mkdir()
    intent = dict(schema='controlled_support_pen_GPU:1', config=file_pin(cp), base_config=file_pin(base_path),
                  proposal=file_pin(pp), review=file_pin(rp), sources=sources, neural_updates=0,
                  role='DIAGNOSTIC_ONLY_NOT_EXPERT_NOT_FIT')
    write(args.output / 'intent.json', intent)
    cases = []
    start = time.perf_counter()
    for dt in cfg['dt']:
        folder = args.output / f'{cfg["name"]}_{dt}'
        result = evaluate_case('hand2-pen-spin', 'CONSTANT_RESIDUAL', None,
            {**base, 'physics_dt': dt}, base_path.parent, registry, folder,
            initializer=lambda env: initialize_and_query_pen(env, folder, z['source_qpos'], z['qpos'],
                z['delta20'], z['target_BODY'], support_all_robot_contacts=True),
            constant_action=2 * z['delta20'])
        result['name'] = cfg['name']
        cases.append(result)
        write(folder / 'report.json', result)
        write(args.output / 'progress.json', dict(cases=cases, wall_s=time.perf_counter() - start))
        print(json.dumps(dict(dt=dt, duration=result['summary']['metrics']['actual_duration_s'],
                              qualified=result['summary']['qualified'])), flush=True)
    for pin in sources:
        verify_file_pin(pin)
    used = []
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name.startswith('dex_supermimic') and path:
            if not Path(path).resolve().is_relative_to(ROOT / 'src'):
                raise ValueError('Unexpected historical task source')
            used.append(file_pin(path))
    write(args.output / 'report.json', dict(**intent, cases=cases, task_sources=used,
        GPU=torch.cuda.get_device_name(), wall_s=time.perf_counter() - start,
        scope='One selected support-aware initial joint pose and independent constant PD residual, original source velocities/clock and first termination, all actual robot contacts eligible for support diagnosis. No new Actor, learning, force aid or source-time changes.'))


if __name__ == '__main__':
    import torch
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    with torch.jit.optimized_execution(False):
        main(p.parse_args())
