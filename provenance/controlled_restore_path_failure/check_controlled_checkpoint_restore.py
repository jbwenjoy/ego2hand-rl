"""Restore an existing controlled cube/pen runner and round-trip it; no training."""
from pathlib import Path
import argparse
import json
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin, verify_file_pin
from dex_supermimic.rl.dex_supermimic.config import DexSuperMimicCfg


def exact_tree(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and left.dtype == right.dtype and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(exact_tree(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(exact_tree(a, b) for a, b in zip(left, right))
    return left == right


def main(args):
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT / 'runs'):
        raise ValueError('Fresh unified output required')
    checkpoint_pin = file_pin(args.checkpoint)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if saved['config']['task'] not in ['hand2-reorient', 'hand2-pen-spin']:
        raise ValueError('Controlled Hand2 checkpoint required')
    out.mkdir()
    intent = dict(schema='controlled_runner_restore_only:1', checkpoint=checkpoint_pin,
        task=saved['config']['task'], source=file_pin(__file__), neural_updates=0,
        scope='Construct real GPU task/runner, strict load modules/optimizers/RNG/counters, save/load roundtrip. No learn(), no task success or exact environment/replay resume claim.')
    (out / 'intent.json').write_text(json.dumps(intent, indent=2) + '\n', encoding='utf8')
    start = time.perf_counter()
    from mjlab.envs import ManagerBasedRlEnv
    import warp as wp
    from dex_supermimic.tasks.dex_reorient.config.task import make_env_cfg
    from dex_supermimic.rl.dex_supermimic.runner import DexSuperMimicRunner
    if not torch.cuda.is_available():
        raise ValueError('CUDA required by original checkpoint')
    torch.set_num_threads(4)
    wp.set_device('cuda:0')
    settings = {**saved['config'], 'logger': 'none', 'checkpoint': str(args.checkpoint.resolve()),
                'compile': False, 'compile_policy': False}
    if args.clips_root:
        settings['clips_root'] = str(args.clips_root.resolve())
    cfg = DexSuperMimicCfg(**settings)
    env_cfg = make_env_cfg(cfg.num_envs, cfg.control_hz, play=True, task=cfg.task,
                           recipe_path=cfg.recipe_path, clips_root=cfg.clips_root)
    env_cfg.seed = cfg.seed
    env_cfg.auto_reset = False
    env, runner = None, None
    try:
        env = ManagerBasedRlEnv(env_cfg, device=cfg.device)
        runner = DexSuperMimicRunner(env, cfg, out / 'runner')
        checks = dict(
            modules_exact=all(exact_tree(mod.state_dict(), saved['modules'][k]) for k, mod in runner.modules().items()),
            optimizers_exact=all(exact_tree(opt.state_dict(), saved['optimizers'][k]) for k, opt in runner.optimizers.items()),
            torque_norm_exact=exact_tree(runner.torque_norm.state_dict(), saved['torque_norm']),
            iteration_exact=runner.iteration == saved['iteration'],
            sim_steps_exact=runner.sim_steps == saved['sim_steps'],
            common_step_exact=env.common_step_counter == saved['common_step'],
            torch_rng_exact=exact_tree(torch.get_rng_state(), saved['torch_rng']),
            cuda_rng_exact=exact_tree(torch.cuda.get_rng_state_all(), saved['cuda_rng']),
            numpy_rng_exact=exact_tree(np.random.get_state(), saved['numpy_rng']),
            python_rng_exact=exact_tree(random.getstate(), saved['python_rng']),
            reference_signature_exact=getattr(runner.mdp, 'reference_signature', None) == saved.get('reference_signature'),
            finite_parameters=all(bool(torch.isfinite(p).all()) for mod in runner.modules().values() for p in mod.parameters()),
            fixed_wrist=int(env.sim.mj_model.body('robot/right_palm_link').jntnum.item()) == 0,
        )
        runner.save(out / 'restored_roundtrip.pt')
        roundtrip = torch.load(out / 'restored_roundtrip.pt', map_location='cpu', weights_only=False)
        for key in ['modules', 'optimizers', 'torque_norm', 'iteration', 'sim_steps', 'common_step',
                    'cuda_rng', 'torch_rng', 'numpy_rng', 'python_rng', 'reference_signature']:
            checks[f'roundtrip_{key}'] = exact_tree(roundtrip[key], saved[key])
        for name, module in list(sys.modules.items()):
            path = getattr(module, '__file__', None)
            if name.startswith('dex_supermimic') and path and not Path(path).resolve().is_relative_to(ROOT / 'src'):
                raise ValueError('Unexpected historical package import: ' + path)
        sources = [file_pin(getattr(module, '__file__')) for name, module in list(sys.modules.items())
                   if name.startswith('dex_supermimic') and getattr(module, '__file__', None)]
        verify_file_pin(checkpoint_pin)
        result = dict(**intent, checks={k: bool(v) for k, v in checks.items()},
            passed=sum(bool(v) for v in checks.values()), total=len(checks),
            iteration=runner.iteration, sim_steps=runner.sim_steps,
            GPU=torch.cuda.get_device_name(), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            sources=sources, roundtrip=file_pin(out / 'restored_roundtrip.pt'), wall_s=time.perf_counter() - start)
        (out / 'report.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
        print(json.dumps(dict(output=str(out / 'report.json'), passed=result['passed'], total=result['total'], wall_s=result['wall_s'])), flush=True)
        if not all(checks.values()):
            raise ValueError('Preserve failed restore checks')
    finally:
        if runner is not None:
            runner.close()
        if env is not None:
            env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--clips-root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
