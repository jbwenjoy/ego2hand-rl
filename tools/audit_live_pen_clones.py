"""Live read-only GPU model binding versus reconstructed-model step fidelity."""
from pathlib import Path
import argparse
import copy
import dataclasses
import hashlib
import json
import sys
import time
import tomllib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from supermimic_research.artifacts import ArtifactRegistry, file_pin, verify_file_pin
from supermimic_research.runtime.controlled_evaluation import evaluate_case, write
from supermimic_research.runtime.pen_gpu_initialization import initialize_and_query_pen
from supermimic_research.runtime.gpu_pen_rollout import GPUPenRollout
from tools.audit_gpu_pen_steps import discrepancies


def model_arrays(model):
    import warp as wp
    result = {}
    for field in dataclasses.fields(model):
        value = getattr(model, field.name)
        if isinstance(value, wp.array):
            result[field.name] = value.numpy()
        elif dataclasses.is_dataclass(value):
            for subfield in dataclasses.fields(value):
                subvalue = getattr(value, subfield.name)
                if isinstance(subvalue, wp.array):
                    result[field.name + '.' + subfield.name] = subvalue.numpy()
    return result


def fingerprints(arrays):
    return {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}


def main(args):
    import torch
    import warp as wp
    import mujoco
    import mujoco_warp as mw
    if args.output.exists():
        raise ValueError('Fresh live model fidelity audit')
    cp = args.config.resolve()
    cfg = tomllib.loads(cp.read_text())
    bp = (cp.parent / cfg['base_config']).resolve()
    base = tomllib.loads(bp.read_text())
    gp = (cp.parent / cfg['gates']).resolve()
    gates = tomllib.loads(gp.read_text())
    pp = (cp.parent / cfg['proposal']).resolve()
    proposal = json.loads(pp.read_text())
    selected = np.load(verify_file_pin(proposal['artifacts']['selected']))
    qref = np.load(verify_file_pin(proposal['clip']))['reference_qpos']
    torch.set_num_threads(4)
    wp.set_device('cuda:0')
    registry = ArtifactRegistry(bp.parent / base['artifact_manifest'])
    args.output.mkdir()
    sources = [file_pin(__file__)] + [file_pin(ROOT / p) for p in [
        'src/supermimic_research/runtime/gpu_pen_rollout.py',
        'src/supermimic_research/runtime/gpu_contact_oracle.py',
        'src/supermimic_research/runtime/controlled_evaluation.py',
        'src/supermimic_research/runtime/pen_gpu_initialization.py', 'tools/audit_gpu_pen_steps.py']]
    intent = dict(schema='live_GPU_pen_clone_audit:1', config=file_pin(cp), base_config=file_pin(bp),
                  gates=file_pin(gp), proposal=file_pin(pp), sources=sources, neural_updates=0)
    write(args.output / 'intent.json', intent)
    cases = []
    clone_cases = []
    for dt in cfg['dt']:
        folder = args.output / f'actual_{dt}'
        captures = []
        inventory = {}

        def initialize(env):
            receipt = initialize_and_query_pen(env, folder, selected['source_qpos'], selected['qpos'],
                        selected['delta20'], selected['target_BODY'], support_all_robot_contacts=True)
            native_model = copy.copy(env.sim.mj_model)
            for key in ['body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'geom_size', 'geom_friction', 'geom_solref', 'geom_solimp',
                        'actuator_gainprm', 'actuator_biasprm', 'actuator_forcerange', 'actuator_ctrlrange', 'jnt_range', 'dof_damping', 'dof_armature']:
                getattr(native_model, key)[:] = getattr(env.sim.wp_model, key).numpy()[0]
            native_data = mujoco.MjData(native_model)
            mujoco.mj_setConst(native_model, native_data)
            mw.get_data_into(native_data, native_model, env.sim.wp_data)
            actual_arrays = model_arrays(env.sim.wp_model)
            before = fingerprints(actual_arrays)
            np.savez_compressed(folder / 'actual_GPU_model_arrays.npz', **actual_arrays)
            reconstructed_arrays = model_arrays(mw.put_model(native_model))
            different = []
            for key in sorted(set(actual_arrays) | set(reconstructed_arrays)):
                a, b = actual_arrays.get(key), reconstructed_arrays.get(key)
                if a is None or b is None or a.shape != b.shape:
                    different.append(dict(field=key, shape_actual=None if a is None else list(a.shape), shape_reconstructed=None if b is None else list(b.shape)))
                elif not np.array_equal(a, b):
                    different.append(dict(field=key, shape=list(a.shape), max_absolute=float(np.max(abs(a.astype(float) - b.astype(float))))))
            for nw in cfg['nworld']:
                clone = GPUPenRollout(native_model, native_data, nw, selected['delta20'], device_model=env.sim.wp_model, nconmax=128, njmax=512)
                reference = torch.as_tensor(qref, device='cuda', dtype=torch.float32)
                raw = torch.as_tensor(2 * selected['delta20'], device='cuda', dtype=torch.float32)[None].expand(nw, -1)
                rows = []
                for phase in range(1, cfg['horizon_controls'] + 1):
                    clone.set_action(raw, reference[phase])
                    for _ in range(clone.decimation):
                        clone.physics_step()
                        rows.append(clone.readback(clone.output()))
                    clone.refresh_boundary()
                actual = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
                path = folder / f'live_clone_{nw}.npz'
                np.savez_compressed(path, **actual)
                captures.append(dict(nworld=nw, clone=clone, trace=actual, pin=file_pin(path)))
            after = fingerprints(model_arrays(env.sim.wp_model))
            inventory.update(model_unchanged=before == after, model_fields=len(before), differences=different,
                             actual_model_arrays=file_pin(folder / 'actual_GPU_model_arrays.npz'))
            return {**receipt, 'clone_inventory': inventory}

        actual_case = evaluate_case('hand2-pen-spin', 'CONSTANT_RESIDUAL', None, {**base, 'physics_dt': dt},
                                   bp.parent, registry, folder, initializer=initialize, constant_action=2 * selected['delta20'])
        actual_case['name'] = 'LIVE_MODEL_CONSTANT_RESIDUAL'
        cases.append(actual_case)
        saved = np.load(verify_file_pin(actual_case['trace']))
        for capture in captures:
            errors = discrepancies(capture['trace'], saved, capture['clone'], len(capture['trace']['qpos']))
            checks = {key: value <= gates[key] for key, value in errors.items()}
            checks['model_unchanged'] = inventory['model_unchanged']
            checks['original_PD'] = bool(abs(capture['trace']['ctrl'] - saved['post_ctrl'][:len(capture['trace']['ctrl']), None]).max() <= gates['control_rad'])
            clone_cases.append(dict(dt=dt, nworld=capture['nworld'], checks=checks, errors=errors, trace=capture['pin'], inventory=inventory))
        print(json.dumps(dict(dt=dt, clones=[{k: c[k] for k in ['nworld', 'checks', 'errors']} for c in clone_cases if c['dt'] == dt])), flush=True)
    for pin in sources:
        verify_file_pin(pin)
    write(args.output / 'report.json', dict(**intent, cases=cases, clones=clone_cases,
        passed=sum(sum(c['checks'].values()) for c in clone_cases), total=sum(len(c['checks']) for c in clone_cases),
        ready_for_100ms_planning=all(all(c['checks'].values()) for c in clone_cases),
        scope='Fresh actual task execution versus pre-start separate planning worlds sharing the same readonly live device model; model-array hashes checked unchanged. No altered task, object aid, neural training or physical success claim.'))


if __name__ == '__main__':
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    with torch.jit.optimized_execution(False):
        main(parser.parse_args())
