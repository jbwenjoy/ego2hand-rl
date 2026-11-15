"""Execute the v0.90 read-only diagnosis; never fit or recollect physics."""
import argparse
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from audit_pen_candidate_coverage import main as graph_coverage
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.learning.pen_response import PenResponseWM
from supermimic_research.learning.pen_stage import write
from supermimic_research.learning.pen_diagnostics import endpoint_errors
from supermimic_research.learning.pen_fit_diagnostics import (
    source_coverage, oracle_rollout, rollout_block_errors, constant_velocity, actual_action_coverage, Y_BLOCKS)


def load(pin):
    return dict(np.load(verify_file_pin(pin)))


def correlation(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def main(a):
    cfg = tomllib.loads(a.config.read_text())
    paths = {k: (a.config.parent / cfg[k]).resolve() for k in ['prepared', 'evaluation', 'rollout', 'health', 'base_config']}
    prep, evaluation, rollout, health = [json.loads(paths[k].read_text()) for k in ['prepared', 'evaluation', 'rollout', 'health']]
    cp = verify_file_pin(health['checkpoint'])
    if (a.output.exists() or not a.output.resolve().is_relative_to(ROOT / 'runs')
            or cfg['horizons'] != [1, 2, 5, 10] or cfg['oracle_modes'] != ['FULL', 'OBSERVED_WARM', 'OBSERVED_STATE', 'OBSERVED_BOTH']
            or not all(health['checks'].values()) or health['training_running'] or health['update'] != 2000):
        raise ValueError('Frozen diagnostics on healthy final checkpoint, fresh unified output')
    for report in [evaluation, rollout]:
        if verify_file_pin(report['checkpoint']) != cp or verify_file_pin(report['prepared']) != paths['prepared']:
            raise ValueError('Exact common checkpoint and preparation lineage required')
    ck = torch.load(cp, map_location='cpu', weights_only=False)
    if verify_file_pin(ck['prepared']) != paths['prepared'] or prep['outputs'] != ck['dataset_pins']:
        raise ValueError('Unchanged role/scale/physics/data lineage required')
    data = {role: load(pin) for role, pin in prep['outputs'].items()}
    scales = load(prep['scales'])
    tail = load(rollout['records'])
    parent_path = verify_file_pin(prep['parents'])
    parents = json.loads(parent_path.read_text())
    a.output.mkdir()
    receipt = freeze_sources(sorted((ROOT / 'src/supermimic_research').rglob('*.py')) + [
        Path(__file__).resolve(), ROOT / 'tools/audit_pen_candidate_coverage.py', a.config.resolve()],
        a.output / 'source_snapshot', ROOT, subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    write(a.output / 'source_receipt.json', receipt)
    graph_coverage(SimpleNamespace(reports=[parent_path], output=a.output / 'graph_coverage'))
    coverage = source_coverage(data['TRAIN'], data['CHECK'], scales)
    gates = tomllib.loads(paths['base_config'].read_text())
    references, inputs = {}, []
    for case in parents['cases']:
        key = (case['name'], case['dt'])
        references[key] = load(case['initialization']['nominal_mean_wrench'])
        inputs.append(case['initialization']['nominal_mean_wrench'])
    action_coverage = []
    for parent in prep['parents_detail']:
        case = next(c for c in parents['cases'] if c['name'] == parent['mode'] and c['dt'] == parent['dt'])
        action_coverage.append(actual_action_coverage(data[parent['role']], parent, case,
                                                     references[parent['mode'], parent['dt']], gates))
    device = torch.device(cfg['device'])
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reports, outputs = {}, {}
    for mode, saved in ck['arms'].items():
        prediction = load(evaluation['outputs'][mode])['all_predictions']
        per_parent = []
        for record in coverage['CHECK_parents']:
            mask = (data['CHECK']['parent_id'] == record['parent_id']) & data['CHECK']['valid']
            mse = {key: float(np.mean(((prediction[mask, sl] - data['CHECK']['y'][mask, sl]) / scales['y_scale'][sl])**2))
                   for key, sl in Y_BLOCKS.items()}
            per_parent.append(dict(**record, normalized_MSE=mse,
                physical=evaluation['models'][mode]['physical_finite_response']['summary'].get(str(record['parent_id']))))
        model = PenResponseWM(saved['scales'], saved['head_scales'], saved['physics'],
                              width=saved['width'], coupled=saved['coupled']).to(device)
        model.load_state_dict(saved['model'], strict=True)
        model.eval()
        previous = load(rollout['outputs'][mode])
        recursive, arrays, parity = {}, {}, {}
        for h in cfg['horizons']:
            ids = previous[f'H{h}_row_indices']
            if not len(ids):
                recursive[str(h)] = dict(windows=0)
                continue
            summary = dict(windows=len(ids), modes={})
            for intervention in cfg['oracle_modes']:
                states, memories, preds = oracle_rollout(model, tail, ids, intervention, device)
                if intervention == 'FULL':
                    parity[str(h)] = dict(states_exact=np.array_equal(states, previous[f'H{h}_states']),
                                          predictions_exact=np.array_equal(preds, previous[f'H{h}_predictions']))
                summary['modes'][intervention] = rollout_block_errors(tail, ids, states, memories, scales)
                arrays[f'H{h}_{intervention}_states'] = states
                arrays[f'H{h}_{intervention}_memories'] = memories
            initial = torch.as_tensor(tail['x'][ids[:, 0], :55], dtype=torch.float64)
            cv = constant_velocity(initial, h * .02).numpy()
            target = previous[f'H{h}_targets']
            summary['constant_velocity'] = endpoint_errors(cv, target)
            summary['persistence'] = endpoint_errors(initial.numpy(), target)
            recursive[str(h)] = summary
        if not all(all(p.values()) for p in parity.values()):
            raise ValueError('Diagnostic FULL changed the previously frozen recursive evaluation')
        target = a.output / (mode + '_interventions.npz')
        np.savez_compressed(target, **arrays)
        outputs[mode] = file_pin(target)
        reports[mode] = dict(per_CHECK_parent=per_parent, recursive=recursive, previous_FULL_parity=parity,
            descriptive_source_distance_MSE_correlation={block: correlation(
                [r['nearest_source_RMS'] for r in per_parent], [r['normalized_MSE'][block] for r in per_parent]) for block in Y_BLOCKS})
    verify_source_snapshot(receipt, live=True)
    result = dict(schema='candidate_fit_failure_diagnosis:1', config=file_pin(a.config),
        inputs={k: file_pin(p) for k, p in paths.items()}, references=inputs,
        source_receipt=file_pin(a.output / 'source_receipt.json'), checkpoint=file_pin(cp),
        coverage=coverage, actual_action_coverage=action_coverage,
        graph_coverage=file_pin(a.output / 'graph_coverage/report.json'),
        models=reports, outputs=outputs, native_integrations=0, optimizer_updates=0,
        scope=cfg['scope'] + ' All CHECK parents, all quality-valid recursive windows and all36 declared actual parent action tables; original task gates. No causal conclusion from source-distance correlations or global infeasibility from sampled actions.')
    write(a.output / 'report.json', result)
    print(json.dumps(dict(report=file_pin(a.output / 'report.json'), coverage=coverage['counts'],
        actual_parent_joint_available=sum(p['joint_pose_candidates'] > 0 for p in action_coverage),
        actual_parent_tables=len(action_coverage),
        oracle200ms={mode: {i: dict(COM_mm=r[-1]['physical']['COM_error_m']['mean'] * 1000,
            angle_deg=r[-1]['physical']['angle_error_deg']['mean'], memory_RMS=r[-1]['next_memory_normalized_RMS'])
            for i, r in m['recursive']['10']['modes'].items()} for mode, m in reports.items()})))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
