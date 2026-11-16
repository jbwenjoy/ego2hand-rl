"""Independent array/recurrence review of the sealed first18 CPU diagnostic."""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.learning.pen_path_response import PathResponseWM, ARMS
from supermimic_research.learning.world_model import raw_tree_equal
from supermimic_research.strict_json import write_json


def read(path):
    return json.loads(Path(path).read_text())


def rotation(state):
    a, b = state[..., 43:46], state[..., 46:49]
    return np.stack([a, b, np.cross(a, b)], axis=-1)


def decode(state, output):
    r = Rotation.from_rotvec(output[..., 43:46].reshape(-1, 3)).as_matrix()
    r = r.reshape(state.shape[:-1] + (3, 3)) @ rotation(state)
    return np.concatenate([state[..., :43] + output[..., :43], r[..., :, 0],
                           r[..., :, 1], state[..., 49:] + output[..., 46:52]], axis=-1)


def stats(values):
    return dict(mean=float(np.mean(values)), median=float(np.median(values)),
                p95=float(np.quantile(values, .95)), max=float(np.max(values)))


def errors(predicted, target):
    r = rotation(predicted) @ rotation(target).swapaxes(-1, -2)
    return dict(COM_error_m=np.linalg.norm(predicted[:, 40:43] - target[:, 40:43], axis=1),
                angle_error_deg=np.rad2deg(Rotation.from_matrix(r).magnitude()),
                joint_position_RMS_rad=np.sqrt(np.mean((predicted[:, :20] - target[:, :20])**2, axis=1)),
                COM_velocity_error_m_s=np.linalg.norm(predicted[:, 49:52] - target[:, 49:52], axis=1),
                omega_error_rad_s=np.linalg.norm(predicted[:, 52:55] - target[:, 52:55], axis=1))


def main():
    started = time.perf_counter()
    out = ROOT / 'runs/pen_selected_rollout_diagnostic_review_v1'
    if out.exists():
        raise ValueError('Fresh immutable review output required')
    rp = ROOT / 'runs/pen_selected_rollout_diagnostic_v1/report.json'
    report = read(rp)
    prep = read(verify_file_pin(report['preparation']))
    if report['passed'] != report['total'] or not all(report['checks'].values()):
        raise ValueError('Completed forecast execution required')
    inputs = [file_pin(rp), report['preparation'], report['arrays'], prep['data'],
              prep['metadata'], report['inputs']['model_checkpoint']]
    for receipt_pin in [report['source_receipt'], prep['source_receipt']]:
        verify_source_snapshot(read(verify_file_pin(receipt_pin)), live=True)
    data = dict(np.load(verify_file_pin(prep['data'])))
    metadata = read(verify_file_pin(prep['metadata']))['rows']
    arrays = dict(np.load(verify_file_pin(report['arrays'])))
    checkpoint = torch.load(verify_file_pin(report['inputs']['model_checkpoint']),
                            map_location='cpu', weights_only=False)
    torch.set_num_threads(4)
    checks = {}
    groups = {}
    case_rows = []
    expected_keys = set()
    for name in ARMS:
        saved = checkpoint['arms'][name]
        ctx = saved['context']
        model = PathResponseWM(ctx['scales'], ctx['physics'], ctx['path_units'], width=512)
        model.load_state_dict(saved['model'], strict=True)
        model.eval().requires_grad_(False)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        groups[name] = {}
        for h in [1, 2, 4, 8, 16]:
            prefix = name + '_H' + str(h)
            expected_keys.update(prefix + '_' + k for k in ['row_indices', 'states', 'predictions', 'targets'])
            ids, states, predictions, targets = [arrays[prefix + '_' + k]
                                                for k in ['row_indices', 'states', 'predictions', 'targets']]
            # Separate enumeration, with no use of the producer window helper.
            windows = []
            for ci in range(18):
                case_ids = sorted(np.flatnonzero(data['case_id'] == ci), key=lambda j: data['control_index'][j])
                for start in range(len(case_ids) - h + 1):
                    w = case_ids[start:start+h]
                    if all(data['valid'][j] for j in w) and all(
                            data['control_index'][w[k]] == data['control_index'][w[0]] + k
                            and data['x'][w[k], -1] == data['x'][w[0], -1] for k in range(h)):
                        windows.append(w)
            checks[prefix + '_all_windows'] = np.array_equal(ids, np.asarray(windows))
            checks[prefix + '_shapes'] = (states.shape == (len(ids), h+1, 55)
                                         and predictions.shape == (len(ids), h, 90)
                                         and targets.shape == (len(ids), 55))
            checks[prefix + '_finite'] = bool(all(np.isfinite(v).all() for v in [states, predictions, targets]))
            checks[prefix + '_true_start_only'] = np.array_equal(states[:, 0], data['x'][ids[:, 0], :55])
            replay = decode(states[:, :-1], predictions)
            checks[prefix + '_independent_state_recursion'] = bool(np.all(
                abs(replay - states[:, 1:]) <= 1e-12 * (1 + abs(states[:, 1:]))))
            checks[prefix + '_native_endpoint'] = bool(np.all(
                abs(targets - data['next55'][ids[:, -1]]) <= 1e-12 * (1 + abs(targets))))
            forward_equal = True
            for k in range(h):
                warm = data['x'][ids[:, 0], 55:81] if k == 0 else predictions[:, k-1, 52:78]
                x = np.concatenate([states[:, k], warm, data['x'][ids[:, k], 81:101],
                                    data['x'][ids[:, 0], -1:]], axis=-1).astype(np.float32)
                with torch.no_grad():
                    recomputed = model.prediction(torch.from_numpy(x)).numpy()
                forward_equal &= np.array_equal(recomputed, predictions[:, k])
            checks[prefix + '_state_warm_control_forward_exact'] = bool(forward_equal)
            r = rotation(states)
            defect = np.linalg.norm(r.swapaxes(-1, -2) @ r - np.eye(3), axis=(-1, -2))
            det = np.linalg.det(r)
            checks[prefix + '_rotation_chart'] = bool(defect.max() < 5e-12 and abs(det - 1).max() < 5e-12)
            e, p = errors(states[:, -1], targets), errors(states[:, 0], targets)
            reported = report['models'][name][str(h)]
            checks[prefix + '_endpoint_statistics'] = all(
                np.allclose(list(stats(v).values()), list(reported['model'][key].values()),
                            rtol=1e-11, atol=1e-11) for key, v in e.items())
            checks[prefix + '_persistence_statistics'] = all(
                np.allclose(list(stats(v).values()), list(reported['persistence'][key].values()),
                            rtol=1e-11, atol=1e-11) for key, v in p.items())
            wrench_checks = []
            for key, sl in [('world_force_RMS_N', slice(78,81)), ('world_torque_RMS_Nm', slice(81,84))]:
                pw = np.einsum('...ij,...j->...i', rotation(states[:, :-1]), predictions[:, :, sl])
                tw = np.einsum('...ij,...j->...i', rotation(data['x'][ids, :55]), data['y'][ids, sl])
                wrench_checks.append(np.isclose(float(np.sqrt(np.mean((pw-tw)**2))), reported[key], rtol=1e-12))
            checks[prefix + '_own_source_frame_wrench'] = bool(all(wrench_checks))
            groups[name][str(h)] = dict(windows=len(ids), duration_s=h*.02,
                model={k: stats(v) for k, v in e.items()}, persistence={k: stats(v) for k, v in p.items()},
                max_rotation_defect=float(defect.max()), min_det=float(det.min()), max_det=float(det.max()))
            for ci in range(18):
                mask = data['case_id'][ids[:, 0]] == ci
                case = next(m['case'] for m in metadata if m['case_id'] == ci)
                row = dict(model=name, horizon=h, duration_s=h*.02, case=case, windows=int(mask.sum()))
                for label, values in [('model', e), ('persistence', p)]:
                    for key, v in values.items():
                        for stat, value in stats(v[mask]).items():
                            row[label+'_'+key+'_'+stat] = value
                case_rows.append(row)
        checks[name + '_unchanged_model'] = raw_tree_equal(before, model.state_dict())
    checks['exact_array_inventory'] = set(arrays) == expected_keys
    checks['all_declared_case_groups'] = len(case_rows) == 270 and all(r['windows'] > 0 for r in case_rows)
    # Rehash all sealed preparation dependencies, including native packets.
    for pin in inputs + prep['packet_inputs']:
        verify_file_pin(pin)
    out.mkdir()
    with (out / 'cases.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    times = np.array([1,2,4,8,16]) * 20
    for ax, key, stat, scale, label in [
            (axes[0,0], 'COM_error_m', 'median', 1000, 'COM median error (mm)'),
            (axes[0,1], 'COM_error_m', 'p95', 1000, 'COM 95th percentile error (mm)'),
            (axes[1,0], 'angle_error_deg', 'median', 1, 'Orientation median error (degrees)'),
            (axes[1,1], 'angle_error_deg', 'p95', 1, 'Orientation 95th percentile error (degrees)')]:
        for name in ARMS:
            ax.plot(times, [groups[name][str(h)]['model'][key][stat]*scale for h in [1,2,4,8,16]],
                    marker='o', label=name)
        ax.plot(times, [groups[ARMS[0]][str(h)]['persistence'][key][stat]*scale for h in [1,2,4,8,16]],
                'k--', label='Persistence (unchanged initial pose)')
        ax.set(xlabel='Prediction horizon (ms)', ylabel=label, ylim=(0, None))
        ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Recorded-control recursive forecasts: all 3 models, sealed first 18 development trajectories')
    fig.text(.5, .015, 'Qualified windows at 20/40/80/160/320 ms: 1757/1739/1703/1631/1487.\n'
             'State and warmstart observed once; future recorded PD controls supplied. No policy or ego evaluation.\n'
             'Horizon cohorts differ after stopping/masking; overlapping windows are not independent trials.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.95))
    fig.savefig(out/'recursive_errors.png', dpi=150)
    plt.close(fig)
    receipt = freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__)],
                             out/'source_snapshot', ROOT,
                             subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip())
    write_json(out/'source_receipt.json', receipt)
    verify_source_snapshot(receipt, live=True)
    result = dict(schema='pen_selected_recursive_review:1', inputs=inputs,
        source_receipt=file_pin(out/'source_receipt.json'), checks=checks,
        passed=sum(checks.values()), total=len(checks), groups=groups,
        per_case=file_pin(out/'cases.csv'), figure=file_pin(out/'recursive_errors.png'),
        native_integrations=0, optimizer_steps=0, normalizer_fits=0,
        wall_s=time.perf_counter()-started,
        scope='Read-only development diagnostics. Independent SciPy state recurrence and native endpoints; '
              'exact CPU model forwards with recursive predicted warmstart; no new physical/policy/ego admission.')
    write_json(out/'report.json', result)
    print(json.dumps(dict(report=file_pin(out/'report.json'), checks=[result['passed'], result['total']])), flush=True)
    if not all(checks.values()):
        raise ValueError('Saved independent review discrepancy: '+repr([k for k,v in checks.items() if not v]))


if __name__ == '__main__':
    main()
