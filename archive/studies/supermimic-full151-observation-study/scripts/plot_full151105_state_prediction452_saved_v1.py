"""Plot every saved fixed-WM prediction row; no model or physics execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def pin(path):
    p = Path(path).resolve(strict=True)
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    files = [run / 'prediction_arrays.npz', run / 'group_identities452.json', run / 'report.json']
    inputs = [pin(p) for p in files]
    rows = json.loads(files[1].read_text(encoding='utf-8-sig'))
    report = json.loads(files[2].read_text(encoding='utf-8-sig'))
    with np.load(files[0], allow_pickle=False) as z:
        predicted, target, recorded = [z[k].copy() for k in ('predicted_x452_FP32', 'target_x452_FP32', 'signed_error452x58_F64')]
    assert predicted.dtype == target.dtype == np.float32 and predicted.shape == target.shape == (452, 58)
    assert all(np.isfinite(a).all() for a in (predicted, target, recorded))
    delta = predicted.astype('f8') - target.astype('f8')
    assert recorded.dtype == np.float64 and recorded.shape == delta.shape and recorded.tobytes() == delta.tobytes()
    assert len(rows) == report['prediction_rows'] == 452
    assert [r['group_id'] for r in rows] == list(range(452))
    args.output.mkdir(parents=True, exist_ok=False)
    blocks = [('q', 0, 20, 1000, 'Joint error (component RMS, mrad)', 'component_RMS'),
              ('qd', 20, 40, 1, 'Joint velocity error (component RMS, rad/s)', 'component_RMS'),
              ('p', 40, 43, 1000, 'Position error (Euclidean norm, mm)', 'norm'),
              ('R', 43, 52, 1, 'Rotation matrix error (Frobenius norm)', 'norm'),
              ('v', 52, 55, 1, 'Linear velocity error (norm, m/s)', 'norm'),
              ('omega', 55, 58, 1, 'Angular velocity error (norm, rad/s)', 'norm')]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    descriptions = []
    masks = []
    for path, color in [('A0', '#2463a8'), ('A2', '#c36629')]:
        for mode, label in [('donor', 'donor'), ('baseline', 'restored baseline'), ('direction', 'restored perturbation')]:
            mask = np.array([r['path'] == path and r['mode'] == mode for r in rows])
            masks.append((path, mode, label, color, mask))
    assert np.stack([m[-1] for m in masks]).sum(0).tolist() == [1] * 452
    times = np.array([r['k'] * .02 for r in rows], dtype='f8')
    for ax, (name, lo, hi, scale, ylabel, kind) in zip(axes.flat, blocks):
        square = delta[:, lo:hi] ** 2
        values = np.sqrt(square.mean(-1) if kind == 'component_RMS' else square.sum(-1)) * scale
        for path, mode, label, color, mask in masks:
            text = f'{path} {label} (n={int(mask.sum())})'
            if mode == 'donor':
                assert [r['k'] for r in rows if r['path'] == path and r['mode'] == mode] == list(range(150))
                ax.plot(times[mask], values[mask], color=color, linewidth=1.3, label=text)
            elif mode == 'baseline':
                ax.scatter(times[mask], values[mask], facecolors='none', edgecolors=color, s=30, linewidth=.8, label=text)
            else:
                ax.scatter(times[mask], values[mask], color=color, marker='x', s=13, linewidth=.7, alpha=.45, label=text)
        ax.set(xlabel='Pre-action time k x 20 ms (s)', ylabel=ylabel, xlim=(-.04, 3.02))
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.15)
        descriptions.append({'block': name, 'all_rows_plotted': 452, 'row_metric': kind, 'display_scale': scale,
                             'display_maximum': float(values.max()), 'maximum_group_ids': np.flatnonzero(values == values.max()).tolist()})
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(.5, .025))
    fig.suptitle('Fixed A2 WM: all 452 matched 20 ms state prediction errors (FP32 CPU shadow)', fontsize=14, y=.99)
    fig.text(.02, .006, 'Donor lines only; restored branches are points. Dependent saved rows; no task score or gradient qualification.', fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .95))
    fig.savefig(args.output / 'all452_state_prediction_error_curves.png', dpi=180)
    plt.close(fig)
    assert inputs == [pin(p) for p in files]
    result = {'schema': 'full151105_state_prediction452_saved_figures:1', 'input_pins': inputs, 'renderer': pin(__file__),
              'all_rows': 452, 'panels': descriptions, 'model_or_physics_calls': 0,
              'scope': 'Raw state errors by original units; full cohort retained; one-step CPU shadow, dependent branches, not method benefit.'}
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'all_rows': 452, 'panels': 6}))


if __name__ == '__main__':
    main()
