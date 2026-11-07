"""Describe all saved dense proxy rows; no model, conversion, or physics calls."""
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
    files = [run / 'all1808_proxy_scores.npz', run / 'rows_all1808.json', run / 'report.json']
    inputs = [pin(p) for p in files]
    rows = json.loads(files[1].read_text(encoding='utf-8-sig'))
    report = json.loads(files[2].read_text(encoding='utf-8-sig'))
    with np.load(files[0], allow_pickle=False) as z:
        P, sdf, native = (z[k].copy() for k in ('P', 'minimum_sdf_m', 'native_HO_depth_m'))
        recorded_positive = z['proxy_P_positive'].copy()
    assert len(rows) == len(P) == 1808
    assert P.shape == sdf.shape == native.shape == (1808,)
    assert all(np.isfinite(a).all() for a in (P, sdf, native))
    assert (P >= 0).all() and (native >= 0).all()
    assert np.array_equal(P > 0, recorded_positive)
    assert [r['row_id'] for r in rows] == list(range(1808))
    assert report['scored_rows'] == 1808
    for i, row in enumerate(rows):
        assert row['P'] == float(P[i]) and row['native_generated_HO_depth_m'] == float(native[i])
    positive, label = P > 0, native > .003
    proxy_depth = np.maximum(0., -sdf) * 1000
    modes = [('donor', 'Donor'), ('baseline', 'Restored baseline'), ('direction', 'Restored perturbation')]
    paths = [('A0', '#2463a8'), ('A2', '#c36629')]
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, (mode, title) in zip(axes, modes):
        for path, color in paths:
            mask = np.array([r['mode'] == mode and r['path'] == path for r in rows])
            ax.scatter(native[mask] * 1000, proxy_depth[mask], s=13, alpha=.45, color=color, label=f'{path} (n={mask.sum()})')
        ax.axvline(3, color='#666666', linestyle='--', linewidth=.8)
        ax.axhline(3, color='#666666', linestyle='--', linewidth=.8)
        ax.set(title=title, xlabel='Native generated HO depth (mm)', ylabel='Maximum sampled proxy depth (mm)',
               xlim=(-.3, max(4., native.max() * 1000 * 1.04)), ylim=(-.3, max(4., proxy_depth.max() * 1.04)))
        ax.legend(fontsize=8)
    fig.suptitle('Same solver pre-state: all 1808 saved rows; original 3 mm reference lines')
    fig.savefig(args.output / 'all1808_same_phase_depth.png', dpi=180)
    plt.close(fig)
    categories = [label & positive, label & ~positive, ~label & positive, ~label & ~positive]
    names = ['Native >3 mm, P >0', 'Native >3 mm, P =0', 'Native <=3 mm, P >0', 'Native <=3 mm, P =0']
    colors = ['#33836b', '#b94444', '#d5a038', '#b3bdca']
    masks, groupnames = [], []
    for path, _ in paths:
        for mode, title in modes:
            masks.append(np.array([r['path'] == path and r['mode'] == mode for r in rows]))
            groupnames.append(path + '\n' + title)
    values = np.array([[int((mask & category).sum()) for mask in masks] for category in categories])
    assert values.sum() == 1808 and sum(masks).min() == sum(masks).max() == 1
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    bottom = np.zeros(len(masks), dtype=int)
    for value, name, color in zip(values, names, colors):
        ax.bar(np.arange(len(masks)), value, bottom=bottom, label=name, color=color)
        for x, v, b in zip(range(len(masks)), value, bottom):
            if v:
                ax.text(x, b + v / 2, str(v), ha='center', va='center', fontsize=9)
        bottom += value
    ax.set(xticks=np.arange(len(masks)), xticklabels=groupnames, ylabel='Saved rows (dependent)',
           title='All rows retained; raw P >0 and strict native generated HO depth >3 mm')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9)
    fig.savefig(args.output / 'all1808_same_phase_counts.png', dpi=180)
    plt.close(fig)
    assert inputs == [pin(p) for p in files]
    result = {'schema': 'full151105_dense_penalty_saved_figures:1', 'input_pins': inputs,
              'renderer': pin(__file__), 'all_rows': 1808, 'group_names': groupnames,
              'category_names': names, 'category_counts_by_group': values.tolist(),
              'model_or_physics_calls': 0, 'scope': 'Descriptive dependent saved rows; no global geometry truth, independent N, task success, or method benefit inference.'}
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'all_rows': 1808, 'category_totals': values.sum(axis=1).tolist()}))


if __name__ == '__main__':
    main()
