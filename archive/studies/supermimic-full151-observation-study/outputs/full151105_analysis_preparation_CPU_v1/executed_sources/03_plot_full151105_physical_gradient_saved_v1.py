"""Saved-only scientific render; no model, Env, physics or backend import."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pin(path): return {'path': str(Path(path).resolve()), 'sha256': sha(path)}
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--run', action='store_true')
    args = p.parse_args(); root = args.analysis.resolve(strict=True)
    report, rows, groups = (read(root / n) for n in ('report.json', 'comparisons.json', 'classifications.json'))
    if report['schema'] != 'full151105_physical_gradient_CPU_analysis:1' or report['status'] != 'COMPLETED_CPU_DIAGNOSTIC' or len(rows) != 64 or len(groups) != 32 or not all(report['invariants'].values()):
        raise ValueError('Actual complete invariant-preserving analysis required')
    output_pins = read(root / 'output_pins.json')
    for n in ('report.json', 'comparisons.json', 'classifications.json'):
        if output_pins[n] != pin(root / n): raise ValueError('Saved analysis artifact drift: ' + n)
    source = pin(__file__); inputs = {n: pin(root / n) for n in ('report.json', 'comparisons.json', 'classifications.json', 'output_pins.json')}
    if not args.run:
        print(json.dumps({'status': 'PASS_SAVED_METADATA_ONLY', 'source': source, 'inputs': inputs, 'rows': 64, 'groups': 32, 'rendered': False}, indent=2)); return
    if args.output.exists(): raise ValueError('Fresh plot output required')
    out = args.output.resolve(); out.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    x = np.array([r['physical_slope'] for r in rows], float); y = np.array([r['WM_AD_effective'] for r in rows], float)
    if not (np.isfinite(x).all() and np.isfinite(y).all()): raise ValueError('All64 finite rows required; no outlier filtering')
    low, high = min(x.min(), y.min(), 0), max(x.max(), y.max(), 0)
    span = max(high - low, 1e-6); limits = (low - .07 * span, high + .07 * span)
    styles = {'STABLE_NONFLAT': ('#2677b6', 'o'), 'NONFLAT_UNSTABLE': ('#da6b22', 'X'), 'FLAT_OR_BELOW_NOISE': ('#737373', 'v')}
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 6.4), constrained_layout=False)
    for ei, eps in enumerate((.002, .001)):
        ax = axes[ei]
        for label, (color, marker) in styles.items():
            ids = [i for i, r in enumerate(rows) if r['epsilon_index'] == ei and r['physical_label'] == label]
            ax.scatter(x[ids], y[ids], c=color, marker=marker, s=48, alpha=.85, linewidths=.5, edgecolors='white', zorder=3)
        clipped = [i for i, r in enumerate(rows) if r['epsilon_index'] == ei and r['clipped']]
        terminal = [i for i, r in enumerate(rows) if r['epsilon_index'] == ei and r['k'] == 149]
        ax.scatter(x[clipped], y[clipped], facecolors='none', edgecolors='#c21a45', marker='s', s=110, linewidths=1.3, zorder=4)
        ax.scatter(x[terminal], y[terminal], facecolors='none', edgecolors='black', marker='o', s=150, linewidths=.7, zorder=5)
        ax.plot(limits, limits, '--', color='#555555', lw=1)
        ax.axhline(0, c='#aaaaaa', lw=.6); ax.axvline(0, c='#aaaaaa', lw=.6)
        ax.set(xlim=limits, ylim=limits, xlabel='Physical controlled20ms secant [score/rad]', ylabel='WM AD projected onto actual d_eff [score/rad]')
        ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=.18)
        stats = report['agreement'][str(eps)]; eligible = stats['eligible']
        fraction = 'NA' if stats['sign_fraction'] is None else '%.1f%%' % (100 * stats['sign_fraction'])
        corr = 'NA' if stats['pearson'] is None else '%.3f' % stats['pearson']
        ax.set_title('epsilon = %.1f mrad | all32 groups\nstable n=%d, sign=%s, Pearson=%s' % (eps * 1000, eligible, fraction, corr), fontsize=10)
    legend = [Line2D([], [], color=color, marker=marker, linestyle='none', markersize=7, label=label.replace('_', ' ').lower()) for label, (color, marker) in styles.items()]
    legend += [Line2D([], [], marker='s', color='#c21a45', markerfacecolor='none', linestyle='none', markersize=9, label='clipped endpoint'),
               Line2D([], [], marker='o', color='black', markerfacecolor='none', linestyle='none', markersize=10, label='k149 terminal response')]
    fig.suptitle('Frozen105 A2 WM vs fresh source-action donor responses', fontsize=13, y=.985)
    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5, .107), ncol=3, fontsize=8, frameon=False)
    caption = ('One source, seed0; 8 dependent states x 4 directions = 32 groups, each shown at two eps (64 rows; not N64 independent).\n'
               'All flat/unstable/clipped/negative/zero/terminal rows retained. Controlled post20ms score and preaction goal k; not original15ms reward.\n'
               'Finite secants are not unique physical Jacobians. Graph scratch UNKNOWN; self15 is not HO force/contact. No generalization claim.')
    fig.text(.5, .015, caption, ha='center', va='bottom', fontsize=8, linespacing=1.45)
    fig.subplots_adjust(left=.075, right=.985, top=.84, bottom=.29, wspace=.29)
    fig.savefig(out / 'physical_WM_controlled20ms_comparison.png', dpi=190)
    fig.savefig(out / 'physical_WM_controlled20ms_comparison.svg')
    plt.close(fig)
    fields = ['snapshot_id', 'path', 'k', 'joint_index', 'epsilon_rad', 'physical_slope', 'WM_AD_effective', 'WM_AD_nominal', 'physical_label', 'clipped', 'center_offset_maxabs', 'signal_floor']
    with (out / 'all64_comparisons.csv').open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    receipt = dict(schema='full151105_physical_gradient_saved_figure:1', status='RENDERED_AWAITING_ACTUAL_VIEW_QA', source=source, inputs=inputs,
                   rows_included=64, groups_included=32, outliers_dropped=0, shared_axis_limits=list(limits), source_unchanged=pin(__file__) == source,
                   Torch_imported='torch' in sys.modules, Warp_imported='warp' in sys.modules, WM_forwards=0, physics_steps=0,
                   output_pins={v.name: pin(v) for v in out.iterdir() if v.is_file()}, scope=caption)
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
