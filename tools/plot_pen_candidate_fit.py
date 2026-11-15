"""Scientific summary of the fixed-budget candidate fit and actual recursion."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from supermimic_research.artifacts import file_pin
from supermimic_research.learning.pen_stage import write


def main(a):
    if a.output.exists():
        raise ValueError('Fresh figure directory required')
    e, r = [json.loads(p.read_text()) for p in (a.evaluation, a.rollout)]
    colors = dict(DIRECT='#2878b5', IMPULSE='#d75a26')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    for i, mode in enumerate(colors):
        values = [e['models'][mode]['one_step'][role]['valid']['normalized_MSE']['object12'] for role in ['TRAIN', 'CHECK']]
        ax.bar(np.arange(2) + (i - .5) * .32, values, width=.32, color=colors[mode], label=mode)
    ax.set_xticks([0, 1], ['TRAIN (24 parents)', 'CHECK (12 parents)'])
    ax.set_yscale('log')
    ax.set_ylabel('Normalized object12 MSE')
    ax.set_title('Low fit error does not transfer to later parents')
    ax.legend()
    ax = axes[0, 1]
    for i, mode in enumerate(colors):
        summary = e['models'][mode]['physical_finite_response']['summary']['ALL']
        values = [summary[method][block]['all_finite']['pooled_relative_L2_error']
                  for method in ['AD', 'secant'] for block in ['object12', 'source_BODY_wrench6']]
        ax.bar(np.arange(4) + (i - .5) * .32, values, width=.32, color=colors[mode], label=mode)
    ax.axhline(1, color='#555555', ls='--', label='Zero response')
    ax.set_xticks(np.arange(4), ['AD object', 'AD wrench', 'Secant object', 'Secant wrench'])
    ax.set_ylabel('Pooled relative L2 error')
    ax.set_title('196 quality-valid native finite-action pairs')
    ax.legend(fontsize=8)
    for ax, quantity, factor, title in [
            (axes[1, 0], 'COM_error_m', 1000., 'Actual-control recursive COM error'),
            (axes[1, 1], 'angle_error_deg', 1., 'Actual-control full SO(3) error')]:
        for mode in colors:
            model = r['models'][mode]
            ax.plot([20, 40, 100, 200], [model[str(h)]['model'][quantity]['mean'] * factor for h in [1, 2, 5, 10]],
                    'o-', color=colors[mode], label=mode)
        baseline = r['models']['DIRECT']
        ax.plot([20, 40, 100, 200], [baseline[str(h)]['persistence'][quantity]['mean'] * factor for h in [1, 2, 5, 10]],
                'x--', color='#555555', label='State persistence')
        ax.set_xlabel('Prediction horizon (ms); windows 63 / 58 / 43 / 18')
        ax.set_ylabel('Mean COM error (mm)' if factor == 1000. else 'Mean full-angle error (deg)')
        ax.set_title(title)
        ax.legend(fontsize=8)
        if factor == 1000.:
            ax.set_yscale('log')
    fig.suptitle('Fixed 2000-update WM components: predictive/gradient readiness fails', fontsize=14)
    fig.text(.5, -.025, 'Same-clip development parents; correlated windows; failed source tasks. No Actor, ego, or task-success claim.', ha='center')
    a.output.mkdir()
    path = a.output / 'summary.png'
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)
    write(a.output / 'report.json', dict(schema='candidate_fit_summary_figure:1', source=file_pin(__file__),
          evaluation=file_pin(a.evaluation), rollout=file_pin(a.rollout), figure=file_pin(path),
          scope='Descriptive component diagnostics, not statistical independence or successful policy control.'))
    print(json.dumps(file_pin(path)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--rollout', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
