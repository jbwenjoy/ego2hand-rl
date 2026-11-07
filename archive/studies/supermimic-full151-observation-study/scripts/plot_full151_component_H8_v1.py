"""Saved-only H8 curves, all models/horizons/cohorts and six physical units."""
from pathlib import Path
import argparse, csv, hashlib, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROLES = ('initial_unmodified_A2', 'pointwise_final1000', 'pointwise_plus_response_final1000')
LABELS = ('Initial A2', 'Pointwise', 'Pointwise + response')
COLORS = ('#687481', '#2675b8', '#d77826')
BLOCKS = (
    ('q', 'Joint angle component RMSE [rad]', 'component_RMSE', 1),
    ('qd', 'Joint velocity component RMSE [rad/s]', 'component_RMSE', 1),
    ('p', 'Object position vector RMS [mm]', 'vector_or_Frobenius_RMS', 1000),
    ('R_raw9', 'Raw rotation matrix Frobenius RMS [1]', 'vector_or_Frobenius_RMS', 1),
    ('v', 'Object velocity vector RMS [m/s]', 'vector_or_Frobenius_RMS', 1),
    ('omega', 'Object angular velocity vector RMS [rad/s]', 'vector_or_Frobenius_RMS', 1),
)


def pin(path):
    p = Path(path).resolve(strict=True)
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--independent-review', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    base = a.run_dir.resolve(strict=True)
    report = json.loads((base / 'report.json').read_text(encoding='utf8'))
    review = json.loads(a.independent_review.read_text(encoding='utf8'))
    assert review['passed'] == review['total'] == 24
    assert all(value is True for value in review['checks'].values())
    rows = report['statistics']
    index = {(r['path'], r['block'], r['model_role'], r['horizon']): r for r in rows}
    assert len(index) == len(rows) == 432
    assert report['all286_windows_all8_horizons_used'] is True
    out = a.output.resolve()
    assert not out.exists()
    out.mkdir()
    exported = []
    for path in ('ALL', 'A0', 'A2'):
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout='constrained')
        for ax, (block, title, field, scale) in zip(axes.flat, BLOCKS):
            for role, label, color in zip(ROLES, LABELS, COLORS):
                values = [index[path, block, role, h][field] * scale for h in range(1, 9)]
                assert all(index[path, block, role, h]['window_count'] == (286 if path == 'ALL' else 143) for h in range(1, 9))
                ax.plot(range(1, 9), values, marker='o', ms=4, lw=1.8, c=color, label=label)
                for h, value in enumerate(values, 1):
                    exported.append({'cohort': path, 'model': role, 'block': block, 'horizon': h,
                                     'metric': field, 'plot_unit_title': title, 'value': value})
            ax.set_title(title, fontsize=11, loc='left')
            ax.set_xlabel('Horizon [20 ms control intervals]')
            ax.set_xticks(range(1, 9))
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=10, loc='upper left')
        count = 286 if path == 'ALL' else 143
        fig.suptitle(f'Fixed recorded controls, {path}: all {count} donor windows, all h1..8', fontsize=16)
        fig.text(.5, -.025,
                 'Overlapping windows from one cached reference lineage; no independent trajectory or ego generalization.\n'
                 'Both x and dynamics-state recur from the previous prediction. R9 is a matrix error, not an angle. No policy/accuracy pass gate.',
                 ha='center', fontsize=10)
        fig.savefig(out / f'{path}_all6_units_H1_to_H8.png', dpi=140, bbox_inches='tight')
        plt.close(fig)
    with (out / 'all432_plotted_metrics.csv').open('x', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    result = {'schema': 'full151105_three_component_H8_figures:1', 'source': pin(base/'report.json'),
              'independent24': pin(a.independent_review), 'renderer': pin(__file__),
              'statistics_rendered': len(exported), 'all3_models_all8_horizons_all3_cohorts_all6_blocks': True,
              'new_model_FW_AD_MJB_physics_GPU': 0,
              'outputs': {p.name: pin(p) for p in out.iterdir()}}
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps({'output': str(out), 'statistics_rendered': len(exported)}))


if __name__ == '__main__':
    main()
