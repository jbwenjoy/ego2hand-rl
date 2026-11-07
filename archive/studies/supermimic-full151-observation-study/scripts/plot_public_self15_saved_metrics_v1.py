"""Scientific figure from already saved descriptive metrics, no model runtime."""
from pathlib import Path
import argparse
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def pin(p):
    return {'path': str(Path(p).absolute()), 'sha256': hashlib.sha256(Path(p).read_bytes()).hexdigest()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--description', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args();r = json.loads(a.description.read_text(encoding='utf-8-sig'))
    assert r['status'] == 'COMPLETED_FIXED_PUBLIC15_AFTER1000_ALL152_NOT_GENERALIZATION'
    out = a.output.absolute();out.mkdir(exist_ok=False)
    roles = r['roles'];groups = ['ALL', 'A0', 'A2']
    panels = [
        ('Object position residual', 'Vector RMS [mm]', lambda s: 1000*s['state_native_delta']['p']['vector_RMS']),
        ('Native LogR residual on raw matrices', 'Vector RMS [deg]', lambda s: 180/np.pi*s['state_native_delta']['native_LogR']['vector_RMS']),
        ('Angular velocity residual', 'Vector RMS [rad/s]', lambda s: s['state_native_delta']['omega']['vector_RMS']),
        ('Two-source normalized impulse', 'z MSE [dimensionless]', lambda s: s['impulse_z_MSE_F64']),
        ('Match the observed discrete defect', 'Consistency MSE [dimensionless]', lambda s: s['saved_defect_consistency_MSE_F64']),
        ('Original self15 cached-sensor labels', 'Mean BCE [F64 rescore]', lambda s: s['cached_self15']['BCE_mean_entry_F64']),
    ]
    labels = ['Original initial', 'Public incl. self15', 'Public + consistency']
    colors = ['#667887', '#2479b3', '#d47825']
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout='constrained')
    fig.get_layout_engine().set(rect=(0, .09, 1, .90))
    table = {}
    for ax, (title, unit, getter) in zip(axes.flat, panels):
        values = np.asarray([[getter(r['statistics'][role][g]) for g in groups] for role in roles])
        assert np.isfinite(values).all() and (values >= 0).all()
        table[title] = values.tolist()
        for j, role in enumerate(roles):
            x = np.arange(3)+(j-1)*.25
            ax.bar(x, values[j], .23, label=labels[j], color=colors[j])
        if (values > 0).all() and values.max()/values.min() > 12:
            ax.set_yscale('log')
        else:
            ax.set_ylim(bottom=0)
        ax.set(title=title, ylabel=unit, xticks=np.arange(3), xticklabels=groups)
        ax.grid(axis='y', alpha=.2)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('Fixed AFTER1000 public-self15 components: known training rows', fontsize=15)
    fig.text(.5, .026, 'All152 = A0(76) + A2(76), including 24 baselines and 128 dependent direction rows. Same fixed inputs and context.\nNative LogR is not a projected SO(3) certificate; self15 targets are cached ~15ms sensors, state targets ~20ms. No task, geometry, ego or generalization benefit claim.', ha='center', fontsize=10)
    p = out/'fixed_after1000_public_self15_tradeoffs.png'
    fig.savefig(p, dpi=145);plt.close(fig)
    report = {'schema':'full151105_public_self15_metrics_figure:1','description':pin(a.description),'renderer':pin(__file__),
              'figure':pin(p),'plotted_values_role_then_group':table,'roles':roles,'groups':groups,
              'new_model_FW_AD_physics':0,'whole_figure_visual_QA_pending':True}
    with (out/'report.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({'figure':pin(p),'report':pin(out/'report.json')}))


if __name__ == '__main__':
    main()
