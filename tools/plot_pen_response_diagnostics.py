"""Standalone scientific summary of the frozen component/response experiment."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin


def main(a):
    health,diagnostic,native=[json.loads(p.read_text()) for p in [a.health,a.diagnostics,a.native]]
    if a.output.exists():raise ValueError('Fresh plot output')
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    colors={'DIRECT':'#276fbf','IMPULSE':'#d16b19'}
    for mode,color in colors.items():
        updates=[int(k) for k in health['CHECK_curve']]
        axes[0,0].plot(updates,[health['CHECK_curve'][str(k)][mode]['object12'] for k in updates],'-o',label=mode,color=color)
        rows=diagnostic['models'][mode]['recorded_control_rollout'];time=[20*int(k) for k in rows]
        axes[0,1].plot(time,[1000*r['model']['COM_error_m']['mean'] for r in rows.values()],'-o',label=mode,color=color)
        axes[1,0].plot(time,[r['model']['angle_error_deg']['mean'] for r in rows.values()],'-o',label=mode,color=color)
    axes[0,0].set(title='Single-step CHECK object12 error',xlabel='Optimizer updates per model',ylabel='TRAIN-scale-normalized MSE')
    axes[0,1].set(title='Recursive COM error (recorded controls)',xlabel='Prediction horizon (ms)',ylabel='Mean COM error (mm)')
    axes[1,0].set(title='Recursive full orientation error',xlabel='Prediction horizon (ms)',ylabel='Mean geodesic angle (degrees)')
    labels=[]
    for i,(mode,block) in enumerate([('DIRECT','object12'),('IMPULSE','object12'),('DIRECT','wrench6'),('IMPULSE','wrench6')]):
        values=[c[block]['relative_L2_error'] for c in native['response_comparisons'] if c['mode']==mode]
        axes[1,1].scatter(np.full(len(values),i)+np.linspace(-.12,.12,len(values)),values,color=colors[mode],s=28)
        axes[1,1].plot([i-.2,i+.2],[np.median(values)]*2,color='black',lw=2)
        labels.append(mode+'\n'+block)
    axes[1,1].set(title='Model AD vs native central response',ylabel='Relative normalized L2 error',xticks=range(4),xticklabels=labels)
    for ax in axes.flat:ax.grid(alpha=.2)
    for ax in axes.flat:
        if ax is not axes[1,1]:ax.legend(fontsize=9)
    fig.suptitle('Pen world-model diagnostics: 2000 updates; no learned controller or ego validation\n'
                 'Rollout windows H1/H2/H5/H10 = 80/72/48/8 (overlapping); native probes = 2 directions x 2 amplitudes x 2 dt',fontsize=12)
    a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=150);plt.close(fig)
    a.output.with_suffix('.json').write_text(json.dumps(dict(source=file_pin(__file__),health=file_pin(a.health),diagnostics=file_pin(a.diagnostics),native=file_pin(a.native),figure=file_pin(a.output)),indent=2)+'\n',encoding='utf8')
    print(json.dumps(file_pin(a.output)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['health','diagnostics','native','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
