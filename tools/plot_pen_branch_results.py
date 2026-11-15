"""Standalone scientific figure of branch gain / sequence and gradient tradeoffs."""
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
    r,audit=[json.loads(p.read_text()) for p in [a.evaluation,a.audit]]
    if a.output.exists():raise ValueError('Fresh figure')
    keys=['V1_DIRECT','V1_IMPULSE','V2_DIRECT','V2_IMPULSE'];colors=['#6999ce','#e5a56d','#276fbf','#d16b19']
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for ax,role,title in [(axes[0,0],'CHECK','New branch CHECK (64 rows)'),(axes[0,1],'SEQUENCE_CHECK','Original sequence CHECK (80 rows)')]:
        for i,block in enumerate(['object12','wrench6']):
            ax.bar(np.arange(4)+(i-.5)*.34,[r['models'][k]['one_step'][role]['common_v1_scale_MSE'][block] for k in keys],width=.34,label=block,alpha=1 if i==0 else .55,color=colors)
        ax.set(xticks=range(4),xticklabels=[k.replace('_','\n') for k in keys],ylabel='MSE on common frozen v1 TRAIN scales',title=title);ax.legend()
    ax=axes[1,0]
    for i,block in enumerate(['object12','wrench6']):
        ax.bar(np.arange(4)+(i-.5)*.34,[audit['models'][k]['summary']['ALL'][block]['local_AD']['pooled_relative_L2_error'] for k in keys],width=.34,label=block,alpha=1 if i==0 else .55,color=colors)
    ax.axhline(1,color='black',ls='--',lw=1,label='Zero-response comparator')
    ax.set(xticks=range(4),xticklabels=[k.replace('_','\n') for k in keys],title='Physical AD check: 11 exact-source CHECK pairs',ylabel='Pooled relative L2 error');ax.legend(fontsize=8)
    for key,color in zip(keys,colors):
        rows=r['models'][key]['recorded_control_rollout']
        axes[1,1].plot([20*int(h) for h in rows],[1000*v['model']['COM_error_m']['mean'] for v in rows.values()],'-o' if key.startswith('V2') else '--o',color=color,label=key)
    axes[1,1].set(title='Recorded-control recursive COM prediction',xlabel='Horizon (ms)',ylabel='Mean COM error (mm)');axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('More action coverage improves branch predictions, without qualifying control gradients\n'
        'V1:157 TRAIN; V2:349 TRAIN; both2000 updates. Fixed clip/initial pose. No Actor or real ego validation.',fontsize=13)
    a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=150);plt.close(fig)
    a.output.with_suffix('.json').write_text(json.dumps(dict(source=file_pin(__file__),evaluation=file_pin(a.evaluation),audit=file_pin(a.audit),figure=file_pin(a.output)),indent=2)+'\n',encoding='utf8')
    print(json.dumps(file_pin(a.output)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['evaluation','audit','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
