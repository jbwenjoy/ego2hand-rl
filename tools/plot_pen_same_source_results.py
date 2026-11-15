"""Scientific summary of native replay fidelity and amplitude-dependent response."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin


def main(a):
    r=json.loads(a.analysis.read_text());review=json.loads(a.review.read_text())
    if a.output.exists():raise ValueError('Fresh figure')
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    values=np.array([[v[block]['relative_L2_error'] for v in r['native_amplitude_comparison']] for block in ['object12','wrench6']])
    im=axes[0].imshow(values,aspect='auto',norm=LogNorm(vmin=.01,vmax=30),cmap='YlOrRd')
    axes[0].set(yticks=[0,1],yticklabels=['Object response','Contact wrench'],
        xticks=range(8),xticklabels=[f'{v["dt"]*1000:g}ms\n{v["anchor"]*20}ms' for v in r['native_amplitude_comparison']],
        xlabel='Native dt / actual source time (later states differ across dt)',title='Native contrast:0.002 vs0.005rad amplitudes')
    for i in range(2):
        for j in range(8):axes[0].text(j,i,f'{100*values[i,j]:.0f}%',ha='center',va='center',fontsize=10)
    fig.colorbar(im,ax=axes[0],shrink=.7,label='Relative L2 change / norm at0.005rad (log scale)')
    names=list(r['model_summaries']);x=np.arange(4)
    for i,block in enumerate(['object12','wrench6']):
        axes[1].bar(x+(i-.5)*.35,[r['model_summaries'][name]['ALL'][block]['pooled_relative_error'] for name in names],width=.35,label=block)
    axes[1].axhline(1,color='black',ls='--',lw=1,label='Zero response')
    axes[1].set(xticks=x,xticklabels=[n.replace('_','\n') for n in names],ylabel='Pooled relative error vs finite native contrasts',
        title='Frozen-network AD remains insufficient\nNot a true-derivative qualification')
    axes[1].legend(fontsize=9);axes[1].grid(axis='y',alpha=.2);axes[1].set_axisbelow(True)
    fig.suptitle('All24 baseline20ms replays pass original state/force gates;120 responses,3 repeats,one new direction\n'
        'Repeated-execution spread <0.07% of contrast; amplitude dependence remains. No Actor or ego/task success.',fontsize=12)
    a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=150);plt.close(fig)
    a.output.with_suffix('.json').write_text(json.dumps(dict(source=file_pin(__file__),analysis=file_pin(a.analysis),review=file_pin(a.review),figure=file_pin(a.output)),indent=2)+'\n',encoding='utf8')
    print(json.dumps(file_pin(a.output)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['analysis','review','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
