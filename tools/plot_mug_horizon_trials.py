"""Paired seed trajectories and an independent same-state continuation audit."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw(root,out):
    runs=json.loads((root/'mug_model_horizon_trials_v1/report.json').read_bytes())['rows']
    audit=json.loads((root/'mug_horizon_choice_audit_v1/report.json').read_bytes())['rows']
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.5),gridspec_kw={'width_ratios':[1,1,1.12]})
    colors=['#215b78','#0d826e']
    for i,clip in enumerate((1884,1888)):
        ax=axes[i];cases=[r for r in runs if r['clip']==clip]
        groups=('h1','h3_pre','h3','shifted')
        data=np.array([[next(r['valid_s'] for r in cases if r['seed']==s and r['mode']==m) for m in groups]
                       for s in sorted({r['seed'] for r in cases})])
        ax.plot(np.arange(4),data.T,color='#a9b5bb',lw=.8,alpha=.72)
        ax.plot(np.arange(4),np.median(data,axis=0),color=colors[i],marker='o',lw=2.6)
        ax.set_xticks(range(4),['20 ms','仅预筛','60 ms','错配']);ax.set_ylim(0,.57)
        ax.set_title(f'{clip:06d}　连续合格时长');ax.set_ylabel('秒')
        samples=[r for r in audit if r['clip']==clip and r['any_feasible']]
        for j,k in enumerate(('greedy_feasible','chosen_feasible')):
            n=sum(r[k] for r in samples);x=j+(.32 if i else 0.)
            axes[2].bar(x,n/len(samples),width=.28,color=colors[i],label=f'{clip:06d}' if j==0 else None)
            axes[2].text(x,n/len(samples)+.025+.05*i,f'{n}/{len(samples)}',ha='center',fontsize=9)
    axes[2].set_xticks([.16,1.16],['按第一步选择','加入预测后续'])
    axes[2].set_title('同一状态选中可行三步的比例');axes[2].set_ylim(0,1.17)
    axes[2].legend(frameon=False,loc='lower center')
    for ax in axes:
        ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2);ax.spines[['right','top']].set_visible(False)
    fig.suptitle('在每步八次原生验证预算下比较预测时长',fontsize=16)
    fig.subplots_adjust(top=.78,bottom=.26,wspace=.33)
    fig.text(.065,.065,'左两图细线为八个候选种子，粗线为中位数。右图仅计八个候选中存在三步位姿可行序列的状态，按相同指令在原生物理中核验。',fontsize=9)
    fig.savefig(out,dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();draw(a.root,a.out)
