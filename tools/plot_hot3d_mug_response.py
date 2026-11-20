"""Visualize native-oracle coverage and closed-loop gradient outcomes on the two windows."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def draw(root,out):
    read=lambda name:json.loads((root/name/'report.json').read_bytes())['rows']
    coverage=read('mug_shortlist_audit_v1')
    gradient=read('mug_gradient_plans_v1')
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),gridspec_kw={'wspace':.35})
    colors=['#355e80','#0b8b80']
    for i,clip in enumerate((1884,1888)):
        rows=[r for r in coverage if r['clip']==clip and not r['anchor']]
        for k,name in enumerate(('learned','random')):
            val=sum(r[name] for r in rows)
            x=k+(i-.5)*.33
            axes[0].bar(x,val/len(rows),.30,color=colors[i],label=f'{clip:06d}' if k==0 else None)
            axes[0].text(x,val/len(rows)+.03+.045*i,f'{val}/{len(rows)}',ha='center',fontsize=9)
        ax=axes[i+1];ax.set_title(f'{clip:06d}　闭环有效时长')
        g=[r for r in gradient if r['clip']==clip]
        for seed in sorted({r['seed'] for r in g}):
            values=[next(r['valid_s'] for r in g if r['mode']==m and r['seed']==seed) for m in ('subset','random','gradient')]
            ax.plot(range(3),values,color='#84949e',alpha=.5,linewidth=1)
            ax.scatter(range(3),values,s=17,color=colors[i],alpha=.65)
        ax.set_ylim(0,.55);ax.set_ylabel('秒');ax.set_xticks(range(3),['原候选','随机微调','梯度微调'])
    axes[0].set_title('命中原生最优动作');axes[0].set_xticks([0,1],['模型预筛','随机预筛'])
    axes[0].set_ylim(0,1.1);axes[0].set_ylabel('命中比例');axes[0].legend(frameon=False,loc='upper right')
    for ax in axes:
        ax.grid(axis='y',alpha=.22);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    fig.subplots_adjust(top=.8,bottom=.24)
    fig.suptitle('真实第一视角杯子参考上的动作预测与原生验证',fontsize=16)
    fig.text(.07,.055,'左图排除固定四个候选已经命中最优动作的状态。右图每条线为一个新候选种子，各方法每次验证八个动作。两段完整参考为 2.00 和 0.92 秒。',fontsize=9)
    fig.savefig(out,dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args();draw(args.root,args.out)
