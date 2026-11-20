"""Plot the full native paired rehearsal evaluation with both repetitions visible."""
import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def draw(report,out):
    runs=json.loads(report.read_bytes())['native']
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(11,4.4),gridspec_kw={'wspace':.32})
    names=['parent','no_rehearsal','rehearsal_1','rehearsal_10']
    colors=['#4c6679','#dc823c','#58a699','#006f73']
    for ax,(clip,maximum) in zip(axes,[('clip_001',8.),('clip_002',.48),('clip_003',1.5)]):
        for i,(arm,color) in enumerate(zip(names,colors)):
            values=[x['passing_s'] for x in runs[clip] if x['arm']==arm]
            assert len(values)==2
            median=np.median(values)
            ax.hlines(median,i-.22,i+.22,color=color,lw=2)
            ax.scatter([i-.075,i+.075],values,s=36,c=color,zorder=4)
        ax.set_ylim(0,maximum);ax.set_xlim(-.5,3.5)
        ax.set_title(clip,pad=8)
        ax.set_xticks(range(4),['母策略','蒸馏','保留×1','保留×10'],rotation=16)
        ax.grid(axis='y',alpha=.25)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('连续通过物理与任务检查（秒）')
    fig.suptitle('把扳手规划器蒸馏为单次策略前向',fontsize=17,y=.98)
    fig.subplots_adjust(top=.78,bottom=.23)
    fig.text(.075,.045,'每点为一次从头启动的原生复测。三幅图使用各自的时间刻度。001 完整参考为 7.36 秒，所有方案的严格成功次数均为零。',fontsize=9)
    fig.savefig(out,dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    r=p.parse_args().run;draw(r/'report.json',r/'retention.png')
