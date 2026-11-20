"""Compare frozen penetration forecasts with a separate prospective online filter test."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np

from plot_pen_mug_prefix_evidence import select_font


ROOT = Path(__file__).resolve().parents[1] / 'runs'
CHOICES = ('hold', 'hold_finger4', 'actor_finger4', 'abrupt')
FILTERS = ('actor', 'paired', 'cpu_dynamics')
CLIPS = ('clip_002', 'clip_003')
INK, NOTE, GRID = '#183644', '#4d6675', '#d6e0e3'
COLORS = ('#173e50', '#bf6026', '#00887a', '#8763aa')


def inputs():
    frozen = json.loads((ROOT/'pen_transfer_contact_forecasts_v1/forecasts.json').read_bytes())
    frozen = {r['arm']:r for r in frozen if r['split']=='clip_003_handoff' and r['rep']==0}
    with np.load(ROOT/'pen_action_depth_pair_v1/native_predictions.npz', allow_pickle=False) as recent:
        choice = (recent['clip']=='clip_003') & (recent['phase']==9) & (recent['rep']==0)
        depths = []
        for arm in CHOICES:
            i, = np.flatnonzero(choice & (recent['arm']==arm))
            w1 = frozen[arm]['warp'][8][0]
            cpu = frozen[arm]['estimates']['cpu_dynamics'][8][0]
            np.testing.assert_allclose([w1,cpu], [recent['actual_mm'][i,0],recent['cpu_dynamics'][i,0]], atol=1e-8)
            depths.append([w1,cpu,frozen[arm]['estimates']['paired'][8][0],recent['state_action'][i,0]])
    online = json.loads((ROOT/'pen_online_contact_filter_v1/report.json').read_bytes())
    times = np.empty((2,3,3))
    strict, halfway, complete = [], [], []
    for c, clip in enumerate(CLIPS):
        for j, key in enumerate(FILTERS):
            rows = sorted(online['clips'][clip]['cases'][key],key=lambda r:r['rep'])
            if [r['rep'] for r in rows] != [0,1,2]:
                raise ValueError('Expected three native numerical repeats per arm')
            times[c,j] = [r['passing_s'] for r in rows]
            strict.extend(r['strict'] for r in rows)
            halfway.extend(r['passed_halfway'] for r in rows)
            complete.extend(r['complete'] for r in rows)
    return np.asarray(depths),times,(sum(halfway),sum(complete),sum(strict),len(strict))


def render():
    depths, times, qualification = inputs()
    plt.rcParams.update({'font.family':select_font(),'axes.unicode_minus':False,'figure.facecolor':'white'})
    fig = plt.figure(figsize=(14,6.9))
    def text(x,y,value,**kw):
        fig.text(x,y,value,va='top',color=kw.pop('color',INK),**kw)
    text(.046,.960,'转笔接管时的穿透预测与在线筛选',fontsize=21.5,weight='bold')
    text(.047,.882,'每个控制周期 20 毫秒。左图比较一次决策的预测，右图测量从起点连续通过原生验收的时长。',fontsize=11.2,color=NOTE)
    fig.add_artist(Line2D([.046,.960],[.815,.815],transform=fig.transFigure,color=GRID,lw=1))
    text(.047,.777,'第 9 个控制周期的笔手穿透',fontsize=15,weight='bold')
    text(.047,.719,'clip_003，原始策略，首次复测。“保持”表示沿用前一周期的原始动作。',fontsize=10.5,color=NOTE)
    text(.559,.777,'在线筛选后的有效前缀',fontsize=15,weight='bold')
    text(.559,.719,'两轨迹均衡策略从头执行。每点为一次复测，短线及数字为中位数。',fontsize=10.5,color=NOTE)
    left = fig.add_axes([.158,.190,.333,.367])
    symbols = ('o','s','^','D')
    legend = []
    for j,(name,marker) in enumerate(zip(('原生 Warp 实测','CPU 动力学重放','既有 paired','新 state_action'),symbols)):
        y = np.arange(4) + (j-1.5)*.197
        left.scatter(depths[:,j],y,marker=marker,s=34,color=COLORS[j],zorder=3)
        for x,yy in zip(depths[:,j],y):
            left.annotate(f'{x:.2f}',(x,yy),xytext=(5,0),textcoords='offset points',va='center',
                          fontsize=8.9,color=COLORS[j])
        legend.append(Line2D([],[],marker=marker,linestyle='none',color=COLORS[j],label=name,ms=6.5))
    fig.legend(handles=legend,loc='upper left',bbox_to_anchor=(.041,.668),ncol=2,
               frameon=False,fontsize=10.1,columnspacing=2.0,handletextpad=.4)
    left.axvline(2,ls=(0,(4,3)),lw=1.2,color='#9b3d3b',zorder=2)
    left.text(2, -.55,'2 mm 阈值',ha='center',va='bottom',color='#9b3d3b',fontsize=9.5)
    for y in (.5,1.5,2.5):
        left.axhline(y,color=GRID,lw=.6,zorder=0)
    left.set_xlim(0,3.75)
    left.set_ylim(3.52,-.52)
    left.set_yticks(range(4),['全部保持','无名指保持\n其余用策略','无名指用策略\n其余保持','全部用策略'])
    left.xaxis.set_major_locator(MultipleLocator(.5))
    left.set_xlabel('周期内最大笔手穿透深度（mm）',fontsize=10.5,color=NOTE,labelpad=7)
    right = fig.add_axes([.590,.190,.359,.367])
    x = np.arange(3)
    clip_colors = ('#596f89','#087d74')
    for c,(clip,color) in enumerate(zip(CLIPS,clip_colors)):
        centers = x + (-.14,.14)[c]
        for j,center in enumerate(centers):
            samples = times[c,j]
            median = float(np.median(samples))
            right.scatter(center+np.array([-.052,0,.052]),samples,s=32,c=color,zorder=4)
            right.hlines(median,center-.094,center+.094,color=color,lw=1.5,zorder=3)
            right.annotate(f'{median:.2f}',(center,median),xytext=(0,8),textcoords='offset points',
                           ha='center',va='bottom',color=color,fontsize=10,weight='bold')
    fig.legend(handles=[Line2D([],[],marker='o',linestyle='none',color=c,label=s) for s,c in zip(CLIPS,clip_colors)],
               loc='upper left',bbox_to_anchor=(.552,.663),ncol=2,frameon=False,fontsize=10.4,handletextpad=.4)
    text(.560,.600,'在线使用既有 paired。新 state_action 只做了左侧离线评测。',fontsize=9.6,color=NOTE)
    right.set_ylim(0,1.40)
    right.set_xlim(-.51,2.51)
    right.set_xticks(x,['直接执行策略','既有 paired 筛选','CPU 筛选'])
    right.set_ylabel('有效前缀时长（秒）',fontsize=10.5,color=NOTE)
    right.yaxis.set_major_locator(MultipleLocator(.2))
    for ax,grid_axis in ((left,'x'),(right,'y')):
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color(GRID)
        ax.set_axisbelow(True)
        ax.grid(axis=grid_axis,color=GRID,lw=.7)
        ax.tick_params(axis='both',length=0,pad=5,labelsize=10,colors=INK)
    fig.add_artist(Line2D([.046,.960],[.092,.092],transform=fig.transFigure,color=GRID,lw=1))
    half,full,strict,n = qualification
    text(.047,.066,f'在线共 {n} 次原生复测，进入后半程 {half} 次，运行完整参考 {full} 次，严格成功 {strict} 次。',fontsize=10.2,color=NOTE)
    fig.savefig(ROOT/'pen_online_contact_filter_v1/decision_evidence.png', dpi=175)
    plt.close(fig)


if __name__=='__main__':
    render()
