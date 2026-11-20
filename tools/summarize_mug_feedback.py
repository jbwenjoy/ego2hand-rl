"""Pair every previously frozen controller with its exactly matched perturbed open loop."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from test_mug_witness_mpc import h3


def summarize(out):
    root=Path(__file__).resolve().parents[1]/'runs'
    sources={name:h3.file_pin(root/name/'report.json') for name in ('mug_distilled_feedback_test_v1','mug_feedback_shortlists_v1')}
    student,teacher=[h3.physics.read_json(pin)['rows'] for pin in sources.values()]
    out.mkdir(exist_ok=False)
    results={}
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
    fig,axes=plt.subplots(1,2,figsize=(10.,4.7))
    colors={'dense12':'#005f73','wrench':'#dc7629','students4':'#8e6a9e'}
    labels={'dense12':'12 项在线规划','wrench':'蒸馏策略','students4':'学生提案＋4 项验证'}
    for ax,clip,span in zip(axes,(1884,1888),(2.,.92)):
        conditions=sorted(r['condition'] for r in student if r['clip']==clip and r['arm']=='open' and r['condition'].startswith('generated'))
        assert len(conditions)==len(set(conditions))==12
        def fetch(arm,condition):
            arr=student if arm in ('open','pose','wrench') else teacher
            row,=[r for r in arr if r['clip']==clip and r['condition']==condition and r.get('mode',r.get('arm'))==arm]
            return row
        base=np.array([fetch('open',c)['passing'] for c in conditions])
        summary={}
        for arm in ('open','pose','wrench','analytic4','students4','dense12'):
            rows=[fetch(arm,c) for c in conditions]
            valid=np.array([v['passing'] for v in rows]);diff=valid-base
            elapsed=sum(v['wall_s'] if 'wall_s' in v else v['decision_wall_s'] for v in rows)
            decisions=sum(v['controls'] if 'controls' in v else v['decisions'] for v in rows)
            summary[arm]=dict(median_s=.02*np.median(valid),median_gain_s=.02*np.median(diff),
                wins=int((diff>0).sum()),ties=int((diff==0).sum()),losses=int((diff<0).sum()),
                complete=sum(v['full'] for v in rows),decision_ms=1000*elapsed/decisions,valid_s=(.02*valid).tolist())
            if arm in colors:
                ax.scatter(.02*base,.02*valid,label=labels[arm],color=colors[arm],s=40,alpha=.75,edgecolor='white',lw=.4)
        ax.plot([0,span],[0,span],'--',lw=.9,c='#576474')
        ax.set(xlim=(-.025*span,1.04*span),ylim=(-.025*span,1.04*span),xlabel='冻结指令的有效时长（秒）',
               ylabel='加入反馈的有效时长（秒）',title=f'HOT3D 片段 {clip:06d}')
        ax.grid(alpha=.17);ax.set_aspect('equal');ax.spines[['top','right']].set_visible(False)
        results[str(clip)]=dict(conditions=conditions,summary=summary)
    fig.suptitle('同一扰动下，反馈能否延长连续合格时长？',fontsize=16)
    handles,names=axes[0].get_legend_handles_labels();fig.legend(handles,names,loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=[0,.10,1,.92])
    path=out/'paired_feedback.png';fig.savefig(path,dpi=170);plt.close(fig)
    h3.write_json(out/'report.json',dict(sources=sources,code=h3.file_pin(__file__),clips=results,plot=h3.file_pin(path)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);summarize(p.parse_args().out.resolve())
