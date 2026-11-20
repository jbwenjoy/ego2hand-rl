"""Summarize paired survival and command timing without treating controls as independent trials."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from test_mug_event_budget import h3


def main(out):
    root=Path(__file__).resolve().parents[1]/'runs'
    sources={n:h3.file_pin(root/f'{n}_v1/report.json') for n in ('mug_event_budget','mug_learned_schedule_test','mug_delayed_feedback','mug_centered_screen','mug_tick_pipeline')}
    out.mkdir(exist_ok=False);summary={}
    for name,pin in sources.items():
        rows=h3.physics.read_json(pin)['rows'];summary[name]={}
        for clip in (1884,1888):
            cohort=[r for r in rows if r['clip']==clip and r['condition']!='nominal']
            modes=list(dict.fromkeys(r['mode'] for r in cohort));paired={}
            for mode in modes:
                trials=sorted([r for r in cohort if r['mode']==mode],key=lambda r:r['condition'])
                passes=np.array([r['passing'] for r in trials]);logs=[d for r in trials for d in r.get('decisions',r.get('records',r.get('events')))]
                t=np.array([d['seconds']*1000 if 'seconds' in d else d.get('decision_ms',d.get('ms')) for d in logs])
                same=sorted([r for r in cohort if r['mode']=='open'],key=lambda r:r['condition'])
                assert [r['condition'] for r in same]==[r['condition'] for r in trials]
                delta=passes-np.array([r['passing'] for r in same])
                stat=dict(n=len(passes),med_s=.02*np.median(passes),paired_gain_over_open_s=.02*np.median(delta),
                    improved=int((delta>0).sum()),same=int((delta==0).sum()),worse=int((delta<0).sum()),full=sum(r['full'] for r in trials),
                    controls=len(t),mean_ms=float(t.mean()),p95_ms=float(np.percentile(t,95)),max_ms=float(t.max()),decision_gt20=int((t>20).sum()))
                if 'periodic' in modes:
                    times=sorted([r for r in cohort if r['mode']=='periodic'],key=lambda r:r['condition'])
                    stat['paired_gain_over_periodic_s']=.02*np.median(passes-np.array([r['passing'] for r in times]))
                ready=[d.get('ready_ms',d.get('scheduled_to_ready_ms')) for d in logs]
                ready=np.asarray([x for x in ready if x is not None])
                if len(ready):
                    sim=[d.get('simulated_ms',d.get('scheduled_to_simulated_ms')) for d in logs]
                    stat.update(max_ready_ms=ready.max(),late_ready=int((ready>20).sum()),late_sim=int((np.asarray(sim)>20).sum()),
                        applied=sum(d.get('status') in ('received','applied') for d in logs),
                        max_sim_ms=max(sim))
                if 'jobs' in trials[0]:
                    jobs=[j for r in trials for j in r['jobs']]
                    stat.update(submitted_jobs=len(jobs),p95_job_ms=float(np.percentile([j['ms'] for j in jobs],95)) if jobs else None)
                paired[mode]=stat
            summary[name][str(clip)]=paired
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
    fig,ax=plt.subplots(1,2,figsize=(11,4.5))
    for j,clip in enumerate(('1884','1888')):
        a=ax[j];sd=summary['mug_event_budget'][clip]
        names=['periodic','event','dense'];x=np.arange(3)
        a.bar(x-.18,[sd[n]['mean_ms'] for n in names],.36,color='#387490',label='平均决策耗时')
        a.bar(x+.18,[sd[n]['p95_ms'] for n in names],.36,color='#db8849',label='第 95 百分位')
        a.axhline(20,color='#263c45',ls='--',lw=1)
        a.set_xticks(x,['定时触发','偏差触发','逐步搜索']);a.set_ylim(0,72);a.set_ylabel('毫秒')
        a.set_title(f'HOT3D {int(clip):06d}');a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.18);a.set_axisbelow(True)
    fig.legend(*ax[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False)
    fig.suptitle('稀疏调用降低平均耗时，搜索发生时仍超过 20 毫秒',fontsize=16)
    fig.tight_layout(rect=[0,.10,1,.92])
    img=out/'search_latency.png';fig.savefig(img,dpi=165);plt.close(fig)
    h3.write_json(out/'report.json',dict(sources=sources,code=h3.file_pin(__file__),results=summary,figure=h3.file_pin(img)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);main(p.parse_args().out.resolve())
