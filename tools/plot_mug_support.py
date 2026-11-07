"""Scientific support/partial-hold comparison, with no additional simulation."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin


def main(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    paths={name:ROOT/'runs'/run/'report.json' for name,run in dict(support='mug_support_v2',native='mug_support_native_v1',
        equilibrium='mug_equilibrium_v1',gravity='mug_gravity_hold_v1',engagement='mug_gravity_engagement_v2').items()}
    data={name:json.loads(path.read_text()) for name,path in paths.items()}
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
    for depth,color in [(0.,'#2563eb'),(.002,'#d97706')]:
        for dt,style in [(.0003125,'-'),(.00015625,'--')]:
            rows=[r for r in data['support']['records'] if r['overclosure_m']==depth and r['source_dt_s']==dt]
            values=[r['domains'][1]['maximum']['support_weight_N'] for r in rows]
            axes[0].plot([r['source_time_s'] for r in rows],[np.nan if x is None else x for x in values],style+'o',
                color=color,markersize=3,label=f'{depth*1000:g}mm / {dt*1000:g}ms')
    axes[0].axhline(2.4525,color='#dc2626',ls=':',label='weight2.4525 N');axes[0].set(xlabel='Saved pose time (s)',ylabel='Max static support (N)',title='Frozen contact/PD approximation');axes[0].legend(fontsize=7)
    spans=[]
    for index,label in enumerate(['new-pose baseline','LP observed load','LP free dry friction']):
        values=[c['summary']['actual_duration_s'] for c in data['native']['cases'] if c['target_index']==index];spans.append((label,min(values),max(values)))
    for index,r in enumerate(data['equilibrium']['results']):
        values=[c['summary']['actual_duration_s'] for c in r['native_cases']];spans.append((f'local search seed{index}',min(values),max(values)))
    values=[c['summary']['actual_duration_s'] for c in data['gravity']['cases']];spans.append(('Z-up branch,10 cases',min(values),max(values)))
    for y,(label,lo,hi) in enumerate(spans):
        axes[1].plot([lo,hi],[y,y],'-o',color='#475569',markersize=4)
    axes[1].axvline(2,color='#dc2626',ls=':');axes[1].set(yticks=range(len(spans)),yticklabels=[x[0] for x in spans],
        xlabel='Termination time (s)',xlim=(0,2.1),title='All20 native cases failed');axes[1].invert_yaxis();axes[1].tick_params(axis='y',labelsize=8)
    rows=data['engagement']['results'];x=np.arange(3)
    for digit,color in enumerate(['#d97706','#dc2626','#16a34a']):
        axes[2].bar(x+(digit-1)*.24,[r['native_normal_N'][digit] for r in rows],width=.24,color=color,label=['thumb','index','middle'][digit])
    axes[2].set(xticks=x,xticklabels=['-20','-50','-75'],xlabel='Initial component gap target (um)',ylabel='Native normal force (N)',title='All3 fingers active; no6D static balance');axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Static cone feasibility, native compliance and contact placement are separate checks',fontsize=12)
    out.mkdir(parents=True);fig.savefig(out/'mug_support_results.png',dpi=155);plt.close(fig)
    result=dict(source=file_pin(__file__),inputs={k:file_pin(v) for k,v in paths.items()},figure=file_pin(out/'mug_support_results.png'),neural_updates=0,integration_steps=0)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args())
