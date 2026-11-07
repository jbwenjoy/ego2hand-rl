"""Show negative native dynamics after support-selected static qualification."""
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
    fit=json.loads(args.fit.read_text());review=json.loads(args.review.read_text());row=next(r for r in fit['results'] if r['native_cases'])
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.7));colors=['#3976a3','#df8543'];x=np.arange(2)
    for i,dt in enumerate([.0003125,.00015625]):
        values=[next(c['summary']['actual_duration_s'] for c in row['native_cases'] if c['control']==name and c['dt_s']==dt) for name in ['constant_q','inverse_PD_strict_LP']]
        bars=axes[0].bar(x+(i-.5)*.3,values,.28,color=colors[i],label=f'dt={dt*1e6:g} us');axes[0].bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
    axes[0].axhline(2,ls='--',color='#555555',lw=1,label='Required duration')
    axes[0].set(xticks=x,xticklabels=['Constant q target','Inverse-PD LP target'],ylim=(0,2.15),ylabel='Time to failure (s)',title='All four native cases failed')
    axes[0].legend(fontsize=8,loc='center right')
    for i,d in enumerate(review['reset_diagnostics']):
        values=np.array(d['actual_COM_torque_world_Nm'])*1000
        bars=axes[1].bar(np.arange(3)+(i-.5)*.3,values,.28,color=colors[i],label=d['control'].replace('_',' '));axes[1].bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
    axes[1].axhline(0,color='#555555',lw=1)
    axes[1].set(xticks=np.arange(3),xticklabels=['World X','World Y','World Z'],ylim=(-40,45),ylabel='Actual reset COM torque (mN m)',title='Cone/actuator feasibility is not native equilibrium')
    axes[1].legend(fontsize=8,loc='upper right')
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Support-selected source-candidate contacts: static vs native',fontsize=14)
    fig.text(.5,.018,'Same object pose, original gravity/material and fixed wrist. Source patches are uncertain annotation geometry.\nNo complete hold, neural training, measured-force claim or policy benefit.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.94));out.mkdir(parents=True);fig.savefig(out/'mug_support_selection.png',dpi=170);plt.close(fig)
    report=dict(schema='mug_support_selection_figure:1',source=file_pin(__file__),inputs=[file_pin(args.fit),file_pin(args.review)],output=file_pin(out/'mug_support_selection.png'))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['fit','review','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
