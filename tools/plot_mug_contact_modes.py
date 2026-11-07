"""Static research figure from retained negative trials and candidate diagnostics."""
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
    modes=json.loads(args.modes.read_text());patches=json.loads(args.patches.read_text());out.mkdir(parents=True)
    fig,axes=plt.subplots(2,2,figsize=(12,8.2));colors=['#3976a3','#df8543'];x=np.arange(3)
    for i,robot in enumerate(patches['robot']):
        ax=axes[0,i]
        for j,dt in enumerate([.0003125,.00015625]):
            y=[next(c['summary']['actual_duration_s'] for c in modes['cases'] if c['seed']==robot['seed'] and c['mode']==mode and c['dt_s']==dt) for mode in ['point3','spin4_1mm','spin4_3mm']]
            bars=ax.bar(x+(j-.5)*.32,y,.30,label=f'dt = {dt*1e6:g} us',color=colors[j])
            ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
        ax.axhline(2,color='#555555',ls='--',lw=1,label='Required duration (2 s)')
        ax.set(xticks=x,xticklabels=['Point force\ncondim 3','Spin prior\n1 mm','Spin prior\n3 mm'],ylim=(0,2.15),ylabel='Time to failure (s)',
               title=['A: local force-search pose','B: conditional Z-up pose'][i])
        ax.text(.5,.81,'All trials failed; no qualified hold',transform=ax.transAxes,ha='center',fontsize=10,color='#9a3131')
        if i==0:ax.legend(fontsize=8,loc='center right',bbox_to_anchor=(1,.55))
    ax=axes[1,0]
    for i,robot in enumerate(patches['robot']):
        values=[robot['geometric_similarity_only']['all_oriented_candidates'],*robot['per_finger_geometric_similarity']['all_oriented_candidates']]
        bars=ax.bar(np.arange(4)+(i-.5)*.32,values,.30,color=colors[i],label=['Pose A','Pose B'][i]);ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    ax.set(xticks=np.arange(4),xticklabels=['Whole hand','Thumb','Index','Middle'],ylim=(0,1.08),ylabel='Diagnostic similarity (unit-force proxy)',title='Whole-hand similarity can hide finger mismatch')
    ax.legend(fontsize=9)
    ax=axes[1,1]
    for i,robot in enumerate(patches['robot']):
        gaps=[min(c['nearest_source_patch_distance_m'] for c in robot['contact_comparisons'] if c['group']==g)*1000 for g in [1,2,3]]
        bars=ax.bar(x+(i-.5)*.32,gaps,.30,color=colors[i],label=['Pose A','Pose B'][i]);ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
    ax.axhline(3,color='#555555',ls='--',lw=1,label='Exploratory proximity threshold')
    ax.set(xticks=x,xticklabels=['Thumb','Index','Middle'],ylim=(0,43),ylabel='Minimum same-finger patch distance (mm)',title='Baseline native contact vs source candidate region')
    ax.legend(fontsize=8,loc='upper left')
    for ax in axes.flat:ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Matched contact-mode sensitivity and source patch diagnostics',fontsize=15,y=.985)
    fig.text(.5,.018,'Unmeasured material priors; first-frame annotation geometry only. Candidate patches are not contact truth.\nSame pose/PD command across modes. No policy training, calibrated-force claim or full-trajectory success.',ha='center',fontsize=9,color='#444444')
    fig.tight_layout(rect=(0,.065,1,.96));fig.savefig(out/'mug_contact_modes.png',dpi=170);fig.savefig(out/'mug_contact_modes.svg');plt.close(fig)
    report=dict(schema='mug_contact_modes_figure:1',source=file_pin(__file__),inputs=[file_pin(args.modes),file_pin(args.patches)],outputs=[file_pin(out/'mug_contact_modes.png'),file_pin(out/'mug_contact_modes.svg')])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['modes','patches','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
