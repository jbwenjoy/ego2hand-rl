"""Artifact-only annotations/depth plots from frozen analysis and saved raw logs.

No analysis thresholds, outcomes, policy, source data or physical state change.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

METHODS=('A1L','P_ONLY','A2')
SEEDS=(4,5,6)
COLORS={'A1L':'#1764ab','P_ONLY':'#15845c','A2':'#d55e00'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(analysis,output):
    output.mkdir(parents=True,exist_ok=False)
    source=analysis/'comparison.json'
    data=json.loads(source.read_text(encoding='utf-8'))
    if data['status']!='COMPLETE' or data['training_seeds']!=list(SEEDS) or len(data['results'])!=27:
        raise ValueError('Require the complete frozen27-job confirmation analysis')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none'})
    cp=data['checkpoint_summaries']
    fig,axes=plt.subplots(1,4,figsize=(13,4.8))
    fig.subplots_adjust(left=.055,right=.99,bottom=.16,top=.71,wspace=.29)
    for ax,key,label in zip(axes,['position_mean_mm','shaft_max_deg','joint_state_max_excess_mrad','support512_score'],
                           ['Position mean (mm)','Shaft maximum (deg)','Actual-state excess (mrad)','Auxiliary support score512']):
        for seed,offset,marker in zip(SEEDS,(-.12,0,.12),('o','s','^')):
            values=[]
            for x,method in enumerate(METHODS):
                row=next(v for v in cp.values() if v['method']==method and v['training_seed']==seed)
                if not row['valid_three_processes']:
                    raise ValueError('Unqualified checkpoint cannot be plotted as valid')
                v=row['metrics'][key]
                ax.plot([x+offset,x+offset],[v['min'],v['max']],color=COLORS[method],alpha=.7)
                ax.scatter(x+offset,v['mean'],color=COLORS[method],marker=marker,s=30)
                values.append((x+offset,v['mean']))
            ax.plot(*np.array(values).T,color='.5',alpha=.35,linewidth=.7)
        ax.set_xticks(range(3),METHODS);ax.set_title(label);ax.grid(alpha=.18)
        if key=='support512_score':
            ax.text(.02,.03,'Y-axis zoomed; full score range0–1',transform=ax.transAxes,fontsize=7)
    legend=[Line2D([],[],color='.4',marker=marker,linestyle='none',label=f'Training seed{seed}')
            for seed,marker in zip(SEEDS,('o','s','^'))]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.865),ncol=3)
    fig.suptitle('Method N=3 training seeds; each marker is a three-process mean\nBars=min–max of those processes, not CI; connections are descriptive',y=.99)
    for ext in ('png','svg'):
        fig.savefig(output/f'seed_summary_annotated.{ext}',dpi=175,bbox_inches='tight',pad_inches=.18)
    plt.close(fig)

    criterion=data['results'][0]['directory']
    ref=json.loads((Path(criterion)/'reference.json').read_text(encoding='utf-8-sig'))
    gate_mm=1000*ref['task_acceptance']['penetration_max_m']
    records=[];input_hashes={};latches=[];aggregate_error=0.
    for row in data['results']:
        folder=Path(row['directory']);path=folder/'rollout.npz'
        if sha(path)!=row['hashes']['rollout.npz']:
            raise ValueError('Frozen raw rollout changed')
        with np.load(path,allow_pickle=False) as z:
            a={k:z[k].copy() for k in ('time_s','hand_object_penetration_m','hand_self_penetration_m','other_penetration_m','penetration_m','terminated')}
        maximum=np.maximum.reduce([a[k] for k in ('hand_object_penetration_m','hand_self_penetration_m','other_penetration_m')])
        aggregate_error=max(aggregate_error,float(np.abs(maximum-a['penetration_m']).max()))
        metadata=json.loads((folder/'rollout.json').read_text(encoding='utf-8-sig'))
        records.append({'row':row,'arrays':a});input_hashes[row['job_id']]=sha(path)
        if row['model_id']=='P_ONLY_trainseed4':
            latches.append({'job_id':row['job_id'],'frames':len(a['time_s']),'terminated_latch_samples':int(a['terminated'].sum()),
                            'rollout_censored':metadata['rollout_censored'],'first_failure':metadata['first_failure']})
    if aggregate_error>1e-10:
        raise ValueError('Depth components do not reproduce frozen aggregate depth')
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey='row')
    fig.subplots_adjust(left=.075,right=.99,bottom=.10,top=.79,wspace=.12,hspace=.22)
    for col,seed in enumerate(SEEDS):
        axes[0,col].set_title(f'Training seed{seed}')
        for method in METHODS:
            group=[v for v in records if v['row']['training_seed']==seed and v['row']['trained_method']==method]
            if len(group)!=3:
                raise ValueError('Need all3processes per checkpoint')
            for line,(key,label) in enumerate([('hand_object_penetration_m','Hand-object depth (mm)'),('hand_self_penetration_m','Hand self depth (mm)')]):
                ax=axes[line,col]
                for item in group:
                    ax.plot(item['arrays']['time_s'],1000*item['arrays'][key],color=COLORS[method],alpha=.22,linewidth=.8)
                ax.plot(group[0]['arrays']['time_s'],1000*np.mean([v['arrays'][key] for v in group],axis=0),color=COLORS[method],linewidth=1.5)
                ax.axhline(gate_mm,color='.2',linestyle=':',linewidth=.9)
                ax.grid(alpha=.15)
                if col==0:ax.set_ylabel(label)
        axes[-1,col].set_xlabel('Canonical time (s)')
    handles=[Line2D([],[],color=COLORS[m],label=m) for m in METHODS]
    handles.append(Line2D([],[],color='.2',linestyle=':',label=f'{gate_mm:g}mm aggregate-depth gate'))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.90),ncol=4)
    fig.suptitle('Penetration diagnostics: all27 full3s traces; thin=process, thick=three-process mean\n50Hz samples, not continuous-time safety; frozen task applies the3mm gate to the maximum category',y=.99)
    for ext in ('png','svg'):
        fig.savefig(output/f'penetration_diagnostic.{ext}',dpi=175,bbox_inches='tight',pad_inches=.18)
    plt.close(fig)
    report={'scope':'Artifact-only annotation/depth render; frozen analysis/source/criteria/results unchanged',
            'analysis_sha256':sha(source),'renderer_sha256':sha(__file__),'raw_rollout_hashes':input_hashes,
            'depth_component_aggregate_max_abs_error_m':aggregate_error,'depth_gate_mm':gate_mm,
            'P_ONLY_seed6_hand_object_peak_mm_by_repeat':{str(v['row']['evaluation_repeat']):float(1000*v['arrays']['hand_object_penetration_m'].max()) for v in records if v['row']['model_id']=='P_ONLY_trainseed6'},
            'complete_clip_explanation':'P_ONLY seed4 complete_clip=false is caused by the recorded terminated latch after shaft_angle_60deg, not missing/censored physics. All3 saved151states/150intervals continue to3s; raw latch/gates remain unchanged.',
            'P_ONLY_seed4_latches':latches,'figures_sha256':{name:sha(output/name) for name in ['seed_summary_annotated.png','seed_summary_annotated.svg','penetration_diagnostic.png','penetration_diagnostic.svg']},
            'GPU_used':False,'new_physics_steps':0,'visual_QA':'pending'}
    (output/'render.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();render(args.analysis,args.output)
