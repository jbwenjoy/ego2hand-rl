"""H8 scientific curves from the qualified saved-only report and frozen JSON."""
import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def pin(p):
    p=Path(p).resolve(strict=True);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--review',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();rp=Path(a.review).resolve(strict=True);r=json.loads(rp.read_text())
    if not (r['schema']=='full151105_public_self15_H8_donor_saved_independent:1' and r['engineering_qualified'] is True and r['passed']==r['total']):raise ValueError('Qualified saved H8 review required')
    pp=r['producer_report'];p=Path(pp['path']).resolve(strict=True)
    if pin(p)['sha256']!=pp['sha256']:raise ValueError('Producer changed')
    run=p.parent;s=json.loads((run/'all432_state_statistics.json').read_text());c=json.loads((run/'all72_cached_self15_statistics.json').read_text());out=Path(a.output).resolve();out.mkdir(exist_ok=False)
    roles=['unwrapped_baseline','shared_public_losses','shared_public_plus_cross_target_consistency'];names=['Original A2 WM','Public + auxiliary','Public + auxiliary + consistency'];colors=['#858b94','#2176b5','#d77825'];h=list(range(1,9));fig,axes=plt.subplots(2,3,figsize=(15,8.8))
    panels=[('p','Vector RMS error (mm)','A  Object position',1000),('R_raw9','Frobenius RMS (dimensionless)','B  Raw rotation matrix',1),('omega','Vector RMS error (rad/s)','C  Object angular velocity',1)]
    for role,label,color in zip(roles,names,colors):
        for j,(block,ylabel,title,factor) in enumerate(panels):
            values=[factor*next(q['vector_or_Frobenius_RMS'] for q in s if q['model_role']==role and q['path']=='ALL' and q['horizon']==k and q['block']==block) for k in h];axes[0,j].plot(h,values,'o-',color=color,label=label,linewidth=2,markersize=4)
        for j,key in enumerate(['BCE_mean_entry_F64','positive_recall','positive_precision']):
            values=[next(q[key] for q in c if q['model_role']==role and q['cohort']=='ALL' and q['horizon']==k) for k in h];axes[1,j].plot(h,values,'o-',color=color,label=label,linewidth=2,markersize=4)
    for j,(_,ylabel,title,_) in enumerate(panels):axes[0,j].set_ylabel(ylabel);axes[0,j].set_title(title,loc='left',fontweight='bold')
    for j,(title,ylabel) in enumerate([('D  Cached native15 BCE','Mean BCE per element'),('E  Positive recall','TP / (TP + FN)'),('F  Positive precision','TP / (TP + FP)')]):axes[1,j].set_title(title,loc='left',fontweight='bold');axes[1,j].set_ylabel(ylabel)
    for ax in axes.flat:ax.set_xlabel('Prediction step h');ax.set_xticks(h);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False);ax.set_xlim(.8,8.2);ax.set_ylim(bottom=0)
    axes[1,1].set_ylim(0,1);axes[1,2].set_ylim(0,1)
    fig.suptitle('Public-self15 fixed AFTER1000: donor H8 predictions',fontsize=16,fontweight='bold',y=.985);handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.955),ncol=3,frameon=False)
    fig.text(.055,.043,'ALL: 286 overlapping windows from two known donor paths. Each h uses recorded controls and its own previous predicted state.\nState targets are native POST20; sensor labels are cached lastsolver ~15ms. H>1 model states differ; label errors include rollout drift.\nTarget-phase populations change with h; a falling BCE curve does not prove greater long-horizon accuracy.\nNo clamping, state forcing, new physics or training. This is not Actor imagination, independent-trajectory generalization or task success.',fontsize=9,va='bottom');fig.subplots_adjust(left=.07,right=.98,top=.87,bottom=.19,wspace=.32,hspace=.43)
    png=out/'fixed_public15_H8_diagnostics.png';fig.savefig(png,dpi=180);fig.savefig(out/'fixed_public15_H8_diagnostics.pdf');plt.close(fig)
    with (out/'manifest.json').open('x',encoding='utf-8') as f:json.dump({'schema':'full151105_public_self15_H8_saved_figure:1','source':pin(__file__),'saved_review':pin(rp),'producer_report':pp,'state_statistics':pin(run/'all432_state_statistics.json'),'cached_statistics':pin(run/'all72_cached_self15_statistics.json'),'PNG':pin(png),'PDF':pin(out/'fixed_public15_H8_diagnostics.pdf'),'new_model_FW_AD_physics':0},f,indent=2)
    print(json.dumps({'figure':pin(png),'manifest':pin(out/'manifest.json')}))
if __name__=='__main__':main()
