"""Scientific figure from the qualified saved-only fresh report; no models."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def pin(p):
    p=Path(p).resolve(strict=True);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def main():
    a=argparse.ArgumentParser();a.add_argument('--report',required=True);a.add_argument('--output',required=True);args=a.parse_args();p=Path(args.report).resolve(strict=True);r=json.loads(p.read_text(encoding='utf-8'))
    if not (r['schema']=='full151105_public_self15_fresh_gradient_saved_independent:1' and r['engineering_qualified'] is True and r['passed']==r['total']==11):raise ValueError('Qualified fixed saved11 report required')
    out=Path(args.output).resolve();out.mkdir(exist_ok=False);roles=['unwrapped_baseline','shared_public_losses','shared_public_plus_cross_target_consistency'];names=['Original A2 WM','Public + auxiliary','Public + auxiliary\n+ consistency'];colors=['#858b94','#2176b5','#d77825'];rows=[r['independent_summary'][k] for k in roles]
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(15,9));x=np.arange(3)
    def bars(ax,values,title,ylabel,log=False):
        ax.bar(x,values,color=colors,width=.66);ax.set_xticks(x,names,fontsize=8);ax.set_title(title,loc='left',fontweight='bold');ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        if log:ax.set_yscale('log');ax.set_ylim(min(values)*.6,max(values)*2.2)
        else:ax.set_ylim(0,max(values)*1.27)
        for j,v in enumerate(values):ax.annotate(f'{v:.4g}',(j,v),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
    bars(axes[0,0],[v['point_scaled_MSE_F64'] for v in rows],'A  Native52 endpoint error: all152','Normalized MSE (F64 rescore)',True)
    bars(axes[0,1],[v['common_pre_response64_MSE_F64'] for v in rows],'B  Common-pre response: all64','Normalized response MSE')
    bars(axes[0,2],[1000*v['raw58']['p']['vector_RMS'] for v in rows],'C  Object position: all152','Vector RMS error (mm)')
    bars(axes[1,0],[v['raw58']['omega']['vector_RMS'] for v in rows],'D  Object angular velocity: all152','Vector RMS error (rad/s)')
    ax=axes[1,1];cats=['All64\nactual coordinate','Stable6\neps=.002','Stable6\neps=.001'];yy=[]
    for v in rows:
        g=v['object_gradient'];yy.append([g['all_defined_actual_coordinate']['RMSE'],g['stable_physical_groups_per_epsilon']['0.002']['RMSE'],g['stable_physical_groups_per_epsilon']['0.001']['RMSE']])
    for j,values in enumerate(yy):ax.bar(x+(j-1)*.24,values,width=.23,color=colors[j])
    ax.set_xticks(x,cats,fontsize=8);ax.set_ylabel('Score/rad RMSE');ax.set_title('E  Physical secant vs captured uVJP',loc='left',fontweight='bold');ax.set_ylim(0,np.max(yy)*1.22);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for j,values in enumerate(yy):
        for k,v in enumerate(values):ax.annotate(f'{v:.3f}',(k+(j-1)*.24,v),xytext=(0,4),textcoords='offset points',ha='center',fontsize=7)
    values=[v['cached15_ALL']['BCE_mean_elements_F64'] for v in rows];bars(axes[1,2],values,'F  Cached native15 labels: all152','Mean BCE per sensor element')
    for j,v in enumerate(rows):
        s=v['cached15_ALL'];axes[1,2].text(j,.015,f"TP={s['TP']}\nFN={s['FN']}",ha='center',va='bottom',fontsize=8,color='white' if j==0 else '#202020')
    fig.suptitle('Public-self15 fixed AFTER1000: fresh phase diagnostics',fontsize=16,fontweight='bold',y=.985)
    fig.text(.5,.947,'Same known donor lineage; phases 24 / 74 / 124 / 148. Three fixed models; no new training or physics.',ha='center',fontsize=10)
    fig.text(.035,.052,'Endpoint inputs are snapshot8 repeated19: a common-visible design, not qualified actual branch PRE152.\nPhysical groups: 6 stable / 25 weak / 1 unstable; all64 pairs retained. Stable6 repeats across eps are dependent.\nObject score contains position2 + shaft8 only. Cached sensor truth is ~15ms; state targets are POST20. No physical-Jacobian or Actor benefit claim.',fontsize=9,va='bottom')
    fig.subplots_adjust(left=.065,right=.98,top=.895,bottom=.19,wspace=.34,hspace=.52);path=out/'fixed_public15_fresh_diagnostics.png';fig.savefig(path,dpi=180);fig.savefig(out/'fixed_public15_fresh_diagnostics.pdf');plt.close(fig)
    with (out/'manifest.json').open('x',encoding='utf-8') as f:json.dump({'schema':'full151105_public_self15_fresh_saved_figure:1','source':pin(__file__),'saved_independent_report':pin(p),'PNG':pin(path),'PDF':pin(out/'fixed_public15_fresh_diagnostics.pdf'),'new_model_FW_AD_physics':0},f,indent=2)
    print(json.dumps({'figure':pin(path),'manifest':pin(out/'manifest.json')}))
if __name__=='__main__':main()
