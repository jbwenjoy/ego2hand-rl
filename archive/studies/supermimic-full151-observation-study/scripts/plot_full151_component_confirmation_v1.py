"""All152/all64 saved confirmation figures, with original units kept separate."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS=('pointwise','pointwise_plus_common_pre_response')
LABELS=('Pointwise only','Pointwise + response')
COLORS=('#2675b8','#d77826')


def pin(p):
    p=Path(p).resolve(strict=True)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--independent-review',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();base=a.run_dir.resolve(strict=True);out=a.output.resolve()
    report=json.loads((base/'report.json').read_text(encoding='utf8'))
    review=json.loads(a.independent_review.read_text(encoding='utf8'))
    assert review['passed']==review['total'] and all(v is True for v in review['checks'].values())
    assert report['status']=='COMPLETED_ALL152_ALL64_COMPONENT_CONFIRMATION_NOT_POLICY_BENEFIT'
    rows=json.loads((base/'all152_identities.json').read_text(encoding='utf8'))
    pairs=json.loads((base/'all64_pairs.json').read_text(encoding='utf8'))
    assert len(rows)==152 and len(pairs)==64 and not out.exists();out.mkdir()
    arrays={arm:load(base/(arm+'_evaluation_arrays.npz')) for arm in ARMS}
    pair_rms={};tables=[]
    for arm in ARMS:
        z=arrays[arm];mi,pl=z['minus_indices'],z['plus_indices']
        delta=(z['pred_common52_FP32'][pl].astype('f8')-z['pred_common52_FP32'][mi].astype('f8'))-(z['target_common52_FP32'][pl].astype('f8')-z['target_common52_FP32'][mi].astype('f8'))
        residual=(delta/z['scale52_FP32'].astype('f8'))/z['actual_du_rad_F64'][:,None]
        pair_rms[arm]=np.sqrt(np.mean(residual*residual,axis=1))
        assert pair_rms[arm].shape==(64,)
        for p,value in zip(pairs,pair_rms[arm]):
            tables.append({'arm':arm,**{k:p[k] for k in ('pair_id','snapshot_id','path','k','j','epsilon_index','minus_branch_id','plus_branch_id','actual_du_rad','clipped')},'normalized_response_residual_component_RMS':float(value)})
    with (out/'all64_pairs_both_arms.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(tables[0]));w.writeheader();w.writerows(tables)
    fig=plt.figure(figsize=(16,9),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=(1,1.45))
    for ax,key,title in ((fig.add_subplot(gs[0,0]),'common_pre_pair_loss','All64 normalized finite-response MSE'),(fig.add_subplot(gs[0,1]),'pointwise_loss','All152 native normalized pointwise loss')):
        values=[report['native52_frozen_criterion_metrics'][arm][key] for arm in ARMS]
        bars=ax.bar(LABELS,values,color=COLORS,width=.6)
        ax.bar_label(bars,labels=[f'{v:.7g}' for v in values],padding=5,fontsize=11)
        ax.set_title(title,loc='left');ax.set_ylim(0,max(values)*1.2);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax=fig.add_subplot(gs[1,:])
    for arm,label,color in zip(ARMS,LABELS,COLORS):
        ax.scatter(np.arange(64),pair_rms[arm],s=20,label=label,color=color,alpha=.8)
    for s in range(1,8):ax.axvline(s*8-.5,color='gray',lw=.6,alpha=.5)
    phase_labels=[f"{pairs[s*8]['path']} / k={pairs[s*8]['k']}" for s in range(8)]
    ax.set_xticks(np.arange(8)*8+3.5,phase_labels);ax.set_xlim(-1,64);ax.set_ylim(bottom=0)
    ax.set_ylabel('Per-pair normalized response residual RMS [52 components]')
    ax.set_xlabel('All64 pairs in original snapshot / joint / epsilon order');ax.grid(axis='y',alpha=.2);ax.legend()
    fig.suptitle('Frozen confirmation: response MSE improves; pointwise loss worsens — criterion NOT met',fontsize=15)
    fig.text(.5,-.025,'All152 rows and all64 clipped/asymmetric pairs retained. Response uses the common-PRE native52 chart and actual control delta / (1 rad).\nSingle dependent cache lineage; no policy/physical-gradient or unseen-state claim.',ha='center',fontsize=10)
    fig.savefig(out/'all64_response_and_frozen_criterion.png',dpi=160,bbox_inches='tight');plt.close(fig)
    specs=[('q',0,20,'Joint q row component RMS [mrad]',1000,False),
           ('qd',20,40,'Joint velocity row component RMS [rad/s]',1,False),
           ('p',40,43,'Object position error norm [mm]',1000,True),
           ('R_raw9',43,52,'Raw rotation-matrix error Frobenius norm [dimensionless]',1,True),
           ('v',52,55,'Object linear-velocity error norm [m/s]',1,True),
           ('omega',55,58,'Object angular-velocity error norm [rad/s]',1,True)]
    fig,axes=plt.subplots(3,2,figsize=(17,11),layout='constrained');all_rows=[]
    labels=[f"{rows[s*19]['path']} / k={rows[s*19]['k']}" for s in range(8)]
    for ax,(name,lo,hi,title,factor,norm) in zip(axes.flat,specs):
        for arm,label,color in zip(ARMS,LABELS,COLORS):
            z=arrays[arm];err=z['pred152_FP32'].astype('f8')-z['target152_FP32'].astype('f8')
            assert np.array_equal(err,z['state_error58_F64'])
            block=err[:,lo:hi];values=np.sqrt((np.sum if norm else np.mean)(block*block,axis=1))*factor
            ax.scatter(np.arange(152),values,s=9,label=label,c=color,alpha=.7)
            for r,value in zip(rows,values):all_rows.append({'arm':arm,'branch_id':r['row_id'],'snapshot_id':r['snapshot_id'],'path':r['path'],'k':r['k'],'block':name,'figure_value':float(value),'figure_unit':title})
        for s in range(1,8):ax.axvline(s*19-.5,c='gray',lw=.5,alpha=.4)
        ax.set_xticks(np.arange(8)*19+9,labels,fontsize=8);ax.set_xlim(-2,153);ax.set_ylim(bottom=0)
        ax.set_title(title,loc='left',fontsize=11);ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=9)
    fig.suptitle('All152 saved confirmation predictions: six original-unit error blocks',fontsize=15)
    fig.text(.5,-.02,'Each snapshot has three baseline repeats plus sixteen perturbation endpoints. Points are separate restored branches; no temporal curve is inferred.\nR9 is a raw matrix error, separate from the native Log3 coordinates used by the normalized response criterion.',ha='center',fontsize=10)
    fig.savefig(out/'all152_state_errors_original_units.png',dpi=160,bbox_inches='tight');plt.close(fig)
    with (out/'all152_original_unit_plot_values.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    metrics=report['native52_frozen_criterion_metrics'];a0,a1=(metrics[a] for a in ARMS)
    change={k:100*(a1[k]/a0[k]-1) if a0[k] else None for k in ('pointwise_loss','common_pre_pair_loss')}
    result={'schema':'full151105_component_confirmation_figures:1','producer':pin(base/'report.json'),'independent_review':pin(a.independent_review),
            'renderer':pin(__file__),'all152_rows':True,'all64_pairs':True,'metric_percent_change_response_vs_pointwise':change,
            'criterion':report['decision'],'outputs':{p.name:pin(p) for p in out.iterdir()},'new_model_FW_AD_nativeLog_physics_GPU':0}
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({'output':str(out),'figures':2,'metric_percent_change':change}))


if __name__=='__main__':main()
