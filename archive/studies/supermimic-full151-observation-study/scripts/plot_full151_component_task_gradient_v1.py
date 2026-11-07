"""All64 exploratory score-gradient comparisons; saved data only."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS=('pointwise','pointwise_plus_common_pre_response')
LABELS=('Pointwise only','Pointwise + response')
COLORS=('#2675b8','#d77826')


def pin(p):return {'path':str(Path(p).resolve(strict=True)),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--independent-review',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();base=a.run_dir.resolve(strict=True);out=a.output.resolve()
    report=json.loads((base/'report.json').read_text(encoding='utf8'))
    review=json.loads(a.independent_review.read_text(encoding='utf8'))
    assert review['passed']==review['total']==27 and all(v is True for v in review['checks'].values())
    data={arm:json.loads((base/(arm+'_comparison64.json')).read_text(encoding='utf8')) for arm in ARMS}
    assert all(len(v)==64 for v in data.values()) and not out.exists();out.mkdir()
    modes=[('physical_actual_coordinate_secant','WM_AD_actual_delta_projected_per_coordinate','Actual control delta: FD vs baseline u-gradient projection'),
           ('physical_legacy_nominal_epsilon_slope','WM_AD_legacy_effective_direction','Nominal epsilon: FD vs gradient dot legacy d_eff')]
    fig,axes=plt.subplots(2,2,figsize=(15,11),layout='constrained')
    exported=[]
    limit=max(abs(r[k]) for rows in data.values() for r in rows for mode in modes for k in mode[:2])*1.08
    limit=max(limit,1e-12)
    for ax,(xkey,ykey,title) in zip(axes[0],modes):
        for arm,label,color in zip(ARMS,LABELS,COLORS):
            for ei,marker in ((0,'o'),(1,'x')):
                rows=[r for r in data[arm] if r['epsilon_index']==ei]
                assert len(rows)==32
                ax.scatter([r[xkey] for r in rows],[r[ykey] for r in rows],s=25,marker=marker,c=color,alpha=.75,label=f'{label}, eps={.002 if ei==0 else .001}')
            for r in data[arm]:
                exported.append({'arm':arm,'mode':xkey,'pair_id':r['pair_id'],'snapshot_id':r['snapshot_id'],'path':r['path'],'k':r['k'],'j':r['j'],'epsilon_index':r['epsilon_index'],'physical_FD_score_per_rad':r[xkey],'model_AD_score_per_rad':r[ykey],'physical_group_label':r['physical_group_label'],'clipped':r['clipped']})
        ax.plot([-limit,limit],[-limit,limit],c='gray',ls='--',lw=1,label='Identity')
        ax.axhline(0,c='gray',lw=.4);ax.axvline(0,c='gray',lw=.4)
        ax.set_xlim(-limit,limit);ax.set_ylim(-limit,limit);ax.set_aspect('equal',adjustable='box')
        ax.set_title(title,fontsize=11,loc='left');ax.set_xlabel('Saved physical finite chord [score/rad]');ax.set_ylabel('Model input AD projection [score/rad]')
    axes[0,0].legend(fontsize=8,loc='upper left')
    ax=axes[1,0];epskeys=('0.002','0.001');x=np.arange(2)
    for i,(arm,label,color) in enumerate(zip(ARMS,LABELS,COLORS)):
        values=[report['comparison_summaries'][arm]['stable_physical_groups_per_epsilon'][eps]['RMSE'] for eps in epskeys]
        bars=ax.bar(x+(i-.5)*.32,values,width=.32,label=label,color=color)
        ax.bar_label(bars,labels=[f'{v:.5f}' for v in values],padding=4,fontsize=10)
    ax.set_xticks(x,['epsilon 0.002 rad','epsilon 0.001 rad']);ax.set_ylim(bottom=0,top=1.15)
    ax.set_ylabel('Legacy projected slope RMSE [score/rad]');ax.set_title('Shared physical stable subset: N=6 directions, both eps shown',fontsize=11,loc='left');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=9)
    ax=axes[1,1];keys=('FLAT_OR_BELOW_NOISE','STABLE_NONFLAT','NONFLAT_UNSTABLE')
    counts=[report['physical_group_counts'].get(k,0) for k in keys];assert sum(counts)==32
    bars=ax.bar(['Below old signal floor','Stable nonflat','Unstable nonflat'],counts,color=['#a3acb5','#549e77','#c7885e'])
    ax.bar_label(bars,padding=4,fontsize=12);ax.set_ylim(0,30);ax.set_ylabel('Shared physical direction groups [N=32]');ax.set_title('All physical groups retained for both models',fontsize=11,loc='left');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Exploratory task-score gradient: full64 and shared physical-subset results differ',fontsize=15)
    fig.text(.5,-.022,'Each epsilon contributes 32 rows; paired epsilon results are dependent and do not turn N=6 into N=12.\nOld floor/stability labels are empirical descriptors, not a noise bound or a true-Jacobian certificate. Original confirmation FAIL remains unchanged.',ha='center',fontsize=10)
    fig.savefig(out/'all64_task_score_gradient_comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)
    with (out/'all64_both_modes_both_arms.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(exported[0]));w.writeheader();w.writerows(exported)
    result={'schema':'full151105_component_task_gradient_figures:1','producer':pin(base/'report.json'),'independent_review':pin(a.independent_review),'renderer':pin(__file__),
            'all64_rows_each_arm_and_mode':True,'physical_shared_groups':32,'physical_stable_groups':6,'two_eps_do_not_double_independent_N':True,
            'parent_confirmation_decision':report['parent_ea22_decision_unchanged'],'new_model_FW_AD_physics_GPU':0,'outputs':{p.name:pin(p) for p in out.iterdir()}}
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({'figure':str(out/'all64_task_score_gradient_comparison.png'),'all64_per_arm_mode':True,'stable_direction_N':6}))


if __name__=='__main__':main()
