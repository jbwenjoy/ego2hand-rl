"""Author-side saved-output description and figures, NOT an independent audit.

Reads only saved JSON/NPZ; no Torch, checkpoint, objective, WM or native API.
All160 physical rows,8 predictions and320 comparison rows remain represented.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path('D:/workspace/supermimic-full151-observation-study')
CHANNELS=('raw_P','raw_S','BP','CS','A2')
LABELS=['A0/0','A0/49','A0/99','A0/149','A2/0','A2/49','A2/99','A2/149']
CHANNEL_LABEL={'raw_P':'Raw P (normalized penetration²)','raw_S':'Raw support match S','BP':'P_ONLY bonus: −0.1P','CS':'Weighted support: W·S','A2':'A2 total: W·S − 0.1P'}


def require(x,m):
    if not x:raise ValueError(m)


def pin(path):
    path=Path(path).resolve(strict=True);h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return {'path':str(path),'sha256':h.hexdigest()}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def write(out,name,v):
    with (out/name).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)


def csvwrite(path,rows):
    with path.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def savefig(fig,out,name,plt):
    fig.savefig(out/(name+'.png'),dpi=190,facecolor='white')
    fig.savefig(out/(name+'.svg'),facecolor='white');plt.close(fig)


def draw_levels(out,physical,predicted,plt):
    from matplotlib.lines import Line2D
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
    for ax,k in zip(axes,('CS','BP','A2')):
        y=physical[k]
        for s in range(8):
            values=y[8+s*19:8+(s+1)*19]
            ax.scatter(s+np.linspace(-.18,.18,19),values,c='#9ca3af',s=15,alpha=.8,zorder=2)
            ax.scatter(s+np.linspace(-.18,-.14,3),values[:3],c='#0f766e',marker='^',s=27,zorder=3)
        ax.scatter(np.arange(8),y[:8],s=52,c='#2563eb',edgecolors='white',linewidths=.6,zorder=4)
        ax.scatter(np.arange(8)+.055,predicted[k],s=49,c='#d94801',marker='D',edgecolors='white',linewidths=.6,zorder=5)
        ax.set_ylabel(CHANNEL_LABEL[k],fontsize=10);ax.grid(axis='y',color='.88',zorder=0)
        ax.axvline(3.5,color='.7',lw=1)
        if k=='BP':
            ax.set_ylim(-.03,.03);ax.set_yticks([-.02,0,.02]);ax.text(.02,.85,'P = 0 for all160 physical states and8 predictions',transform=ax.transAxes,fontsize=10)
        else:ax.set_ylim(-.045,1.045)
    axes[-1].set_xticks(range(8),LABELS);axes[-1].set_xlabel('Fresh replay snapshot / preaction phase k (nextstate scored with target k)')
    handles=[Line2D([],[],color='#9ca3af',marker='o',ls='',label='All152 physical branch states'),
             Line2D([],[],color='#0f766e',marker='^',ls='',label='24 baseline branches (included)'),
             Line2D([],[],color='#2563eb',marker='o',ls='',label='8 fresh physical donors'),
             Line2D([],[],color='#d94801',marker='D',ls='',label='8 fixed-A2 WM predictions')]
    fig.legend(handles=handles,ncol=2,loc='upper center',bbox_to_anchor=(.5,.945),frameon=False,fontsize=9)
    fig.suptitle('Saved bonus levels: penetration term inactive; A2 total equals support here',fontsize=14,y=.995)
    fig.text(.03,.015,'Actual trained64-basis functions, CPU FP32shadow. All rows retained, including far/dropped states. No force/contact truth or GPUAMP-gradient claim.',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.89));savefig(fig,out,'bonus_levels_all160_plus8',plt)


def draw_projections(out,rows,classes,plt):
    from matplotlib.lines import Line2D
    fig,axes=plt.subplots(2,3,figsize=(14,8.5));flat=axes.ravel()
    colors={0:'#2563eb',1:'#d94801'};markers={0:'o',1:'^'}
    for ax,k in zip(flat,CHANNELS):
        rr=[r for r in rows if r['channel']==k]
        maxabs=max([abs(r[z]) for r in rr for z in ('physical_slope','WM_AD_effective')]+[.02])*1.15
        for ei in (0,1):
            selected=[r for r in rr if r['epsilon_index']==ei]
            ax.scatter([r['physical_slope'] for r in selected],[r['WM_AD_effective'] for r in selected],
                       s=34,marker=markers[ei],c=colors[ei],alpha=.7,edgecolors='white',linewidth=.35,zorder=3)
        clipped=[r for r in rr if r['clipped']]
        ax.scatter([r['physical_slope'] for r in clipped],[r['WM_AD_effective'] for r in clipped],
                   s=76,marker='s',facecolors='none',edgecolors='#111827',linewidth=.9,zorder=4)
        ax.plot([-maxabs,maxabs],[-maxabs,maxabs],color='.65',lw=1,zorder=0)
        ax.axvline(0,color='.85',lw=.8);ax.axhline(0,color='.85',lw=.8)
        threshold=.1 if k=='raw_P' else .01
        ax.set_xscale('symlog',linthresh=threshold);ax.set_yscale('symlog',linthresh=threshold)
        ax.set_xlim(-maxabs,maxabs);ax.set_ylim(-maxabs,maxabs);ax.grid(color='.9',lw=.6)
        cr=[r for r in classes if r['channel']==k];stable=sum(r['label']=='STABLE_NONFLAT' for r in cr)
        below=sum(r['label']=='FLAT_OR_BELOW_NOISE' for r in cr)
        ax.set_title(CHANNEL_LABEL[k],fontsize=10)
        note=f'All64 rows /32 dependent groups\n{stable} stable; {below} below floor'
        if all(r['physical_slope']==0 and r['WM_AD_effective']==0 for r in rr):note+='\nAll64 coincide at (0,0)'
        ax.text(.035,.965,note,va='top',transform=ax.transAxes,fontsize=8,bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
        ax.set_xlabel('Empirical physical secant (component units/rad)',fontsize=8)
        ax.set_ylabel('Fixed-A2 WM AD · actual d_eff',fontsize=8)
    flat[-1].axis('off')
    flat[-1].text(0,.93,'Every channel keeps both eps and all32 groups.\n\n'
        'P and −0.1P have no above-floor signal.\nSupport and A2 share10 stable groups:\n8/10 signs at each eps, including3 clipped.\nUnclipped stable secondary:5/7 each eps.\n\n'
        'These are dependent repeats, not N20 or N64.\nSymlog axes expose nearzero and extremes.\nNo physical Jacobian / HO-force qualification.\nCPU shadow is not native GPUAMP gradient.',va='top',fontsize=10,linespacing=1.55)
    handles=[Line2D([],[],color=colors[i],marker=markers[i],ls='',label=f'eps={e:g} rad') for i,e in enumerate((.002,.001))]
    handles.append(Line2D([],[],color='#111827',marker='s',markerfacecolor='none',ls='',label='clipped endpoint (retained)'))
    fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.5,.948),frameon=False,fontsize=9)
    fig.suptitle('All320 saved secant/AD rows; no stable-only plotting',fontsize=14,y=.99)
    fig.tight_layout(rect=(0,.02,1,.9));savefig(fig,out,'bonus_secant_AD_all320',plt)


def draw_surface(out,physical,predicted,plt):
    fig,axes=plt.subplots(4,1,figsize=(12,10),sharex=True)
    metrics=[('minimum_sdf_m','Minimum sampled SDF (mm)',1000),('sample_count_abs_sdf_le_1mm','Samples with |SDF| ≤ 1mm',1),
             ('maximum_soft_weight','Maximum soft contact weight',1),('support_live_count_gt_1e_3','Live support directions /64',1)]
    for ax,(key,label,scale) in zip(axes,metrics):
        values=np.asarray(physical[key])*scale;pv=np.asarray(predicted[key])*scale
        for s in range(8):ax.scatter(s+np.linspace(-.18,.18,19),values[8+s*19:8+(s+1)*19],s=11,c='#9ca3af',alpha=.75)
        ax.plot(range(8),values[:8],c='#2563eb',marker='o',label='Physical donor8',ms=6,lw=.9)
        ax.plot(range(8),pv,c='#d94801',marker='D',label='WM prediction8',ms=5,lw=.9)
        ax.axvline(3.5,color='.7',lw=1);ax.grid(axis='y',color='.88');ax.set_ylabel(label,fontsize=9)
        if key=='minimum_sdf_m':
            ax.set_yscale('symlog',linthresh=3);ax.axhline(-3,color='#b91c1c',ls='--',lw=.8);ax.text(.02,.88,'Symlog: ±3mm linear region; all far states retained',transform=ax.transAxes,fontsize=9)
        if key=='maximum_soft_weight':ax.set_ylim(-.03,1.08)
        if key=='support_live_count_gt_1e_3':ax.set_ylim(-2,67)
    axes[0].legend(loc='upper right',ncol=2,frameon=False,fontsize=9)
    axes[-1].set_xticks(range(8),LABELS);axes[-1].set_xlabel('Snapshot / preaction k; 1344 fixed sampled hand points per row')
    fig.suptitle('Exact captured proxy outputs, not inference from position RMS',fontsize=14,y=.99)
    fig.text(.03,.015,'Gray points show all152 branch states. SDF/weights/support are sampled proxy quantities, not solver-active contact/force. No geometric calls were repeated.',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.96));savefig(fig,out,'proxy_surface_all160_plus8',plt)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    run=Path(args.run_dir).resolve(strict=True);out=Path(args.output).resolve()
    require(not out.exists(),'Fresh author-description directory required')
    out.mkdir(parents=True);(out/'source.py').write_bytes(Path(__file__).read_bytes())
    names=['report.json','physical_scores.npz','predicted_scores.npz','classifications.json','comparisons.json','statistics.json','arithmetic_diagnostics.json',
           'component_noise.json','physical_proxy_diagnostics.json','predicted_proxy_diagnostics.json','WM_predictions_gradients.npz','output_pins.json']
    before={name:pin(run/name) for name in names};savedpins=read(run/'output_pins.json')
    for n,v in before.items():
        if n!='output_pins.json':require(v['sha256']==savedpins[n]['sha256'],'Saved output pin mismatch: '+n)
    try:
        report=read(run/'report.json');require(report['status']=='COMPLETED_CPU_SHADOW_TRAINED_BONUS_SECANT_COMPARISON','Not completed')
        physical=load(run/'physical_scores.npz');pred=load(run/'predicted_scores.npz')
        rows=read(run/'comparisons.json');classes=read(run/'classifications.json');stats=read(run/'statistics.json')
        pd=read(run/'physical_proxy_diagnostics.json');wd=read(run/'predicted_proxy_diagnostics.json')
        require(set(physical)==set(pred)==set(CHANNELS) and all(v.shape==(160,) for v in physical.values()) and all(v.shape==(8,) for v in pred.values()),'Full160+8 score rows required')
        require(len(rows)==320 and len(classes)==160 and pd['rows']==160 and wd['rows']==8,'Full saved rows required')
        ids=np.concatenate((np.arange(8),np.repeat(np.arange(8),19)))
        physical_rows=[]
        for i in range(160):
            s=int(ids[i]);offset=None if i<8 else (i-8)%19
            kind='donor' if offset is None else 'baseline' if offset<3 else 'perturbation'
            physical_rows.append({'row':i,'kind':kind,'snapshot':LABELS[s],'source_action_path':LABELS[s].split('/')[0],'preaction_k':int(LABELS[s].split('/')[1]),
               **{k:float(physical[k][i]) for k in CHANNELS},'minimum_proxy_sdf_mm':1000*pd['minimum_sdf_m'][i],
               'near_surface_samples_abs_sdf_le_1mm':pd['sample_count_abs_sdf_le_1mm'][i],
               'max_softweight':pd['maximum_soft_weight'][i],'live_support64':pd['support_live_count_gt_1e_3'][i]})
        predicted_rows=[{'snapshot':LABELS[s],**{k:float(pred[k][s]) for k in CHANNELS},'minimum_proxy_sdf_mm':1000*wd['minimum_sdf_m'][s],
                        'near_surface_samples_abs_sdf_le_1mm':wd['sample_count_abs_sdf_le_1mm'][s],'max_softweight':wd['maximum_soft_weight'][s],
                        'live_support64':wd['support_live_count_gt_1e_3'][s]} for s in range(8)]
        csvwrite(out/'physical_all160.csv',physical_rows);csvwrite(out/'WM_all8.csv',predicted_rows)
        fields=['channel','snapshot_id','path','k','joint_index','epsilon_rad','physical_slope','WM_AD_effective','WM_AD_nominal_secondary','clipped','physical_label','numerator_floor','absolute_error','normalized_error','sign_agreement','comparison_eligible']
        csvwrite(out/'comparisons_all320.csv',[{k:r[k] for k in fields} for r in rows]);csvwrite(out/'all_statistics.csv',stats)
        class_counts={k:{l:sum(r['channel']==k and r['label']==l for r in classes) for l in ('STABLE_NONFLAT','FLAT_OR_BELOW_NOISE','NONFLAT_UNSTABLE')} for k in CHANNELS}
        factual={'rawP_all168_zero':bool(np.all(physical['raw_P']==0) and np.all(pred['raw_P']==0)),
                 'BP_all168_zero':bool(np.all(physical['BP']==0) and np.all(pred['BP']==0)),
                 'A2_equals_CS_all168':bool(np.array_equal(physical['A2'],physical['CS']) and np.array_equal(pred['A2'],pred['CS'])),
                 'rawS_equals_CS_all168':bool(np.array_equal(physical['raw_S'],physical['CS']) and np.array_equal(pred['raw_S'],pred['CS']))}
        # The prose and figure annotations below state these actual saved facts.
        require(all(factual.values()),'Annotations require these saved facts; do not silently generalize')
        for channel in ('CS','A2'):
            require(class_counts[channel]['STABLE_NONFLAT']==10 and class_counts[channel]['FLAT_OR_BELOW_NOISE']==22,'Fixed descriptive caption does not match data')
            require(all(r['sign_agreement_count']==8 and r['rows']==10 and r['clipped_rows']==3 for r in stats if r['channel']==channel and r['subset']=='stable_secondary'),'Caption sign counts mismatch')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
        draw_levels(out,physical,pred,plt);draw_projections(out,rows,classes,plt);draw_surface(out,pd,wd,plt)
        donor_lines=['|Snapshot|Physical support S|WM support S|Physical min SDF mm|WM min SDF mm|Physical / WM samples within 1mm|',
                     '|---|---:|---:|---:|---:|---:|']
        for s,label in enumerate(LABELS):donor_lines.append(f"|{label}|{physical['raw_S'][s]:.6g}|{pred['raw_S'][s]:.6g}|{1000*pd['minimum_sdf_m'][s]:.6g}|{1000*wd['minimum_sdf_m'][s]:.6g}|{pd['sample_count_abs_sdf_le_1mm'][s]} / {wd['sample_count_abs_sdf_le_1mm'][s]}|")
        md="""# Actual trained bonus: saved CPU-shadow diagnostic

This is author-side interpretation of the saved consumer outputs, not an independent mathematical audit. No model, objective, geometry query or forward was repeated. Contact's independent saved audit is a separate prerequisite for final conclusions.

All160 physical nextstates (8 donors +24 baseline branches +128 perturbation branches),8 fixed-A2 WM nextstates, and320 component/epsilon comparison rows are retained. The functions are the actual training factories and R64-basis math; the calculation is CPU FP32shadow, not the original GPUAMP policy gradient. Scores use next20ms state with preaction memory k; the k149 transition remains active and there is no u150.

## What this batch can establish

Raw sampled penetration P and the actual P_ONLY bonus −0.1P are exactly0 at every one of168 scored nextstates. Their32 direction groups are all below the predeclared numerator floor. This batch contains no nonzero penetration response with which to validate penalty-gradient accuracy. It does not show that real trajectories never penetrate: the proxy samples geometry and these are eight selected donor phases, not all3 seconds.

Every A2 total equals weighted support here because P=0; phase weights are1. The support and A2 channels therefore have the same10 stable groups and22 below-floor groups, with no nonflat-unstable group. At each epsilon, saved sign agreement is8/10 among stable groups. Three of those10 groups have clipped endpoints; the unclipped stable secondary subset is5/7 at each epsilon. The repeated eps are not20 independent observations and the support/total duplicates are not extra evidence. All64 rows per channel remain in the plots and CSV.

The physical/WM support levels do differ, e.g. A0/0 about0.925/0.836 and A2/0 about0.360/0.598. The exact captured SDF/weight/support outputs below locate those differences. Aggregate position RMS versus1mm bandwidth alone is insufficient. At A2/49 both support values are0; A2/99 and149 are far from the proxy surface (minimum sampled distances about453mm physical versus441mm WM), remain in the data, and provide no support signal. These are not missing rows.

## Same-state proxy diagnostics

"""+'\n'.join(donor_lines)+"""

Distances are sampled proxy SDF, not measured solver-active contact. The allowance is3mm, the contact weighting bandwidth1mm, and the retained64 support directions use the original proxy mu0.1 representation. These are not native mu0.7/condim4 friction/force claims. Full SDF/weights/support captures remain in the original immutable output.

## Fixed analysis and limits

The original pre-score noise formulas, unit scaling, cancellation-protected A2 floor, effective FP32 control directions and all clipped flags are unchanged. RawP,rawS,weighted penalty,weighted support,total and derived dt0.02 quantities keep separate units. Do not infer a P_ONLY-versus-A2 training benefit from this local fixed-WM score comparison. The historical runs already differed before prior activation; Nsource1/Nseed1/Nego0 and the eight states are dependent.

Nearest-surface choice, sampled maxima, live-support threshold and hinge boundaries are nonsmooth. Secant/AD differences can combine WM error, proxy branches, endpoint quantization/clipping and unrecorded graph scratch. No unique physical Jacobian, wholehiddenstate, HO-force gradient, forceclosure or task-success qualification follows. The native self15 head is not HO contact.

Consumer invariance flags assert unchanged RNG/model/buffers/normalizers and inputs in its own runtime; those are source-backed assertions, not independent reconstruction of unsaved pre/post memory. This renderer only checks its saved input file hashes stay unchanged.

## Complete artifacts

"""
        for n in ('bonus_levels_all160_plus8.png','bonus_secant_AD_all320.png','proxy_surface_all160_plus8.png'):
            md+=f'![{n}]({(out/n).as_posix()})\n\n'
        for n in ('physical_all160.csv','WM_all8.csv','comparisons_all320.csv','all_statistics.csv'):
            md+=f'- [{n}]({(out/n).as_posix()})\n'
        (out/'diagnostic.md').write_text(md,encoding='utf-8')
        require(all(pin(run/n)==v for n,v in before.items()),'Saved scientific input file changed')
        require(all(m not in sys.modules for m in ('torch','warp','mujoco')),'Renderer imported a model/backend')
        result={'schema':'full151105_trained_bonus_saved_author_interpretation:1','status':'COMPLETED_SAVED_AUTHOR_INTERPRETATION_NOT_INDEPENDENT_AUDIT',
          'source':pin(__file__),'consumer_report':before['report.json'],'input_pins':before,'physical_rows':160,'WM_rows':8,'comparison_rows':320,'channels':list(CHANNELS),
          'class_counts':class_counts,'saved_factual_summary':factual,'runtime_invariance_evidence':'consumer assertions/source-backed, not independently remeasured memory',
          'own_scope':{'new_model_loads':0,'new_objective_or_geometry_or_WM_or_Actor_forward':0,'new_AD':0,'Env_physics_GPU':0,'Torch_Warp_MuJoCo_imported':False},
          'new_independent_numeric_qualification':False,'physical_gradient_qualified':False,'wholehiddenstate_qualified':False,'figure_visual_QA':'PENDING_ACTUAL_VIEW',
          'output_files':{p.name:pin(p) for p in out.iterdir() if p.is_file()}}
        write(out,'report.json',result)
        print(json.dumps({'report':pin(out/'report.json'),'markdown':pin(out/'diagnostic.md'),'figures':[str(p) for p in out.glob('*.png')]},indent=2))
    except BaseException as e:
        write(out,'failure.json',{'status':'FAILED_SAVED_AUTHOR_RENDER_PRESERVED','error':repr(e),'source':pin(__file__),'input_pins':before,'models_executed':False});raise


if __name__=='__main__':main()
