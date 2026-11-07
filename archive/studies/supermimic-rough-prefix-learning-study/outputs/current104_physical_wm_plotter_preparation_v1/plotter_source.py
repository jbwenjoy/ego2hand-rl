"""Saved-only figures for the frozen physical-secant/three-WM CPU comparison.

Default validates saved artifacts only. --run renders once to a fresh output.
No model import, model forward, simulator, Torch, Warp, fitting or resampling.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,shutil,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
METHODS=('A1L','P_ONLY','A2')
PATHS=('A0','A1L','P_ONLY','A2')
PHASES=(0,4,8,12)
EPS=(.002,.001)
LABELS=('STABLE_NONFLAT','NONFLAT_UNSTABLE','FLAT_OR_BELOW_NOISE','NOT_RUN')
COLORS={'STABLE_NONFLAT':'#2468a0','NONFLAT_UNSTABLE':'#dc8b29','FLAT_OR_BELOW_NOISE':'#9a9a9a','NOT_RUN':'#e5dce9'}
SHORT={'STABLE_NONFLAT':'Stable nonflat','NONFLAT_UNSTABLE':'Nonflat, unstable','FLAT_OR_BELOW_NOISE':'Flat / below noise','NOT_RUN':'Not run'}
CONSUMER_SHA='d89eae2f2712fa00e176048c659fe36eac404a2c81212f52c05efeda6f6199e7'
MATH_SHA='bbc5cda303bcaae75af54f339c96d06e52a0b1eda726bf74ce684a995e10f3ab'


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(Path(p).resolve(strict=True)),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'),parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def checked(p):
    need(type(p) is dict and set(p)=={'path','sha256'} and Path(p['path']).is_absolute() and sha(p['path'])==p['sha256'],'Changed saved input/source pin')
    return Path(p['path'])


def finite(x):return type(x) in (int,float) and math.isfinite(x)


def csv_write(path,rows):
    need(bool(rows),'Refuse empty table without schema')
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def load_saved(directory):
    """Read actual completed artifacts only; never synthesize future results."""
    directory=Path(directory).resolve(strict=True)
    names=('report.json','comparisons.json','descriptive_agreement.json','physical_secants.json',
           'baseline_recomputed.json','WM_predictions_gradients.npz','self15_A1L.json','self15_P_ONLY.json','self15_A2.json')
    pins={name:pin(directory/name) for name in names}
    report=read(directory/'report.json')
    need(report.get('schema')=='current104_physical_probe_CPU_consumption:2' and report.get('status')=='COMPLETED_CPU_SHARED_PHYSICAL_SECANT_COMPARISON','Only completed frozen consumer_v2 result can be rendered')
    need(report.get('entry',{}).get('sha256')==CONSUMER_SHA and sha(checked(report['entry']))==CONSUMER_SHA,'Unexpected consumer source')
    plan=read(checked(report['runplan']))
    need(plan['source_pins']['math']['sha256']==MATH_SHA and sha(checked(plan['source_pins']['math']))==MATH_SHA,'FD classification formula changed')
    for value in plan['source_pins'].values():checked(value)
    checked(report['physical_manifest']);checked(report['proposal'])
    need(report.get('all3_WM_completed') is True and report.get('WM_forwards')==3 and report.get('GPU_used') is False and report.get('physics_steps_added')==0,'Unexpected inference/physics scope')
    need(report.get('source_lineage_N')==1 and report.get('independent_ego_N')==0 and report.get('RNG_unchanged') is True,'Source/RNG scope changed')
    need(report.get('model_statuses')=={m:'COMPLETED' for m in METHODS} and report.get('independent_baseline_all_passed') is True,'Missing model/baseline qualification')
    semantic=report.get('saved_evidence_semantic_qualification',{})
    need(semantic.get('qualified') is True and semantic.get('capacity_semantics',{}).get('qualified') is True and semantic.get('lifecycle_semantics',{}).get('qualified') is True,'Physical raw-evidence gate not qualified')
    baseline=read(directory/'baseline_recomputed.json')
    need(baseline.get('all48_completed') is True and baseline.get('all16_passed') is True and len(baseline['snapshots'])==16,'Missing48 baseline evidence')
    secants=read(directory/'physical_secants.json');comparisons=read(directory/'comparisons.json');groups=read(directory/'descriptive_agreement.json')
    need(type(secants) is list and len(secants)==128 and type(comparisons) is list and len(comparisons)==768,'Expected128 shared direction slots and3x256 comparisons')
    keyed={}
    for row in secants:
        s,j=row['snapshot_id'],row['j']
        need(type(s) is int and 0<=s<16 and type(j) is int and 0<=j<8 and (s,j) not in keyed,'Duplicate/invalid physical direction')
        need(row['path']==PATHS[s//4] and row['k']==PHASES[s%4] and row['classification'] in LABELS,'Physical path/phase/class mismatch')
        need(type(row['complete']) is bool and type(row['stable_nonflat']) is bool and row['stable_nonflat']==(row['classification']=='STABLE_NONFLAT'),'Physical classification mismatch')
        need(row['complete']==(row['classification']!='NOT_RUN') and len(row['slopes'])==2,'Physical completion/epsilon mismatch')
        need(all(finite(x) for x in row['slopes']) if row['complete'] else all(x is None for x in row['slopes']),'Missing/fabricated slope')
        keyed[s,j]=row
    seen=set()
    for row in comparisons:
        key=(row['model'],row['snapshot_id'],row['j'],row['epsilon_rad'])
        need(key[0] in METHODS and key[3] in EPS and key not in seen,'Unknown/duplicate WM comparison')
        seen.add(key);physical=keyed[key[1],key[2]];e=EPS.index(key[3])
        need(row['path']==physical['path'] and row['k']==physical['k'] and row['classification']==physical['classification'] and row['stable_nonflat']==physical['stable_nonflat'],'WM row not paired with shared physical slot')
        need(row['physical_slope']==physical['slopes'][e] and row['status']==('COMPLETED' if physical['complete'] else 'NOT_RUN'),'WM physical slope/status mismatch')
        need(finite(row['AD_effective_FP32']) and finite(row['AD_nominal_secondary']),'Nonfinite saved AD')
        need(row['sign_match']==((row['AD_effective_FP32']>0)-(row['AD_effective_FP32']<0)==(row['physical_slope']>0)-(row['physical_slope']<0)) if row['stable_nonflat'] else row['sign_match'] is None,'Saved stable sign indicator differs')
    # Recount descriptive signs only. Neither gradients nor physical FD are rerun.
    sign_rows=[]
    for method in METHODS:
        for eps in EPS:
            rows=[v for v in comparisons if v['model']==method and v['epsilon_rad']==eps]
            eligible=[v for v in rows if v['stable_nonflat']]
            sign=sum(v['sign_match'] for v in eligible)/len(eligible) if eligible else None
            saved=[v for v in groups if v['model']==method and v['epsilon_rad']==eps and v['path']=='ALL_DEPENDENT_PATHS' and v['k'] is None]
            need(len(saved)==1 and saved[0]['planned_rows']==128 and saved[0]['eligible_rows']==len(eligible) and saved[0]['sign_agreement']==sign,'Sign coverage aggregation mismatch')
            sign_rows.append(dict(model=method,epsilon_rad=eps,planned_dependent_direction_slots=128,eligible_stable_nonflat=len(eligible),
                                  sign_match_count=sum(v['sign_match'] for v in eligible),sign_agreement_percent=100*sign if sign is not None else None,
                                  saved_descriptive_Pearson_r=saved[0]['pearson_r']))
    coverage=[]
    for path in PATHS:
        for label in LABELS:
            coverage.append(dict(path=path,classification=label,count=sum(v['path']==path and v['classification']==label for v in secants),planned_dependent_directions=32))
    self_rows=[];truth_count=None
    for method in METHODS:
        cal=read(directory/('self15_'+method+'.json'))
        need(cal.get('native_self15_only') is True and cal.get('fresh_donor_truth')=='fixed_primary_no_majority_vote' and cal.get('dependent_snapshot_count')==16 and cal.get('binary_slots')==240,'Native self15 calibration identity differs')
        n=cal['true_positive_slots'];need(type(n) is int and 0<=n<=240 and (truth_count is None or truth_count==n),'WM models did not use the same fresh truth')
        truth_count=n
        need(finite(cal['BCE']) and finite(cal['Brier']) and cal['BCE']>=0 and 0<=cal['Brier']<=1,'Invalid calibration metric')
        absent=n in (0,240)
        self_rows.append(dict(model=method,dependent_snapshots=16,native_binary_slots=240,truth_positive_slots=n,
            predicted_positive_slots=cal['predicted_positive_slots'],BCE=cal['BCE'],Brier=cal['Brier'],
            raw_consumer_F1=cal['F1'],display_F1=None if absent else cal['F1'],
            F1_display_note='N/A: ground-truth class absent; raw consumer F1 retained separately' if absent else 'Threshold0.5 descriptive only; dependent slots'))
    # Read frozen prediction/gradient arrays only to validate their finite ABI.
    import numpy as np
    expected={m+'__'+k:shape for m in METHODS for k,shape in [('prediction',(16,58)),('self15_logits',(16,15)),('gradient_refk',(16,20)),('gradient_refkplus1',(16,20))]}
    with np.load(directory/'WM_predictions_gradients.npz',allow_pickle=False) as z:
        need(set(z.files)==set(expected),'Unexpected prediction/gradient fields')
        for key,shape in expected.items():need(z[key].shape==shape and z[key].dtype==np.float32 and np.isfinite(z[key]).all(),'Saved tensor ABI/finite mismatch '+key)
    return report,pins,comparisons,secants,coverage,sign_rows,self_rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=ROOT/'outputs/current104_physical_v2_WM_CPU_v1')
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/current104_physical_v2_WM_summary_v1')
    ap.add_argument('--run',action='store_true');args=ap.parse_args()
    report,pins,rows,secants,coverage,sign_rows,self_rows=load_saved(args.input)
    need(not any(k in sys.modules for k in ('torch','warp','mujoco_warp')),'Framework import not permitted')
    if not args.run:
        print(json.dumps({'status':'SAVED_ONLY_PLOT_INPUTS_VALIDATED','comparison_rows':len(rows),'physical_direction_slots':len(secants),'new_WM_forwards':0,'GPU_used':False}));return
    out=args.output.resolve();need(out.is_relative_to((ROOT/'outputs').resolve()),'Owned fresh plot output required');out.mkdir(parents=True,exist_ok=False)
    began=time.perf_counter();shutil.copy2(__file__,out/Path(__file__).name)
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(20,11.4));gs=fig.add_gridspec(2,3,left=.055,right=.98,bottom=.16,top=.85,hspace=.40,wspace=.33,height_ratios=(1.10,1))
    actual=[v for v in rows if v['status']=='COMPLETED']
    extent=max([abs(v[k]) for v in actual for k in ('physical_slope','AD_effective_FP32')],default=0)
    limit=extent*1.06 if extent else 1.
    markers={.002:'o',.001:'^'}
    for col,method in enumerate(METHODS):
        ax=fig.add_subplot(gs[0,col]);selected=[v for v in rows if v['model']==method and v['status']=='COMPLETED']
        for label in LABELS[:-1]:
            for eps in EPS:
                points=[v for v in selected if v['classification']==label and v['epsilon_rad']==eps]
                ax.scatter([v['physical_slope'] for v in points],[v['AD_effective_FP32'] for v in points],color=COLORS[label],marker=markers[eps],s=34,alpha=.65,linewidths=.35,edgecolors='#303030',zorder=3)
        ax.plot([-limit,limit],[-limit,limit],'--',color='#333333',lw=1,zorder=2)
        ax.axhline(0,color='#dddddd',lw=.6);ax.axvline(0,color='#dddddd',lw=.6)
        ax.set(xlim=(-limit,limit),ylim=(-limit,limit),xlabel='Shared physical central slope [reward/rad]',ylabel='WM AD along effective FP32 direction [reward/rad]',title=f'{method}: all {len(selected)} completed epsilon rows')
        ax.set_aspect('equal',adjustable='box');ax.ticklabel_format(axis='both',style='sci',scilimits=(-3,3));ax.grid(alpha=.15)
    ax=fig.add_subplot(gs[1,0]);left=np.zeros(4)
    for label in LABELS:
        values=np.array([next(v['count'] for v in coverage if v['path']==p and v['classification']==label) for p in PATHS])
        ax.barh(np.arange(4),values,left=left,color=COLORS[label],label=SHORT[label])
        for i,n in enumerate(values):
            if n:ax.text(left[i]+n/2,i,str(n),ha='center',va='center',fontsize=10,color='white' if label=='STABLE_NONFLAT' else 'black')
        left+=values
    ax.set(yticks=np.arange(4),yticklabels=PATHS,xlim=(0,32),xticks=(0,8,16,24,32),xlabel='Shared physical direction slots (32 per path)',title='Physical coverage: 128 directions, counted once');ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    ax=fig.add_subplot(gs[1,1]);matrix=np.array([[next(v['sign_agreement_percent'] for v in sign_rows if v['model']==m and v['epsilon_rad']==e) for e in EPS] for m in METHODS],float)
    cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#ececec');image=ax.imshow(np.ma.masked_invalid(matrix),vmin=0,vmax=100,cmap=cmap,aspect='auto')
    for i,m in enumerate(METHODS):
        for j,e in enumerate(EPS):
            value=next(v for v in sign_rows if v['model']==m and v['epsilon_rad']==e);percent=value['sign_agreement_percent']
            label='N/A' if percent is None else f'{percent:.1f}%'
            ax.text(j,i,label+'\n'+str(value['eligible_stable_nonflat'])+'/128 eligible',ha='center',va='center',fontsize=11,color='white' if percent is not None and percent>62 else 'black')
    ax.set(xticks=(0,1),xticklabels=('epsilon 2 mrad','epsilon 1 mrad'),yticks=np.arange(3),yticklabels=METHODS,title='Sign agreement: stable nonflat only')
    fig.colorbar(image,ax=ax,fraction=.04,pad=.04,label='Descriptive sign agreement [%]')
    ax=fig.add_subplot(gs[1,2]);ax.axis('off');ax.set_title('Native self-collision15 calibration\nSame fixed fresh-donor truth; 240 dependent slots',pad=20)
    body=[[v['model'],f"{v['truth_positive_slots']}/240",f"{v['BCE']:.5f}",f"{v['Brier']:.5f}",'N/A' if v['display_F1'] is None else f"{v['display_F1']:.4f}"] for v in self_rows]
    table=ax.table(cellText=body,colLabels=['WM','Truth +','BCE','Brier','F1*'],cellLoc='center',loc='center',colWidths=[.19,.23,.20,.20,.18]);table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1.05,2.15)
    for (i,j),cell in table.get_celld().items():
        cell.set_edgecolor('#c7c7c7')
        if i==0:cell.set_facecolor('#edf1f4');cell.set_text_props(weight='bold')
    ax.text(.5,.08,'* F1 shown as N/A when either truth class is absent.\nNo hand-object contact or force-prediction claim.',ha='center',va='center',transform=ax.transAxes,fontsize=10)
    handles=[Patch(facecolor=COLORS[k],label=SHORT[k]) for k in LABELS]
    handles += [Line2D([],[],color='#333333',marker=markers[e],linestyle='none',label=f'epsilon {e*1000:g} mrad') for e in EPS]
    handles += [Line2D([],[],color='#333333',linestyle='--',label='AD = physical slope')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.905),ncol=7,frameon=False,fontsize=10)
    fig.suptitle('Current104: saved one-step WM gradients versus same-world physical secants',fontsize=19,y=.978)
    fig.text(.5,.94,'All completed finite rows and outliers retained; common scatter axes/units; no quality selection',ha='center',fontsize=11)
    fig.text(.055,.094,'One seen source, training seed0: 16 dependent snapshots x 8 directions; two epsilon rows reuse each direction. Physical results are shared across all 3 WMs.',fontsize=10)
    fig.text(.055,.068,'Primary objective: native FP32 object2+8 at ref k. AD uses the saved effective FP32 direction; nominal direction and ref k+1 remain separate diagnostics.',fontsize=10)
    fig.text(.055,.042,'No CI/p-values or method-superiority/generalization claim. Sampled interior-coordinate comparison only; not full-motion, full-Jacobian, force, ego or hardware qualification.',fontsize=10)
    png=out/'physical_secants_WM_summary.png';svg=out/'physical_secants_WM_summary.svg'
    fig.savefig(png,dpi=180);fig.savefig(svg);plt.close(fig)
    csv_write(out/'scatter_all_rows.csv',rows);csv_write(out/'physical_coverage.csv',coverage);csv_write(out/'sign_agreement.csv',sign_rows);csv_write(out/'self15_calibration.csv',self_rows)
    for value in pins.values():checked(value)
    inputs_preserved=all(sha(v['path'])==v['sha256'] for v in pins.values())
    need(inputs_preserved,'Plotting mutated input artifacts')
    summary={'schema':'current104_physical_WM_saved_summary:1','status':'SAVED_ONLY_FIGURES_RENDERED_PENDING_VISUAL_QA','plotter':pin(__file__),
        'input_directory':str(args.input.resolve()),'input_pins':pins,'original_consumer':report['entry'],'physical_manifest':report['physical_manifest'],
        'source_lineage_N':1,'independent_ego_N':0,'training_seed':0,'dependent_snapshots':16,'directions_per_snapshot':8,
        'planned_physical_directions':128,'WM_epsilon_comparison_rows':len(rows),'displayed_finite_scatter_rows':len(actual),
        'scatter_common_axis_minmax':[-limit,limit],'all_outliers_retained':True,'physical_rows_shared_not_multiplied':True,
        'classifications_shared_physical':{k:sum(v['classification']==k for v in secants) for k in LABELS},'sign_descriptive':sign_rows,
        'self15_table':self_rows,'input_files_unchanged':inputs_preserved,'new_WM_loads':0,'new_WM_forwards':0,'new_physics_steps':0,'Env_or_GPU_used':False,
        'artifacts':{p.name:pin(p) for p in (png,svg,out/'scatter_all_rows.csv',out/'physical_coverage.csv',out/'sign_agreement.csv',out/'self15_calibration.csv')},
        'visual_QA':'PENDING actual image view; source rendering is not visual verification','wall_seconds':time.perf_counter()-began,
        'scope':'Saved-only descriptive summary;16dependent states/8directions/one seen source seed0. No new qualification or CI; self15 native robot selfcollision only, not handobject force/ego/fullmotion.'}
    (out/'report.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':summary['status'],'PNG':str(png),'SVG':str(svg),'new_WM_forwards':0,'new_physics_steps':0}))


if __name__=='__main__':main()
