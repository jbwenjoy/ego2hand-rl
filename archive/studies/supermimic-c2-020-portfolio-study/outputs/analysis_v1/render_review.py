"""Artifact-only summary layout/review from frozen portfolio comparison.json."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d=read(OUT/'comparison.json');plan=read(ROOT/'configs/c2-portfolio-v1.json');protocol=read(ROOT/'analysis_protocol_v1.json')
checkpoints=d['checkpoint_summaries'];methods=['A1L','P_ONLY','A2'];colors={'A1L':'#1764ab','P_ONLY':'#15845c','A2':'#d55e00'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,4,figsize=(13,4.6),constrained_layout=True)
for ax,key,label in zip(axes,['position_mean_mm','shaft_max_deg','joint_state_max_excess_mrad','support512_score'],['Position mean (mm)','Shaft maximum (deg)','Actual-state excess (mrad)','Auxiliary support score512\nZoomed y-axis']):
    for seed,offset in zip((1,2,3),(-.12,0,.12)):
        values=[]
        for x,method in enumerate(methods):
            cp=next(v for v in checkpoints.values() if v['method']==method and v['training_seed']==seed);v=cp['metrics'][key]
            ax.plot([x+offset,x+offset],[v['min'],v['max']],color=colors[method],alpha=.7)
            ax.scatter(x+offset,v['mean'],color=colors[method],marker=['o','s','^'][seed-1],s=30);values.append((x+offset,v['mean']))
        ax.plot(*np.array(values).T,color='.5',alpha=.35,linewidth=.7)
    ax.set_xticks(range(3),methods);ax.set_title(label);ax.grid(alpha=.18)
fig.suptitle('Each marker=one training seed, mean of three process results; bars=min–max of those processes\nConnected seed pairs are descriptive. N=3 training seeds; no CI, selection or unconditional mechanism attribution.')
handles=[Line2D([],[],marker=marker,color='none',markeredgecolor='black',markerfacecolor='white',label=f'Training seed{seed}') for seed,marker in zip((1,2,3),('o','s','^'))]
fig.legend(handles=handles,ncol=3,loc='outside lower center',frameon=False)
for ext in ('png','svg'):fig.savefig(OUT/f'seed_summary_annotated.{ext}',dpi=175)
plt.close(fig)

table=[]
for cp in checkpoints.values():
    row={'method':cp['method'],'training_seed':cp['training_seed'],'process_passes_of3':cp['process_passes'],'outcome_class':cp['outcome_class']}
    for key in ['position_mean_mm','shaft_max_deg','joint_state_max_excess_mrad','support512_score']:
        for measure in ('mean','min','max','worst'):row[key+'_'+measure]=cp['metrics'][key][measure]
    table.append(row)
with (OUT/'checkpoints.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)

sidecars=[read(Path(j['output_dir'])/'capacity_audit.json') for j in plan['jobs']]
doseout=ROOT/'outputs/dose_comparison_v1';dose=read(doseout/'comparison.json')
review={'status':'COMPLETE_20_FOLLOWUP_REVIEWED','scope':'Separate20mrad cohort only, all27 observations; nine checkpoints, method N=3training seeds. Sequential followup after known5results, not independent confirmation.',
        'checks':{'all27_unique_ordered_as_plan':[r['job_id'] for r in d['results']]==[j['job_id'] for j in plan['jobs']],'all128_source_input_checkpoint_checks':len(d['input_source_checks'])==128 and all(d['input_source_checks'].values()),'all27_valid_capacity_zero':all(r['status']=='VALID_CAPACITY_ZERO' for r in d['results']),'all27_read_error_None':all(c['read_error'] is None for c in sidecars),'all9_checkpoint_groups_have3processes':all(c['valid_three_processes'] for c in checkpoints.values()),'all3_method_groups_have3trainingseeds':all(c['valid_three_training_seeds'] and c['N_training_seeds']==3 for c in d['method_summaries'].values()),'cross_dose_model_reference_t0_binding':all(dose['binding_checks'].values())},
        'capacity_scope':'Stickybit0 read once after150steps, singleworld/noreset and raw/runtime/health/condition/source bindings. Does not qualify convergence, physics accuracy, continuous-time safety or generalization.',
        'method_seed_process_pass_profiles':{m:d['method_summaries'][m]['seed_process_pass_profile'] for m in methods},'method_all3pass_training_seed_counts':{m:d['method_summaries'][m]['training_seeds_all3pass'] for m in methods},
        'task_gate_interpretation':'A2 all3seeds pass all3processes in this20cohort. A1Lseed2 fails shaft only; A1Lseed3 fails shaft and actualq. P_ONLYseed3 fails shaft only. No mixed task labels within checkpoints. All other frozen observed gates pass.',
        'continuous_interpretation':'A2 vs A1L mean-of-process maximum shaft error is lower in3/3seeds; vs P_ONLY lower in2/3, withseed2 slightly higher. From5to20, A2 qhardexcess becomes0 but its maximum shaft error increases about0.668/0.794/0.733degrees acrossseeds. No claim of uniform tracking improvement or optimal dose.',
        'support512_raw_recompute_max_abs_error':max(r['real_contacts']['support512_recompute_max_abs_error'] for r in d['results']),
        'support_scope':'Auxiliary frozen-reference unit-cone COM/no-mount geometry, not force matching or task success; support axis explicitly zoomed.',
        'plot_qa':{'original_tracking_and_seed_summary_viewed':True,'full27thin_traces_and9means_retained':True,'original_criteria_and_max12p274mrad_unclipped':True,'annotated_summary_shape_legend':{'circle':1,'square':2,'triangle':3},'range_bars_are_process_minmax_not_CI':True,'annotated_and_dose_figure_visual_review':'PENDING'},
        'evaluation_cost':d['evaluation_cost'],'dose_scope':protocol['dose_comparison'],'prior5_frozen_sha256':dose['baseline5_sha256'],'no54job_pool_or_best_dose':True,'retained_limits':d['retained_limits'],
        'hashes':{'comparison_json':sha(OUT/'comparison.json'),'frozen_source':sha(ROOT/'scripts/summarize_portfolio.py'),'frozen_protocol':sha(ROOT/'analysis_protocol_v1.json'),'render_review_source':sha(Path(__file__)),'tracking_png':sha(OUT/'tracking.png'),'annotated_summary_png':sha(OUT/'seed_summary_annotated.png'),'dose_comparison_json':sha(doseout/'comparison.json')}}

# Two separate dose profiles and three-process ranges per checkpoint. No
# pseudo-pairing of individual numerical-noise realizations between batches.
fig,axes=plt.subplots(2,3,figsize=(11.8,7),sharex=True,sharey='row',constrained_layout=True)
for col,method in enumerate(methods):
    axes[0,col].set_title(method)
    for seed in (1,2,3):
        pair=next(p for p in dose['pairs'] if p['method']==method and p['training_seed']==seed)
        for margin,key,color,offset in [(5,'metrics5','#1764ab',-.08),(20,'metrics20','#d55e00',.08)]:
            for row,metric in enumerate(['joint_state_max_excess_mrad','shaft_max_deg']):
                v=pair[key][metric];ax=axes[row,col];ax.plot([seed+offset]*2,[v['min'],v['max']],color=color,linewidth=1.2);ax.scatter(seed+offset,v['mean'],color=color,s=25,label=f'{margin}mrad' if seed==1 else None)
    for row,limit in enumerate([1,20]):axes[row,col].axhline(limit,color='.25',linestyle=':',linewidth=1);axes[row,col].grid(alpha=.2)
    axes[-1,col].set_xticks([1,2,3],['seed1','seed2','seed3']);axes[-1,col].set_xlabel('Training seed')
axes[0,0].set_ylabel('Actual-state maximum excess (mrad)');axes[1,0].set_ylabel('Shaft maximum error (deg)');axes[0,0].legend(ncol=2)
fig.suptitle('Separate5mrad and20mrad cohorts: marker=three-process mean, bar=process min–max\nSame nine frozen checkpoints; sequential batches after known5results. No pooling, CI, best-dose or unconditional causal inference.')
for ext in ('png','svg'):fig.savefig(doseout/f'checkpoint_doses.{ext}',dpi=180)
plt.close(fig)
review['hashes']['dose_figure_png']=sha(doseout/'checkpoint_doses.png')
(OUT/'review.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')

lines=['20mrad followup: all27 predeclared full3s runs and identity/control/initial/capacity/input/metric contracts passed independent audit. Method N=3training seeds, eachcheckpoint3fresh processes.','All3process-pass training-seed counts: A1L1/3; P_ONLY2/3; A23/3. Profiles by seed1/2/3: A1L3/0/0, P_ONLY3/3/0, A23/3/3. No best-repeat selection.','Failures: A1Lseed2 shaft only; A1Lseed3 shaft+actualjointstate; P_ONLYseed3 shaft only. Other frozen gates pass, including bounded targets, position, progress, drop, penetration, load/rate and logging.','A2 has0 sampled hard-limit excess acrossall20joints in all9processes, with shaftmax8.338–13.998degrees. This is a conditional fixed-task simulation result, not continuous-time or general-domain safety.','Checkpoint metrics: mean[min,max] overthree processes, retained worst/min/max incheckpoints.csv/comparison.json.']
for cp in checkpoints.values():
    v=cp['metrics'];fmt=lambda key:f"{v[key]['mean']:.6f}[{v[key]['min']:.6f},{v[key]['max']:.6f}]"
    lines.append(f"{cp['model_id']}: passes{cp['process_passes']}/3; positionmeanmm={fmt('position_mean_mm')}; shaftmaxdeg={fmt('shaft_max_deg')}; qexcessmrad={fmt('joint_state_max_excess_mrad')}; support512={fmt('support512_score')}.")
lines+=['20mrad method contrasts remain descriptive across3training-seed pairs. A2shaftpeak improves vsA1L in3/3seeds and vsP_ONLY in2/3; it is slightly worse thanP_ONLY onseed2.','Separate dose descriptions (20minus5, checkpoint-level three-process summaries):']
for pair in dose['pairs']:
    delta=pair['delta20minus5_process_mean'];lines.append(f"{pair['model_id']}: passes5={pair['process_passes5']}/3,20={pair['process_passes20']}/3; deltaqmax={delta['joint_state_max_excess_mrad']:+.6f}mrad; deltashaftmax={delta['shaft_max_deg']:+.6f}deg; deltapositionmean={delta['position_mean_mm']:+.6f}mm.")
lines+=['A2 from5to20: actualq excess decreases to0 while mean-of-process shaftmax rises0.668/0.794/0.733deg; all9newprocesses pass originaltask gates. A1Lseed3 still violatesjointstate12.156–12.274mrad;20mrad is not a universalstate constraint.','5and20remain separate cohorts. No pooled54job estimate, union-of-success/best-dose selection, CI or newtraining-seed claim. Followup was declared after knowing5results; changes include sequential numerical/cache contexts, so no unconditionaldosecausality or independentconfirmation.','All27sticky flags0/readonceat150/noreset; NumPyrawcontacts→COM512 maxerror3.33e-16. Capacity0 is not convergence/physicalaccuracy, and supportscore is not force matching/task success.','Evaluation queue280.140s; childsum277.578s; evaluator-reported143.752696s. Historical training costs remain distinct; no newtraining.','Figurestracking/seedsummary preserveall27; annotatedseedsummary addsseedlegend andzoomlabel only. Dosefigure keepsbothcohorts and their ownprocessranges. No observation/palm-root IK candidate enters portfolio inputs.','']
(OUT/'SUMMARY.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'review_checks_passed':all(review['checks'].values()),'checkpoint_rows':len(table),'review_sha256':sha(OUT/'review.json')},indent=2))

