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
review={'status':'COMPLETE_REVIEWED','scope':'All27 portfolio observations only, nine frozen checkpoints, three training seeds per method; historical cohorts excluded.',
        'checks':{'all27_unique_ordered_as_plan':[r['job_id'] for r in d['results']]==[j['job_id'] for j in plan['jobs']],'all125_source_input_checkpoint_checks':len(d['input_source_checks'])==125 and all(d['input_source_checks'].values()),'all27_valid_capacity_zero':all(r['status']=='VALID_CAPACITY_ZERO' for r in d['results']),'all27_read_error_None':all(c['read_error'] is None for c in sidecars),'all9_checkpoint_groups_have3processes':all(c['valid_three_processes'] for c in checkpoints.values()),'all3_method_groups_have3trainingseeds':all(c['valid_three_training_seeds'] and c['N_training_seeds']==3 for c in d['method_summaries'].values())},
        'capacity_scope':'Stickybit0 read once after150 originalsteps, singleworld and noreset; identities/runtime/health/condition/rawhashes bind. No convergence/accuracy/state-safety guarantee.',
        'method_seed_process_pass_profiles':{m:d['method_summaries'][m]['seed_process_pass_profile'] for m in methods},'method_all3pass_training_seed_counts':{m:d['method_summaries'][m]['training_seeds_all3pass'] for m in methods},
        'task_gate_interpretation':'A1L seed2/3 fail shaft tracking and actual-state joint limits; P_ONLY seed3 fails shaft tracking; A2 seed1/3 fail actual-state limits only. All3repeat labels within each checkpoint agree. All other recorded frozen gates pass; no task threshold or physical-state clamp.',
        'continuous_interpretation':'A2 lowers the process-mean maximum shaft error for all3 paired seeds relative to both A1L and P_ONLY, while its overall all3pass seed count equals A1L and is below P_ONLY. This is a peak-error signal with joint-limit tradeoffs, not an unconditional mechanism result.',
        'support512_raw_recompute_max_abs_error':max(r['real_contacts']['support512_recompute_max_abs_error'] for r in d['results']),
        'support_scope':'Auxiliary fixed-reference unit-cone/COM/no-mount geometry; not force matching or task success. The support plot y-axis is zoomed.',
        'plot_qa':{'original_tracking_and_seed_summary_viewed':True,'tracking_all27_thin_traces_and9_means_visible':True,'original_thresholds_and_max14p266mrad_visible':True,'annotated_summary_seed_shape_legend':{'circle':1,'square':2,'triangle':3},'range_bars_are_process_min_max_not_CI':True,'annotation_changes_layout_only':True,'annotated_figure_visual_review':'PENDING'},
        'cost':d['evaluation_cost'],'historical_training_cost_scope':'Individual summary.json recorded process totals remain separate from new evaluation and total engineering; nominal matched updates are not equal compute.',
        'background_only':d['background_only'],'retained_limits':d['retained_limits'],'hashes':{'comparison_json':sha(OUT/'comparison.json'),'frozen_source':sha(ROOT/'scripts/summarize_portfolio.py'),'frozen_protocol':sha(ROOT/'analysis_protocol_v1.json'),'render_review_source':sha(Path(__file__)),'tracking_png':sha(OUT/'tracking.png'),'annotated_summary_png':sha(OUT/'seed_summary_annotated.png')}}
(OUT/'review.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')

lines=['Independent terminal review: all27 complete3s observations and capacity/control/input contracts passed audit. Method N=3training seeds, each with3fresh processes.','All3processes pass per checkpoint: A1L seeds[1]; P_ONLY seeds[1,2]; A2 seeds[2]. Other checkpoints fail all3. There are no mixed task labels in this batch; numerical continuous ranges remain retained.','Thus all3pass training-seed counts are A1L1/3, P_ONLY2/3, A21/3. Do not reinterpret the27processes or frames as training seeds. No best-of-repeat or historical-cohort pooling.','Failures: A1Lseed2/3 shaft+actualq; P_ONLYseed3 shaft only; A2seed1/3 actualq only. Target bounds, position, progress, drop, penetration, load/rate and logging gates pass all27.','Checkpoint summary: values are mean [min,max] across3processes; q is actual-state maximum excess in mrad.']
for cp in checkpoints.values():
    v=cp['metrics'];fmt=lambda key:f"{v[key]['mean']:.6f} [{v[key]['min']:.6f},{v[key]['max']:.6f}]"
    lines.append(f"{cp['model_id']}: passes{cp['process_passes']}/3; positionmeanmm={fmt('position_mean_mm')}; shaftmaxdeg={fmt('shaft_max_deg')}; qexcessmrad={fmt('joint_state_max_excess_mrad')}; support512={fmt('support512_score')}.")
lines+=['Descriptive paired process-mean differences (three training-seed pairs, no CI):']
for name,pair in d['paired_seed_descriptions'].items():
    summary=pair['descriptive_mean_differences'];lines.append(name+': '+', '.join(f"{key}={summary[key]['mean']:+.6f}" for key in ['position_mean_mm','shaft_max_deg','joint_state_max_excess_mrad','support512_score']))
lines+=['A2 maximum shaft error improves in all3seed pairs relative to A1L and P_ONLY; this does not give an overall task-rate improvement over A1L. P_ONLY has the most all3pass seeds here, a descriptive result under the selected common controller/historical training, not general method superiority.','Full rows/gates/identity/control/capacity/contact evidence are in comparison.json/jobs.csv; compact checkpoint means/worst/ranges are in checkpoints.csv.','All27stickycapacity flags are0; support512 raw recomputation maxerror3.33e-16. Capacity0 does not establish convergence, physics accuracy or global safety.','Whole evaluation queue279.391s; child-supervisor sum276.655s; evaluator-reported sum143.107279s. Historical training process costs are reported separately, not equal compute or full engineering cost.','Plots: tracking.png/svg preserves all27traces and original thresholds. seed_summary_annotated.png/svg adds a seed-shape legend and marks support-y zoom without changing data/statistics.','Original0controller and single-checkpoint engineering cohorts are linked background only. No portfolio input uses observation-only IK or labels from a new ego sample.','']
(OUT/'SUMMARY.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'review_checks_passed':all(review['checks'].values()),'checkpoint_rows':len(table),'review_sha256':sha(OUT/'review.json')},indent=2))
