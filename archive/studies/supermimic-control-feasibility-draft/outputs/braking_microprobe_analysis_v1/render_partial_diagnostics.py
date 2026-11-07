"""Artifact-only rendering of terminal partial raw; does not modify frozen analysis."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
RAW=ROOT/'outputs/braking_microprobe_v1'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=read(ROOT/'calibration_plan_v1.json')
analysis=read(OUT/'analysis.json')
runtime=read(RAW/'runtime.json')
watchdog=read(ROOT/'outputs/braking_microprobe_v1_watchdog/state.json')
assert runtime['status']=='failed' and watchdog['status']=='failed'
manifest=read(RAW/'branches_manifest.json')
assert len(manifest['branches'])==1
entry=manifest['branches'][0]
meta=read(RAW/entry['metadata'])
with np.load(RAW/entry['npz'],allow_pickle=False) as archive:
    branch={k:archive[k].copy() for k in archive.files}
source=meta['source_future_slice']
with np.load(RAW/source['npz'],allow_pickle=False) as archive:
    donor={k:archive[k][source['start_state']:source['stop_state_exclusive']].copy() for k in ('q_rad','qd_rad_s')}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none'})
fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True,constrained_layout=True)
for col,case in enumerate(plan['joint_cases']):
    j=meta['joint_names'].index(case['joint'])
    for row,(key,label) in enumerate([('q_rad','Joint position (rad)'),('qd_rad_s','Upper-outward velocity (rad/s)')]):
        ax=axes[row,col]
        ax.plot(branch['elapsed_s'],donor[key][:,j],label='Fresh donor future',color='#1764ab')
        ax.plot(branch['elapsed_s'],branch[key][:,j],label='Restored baseline0',color='#d55e00',linestyle='--')
        ax.set_ylabel(label);ax.grid(alpha=.25)
        ax.set_title(case['joint'].replace('right_','')+f" | max difference={np.abs(donor[key][:,j]-branch[key][:,j]).max():.4g}")
        if row==1:ax.axhline(0,color='0.4',linewidth=.7);ax.set_xlabel('Branch elapsed (s)')
axes[0,0].legend(fontsize=8)
fig.suptitle('Terminal partial attempt: A1L training seed1, evaluation seed0, donor t=0.8s\nOnly baseline0 ran; source-future replay FAILED. No pulses or braking qualification.')
for ext in ('png','svg'):fig.savefig(OUT/f'partial_baseline_vs_donor.{ext}',dpi=180)
plt.close(fig)
coverage=[]
for src in plan['sources']:
    for frame,time in zip(plan['snapshot_frames'],plan['snapshot_times_s']):
        sid=f"{src['method']}_seed1_frame{frame:03d}"
        saved=next((x for x in analysis['snapshots'] if x['snapshot_id']==sid),None)
        coverage.append({'snapshot_id':sid,'method':src['method'],'capturetime_s':time,'recorded_branches':sum(x['snapshot_id']==sid for x in analysis['branches']),'replay_gate':saved['replay_gate'] if saved else None,'execution_status':'RECORDED_BASELINE_REPLAY_FAILED' if saved else 'NOT_RUN','braking_inference':'NOT_QUALIFIED' if saved else 'UNAVAILABLE_NOT_RUN'})
review={'status':'TERMINAL_PARTIAL_ENGINEERING_FAILURE','gpu_used_for_analysis':False,'runtime_status':runtime['status'],'watchdog_status':watchdog['status'],'counts':analysis['counts'],'executed_baselines':1,'executed_pulses':0,'unexecuted_branches':53,'stopping_observation_eligible_rows':sum(x['upper_stopping_observation_eligible'] for x in analysis['responses']),'snapshot_coverage':coverage,'all_recorded_descriptive_rows_preserved':True,'force_attribution':'UNKNOWN','m0':None,'a_brake':None,'parameter_enablement':False,'template_plot_note':'Frozen-analysis template velocity plot has no mean because three baselines are unavailable; response_summary has no pulse points. Supplement partial_baseline_vs_donor shows the actual one recorded trace and its donor future. Blank cells/plots are not zero response.','qualification':'No braking pulse was executed. The recorded baseline descriptive values cannot support pulse efficacy, weak C1 calibration, a hard bound or a method comparison. Unexecuted branches are NOT_RUN, never physics failures.','hashes':{'frozen_analysis_source':sha(ROOT/'scripts/analyze_braking_microprobe.py'),'analysis_json':sha(OUT/'analysis.json'),'runtime_json':sha(RAW/'runtime.json'),'watchdog_state':sha(ROOT/'outputs/braking_microprobe_v1_watchdog/state.json'),'supplement_source':sha(Path(__file__))}}
(OUT/'partial_terminal_review.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':review['status'],'recorded':1,'unexecuted':53,'pulses':0,'eligible':0,'coverage':coverage},indent=2))
