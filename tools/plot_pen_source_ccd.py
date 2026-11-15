"""Per-anchor maxima at unchanged force/state gates, no pooled success claim."""
import argparse
import json
import sys
import tomllib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.runtime.controlled_evaluation import write


def main(a):
    compare=json.loads(a.comparison.read_text())
    if a.output.exists():raise ValueError('Fresh matched-cohort scientific figure')
    gp=ROOT/'configs/pen_gpu_step_audit_v1.toml';gates=tomllib.loads(gp.read_text())
    state=['hand_position_rad','object_position_m','object_rotation_rad','hand_velocity_rad_s','object_velocity_m_s','object_omega_rad_s']
    summaries=[]
    for pin in [compare['baseline_review'],compare['tight_review']]:
        r=json.loads(verify_file_pin(pin).read_text());rows=[e for e in r['comparisons'] if e['used_for_profile_qualification']]+r['all57_repeats'];anchors=[]
        for anchor in r['anchors']:
            selected=[e for e in rows if all(e[k]==anchor[k] for k in ['mode','dt','k'])]
            anchors.append(dict(mode=anchor['mode'],dt=anchor['dt'],k=anchor['k'],comparisons=len(selected),
                maximum_force_gate_ratio=max(e['metrics']['EFC_force_N']/gates['EFC_force_N'] for e in selected),
                maximum_state_gate_ratio=max(e['metrics'][key]/gates[key] for e in selected for key in state)))
        summaries.append(dict(profile=r['ccd_profile'],required_comparisons=len(rows),state_failed=sum(not all(e['checks'][key] for key in state) for e in rows),
            maximum_metrics={key:max(e['metrics'][key] for e in rows) for key in rows[0]['metrics']},anchors=anchors))
    a.output.mkdir();fig,axes=plt.subplots(2,1,figsize=(10.5,6.6),sharex=True)
    x=np.arange(7);width=.34
    for i,s in enumerate(summaries):
        for ax,key in zip(axes,['maximum_force_gate_ratio','maximum_state_gate_ratio']):
            ax.bar(x+(i-.5)*width,[max(1e-8,p[key]) for p in s['anchors']],width,label=s['profile']+' CCD '+('1e-6' if i==0 else '1e-7'),color=['#3978b8','#db8a32'][i])
    for ax in axes:
        ax.set_yscale('log');ax.axhline(1,color='#a22c34',ls='--',lw=1.2,label='Original acceptance gate');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Maximum force error / 1e-4 N gate');axes[1].set_ylabel('Maximum state error / its gate')
    axes[0].legend(ncol=3,loc='upper right',fontsize=9);axes[0].set_title('Matched public sources and57-action tables; maxima over required repeat/path comparisons')
    labels=[f"{p['k']*20} ms\n{p['dt']*1000:g} ms dt" for p in summaries[0]['anchors']]
    axes[1].set_xticks(x,labels);axes[1].set_xlabel('Original source time / native integration timestep')
    fig.suptitle('Tighter CCD reduces large state discrepancies here, but force gates still fail',fontsize=13)
    fig.subplots_adjust(left=.12,right=.985,bottom=.17,top=.88,hspace=.14)
    fig.text(.12,.035,'847 required comparisons/profile; seven saved anchors. Display floor1e-8 only. No task, learned-policy, ego or acceleration qualification.',fontsize=8,color='#444')
    path=a.output/'summary.png';fig.savefig(path,dpi=180);plt.close(fig)
    result=dict(schema='matched_source_CCD_figure:1',comparison=file_pin(a.comparison),gates=file_pin(gp),source=file_pin(__file__),figure=file_pin(path),
        summaries=summaries,native_integrations=0,neural_updates=0,scope='Per-anchor maxima over SAME within-profile cohort. State ratio is max over six original state gates, not a pooled accuracy score. Force failures retained; no profile fully qualified.')
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),figure=result['figure'],summaries=[dict(profile=s['profile'],required=s['required_comparisons'],state_failed=s['state_failed'],maxima=s['maximum_metrics']) for s in summaries])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--comparison',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
