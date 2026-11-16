"""Measured cost-sign failures and all-row arithmetic-error ECDF."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.strict_json import write_json


def main():
    out=ROOT/'runs/pen_paired_ray_friction_figure_v1'
    if out.exists(): raise ValueError('Fresh measured figure required')
    rp=file_pin(ROOT/'runs/pen_paired_ray_friction_analysis_v1/report.json'); d=json.loads(verify_file_pin(rp).read_text())
    if d['passed']!=d['total'] or not all(d['checks'].values()): raise ValueError('Complete arithmetic analysis required')
    z=dict(np.load(verify_file_pin(d['primitive_outputs']))); x=dict(np.load(verify_file_pin(d['fixed_inputs'])))
    errors={name:abs(z[key][:,0].astype(float)-x['FP64_delta']) for name,key in [('Original absolute-cost subtraction','original'),('Same-regime stable difference + fallback','same_regime')]}
    fig,axes=plt.subplots(1,2,figsize=(13,5.5)); ax=axes[0]
    peaks=d['failed_query_peak_rays']; colors=dict(INSTALLED_ORIGINAL_OR='#b75b4c',DROP_IMPROVEMENT_ONLY='#387ca6')
    for alg in colors:
        counts=Counter((r['FP64_initial_delta'],r['GPU_initial_delta']) for r in peaks if r['algorithm']==alg)
        ax.scatter([p[0] for p in counts],[p[1] for p in counts],s=[35+18*n for n in counts.values()],alpha=.65,color=colors[alg],label=alg)
    ax.axhline(0,color='.5',lw=1); ax.axvline(0,color='.5',lw=1); ax.set(xlabel='FP64 fixed-input initial trial cost delta',ylabel='Recorded GPU initial trial cost delta',title=f'All{len(peaks)} failed-query peak rays;size shows multiplicity')
    ax.ticklabel_format(axis='both',style='sci',scilimits=(0,0)); ax.grid(alpha=.2); ax.legend(fontsize=8)
    ax=axes[1]; grid=np.logspace(-18,-4,240)
    for name,error in errors.items(): ax.semilogx(grid,np.searchsorted(np.sort(error),grid,side='right')/len(error),label=name)
    ax.set(xlabel='Absolute friction-row delta error vs FP64 fixed input',ylabel='Fraction of all rows (zeros included)',ylim=(0,1.01),title=f'All{d["primitive_rows"]:,} recorded friction rows')
    ax.grid(alpha=.2); ax.legend(fontsize=8,loc='lower right')
    fig.text(.5,.012,f'{d["same_regime_rows"]:,} same-regime rows;{d["fallback_rows"]:,} explicit unchanged cross-regime fallbacks. One standalone CUDA arithmetic batch;no dynamics/task/Actor/ego improvement inferred.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.055,1,1)); out.mkdir(); fig.savefig(out/'cost_signs_and_friction_error.png',dpi=170); plt.close(fig)
    receipt=freeze_sources([Path(__file__)]+sorted((ROOT/'src').rglob('*.py')),out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()); write_json(out/'source_receipt.json',receipt)
    for pin in [rp,d['fixed_inputs'],d['primitive_outputs']]: verify_file_pin(pin)
    verify_source_snapshot(receipt,live=True)
    write_json(out/'report.json',dict(schema='paired_ray_friction_figure:1',analysis=rp,source_receipt=file_pin(out/'source_receipt.json'),figure=file_pin(out/'cost_signs_and_friction_error.png'),all_failed_peaks=len(peaks),all_friction_rows=d['primitive_rows'],actual_native_integrations=0,optimizer_steps=0,
        scope='Measured fixed-input arithmetic and all failed peaks,not a physical solver modification or success/speed claim.'))
    print(json.dumps(dict(report=file_pin(out/'report.json'),native=0)))


if __name__=='__main__': main()
