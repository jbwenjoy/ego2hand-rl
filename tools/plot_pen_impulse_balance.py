"""All676 residuals and all5 failures;no selected favorable cases or gate edit."""
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.strict_json import write_json


def read(p):return json.loads(Path(p).read_text())


def main():
    rp=ROOT/'runs/pen_impulse_balance_v2/report.json';r=read(rp);base=read(verify_file_pin(r['prior_failed_diagnostic']));out=ROOT/'runs/pen_impulse_balance_figure_v1'
    if out.exists() or r['passed']!=r['total'] or not all(r['checks'].values()):raise ValueError('Fresh figure after corrected recorded partition required')
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__)],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt)
    x=np.array([a['peak_impulse_Ns'] for a in base['rows']]);y=np.array([a['peak_component_norms_Ns']['acceleration_force'] for a in base['rows']]);bad=np.array([not a['original_impulse_pass'] for a in base['rows']]);positive=(x>0)&(y>0);fig,axes=plt.subplots(1,2,figsize=(13,5.2));ax=axes[0]
    for mask,label,color,marker in [(positive&~bad,'Original impulse gate passed','#367aa4','o'),(positive&bad,'Original impulse gate failed','#ce563b','x')]:ax.scatter(x[mask],y[mask],s=20 if marker=='o' else 55,c=color,marker=marker,alpha=.55 if marker=='o' else 1,label=f'{label} ({int(mask.sum())})')
    lo=min(x[positive].min(),y[positive].min())*.6;hi=max(x.max(),y.max())*1.8;ax.plot([lo,hi],[lo,hi],'--',c='.55',lw=1,label='1:1 reference');ax.axvline(5e-7,color='#ce563b',ls=':',lw=1.4,label='Original total-residual gate');ax.set(xscale='log',yscale='log',xlim=(lo,hi),ylim=(lo,hi),xlabel='Peak original impulse residual (N s)',ylabel='Acceleration-force term at that same tick (N s)',title='All 676 fixed recorded queries');ax.grid(alpha=.15);ax.legend(fontsize=8,loc='upper left')
    failed=r['failed_peak_rows'];keys=['acceleration_force','constraint_contact','velocity_update','gravity_representation'];names=['Acceleration / force','Contact labels / native force','Velocity update','Gravity representation'];colors=['#3d799f','#e49a49','#64a079','#9172ae'];ticks=np.arange(5);width=.19;ax=axes[1]
    for j,(key,label,color) in enumerate(zip(keys,names,colors)):ax.bar(ticks+(j-1.5)*width,[v['components_Ns'][key] for v in failed],width,label=label,color=color)
    ax.set(yscale='log',ylabel='Term magnitude (N s)',title='Every failing query: magnitudes at its failing tick',xticks=ticks,xticklabels=['Life\nP058','Actor\nP027','Sham ALT\nP064','Sham BASE\nP110','Random\nP067']);ax.grid(axis='y',alpha=.15);ax.legend(fontsize=8,loc='upper right')
    fig.suptitle('Recorded impulse-gap localization; original 5e-7 N s gate retained',fontsize=13);fig.text(.04,.025,'Term magnitudes do not add as scalars. Implicitfast execution; post-control forward caches are not last-tick solver-gradient measurements.',fontsize=9);fig.tight_layout(rect=(0,.07,1,.95));path=out/'impulse_gap_localization.png';fig.savefig(path,dpi=170);plt.close(fig);verify_source_snapshot(receipt,live=True)
    write_json(out/'report.json',dict(schema='pen_recorded_impulse_partition_figure:1',analysis=file_pin(rp),prior_diagnostic=r['prior_failed_diagnostic'],source_receipt=file_pin(out/'source_receipt.json'),figure=file_pin(path),all_queries=676,all_impulse_failures=5,zero_points_excluded_from_log_display=int((~positive).sum()),actual_native_integrations=0,optimizer_steps=0,scope='All fixed diagnostic rows/all5 failures plotted. Localization only;no new native work,solver-cause proof,gate relaxation or full54 outcome.'))
    print(json.dumps(dict(figure=file_pin(path),log_display_zero_points=int((~positive).sum()))),flush=True)


if __name__=='__main__':main()
