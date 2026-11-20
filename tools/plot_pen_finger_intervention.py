"""Plot only the executed shaft-error traces in the paired finger intervention."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from analyse_pen_axis_contacts import ROOT
from supermimic_research.artifacts import verify_file_pin
from supermimic_research.mechanics.pen_task_contracts import tracking_series

def main(folder):
    report=json.loads((folder/'report.json').read_bytes())
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    seen=set()
    for row in report['rows']:
        case=json.loads(verify_file_pin(row['case']['report']).read_bytes())
        with np.load(verify_file_pin(case['trace'])) as data: series=tracking_series(data,.0025)
        name=row['label']; t=series['nominal_time_s']; y=np.rad2deg(series['shaft_angle_rad']);mask=(t>=4)&(t<=4.55)
        label={'SHAM':'Original action','HALF':'Half residual (fingers 2 and 3)'}[name]
        ax.plot(t[mask],y[mask],color=('#2a7597' if name=='SHAM' else '#c74e26'),lw=1.5,alpha=.8,label=label if name not in seen else None)
        if row['case']['stop']=='TASK_FAILURE':ax.scatter(t[-1],y[-1],s=20,color='#c74e26',zorder=4)
        seen.add(name)
    ax.axvspan(4.10,4.44,color='#888888',alpha=.1,label='Scheduled intervention')
    ax.axhline(60,ls='--',color='#444444',lw=1,label='Shaft termination threshold')
    ax.set(xlabel='Reference time (s)',ylabel='Shaft angle error (deg)',title='Local residual-action intervention | three numerical repeats per condition')
    ax.legend(fontsize=8);fig.savefig(folder/'shaft_error.png',dpi=150);plt.close(fig)
if __name__=='__main__':main(Path(sys.argv[1]))
