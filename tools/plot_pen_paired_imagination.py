"""Visualize the full denominator and terminal checks for the paired imagination pilot."""
import json,sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json


def main(run,out):
    out.mkdir(exist_ok=False)
    path=run/'report.json';report=json.loads(path.read_bytes())
    inputs=[];groups={}
    for record in report['evaluated']:
        ev=json.loads(verify_file_pin(record['evaluation']).read_bytes())
        inputs.append(record['evaluation'])
        groups[record['key']]=[]
        for row in ev['cases']:
            case=json.loads(verify_file_pin(row['report']).read_bytes())
            groups[record['key']].append((row['valid_s'],row['stop'],np.rad2deg(case['summary']['metrics']['angle_RMS_rad']),row['success']))
    order=['parent']+[f'{arm}_{s}' for s in (7411,7412,7413) for arm in ('A1','A2')]
    assert set(order)==groups.keys() and all(len(groups[k])==2 for k in order)
    fig,(left,right)=plt.subplots(1,2,figsize=(11,4.6),sharey=True,layout='constrained')
    for i,k in enumerate(order):
        for rep,(duration,stop,angle,success) in enumerate(groups[k]):
            y=i+(rep-.5)*.18
            color=('#555b65' if k=='parent' else '#367ca1' if k.startswith('A1') else '#ba6041')
            marker={'REFERENCE_COMPLETION':'o','PHYSICAL_AUDIT_STOP':'x','TASK_FAILURE':'v'}[stop]
            left.hlines(y,0,duration,color=color,lw=.8,alpha=.6);left.scatter(duration,y,color=color,marker=marker,s=35)
            if stop=='REFERENCE_COMPLETION':right.scatter(angle,y,color=color,s=35)
    left.set(yticks=range(len(order)),yticklabels=[k.replace('_','  ') for k in order],xlim=(-.15,7.65),xlabel='Valid execution (s)',title='All native trials')
    right.set(xlabel='Full-trajectory SO3 RMS (degrees)',title='Complete trajectories only')
    left.invert_yaxis();left.axvline(7.36,color='#999999',ls='--',lw=.8)
    for ax in (left,right):ax.grid(axis='x',alpha=.2)
    left.legend(handles=[Line2D([],[],marker='o',ls='',color='#666',label='Reference end'),Line2D([],[],marker='x',ls='',color='#666',label='Physical audit stop')],loc='lower left',fontsize=8)
    fig.suptitle('Three-step imagined policy updates | two numerical repeats per checkpoint')
    dest=out/'native_endpoints.png';fig.savefig(dest,dpi=150);plt.close(fig)
    write_json(out/'report.json',dict(run=file_pin(path),inputs=inputs,figure=file_pin(dest)))
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
