import json,sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin
from supermimic_research.strict_json import write_json


def main(out):
    out.mkdir(exist_ok=False);root=ROOT/'runs';a=root/'pen_substep_capacity_eval_v1/report.json';b=root/'pen_substep_secant_capacity_v1/report.json'
    rows=json.loads(a.read_bytes())['pairs']+json.loads(b.read_bytes())['pairs']
    fig,ax=plt.subplots(figsize=(9,4.2),layout='constrained')
    for method,name,color,style in [('absolute','Absolute-pose model','#367ca1','--'),('local_secant','Local action secants','#be653a','-')]:
        data=sorted((r for r in rows if r['method']==method and r['kind']=='mixed'),key=lambda r:r['phase'])
        assert len(data)==10
        if method=='absolute':
            ax.plot(range(10),[r['native_l2'] for r in data],marker='o',color='#20282f',label='Measured GPU response')
            for i,r in enumerate(data):
                if r['changed']:
                    ax.axvspan(i-.22,i+.22,alpha=.13,color='#a54d20',label='Contact set changed')
        ax.plot(range(10),[r['prediction_l2'] for r in data],marker='.',ls=style,color=color,label=name)
    ax.set(xticks=range(10),xticklabels=[r['phase'] for r in data],xlabel='Reference phase (even spacing for comparison)',ylabel='Norm of 72-direction capacity change',title='Same-source mixed-action pairs | held-out pen policy')
    ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
    file=out/'contact_switch.png';fig.savefig(file,dpi=150);plt.close(fig);write_json(out/'report.json',dict(inputs=[file_pin(a),file_pin(b)],figure=file_pin(file)))
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
