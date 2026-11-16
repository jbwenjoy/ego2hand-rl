"""Show source concentration and native component changes without selection claims."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from supermimic_research.artifacts import file_pin
from supermimic_research.strict_json import write_json


def main():
    cp=ROOT/'runs/pen_native_suggestions_comparison_v1/report.json';rp=ROOT/'runs/pen_native_suggestions_review_v1/report.json';c=json.loads(cp.read_text());r=json.loads(rp.read_text());out=ROOT/'runs/pen_native_suggestions_figure_v1'
    if out.exists():raise ValueError('Fresh native utility figure required')
    cell=next(x for x in c['cells'] if (x['role'],x['goal'],x['epsilon'])==('VALIDATE','SUPPORT',.005));records=cell['paired_against_RESPONSE_ONLY']['PATH_POINT']['records']
    source='RESIDUAL_N8861_P042';groups={g['model']:g for g in r['groups'] if g['source_name']==source and g['goal']=='SUPPORT' and g['epsilon']==.005}
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),gridspec_kw={'width_ratios':[1.25,1]})
    labels=[x['source_name'].replace('RESIDUAL_N','R/').replace('SUPPORT_N','S/').replace('_P','/') for x in records];delta=np.array([x['gain_difference'] for x in records])
    axes[0].barh(labels,delta,color=['#1f77b4' if x>0 else '#c05b45' for x in delta]);axes[0].axvline(0,color='#444',lw=.8);axes[0].invert_yaxis();axes[0].set_xlabel('Positive-direction gain difference\nPATH_POINT minus RESPONSE_ONLY');axes[0].set_title('All 8 common physically eligible sources')
    selected=records[1];axes[0].annotate(f"{selected['gain_difference']:+.5f}",(selected['gain_difference'],1),xytext=(-6,0),textcoords='offset points',ha='right',va='center',color='white',fontweight='bold')
    x=np.arange(4);width=.36
    for offset,model,color in [(-width/2,'RESPONSE_ONLY','#c98239'),(width/2,'PATH_POINT','#1f77b4')]:
        g=np.array(groups[model]['positive_component_gain']);values=np.r_[g[0],g[1],g[5],.2*g[6]];axes[1].bar(x+offset,values,width,label=model,color=color)
    axes[1].axhline(0,color='#444',lw=.8);axes[1].set_xticks(x,['COM','SO3','Geometry','0.2 x support']);axes[1].set_ylabel('Native objective component gain vs BASE');axes[1].set_title('R/8861/042: source with largest paired difference');axes[1].legend(fontsize=8)
    fig.suptitle('Recorded one-step physics probes: support goal, PD radius 0.005 rad',fontsize=13)
    fig.text(.5,.02,'Observed source counts, not independent training seeds. Hard gates retained. No trajectory/Actor/ego claim. Hand/action changes < 3e-7 in right panel.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.065,1,.93]);out.mkdir();image=out/'native_gain_concentration.png';fig.savefig(image,dpi=160);plt.close(fig)
    write_json(out/'report.json',dict(schema='pen_native_suggestion_figure:1',comparison=file_pin(cp),physical_review=file_pin(rp),source=file_pin(Path(__file__)),figure=file_pin(image),actual_native_integrations=0,optimizer_steps=0,scope='Descriptive fixed-cell plot;all8 common sources shown,no effect size/confidence or trajectory success claim.'))
    print(json.dumps(dict(figure=file_pin(image))),flush=True)


if __name__=='__main__':main()
