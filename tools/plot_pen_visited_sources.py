"""Scientific figure of recorded force branches and collision-only comparisons."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.runtime.controlled_evaluation import write


def main(a):
    r=json.loads(a.replay.read_text());v=json.loads(a.review.read_text());probe=json.loads(a.collision.read_text())
    if verify_file_pin(v['run'])!=a.replay.resolve() or verify_file_pin(probe['replay'])!=a.replay.resolve() or a.output.exists():raise ValueError('Exact reviewed/probed run and fresh figure')
    original=json.loads(verify_file_pin(r['original_run']).read_text());old_review=json.loads(verify_file_pin(r['original_review']).read_text())
    old_efc=json.loads(verify_file_pin(old_review['EFC']).read_text());efc=json.loads(verify_file_pin(v['EFC']).read_text())
    mode,dt,k='STATE_PLUS_VELOCITY_SUPPORT',.005,9;c=next(c for c in original['cases'] if c['name']==mode and c['dt']==dt)
    d=c['planning'][k];packet=np.load(verify_file_pin(d['prediction']));old_e=next(e for e in old_efc['cases'] if e['name']==mode and e['dt']==dt)
    actual=np.load(verify_file_pin(old_e['wrenches']))['world_COM_wrench'][36:40];time=np.arange(1,5)*5
    a.output.mkdir();fig,(ax,table_ax)=plt.subplots(1,2,figsize=(12,5.4),gridspec_kw={'width_ratios':[1.45,1.]})
    ax.plot(time,np.linalg.norm(actual[:,:3],axis=1),color='black',lw=2.4,label='Original actual')
    ax.plot(time,np.linalg.norm(packet['W'][d['selected'],:,:3],axis=1),'--',color='#e1812c',lw=2.2,label='Original forecast')
    for repeat in range(3):
        entry=next(e for e in r['candidate_replays'] if e['mode']==mode and e['dt']==dt and e['k']==k and e['repeat']==repeat)
        z=np.load(verify_file_pin(entry['trace']));w=z['W'][d['selected'],:,:3]
        ax.plot(time,np.linalg.norm(w,axis=1),color='#3574b9',marker=['o','s','^'][repeat],ls=':',alpha=.65,label=f'57-candidate replay R{repeat}')
        inst=next(e for e in r['cases'] if e['mode']==mode and e['dt']==dt and e['k']==k and e['repeat']==repeat)
        e=next(e for e in efc['cases'] if e['name']==inst['name'] and e['dt']==dt);w=np.load(verify_file_pin(e['wrenches']))['world_COM_wrench'][:,:3]
        ax.plot(time,np.linalg.norm(w,axis=1),color='#27966e',marker=['x','+','d'][repeat],ls='-.',alpha=.65,label=f'Instrumented replay R{repeat}')
    ax.set(xlabel='Elapsed physical time from saved source (ms)',ylabel='WORLD contact-force magnitude (N)',xticks=time,title='Same saved source at 180 ms; held PD for 20 ms')
    ax.grid(alpha=.25);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,fontsize=8,frameon=False)
    table_ax.axis('off');rows=[]
    for p in probe['poses']:
        records=[e for e in probe['records'] if e['pose']==p['name']]
        rows.append([p['name'].replace('INSTRUMENTED_','Instr. '),f"{1e6*p['CPU_unfiltered_signed_distance_m']:.5f}",f"{sum(e['pair_contacts']>0 for e in records)}/5"])
    table=table_ax.table(cellText=rows,colLabels=['Recorded input','CPU distance\n(micrometers)','GPU contact\nqueries'],loc='center',cellLoc='center',colWidths=[.44,.33,.23])
    table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.8)
    for column in range(3):table[(0,column)].set_height(table[(0,column)].get_height()*1.7)
    table_ax.set_title('Separate kinematics + collision only\nPair: pen / index-finger link4',fontsize=11)
    table_ax.text(.5,.12,'Fixed input: 5 repeated results agree.\nFour labels contain two distinct pose images.\nCPU query is an unfiltered geometric comparator,\nnot a contact-force truth oracle.',ha='center',va='center',transform=table_ax.transAxes,fontsize=9)
    fig.suptitle('Visited-source replay splits with contact generation',fontsize=14)
    fig.subplots_adjust(left=.075,right=.985,top=.84,bottom=.28,wspace=.32)
    fig.text(.075,.025,'Original force/replay gates retained. Data static capacities reconstructed; no new policy, successful task or ego claim.',fontsize=8,color='#444444')
    path=a.output/'summary.png';fig.savefig(path,dpi=180);plt.close(fig)
    result=dict(schema='pen_visited_source_figure:1',replay=file_pin(a.replay),review=file_pin(a.review),collision=file_pin(a.collision),source=file_pin(__file__),figure=file_pin(path),native_integrations=0,neural_updates=0)
    write(a.output/'report.json',result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--replay',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--collision',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
