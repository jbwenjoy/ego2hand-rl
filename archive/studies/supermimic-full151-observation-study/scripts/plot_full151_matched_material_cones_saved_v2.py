"""Saved-only scientific rendering of two151-frame LP objective series."""
from pathlib import Path
import argparse,csv,hashlib,json,textwrap
import numpy as np

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):p=Path(p).resolve(strict=True);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();src=a.input.resolve(strict=True);out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'renderer_source.py').write_bytes(Path(__file__).read_bytes())
    data=read(src/'frame_results.json');report=read(src/'report.json');config=read(src/'config.json')
    inputs={name:pin(src/name) for name in ('frame_results.json','report.json','config.json')}
    inputs['material_contract']=report['source_pins']['material_contract'];inputs['fivecue_metadata']=report['source_pins']['cue_metadata']
    for value in inputs.values():
        if pin(value['path'])['sha256']!=value['sha256']:raise ValueError('Saved input pin differs')
    checks={};by={name:sorted([v for v in data if v['condition']==name],key=lambda x:x['frame']) for name in ('H123','H12345')}
    checks['complete_saved302_screen']=len(data)==302 and report['status']=='COMPLETED_MATCHED_CONE_CPU_SCREEN' and report['LP_calls']==302
    checks['both151_frames_not_subselected']=all(len(v)==151 and [x['frame'] for x in v]==list(range(151)) for v in by.values())
    checks['all_conditional_witness_status_retained']=all(x['status']=='FEASIBLE_CONDITIONAL_WITNESS' and x['solver_status']==0 for x in data)
    t=np.array([v['time_s'] for v in by['H123']]);v3=np.array([v['lambda_sum_normal_N'] for v in by['H123']]);v5=np.array([v['lambda_sum_normal_N'] for v in by['H12345']]);delta=v5-v3
    checks['same_time151_full3s']=np.array_equal(t,np.array([v['time_s'] for v in by['H12345']])) and np.array_equal(t,np.arange(151)*.02)
    checks['finite_raw_objectives_no_clip']=np.isfinite(v3).all() and np.isfinite(v5).all() and np.isfinite(delta).all()
    if not all(checks.values()):raise ValueError('Saved plot input contract failed: '+repr(checks))
    with (out/'normal_force_objectives151.csv').open('w',newline='',encoding='utf-8') as stream:
        w=csv.writer(stream);w.writerow(['frame','reference_time_s','H123_LP_objective_sum_lambda_normal_N','H12345_LP_objective_sum_lambda_normal_N','H12345_minus_H123_objective_N','H123_status','H12345_status','H123_residual_inf_N_equivalent','H12345_residual_inf_N_equivalent'])
        w.writerows((k,float(t[k]),float(v3[k]),float(v5[k]),float(delta[k]),by['H123'][k]['status'],by['H12345'][k]['status'],by['H123'][k]['residual_inf_N_equivalent'],by['H12345'][k]['residual_inf_N_equivalent']) for k in range(151))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'svg.fonttype':'none'})
    fig,(ax,dx)=plt.subplots(2,1,figsize=(11,7.8),sharex=True,gridspec_kw={'height_ratios':[2.2,1.]})
    fig.subplots_adjust(left=.115,right=.98,top=.84,bottom=.245,hspace=.14)
    fig.suptitle('Declared geometric cones — saved LP solver objectives',fontsize=16,fontweight='bold',y=.965)
    fig.text(.5,.919,'H123 and H12345: 151 conditional witnesses each · one source · no new LP or physical execution',ha='center',fontsize=10)
    ax.plot(t,v3,color='#2366a8',lw=1.5,marker='o',ms=2.2,label='H123 — first 3 source cues')
    ax.plot(t,v5,color='#dc781f',lw=1.4,marker='o',ms=2.2,label='H12345 — all 5 source cues')
    ax.set_ylabel('LP objective Σλ\n(N, normal-force coefficients)');ax.set_ylim(bottom=0);ax.legend(loc='upper right',framealpha=.93);ax.grid(alpha=.24)
    ax.set_title('Returned minimum-sum normal-force objective; no force cap / no optimality certificate',loc='left',pad=10)
    dx.axhline(0,color='#555',lw=1,ls='--');dx.plot(t,delta,color='#77469a',lw=1.35,marker='o',ms=2)
    dx.fill_between(t,delta,0,color='#9772b2',alpha=.14)
    dx.set_ylabel('Objective difference (N)\nH12345 − H123');dx.set_xlabel('Reference time (s)');dx.grid(alpha=.24)
    dx.set_xlim(0,3);dx.set_xticks(np.arange(0,3.01,.5));dx.set_ylim(min(0,float(delta.min()))-.06*max(1,float(np.ptp(delta))),max(0,float(delta.max()))+.10*max(1,float(np.ptp(delta))))
    notes=[
        'Same saved required inertial wrench and actual material parameters; H123 is an exact subset of the five-cue hypothesis. All 151 samples per set are shown.',
        'These are conditional polyhedral-cone solver objectives, not actual hand forces or certified optima. Contact points/Frisvad frames are hypothetical, not measured contacts.',
        'No hand q, actuator/PD limits, force cap or contact constitutive law is imposed. This does not qualify physical feasibility, task performance, or generalization.'
    ]
    y=.174
    for note in notes:
        wrapped=textwrap.fill(note,width=151);fig.text(.055,y,wrapped,ha='left',va='top',fontsize=8.7);y-=.048 if '\n' not in wrapped else .061
    fig.savefig(out/'matched_material_cone_objectives.png',dpi=180);fig.savefig(out/'matched_material_cone_objectives.svg');plt.close(fig)
    checks['all_input_files_unchanged']=all(pin(v['path'])['sha256']==v['sha256'] for v in inputs.values())
    checks={name:bool(value) for name,value in checks.items()}
    summary={'schema':'full151_matched_material_cone_saved_figure:1','status':'RENDERED_PENDING_ACTUAL_VIEW_QA','checks':checks,'passed':sum(bool(v) for v in checks.values()),'total':len(checks),'renderer':pin(__file__),'input_pins':inputs,'conditions':['H123','H12345'],'frames_per_condition':151,'time_range_s':[float(t[0]),float(t[-1])],'objective_units':'N normal-force coefficients, λ not unit-force-vector amplitude','raw_H123_minmax_N':[float(v3.min()),float(v3.max())],'raw_H12345_minmax_N':[float(v5.min()),float(v5.max())],'difference_minmax_N':[float(delta.min()),float(delta.max())],'difference_mean_N':float(delta.mean()),'physical_reference_qualified':False,'new_LP_Env_Torch_Warp_GPU_calls':0,'artifacts':{n:pin(out/n) for n in ('matched_material_cone_objectives.png','matched_material_cone_objectives.svg','normal_force_objectives151.csv')},'scope':'Saved conditional solver objectives only; no optimization rerun, certified-optimum proof, observed contact, actual hand force, physical-feasibility or method-win claim.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'summary':pin(out/'summary.json'),'png':summary['artifacts']['matched_material_cone_objectives.png'],'checks':f"{summary['passed']}/{summary['total']}"},indent=2))

if __name__=='__main__':main()
