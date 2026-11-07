"""Saved-only scientific comparison; no device, model or physics execution."""
from pathlib import Path
import csv,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def pin(p):return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def rotation(q):
    q=np.asarray(q,float);q=q/np.linalg.norm(q,axis=-1,keepdims=True);w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(q.shape[:-1]+(3,3))
def table(p,header,rows):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)
def main():
    out=ROOT/'outputs/full151_fixed_checkpoint_comparison_v2'
    if out.exists():raise FileExistsError('Fresh figure only')
    sourcepin=pin(Path(__file__));bundle=read(ROOT/'outputs/full151_native_input_v1/bundle.json')
    refpin=bundle['source_pins']['reference_arrays'];ref=npz(Path(refpin['path']))
    if pin(Path(refpin['path']))!=refpin:raise ValueError('Reference changed')
    cases=[('A0',ROOT/'outputs/full151_native_A0_v1','#596b7c'),('A1L final40 (smoke)',ROOT/'outputs/full151_native_A1L_policy_eval_smoke40_v1','#d59132'),('A1L final2000',ROOT/'outputs/full151_native_A1L_policy_eval_primary2000_v1','#206d9f')]
    inputs={'reference':refpin};all_data=[];checks={};t=np.arange(601)*.005
    for label,run,color in cases:
        paths={k:run/n for k,n in [('raw','raw_states.npz'),('solver','solver_samples.json'),('events','control_events.json'),('evaluation','evaluation.json')]}
        for k,p in paths.items():inputs[label+'__'+k]=pin(p)
        raw=npz(paths['raw']);solver=read(paths['solver']);events=read(paths['events']);ev=read(paths['evaluation'])
        if raw['joint_rad'].shape!=(601,20) or len(solver)!=600 or len(events)!=150 or not ev['complete']:raise ValueError('Incomplete comparison episode')
        pos=np.linalg.norm(raw['object_pose_w'][:,:3]-ref['object_COM_world_m'],axis=1)
        R=rotation(raw['object_pose_w'][:,3:]);angle=np.degrees(np.arccos(np.clip(np.sum(R[:,:,0]*ref['object_rotation_w'][:,:,0],axis=1),-1,1)))
        excess=np.maximum(np.maximum(raw['joint_lower_rad']-raw['joint_rad'],raw['joint_rad']-raw['joint_upper_rad']),0).max(1)
        depth=np.array([s['penetration_m'] for s in solver]);force=np.array([s['actuator_force_solver_native'] for s in solver])
        ratios=np.where(force>=0,force/raw['force_upper_Nm'],force/raw['force_lower_Nm']);charge=np.zeros(20);charges=[]
        for dt,row in zip(np.diff(raw['raw_time_s']),ratios):charge=np.clip(charge+dt*(np.minimum(row,1)-charge/2),0,1);charges.append(charge.max())
        u=np.array([e['actualu'][0] for e in events],np.float32).astype(float)
        rates=np.abs(np.diff(np.vstack((raw['ctrl_rad'][0],u)),axis=0)).max(1)/.02
        metrics={'max_object_position_error_151_m':float(pos[::4].max()),'max_object_position_error_601_m':float(pos.max()),'max_shaft_angle_151_deg':float(angle[::4].max()),'max_shaft_angle_601_deg':float(angle.max()),'max_actual_joint_excess_rad':float(excess.max()),'max_penetration_m':float(depth.max()),'max_actual_target_rate_rad_s':float(rates.max()),'max_actuator_load_ratio':float(ratios.max()),'max_load_charge':float(max(charges))}
        for k,v in metrics.items():checks[label+'__'+k]=bool(np.isclose(v,ev[k],rtol=0,atol=1e-10))
        all_data.append(dict(label=label,color=color,pos=pos,angle=angle,excess=excess,depth=depth,rates=rates,charge=np.array(charges),ev=ev,metrics=metrics))
    if not all(checks.values()):raise ValueError('Saved metric mismatch '+repr([k for k,v in checks.items() if not v]))
    out.mkdir(parents=True);(out/'renderer_source_before.py').write_bytes(Path(__file__).read_bytes())
    for i,d in enumerate(all_data):
        table(out/f'case{i}_poststates601.csv',['time_s','position_error_m','shaft_error_deg','joint_excess_rad'],zip(t,d['pos'],d['angle'],d['excess']))
        table(out/f'case{i}_solver600.csv',['preintegration_time_s','penetration_m','charge_postupdate'],zip(t[:-1],d['depth'],d['charge']))
        table(out/f'case{i}_actual_target_rate150.csv',['interval_time_s','max_abs_rate_rad_s'],zip(np.arange(150)*.02,d['rates']))
    table(out/'summary.csv',['case','gates_passed','gates_total',*all_data[0]['metrics']],[[d['label'],sum(d['ev']['gates'].values()),len(d['ev']['gates']),*d['metrics'].values()] for d in all_data])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    for d in all_data:
        style=dict(color=d['color'],label=d['label'],lw=1.1)
        axes[0,0].plot(t,d['pos']*1000,**style);axes[0,1].plot(t,d['angle'],**style);axes[0,2].plot(t[:-1],d['depth']*1000,**style)
        axes[1,0].plot(np.arange(150)*.02,d['rates'],**style);axes[1,1].plot(t,d['excess']*1000,**style);axes[1,2].plot(t[1:],d['charge'],**style)
    settings=[('Position error: 601 states; gate at 151 knots','error (mm)',20),('Directed shaft: 601 states; gate at 151 knots','angle (degrees)',20),('Generated penetration: 600 solver samples','overlap (mm)',3),('Actual target rate: startup + 149 updates','rate (rad/s)',15),('Actual joint hard-limit excess: 601 states','excess (mrad)',1),('Integrated actuator charge: 600 updates','charge',.8)]
    for ax,(title,y,gate) in zip(axes.flat,settings):
        ax.set(title=title,ylabel=y,xlabel='canonical episode time (s)',xlim=(0,3));ax.axhline(gate,color='#a33030',ls='--',lw=1);ax.grid(alpha=.18)
    axes[0,0].set_yscale('symlog',linthresh=1);axes[0,0].set_ylim(bottom=0);axes[0,0].text(.02,.95,'symlog, linear below 1 mm',transform=axes[0,0].transAxes,va='top',fontsize=8)
    fig.suptitle('Full151 known-rough reference | fixed final checkpoints | seed 0',fontsize=16,fontweight='bold',y=.99)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.955),ncol=3,frameon=False)
    scores=';  '.join(f"{d['label']}: {sum(d['ev']['gates'].values())}/13 gates" for d in all_data)
    fig.text(.035,.045,scores+'\nAll three episodes fail task quality. One lineage, one seed; no method-benefit or real-ego claim. Dashed lines: unchanged original gates.\nAll 150 actions retained without quality stop/reset/assist. Zero generated overlap is not positive clearance. Startup target-rate convention: delta / 20 ms.',fontsize=9,va='bottom')
    fig.tight_layout(rect=(.02,.12,.99,.91),h_pad=2,w_pad=2)
    fig.savefig(out/'full151_fixed_checkpoint_comparison.png',dpi=170);fig.savefig(out/'full151_fixed_checkpoint_comparison.svg');plt.close(fig)
    if any(k in sys.modules for k in ('torch','warp','mujoco','mujoco_warp','dex_supermimic')):raise AssertionError('Saved-only renderer imported runtime')
    if any(pin(Path(p['path']))!=p for p in inputs.values()):raise ValueError('Inputs changed')
    report={'schema':'full151_fixed_checkpoint_saved_comparison:1','status':'ALL_EPISODES_TASK_FAIL_RETAINED','source':sourcepin,'inputs':inputs,'checks':checks,'passed':sum(checks.values()),'total':len(checks),'cases':[{k:d[k] for k in ('label','metrics')}|{'gates':d['ev']['gates']} for d in all_data],'scope':'Saved-only plots; no physical replay, Actor/WM inference, new grading thresholds, checkpoint selection or method comparison. All 3 fixed episodes remain failures.','new_physics_or_GPU_calls':0,'visual_QA':'PENDING_ACTUAL_VIEW','artifacts':[pin(p) for p in sorted(out.iterdir()) if p.is_file()]}
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'summary':pin(out/'summary.json'),'PNG':str(out/'full151_fixed_checkpoint_comparison.png'),'saved_metric_checks':len(checks)}))
if __name__=='__main__':main()

