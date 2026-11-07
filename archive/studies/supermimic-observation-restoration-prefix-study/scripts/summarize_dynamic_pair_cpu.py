"""Read completed H0/T0 once; recompute metrics and draw a scientific figure."""
from __future__ import annotations
import argparse,hashlib,json,os,time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):return {'path':str(Path(path).resolve()),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def rotation(q):
    q=q/np.linalg.norm(q,axis=-1,keepdims=True);w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(q.shape[:-1]+(3,3))
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--version',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    started=time.perf_counter();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    export=ROOT/'outputs/restored_prefix_control_export_v1';ref=load(export/'reference_bspline.npz');rows=[];traces=[];pins={'summarizer':pin(__file__),'reference':pin(export/'reference_bspline.npz')};allraw=[];allinitial=[]
    for kind in ('hold','track'):
        rawdir=ROOT/'outputs'/('dynamic_matched_'+kind+'_'+a.version);watch=rawdir.with_name(rawdir.name+'_watchdog')
        planpath=ROOT/'configs'/('dynamic-prefix-'+kind+'-'+a.version+'.json');plan=read(planpath);raw=load(rawdir/'raw_states.npz');solver=read(rawdir/'solver_samples.json');evaluation=read(rawdir/'evaluation.json');cap=read(rawdir/'capacity_audit.json');identity=read(watch/'engineering_identity.json');runtime=read(rawdir/'runtime.json');watched=read(watch/'state.json');init=load(rawdir/'requested_initialization_and_control.npz')
        if not identity['engineering_identity_qualified'] or not evaluation['complete'] or cap['status']!='OK' or len(raw['index'])!=61 or len(solver)!=60:raise ValueError('Unqualified engineering identity/coverage/capacity, no pair outcome')
        if evaluation['plan_sha256']!=pin(planpath)['sha256'] or runtime['plan_sha256']!=pin(planpath)['sha256']:raise ValueError('Raw/plan mismatch')
        pos=np.linalg.norm(raw['object_pose_w'][:,:3]-ref['object_COM_world_m'],axis=1);R=rotation(raw['object_pose_w'][:,3:]);angle=np.rad2deg(np.arccos(np.clip(np.sum(R[:,:,0]*ref['object_rotation_w'][:,:,0],axis=1),-1,1)))
        excess=np.maximum(np.maximum(raw['joint_lower_rad']-raw['joint_rad'],raw['joint_rad']-raw['joint_upper_rad']),0).max(axis=1)
        depth=np.array([s['penetration_m'] for s in solver]);force=np.asarray([s['actuator_force_solver_native'] for s in solver]);ratio=np.where(force>=0,force/init['force_upper_Nm'],force/init['force_lower_Nm'])
        charge=np.zeros(20);maxcharge=charge.copy()
        for dt,row in zip(np.diff(raw['raw_time_s']),ratio):charge=np.clip(charge+dt*(np.minimum(row,1)-charge/2.),0,1);maxcharge=np.maximum(maxcharge,charge)
        primary=np.arange(0,61,4);computed={'max_object_position_error_16_m':float(pos[primary].max()),'max_object_position_error_61_m':float(pos.max()),'max_shaft_angle_16_deg':float(angle[primary].max()),'max_shaft_angle_61_deg':float(angle.max()),'max_actual_joint_excess_rad':float(excess.max()),'max_penetration_m':float(depth.max()),'max_actuator_load_ratio':float(ratio.max()),'max_load_charge':float(maxcharge.max())}
        comparisons={k:abs(v-evaluation[k])<=1e-12 for k,v in computed.items()}
        if not all(comparisons.values()):raise ValueError('Saved metric mismatch '+str(comparisons))
        row={'condition':plan['control_condition'],'short_dynamic_pass':evaluation['short_dynamic_pass'],'original_gates':evaluation['gates'],**computed,'raw_metric_match':comparisons,'capacity':cap['status'],'actual_pid':watched['actual_pid'],'launcher_pid':watched['launcher_pid'],'OS_creation_utc':watched['actual_process_OS_observation']['creation_date_utc'],'worker_seconds':runtime['wall_seconds'],'watchdog_seconds_nested':watched['elapsed_s'],'steps':60,'readback_events':evaluation['control_event_count']}
        rows.append(row);traces.append((kind,pos,angle,excess,depth));allraw.append(raw);allinitial.append(init)
        pins[kind+'__plan']=pin(planpath)
        for path in list(rawdir.glob('*'))+list(watch.glob('*')):
            if path.is_file():pins[kind+'__'+path.parent.name+'__'+path.name]=pin(path)
    commoninit=all(np.array_equal(allinitial[0][k],allinitial[1][k]) for k in allinitial[0] if k!='selected_planned15_FP32_rad')
    actualinitial=all(np.array_equal(allraw[0][k][0],allraw[1][k][0]) for k in ('full_qpos','full_qvel','ctrl_rad','mocap_pos_w','mocap_quat_wxyz'))
    if not commoninit or not actualinitial:raise ValueError('Paired initialization differs')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for kind,pos,angle,excess,depth in traces:
        axes[0,0].plot(ref['planned_time61_s']*1000,pos*1000,label=kind)
        axes[0,0].scatter(ref['time_s']*1000,pos[::4]*1000,s=12)
        axes[0,1].plot(ref['planned_time61_s']*1000,angle,label=kind)
        axes[1,0].plot(ref['planned_time61_s']*1000,excess*1000,label=kind)
        axes[1,1].plot(ref['planned_time61_s'][:-1]*1000,depth*1000,label=kind)
    for ax,threshold,title,ylabel in zip(axes.flat,[20,20,1,3],['Object position vs moving reference','Directed shaft vs moving reference','Actual state hard-range excess','Solved contact depth at preintegration time'],['error [mm]','error [deg]','excess [mrad]','depth [mm]']):
        ax.axhline(threshold,color='.3',ls=':',label='original threshold');ax.set(title=title,xlabel='time [ms]',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Matched nonzero-init hold/track | one process each | 0.3 s diagnostic',fontsize=12)
    fig.savefig(out/'dynamic_pair_diagnostics.png',dpi=160);fig.savefig(out/'dynamic_pair_diagnostics.svg');plt.close(fig)
    report={'status':'PAIRED_COMPLETE_METRICS_RECOMPUTED','conditions':rows,'same_requested_initialization':commoninit,'same_actual_initialization':actualinitial,'source_pins':pins,'original_numeric_gates_unchanged':True,'same_source_lineage_N':1,'attempts_per_condition':1,'learning_started':False,'new_geometry_queries':0,'new_physics_steps':0,'GPU_used_by_summarizer':False,'figure_QA':'PENDING_ACTUAL_VIEW','wall_seconds':time.perf_counter()-started,'scope':'Observed sequential known-history diagnostic only; no full3s/learning/CHORD/independentego benefit. First condition quality was not used to alter second.'}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='source_pins'},indent=2))
if __name__=='__main__':main()
