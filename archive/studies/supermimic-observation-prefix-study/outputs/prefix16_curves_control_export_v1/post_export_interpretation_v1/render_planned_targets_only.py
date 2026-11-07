"""Artifact-only presentation correction; saved data, no curve/control recompute."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json
import numpy as np
BASE=Path('D:/workspace/supermimic-observation-prefix-study/outputs/prefix16_curves_control_export_v1')
OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin=lambda p:{'path':str(p),'sha256':sha(p)}
def render_diagnostics(output,time,q,knots,ticks,qrange,lower,upper,control,params,gates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    def excess(values):return np.maximum(np.maximum(lower-values,values-upper),0).max(axis=1)*1000
    ax[0,0].plot(np.arange(61)*5,excess(ticks['q_goal_rad']),label='61 planned ticks',color='#31688e')
    ax[0,0].scatter(time*1000,excess(q),label='16 knots',color='#35b779',s=18,zorder=3)
    extremetimes=[];extremeexcess=[]
    for row in qrange['all300_segments_joints']:
        for s,v in zip(row['candidate_s'],row['candidate_values_rad']):
            e=max(lower[row['joint']]-v,v-upper[row['joint']],0.)
            if e>0:extremetimes.append((row['segment']+s)*20);extremeexcess.append(e*1000)
    ax[0,0].scatter(extremetimes,extremeexcess,color='#b63679',marker='x',s=22,label='all polynomial extrema with excess')
    ax[0,0].set(title='Continuous q hard-range audit (zero slack)',xlabel='reference time [ms]',ylabel='maximum excess [mrad]');ax[0,0].legend(fontsize=7)
    rates=np.abs(control['actual_target_rate_14_rad_s']).max(axis=1)
    ax[0,1].plot(np.arange(1,15)*20,rates,marker='o',ls='None',label='14 planned update values only');ax[0,1].axhline(15,color='0.4',ls=':',label='15 rad/s cap')
    ax[0,1].set(title='15 post-C2 planned targets / 14 planned updates',xlabel='planned update time [ms]',ylabel='max planned joint rate [rad/s]');ax[0,1].legend(fontsize=7)
    lo=np.maximum(params.actuator_force_bounds[:,0],params.joint_force_bounds[:,0]);hi=np.minimum(params.actuator_force_bounds[:,1],params.joint_force_bounds[:,1])
    for name,color,prefix in [('track','#31688e',''),('hold','#b63679','hold_')]:
        ratios=[]
        for grouping in ['affine','difference']:
            force=control[prefix+'reference_PD_'+grouping+'_60_FP32_Nm'];ratios.append(np.where(force>=0,force/hi,force/lo).max(axis=1))
        ax[1,0].plot(np.arange(60)*5,np.maximum(*ratios),label=name+' max of two FP32 groupings',color=color)
    ax[1,0].axhline(1,color='0.4',ls=':',label='raw PD cap');ax[1,0].set(title='Reference-state raw PD prediction; no dynamics',xlabel='planned solver prestate [ms]',ylabel='max actuator/joint force ratio');ax[1,0].legend(fontsize=7)
    def shaft(rotation):
        v=rotation[:,:,0];return np.rad2deg(np.arccos(np.clip(v@v[0],-1,1)))
    ax[1,1].plot(np.arange(61)*5,shaft(ticks['object_rotation_w']),label='61 continuous reference ticks');ax[1,1].scatter(time*1000,shaft(knots['object_rotation_w']),color='#35b779',s=18,label='16 frozen pose knots')
    ax[1,1].set(title='Reference directed shaft excursion (not tracking)',xlabel='reference time [ms]',ylabel='deg from initial shaft');ax[1,1].legend(fontsize=7)
    for a in ax.flat:a.grid(alpha=.2)
    fig.suptitle('Prefix16 CPU planned targets | failed gates retained\nNo dynamics/geometry qualification; source data unchanged',fontsize=12)
    files=[Path(output)/'prefix_cpu_diagnostics_planned.png',Path(output)/'prefix_cpu_diagnostics_planned.svg']
    for file in files:fig.savefig(file,dpi=160)
    plt.close(fig)
    return [pin(p) for p in files]

r=json.loads((BASE/'receipt.json').read_text());before=sha(BASE/'receipt.json')
with np.load(BASE/'reference_curves.npz') as z:ref={k:z[k].copy() for k in z.files}
with np.load(BASE/'control_schedule.npz') as z:control={k:z[k].copy() for k in z.files}
q0pin=r['source_pins']['q0_source']
with np.load(q0pin['path']) as z:lo=z['joint_lower_rad'].copy();hi=z['joint_upper_rad'].copy()
design=json.loads(Path(r['source_pins']['static_preload_design']['path']).read_text());matrix=Path(r['source_pins']['static_preload_design']['path']).with_name('static_matrices.npz');assert sha(matrix)==design['static_matrices_sha256']
with np.load(matrix) as z:params=SimpleNamespace(actuator_force_bounds=z['actuator_force_bounds'].copy(),joint_force_bounds=z['joint_force_bounds'].copy())
knots={k[len('knot16__'):]:v for k,v in ref.items() if k.startswith('knot16__')};ticks={k[len('tick61__'):]:v for k,v in ref.items() if k.startswith('tick61__')}
files=render_diagnostics(OUT,ref['time_s'],ref['q_goal_rad'],knots,ticks,json.loads((BASE/'whole_q_range.json').read_text()),lo,hi,control,params,r['gates'])
assert sha(BASE/'receipt.json')==before
print(json.dumps({'new_figures':files,'original_receipt_unchanged':True},indent=2))
