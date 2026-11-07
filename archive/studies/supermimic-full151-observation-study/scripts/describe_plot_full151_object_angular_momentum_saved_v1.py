"""All-row saved angular-momentum description, no new model/backend execution."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def pin(p):
    p=Path(p).resolve(strict=True)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--independent-review',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();base=a.run_dir.resolve(strict=True);report=read(base/'report.json');review=read(a.independent_review)
    assert review['passed']==review['total']==9 and all(v is True for v in review['checks'].values())
    rows=[json.loads(x) for x in (base/'all1808_rows.jsonl').read_text(encoding='utf8').splitlines()]
    with np.load(base/'all1808_angular_momentum_arrays.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    assert len(rows)==1808 and data['closure_interpretation_eligible'].all()
    with np.load(base/'actual_inertia_observer.npz',allow_pickle=False) as z:inertia={k:z[k].copy() for k in z.files}
    torque=data['tau_total_Nm_F64'];finite_change=data['delta_L_world_F64']/data['dt_wp32_s_FP32'].astype('f8')[:,None]
    residual=data['torque_residual_Nm_F64'];norm=np.linalg.norm(residual,axis=1)
    out=a.output.resolve();assert not out.exists();out.mkdir()
    summary=[]
    for path in ('ALL','A0','A2'):
        for mode in ('ALL','donor','baseline','direction'):
            mask=np.asarray([(path=='ALL' or x['path']==path) and (mode=='ALL' or x['mode']==mode) for x in rows])
            e=norm[mask];t=torque[mask];d=finite_change[mask]
            trms=float(np.sqrt(np.mean(np.sum(t*t,axis=1))));erms=float(np.sqrt(np.mean(e*e)))
            summary.append({'path':path,'mode':mode,'rows':int(mask.sum()),'torque_residual_vector_RMS_Nm':erms,
                **{f'residual_norm_{label}_Nm':float(np.quantile(e,q)) for label,q in [('min',0),('median',.5),('p90',.9),('p99',.99),('max',1)]},
                'contact_torque_vector_RMS_Nm':trms,'finite_delta_L_over_h_vector_RMS_Nm':float(np.sqrt(np.mean(np.sum(d*d,axis=1)))),
                'ratio_RMS_residual_over_RMS_contact_posthoc':erms/trms if trms else None,'accuracy_gate':None})
    with (out/'all_path_mode_summary.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    np.savez_compressed(out/'all1808_descriptive_arrays.npz',residual_torque_norm_Nm=norm,observed_finite_delta_L_over_h_Nm=finite_change)
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained');fig.get_layout_engine().set(rect=(0,.14,1,.86))
    ax=axes[0]
    for j,(axis,color) in enumerate(zip(('x','y','z'),('#2675b8','#d77826','#549e77'))):
        ax.scatter(torque[:,j]*1e3,finite_change[:,j]*1e3,s=10,alpha=.6,c=color,label=axis)
    bound=float(max(np.abs(torque).max(),np.abs(finite_change).max()))*1.05e3
    ax.plot([-bound,bound],[-bound,bound],c='gray',lw=1,ls='--');ax.set_xlim(-bound,bound);ax.set_ylim(-bound,bound)
    ax.set_aspect('equal',adjustable='box');ax.legend(title='World axis');ax.grid(alpha=.2)
    ax.set(xlabel='All boundary contact moment at solver PRE [mN m]',ylabel='Observed finite delta L / h [mN m]',title='All 1808 substeps, all three components')
    ax=axes[1];ax.plot(range(1808),norm*1e3,'o',ms=2,alpha=.65,c='#4c667b')
    for boundary in (600,904,1504):ax.axvline(boundary,c='gray',lw=.8,ls='--')
    ax.set_yscale('symlog',linthresh=.001);ax.set_ylim(bottom=0);ax.grid(alpha=.2)
    ax.set(xlabel='Original serialized row index (boundaries marked)',ylabel='Equivalent torque residual norm [mN m]',title='All rows; symlog, linear below 0.001 mN m')
    fig.suptitle('Conditional world angular momentum: finite endpoint change vs PRE contact impulse',fontsize=14)
    fig.text(.5,.025,'Normalized-F64 quaternion observer; not native GPU bit replay. All dependent donor/restored rows retained.\nNo residual accuracy gate. Finite orientation step, implicit gyro, solver and observer terms are not separately identified.',ha='center',fontsize=10)
    fig.savefig(out/'all1808_angular_momentum_consistency.png',dpi=145,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained');fig.get_layout_engine().set(rect=(0,.1,1,.9))
    for row,path in enumerate(('A0','A2')):
        ids=[j for j,x in enumerate(rows) if x['path']==path and x['mode']=='donor'];assert len(ids)==600
        t=np.arange(600)*float(data['dt_wp32_s_FP32'][0])
        for axis in range(3):
            ax=axes[row,axis]
            for key,label,color in [('tau_HO_total_Nm_F64','Hand-object','#2675b8'),('tau_other_total_Nm_F64','Other object boundary','#d77826')]:
                ax.plot(t,data[key][ids,axis]*1e3,lw=1.,c=color,label=label)
            ax.plot(t,torque[ids,axis]*1e3,c='#549e77',lw=1.,ls=':',label='Total contact moment')
            ax.plot(t,finite_change[ids,axis]*1e3,c='#4d5863',lw=.9,ls='--',label='Observed delta L / h')
            ax.set(title=f'{path}, world {"xyz"[axis]}: all 600 donor substeps',xlabel='Canonical PRE-tick time [s]',ylabel='COM moment [mN m]');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('All donor axes: HO and terrain moments stay separate; direct torque counted once',fontsize=14)
    fig.text(.5,.025,'The finite momentum change is a chosen endpoint observer, not an exact PRE-torque target.\nContact positions and COM use solver PRE; raw endpoint quaternions and angular velocities remain saved unchanged.',ha='center',fontsize=10)
    fig.savefig(out/'both_donors_all3_moment_axes.png',dpi=140,bbox_inches='tight');plt.close(fig)
    def maxabs(k):return float(np.max(np.abs(data[k])))
    peak=int(np.argmax(norm))
    d={'schema':'full151105_object_angular_momentum_descriptive_figures:1','producer':pin(base/'report.json'),'independent9':pin(a.independent_review),'renderer':pin(__file__),
       'all1808_preserved':True,'all12_path_mode_summaries':summary,'actual_principal_I_kg_m2':inertia['actual_inertia_principal_FP32'].tolist(),
       'actual_raw_body_iquat_wxyz':inertia['actual_iquat_raw_FP32'].tolist(),
       'peak_row':{k:rows[peak][k] for k in ('row_id','path','mode','branch_id','phase','substep')},'peak_residual_Nm':float(norm[peak]),
       'source_phase_descriptors':{'PRE_pose_minus_pre_qpos_position_maxabs_m':maxabs('solver_PRE_minus_pre_qpos_position_F64'),
          'PRE_R_minus_pre_qpos_R_maxabs_entry':maxabs('solver_PRE_minus_pre_qpos_R_F64'),
          'PRE_COM_minus_pre_qpos_position_maxabs_m':maxabs('solver_PRE_COM_minus_pre_qpos_position_F64'),
          'HO_recomputed_minus_original_descriptor_maxabs_Nm':maxabs('computed_minus_original_HO_descriptor_F64'),
          'raw_pre_quaternion_max_abs_norm_minus1':float(np.max(np.abs(data['quat_pre_raw_norm_F64']-1))),
          'raw_post_quaternion_max_abs_norm_minus1':float(np.max(np.abs(data['quat_post_raw_norm_F64']-1)))},
       'quantiles_and_RMS_ratio':'post hoc full-population descriptors, not an accuracy gate or error/causal bound',
       'new_MJB_models_physics_GPU':0,'outputs':{p.name:pin(p) for p in out.iterdir()}}
    (out/'report.json').write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({'ALL':summary[0],'peak':d['peak_row'],'phase_descriptors':d['source_phase_descriptors'],'output':str(out)}))


if __name__=='__main__':main()
