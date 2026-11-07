"""Independent array/witness checks, scientific plots and descriptive report."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,default=ROOT/'outputs/dynamic_wrench_v2');args=ap.parse_args();out=args.input
    report=read(out/'report.json');screens=read(out/'all_frame_LP_results.json');contacts=read(out/'native_geometric_contacts.json')
    z=np.load(out/'designed_wrench_151.npz',allow_pickle=False);t=z['time_s'];checks={};errors={};stats={}
    checks['report16_input_and_API_contracts_pass']=all(report['checks'].values())
    checks['all_LP_statuses_known_not_pending']=all(s['conditions'][m]['solver_status'] in (0,2) for s in screens for m in ('static','raw','sensitivity'))
    checks['151frames_canonical_time']=np.max(np.abs(t-np.arange(151)*.02))<1e-12
    checks['all151_absent_pair_gaps_unknown']=np.isnan(z['native_selected_contact_distance_m'][~z['native_generated_pair_mask']]).all()
    R=z['body_rotation_w'];I=z['I_world_kg_m2'];p=report['physical'];Ri=Rotation.from_quat(np.array(p['body_iquat_wxyz'])[[1,2,3,0]]).as_matrix();Ib=Ri@np.diag(p['principal_inertia_kg_m2'])@Ri.T
    for m in ('raw','sensitivity'):
        w=z[m+'_omega_world_rad_s'];alpha=z[m+'_alpha_world_rad_s2'];wb=np.einsum('tji,tj->ti',R,w);ab=np.einsum('tji,tj->ti',R,alpha)
        independent=np.einsum('tij,tj->ti',R,ab@Ib.T+np.cross(wb,wb@Ib.T))
        error=np.max(np.abs(independent-z[m+'_required_wrench_world_N_Nm'][:,3:]));errors[m+'_world_vs_body_Euler_torque_Nm']=float(error);checks[m+'_world_body_Euler_equivalence']=error<1e-12
    # Independent SciPy SO3 logs, including declared endpoint formula.
    intervals=Rotation.from_matrix(R[1:]@np.swapaxes(R[:-1],-1,-2)).as_rotvec()/.02
    omega=np.empty((151,3));omega[1:-1]=Rotation.from_matrix(R[2:]@np.swapaxes(R[:-2],-1,-2)).as_rotvec()/.04;omega[0]=1.5*intervals[0]-.5*intervals[1];omega[-1]=1.5*intervals[-1]-.5*intervals[-2]
    error=np.max(np.abs(omega-z['raw_omega_world_rad_s']));errors['independent_Scipy_SO3_world_omega_rad_s']=float(error);checks['independent_SO3_world_omega']=error<1e-11
    checks['shortest_relative_rotation_not_near_pi']=np.max(np.linalg.norm(intervals,axis=1)*.02)<np.pi-1e-6
    checks['inertia_eigenvalues_preserved']=np.max(np.abs(np.linalg.eigvalsh(I)-np.sort(p['principal_inertia_kg_m2'])))<1e-12
    # Construct each condim4 edge independently, without calling the LP helper.
    reconstructed=[]
    for i in range(151):
        columns=[]
        for c in [c for c in contacts if c['frame']==i]:
            if c['condim']!=4:raise ValueError('This independent construction explicitly covers actual condim4')
            n,t1,t2=np.array(c['frame_rows_force_on_object_world']);arm=np.array(c['position_world_m'])-z['COM_world_m'][i];f=c['friction5']
            for tangent,mu in ((t1,f[0]),(t2,f[1])):
                for sign in (1.,-1.):
                    F=n+sign*mu*tangent;columns.append(np.r_[F,np.cross(arm,F)])
            for sign in (1.,-1.):columns.append(np.r_[n,np.cross(arm,n)+sign*f[2]*n])
        A=np.array(columns).T;E=int(z['native_edges_per_frame'][i]);reconstructed.append(A)
        if A.shape!=(6,E):raise ValueError('Native edge layout differs')
    matrix_error=max(np.max(np.abs(A-z['native_wrench_matrix_world_padded'][i,:,:A.shape[1]])) for i,A in enumerate(reconstructed))
    errors['independent_native_condim4_matrix_max_abs_mixedunits']=float(matrix_error);checks['independent_condim4_force_torsion_and_COM_moment']=matrix_error<1e-14
    normal_sums={};max_residual={}
    for m in ('static','raw','sensitivity'):
        sums=np.full(151,np.nan);force_errors=[];torque_errors=[];minimum_weights=[]
        for i,s in enumerate(screens):
            result=s['conditions'][m]
            if not result['feasible_in_native_object_wrench_screen']:continue
            weights=np.array(result['edge_weights_N']);A=reconstructed[i];residual=A@weights-z[m+'_required_wrench_world_N_Nm'][i]
            sums[i]=weights.sum();force_errors.append(np.max(np.abs(residual[:3])));torque_errors.append(np.max(np.abs(residual[3:])));minimum_weights.append(weights.min())
        checks[m+'_all_successful_LP_witnesses_nonnegative']=min(minimum_weights)>=-1e-10
        checks[m+'_all_successful_force_residuals']=max(force_errors)<1e-8
        checks[m+'_all_successful_torque_residuals']=max(torque_errors)<1e-9
        normal_sums[m]=sums;max_residual[m]={'force_N':float(max(force_errors)),'torque_Nm':float(max(torque_errors)),'minimum_edge_weight_N':float(min(minimum_weights))}
        stats[m]={'feasible_frames':int(np.isfinite(sums).sum()),'total_frames':151,'sum_normal_force_candidate_median_N':float(np.nanmedian(sums)),'sum_normal_force_candidate_max_N':float(np.nanmax(sums)),'largest_normal_force_frame':int(np.nanargmax(sums)),'largest_normal_force_time_s':float(t[np.nanargmax(sums)])}
    checks['all3_descriptive_feasibility_masks_equal']=np.array_equal(z['static_LP_feasible'],z['raw_LP_feasible']) and np.array_equal(z['raw_LP_feasible'],z['sensitivity_LP_feasible'])
    with np.load(ROOT/'outputs/dynamic_wrench_v1/designed_wrench_151.npz',allow_pickle=False) as old:
        checks['v1_v2_existing_numerical_arrays_unchanged']=all(np.array_equal(old[k],z[k],equal_nan=True) for k in old.files)
    proof={'status':'ALL_ARRAY_AND_WITNESS_CHECKS_PASS' if all(checks.values()) else 'ARRAY_OR_WITNESS_CHECK_FAILED','checks':{k:bool(v) for k,v in checks.items()},'errors':errors,'LP_witness_residuals':max_residual,'unbounded_normal_candidate_force_statistics':stats,'source_report_sha256':sha(out/'report.json'),'code_sha256':sha(__file__),'physics_steps':0,'GPU_used':False,'scope':'Saved-array/rigid-body identity and feasible object-cone witness checks, not new physics. Large candidate normal forces are unbounded LP allocations, not observed/simulated forces or actuator feasibility.'}
    write(out/'independent_array_witness_checks.json',proof)
    with (out/'all_frames.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    for i,row in enumerate(rows):
        for key in ('raw_v_world_m_s','raw_a_world_m_s2','raw_omega_world_rad_s','raw_alpha_world_rad_s2','sensitivity_v_world_m_s','sensitivity_a_world_m_s2','sensitivity_omega_world_rad_s','sensitivity_alpha_world_rad_s2'):
            for axis,component in zip('xyz',z[key][i]):row[key+'_'+axis]=float(component)
            row[key+'_norm']=float(np.linalg.norm(z[key][i]))
        for m in ('static','raw','sensitivity'):
            for axis,component in zip(('Fx_N','Fy_N','Fz_N','tx_Nm','ty_Nm','tz_Nm'),z[m+'_required_wrench_world_N_Nm'][i]):row[m+'_'+axis]=float(component)
            row[m+'_candidate_sum_normal_force_N']=float(normal_sums[m][i]) if np.isfinite(normal_sums[m][i]) else None
    with (out/'all_frames_with_demand.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True,layout='constrained')
    series=[('v_world_m_s','COM speed (m/s)',1),('a_world_m_s2','COM acceleration (m/s²)',1),('omega_world_rad_s','Angular speed (rad/s)',1),('alpha_world_rad_s2','Angular acceleration (rad/s²)',1),('required_wrench_world_N_Nm','Required contact force (N)',1),('required_wrench_world_N_Nm','Required COM torque (mNm)',1000)]
    for index,(ax,(key,label,scale)) in enumerate(zip(axes.ravel(),series)):
        for method,color,line,label2 in (('raw','#1f77b4','-','Raw FD (primary)'),('sensitivity','#e1812c','--','Window9/poly3 sensitivity')):
            a=z[method+'_'+key]
            if index==4:a=a[:,:3]
            elif index==5:a=a[:,3:]
            ax.plot(t,np.linalg.norm(a,axis=1)*scale,color=color,linestyle=line,label=label2,linewidth=1.25)
        ax.axvspan(0,.08,color='gray',alpha=.12);ax.axvspan(2.92,3.,color='gray',alpha=.12);ax.set_ylabel(label);ax.set_xlim(0,3);ax.grid(alpha=.2)
    axes[0,0].legend(loc='upper right',fontsize=9);axes[-1,0].set_xlabel('Designed mapped time (s)');axes[-1,1].set_xlabel('Designed mapped time (s)')
    fig.suptitle('Known-model demand for the fixed-palm mapped object trajectory\nNot observed velocities/forces; shaded boundaries are estimate-sensitive',fontsize=13)
    for ext in ('png','svg'):fig.savefig(out/('designed_demand_tracking.'+ext),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,8.3),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[1,1,1.5]})
    axes[0].step(t,[s['contacts'] for s in screens],where='mid',color='#4c72b0');axes[0].set_ylabel('Generated hand–object\ncontacts (count)');axes[0].grid(alpha=.2)
    raster=np.array([z[m+'_LP_feasible'] for m in ('static','raw','sensitivity')]);axes[1].imshow(raster,aspect='auto',interpolation='nearest',extent=(-.01,3.01,2.5,-.5),cmap=matplotlib.colors.ListedColormap(['#e4b4b4','#4e937a']),vmin=0,vmax=1);axes[1].set_yticks([0,1,2],['Static','Raw FD','SG9/poly3']);axes[1].set_title('Object-only native cone: green feasible / pink infeasible; counts '+', '.join(f'{stats[m]["feasible_frames"]}/{stats[m]["total_frames"]}' for m in ('static','raw','sensitivity')),fontsize=11)
    for m,label,color in (('static','Static','#4c72b0'),('raw','Raw FD','#c44e52'),('sensitivity','SG9/poly3','#dd8452')):axes[2].plot(t,normal_sums[m],'.-',label=label,color=color,linewidth=.8,markersize=2.3)
    axes[2].set_yscale('log');axes[2].set_ylabel('LP candidate sum fn (N)\nlog scale; no force cap');axes[2].set_xlabel('Designed mapped time (s)');axes[2].grid(alpha=.2);axes[2].legend(ncol=3,loc='upper right')
    for ax in axes:ax.set_xlim(0,3)
    fig.suptitle('151 correlated frames of one geometric candidate; zero physics steps\nA feasible object wrench can require unrealistic normal-force allocation',fontsize=13)
    for ext in ('png','svg'):fig.savefig(out/('native_wrench_screen.'+ext),dpi=180)
    plt.close(fig)
    write(out/'render_manifest.json',{'renderer_sha256':sha(__file__),'inputs_sha256':{'report':sha(out/'report.json'),'arrays':sha(out/'designed_wrench_151.npz')},'artifacts':{n:sha(out/n) for n in ('designed_demand_tracking.png','designed_demand_tracking.svg','native_wrench_screen.png','native_wrench_screen.svg','all_frames_with_demand.csv','independent_array_witness_checks.json')},'visual_QA':'Pending actual view_image inspection; no claim of QA from rendering alone.'})
    print(json.dumps({'status':proof['status'],'checks_passed':sum(proof['checks'].values()),'checks_total':len(proof['checks']),'stats':stats},indent=2));raise SystemExit(not all(checks.values()))


if __name__=='__main__':main()
