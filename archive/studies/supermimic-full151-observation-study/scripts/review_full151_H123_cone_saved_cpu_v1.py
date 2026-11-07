"""Saved-only algebra review of the fixed H123 cone diagnostic; no LP/query."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def pin(p):p=Path(p).resolve(strict=True);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def rot(q):
    w,x,y,z=np.asarray(q)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def rays(points,n):
    n=n/np.linalg.norm(n,axis=-1,keepdims=True);sign=np.where(n[:,2]>=0,1.,-1.);a=-1/(sign+n[:,2]);b=n[:,0]*n[:,1]*a
    t1=np.stack((1+sign*n[:,0]**2*a,sign*b,-sign*n[:,0]),-1);t1/=np.linalg.norm(t1,axis=-1,keepdims=True)
    t2=np.stack((b,sign+n[:,1]**2*a,-n[:,1]),-1);t2/=np.linalg.norm(t2,axis=-1,keepdims=True)
    theta=np.arange(8)*2*np.pi/8
    force=n[:,None]+.1*(np.cos(theta)[None,:,None]*t1[:,None]+np.sin(theta)[None,:,None]*t2[:,None]);force/=np.linalg.norm(force,axis=-1,keepdims=True)
    force=np.concatenate((force,n[:,None]),axis=1);return force.reshape(27,3),np.cross(points[:,None],force).reshape(27,3)

def main(output):
    start=time.perf_counter();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);checks={};facts={}
    r=ROOT/'outputs/full151_declared_H123_dynamic_cone_LP_v1';report=read(r/'report.json');rows=read(r/'frame_results.json');data=load(r/'matrices_and_witnesses.npz');src=report['source_pins']
    target=load(src['targets']['path']);ref=load(src['reference']['path']);ctx=load(src['context']['path'])
    def check(k,v):checks[k]=bool(v)
    check('all_source_and_output_pins_current',all(pin(v['path'])['sha256']==v['sha256'] for v in src.values()) and pin(r/'frame_results.json')==report['results'] and pin(r/'matrices_and_witnesses.npz')==report['matrices'])
    obj=report['object_body_id'];mass=float(ctx['body_mass'][0,obj]);Q=rot(ctx['body_iquat'][0,obj]);I=Q@np.diag(ctx['body_inertia'][0,obj].astype(float))@Q.T
    check('actual_mass_inertia_rebuilt_from_context',mass==report['actual_mass_kg'] and np.array_equal(I,data['I_body_kg_m2']) and np.array_equal(I,np.array(report['I_body_kg_m2'])))
    radius=float(target['object_nominal_enclosing_radius_m']);R=ref['object_rotation_w'][::4];acc=ref['object_a_world_m_s2'][::4];w=ref['object_omega_world_rad_s'][::4];alpha=ref['object_alpha_world_rad_s2'][::4];g=data['gravity_world_m_s2']
    F=np.einsum('tji,tj->ti',R,mass*(acc-g));wb=np.einsum('tji,tj->ti',R,w);ab=np.einsum('tji,tj->ti',R,alpha);tau=np.einsum('ij,tj->ti',I,ab)+np.cross(wb,np.einsum('ij,tj->ti',I,wb));required=np.c_[F,tau/radius]
    err=lambda a,b:float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    facts['force_reconstruction_max_N']=err(F,data['required_force_body_N']);facts['tau_reconstruction_max_Nm']=err(tau,data['required_moment_body_Nm']);facts['required_scaled_reconstruction_max_N']=err(required,data['required_wrench_N_equivalent'])
    check('required_force_torque_world_to_BODY_formula',max(facts.values())<=1e-12)
    rebuilt=np.empty_like(data['A_N_equivalent']);support_error=[]
    for k in range(151):
        f,t=rays(target['point_com_body_m'][k],target['inward_normal_body'][k]);rebuilt[k]=np.c_[f,t/radius].T
        support=np.maximum(0,(rebuilt[k].T@target['support_basis'].T).max(axis=0));support_error.append(err(support,target['support512'][k,0,0]))
    facts['A_reconstruction_max']=err(rebuilt,data['A_N_equivalent']);facts['support512_reconstruction_max']=max(support_error)
    check('all151_unit_force_Frisvad27_COM_radius_once',facts['A_reconstruction_max']<=1e-12 and facts['support512_reconstruction_max']<=1e-12)
    eligible=np.array([x['status']=='FEASIBLE' for x in rows]);infeasible=np.array([x['status']=='INFEASIBLE' for x in rows]);unknown=~(eligible|infeasible);lam=data['lambda_N'];res=np.einsum('tij,tj->ti',rebuilt[eligible],lam[eligible])-required[eligible]
    facts['feasible_frames']=np.flatnonzero(eligible).tolist();facts['eligible_real0to149']=np.flatnonzero(eligible[:150]).tolist();facts['eligible_terminal150']=bool(eligible[150]);facts['feasible_lambda_min_N']=float(lam[eligible].min());facts['feasible_residual_inf_N_equivalent']=float(np.abs(res).max());facts['lambda_sum_minmax_N']=[float(lam[eligible].sum(1).min()),float(lam[eligible].sum(1).max())]
    check('23_finite_nonnegative_primal_witnesses_recomputed',eligible.sum()==23 and np.isfinite(lam[eligible]).all() and lam[eligible].min()>=0 and np.abs(res).max()<=1e-7)
    check('128_solverinfeasible_not_zero_imputed_witness',infeasible.sum()==128 and not unknown.any() and np.isnan(lam[infeasible]).all() and all(x['solver_status']==2 for x in rows if x['status']=='INFEASIBLE'))
    check('all151_rows_retained_in_original_order',len(rows)==151 and all(x['frame']==i and x['time_s']==target['time_s'][i] for i,x in enumerate(rows)))
    check('mask_nonPonly_in_real_intervals',np.any(eligible[:150]) and np.all(np.any(target['support512'][:150,0,0,::8][eligible[:150]]>.001,axis=1)))
    check('frame0_not_eligible_and_late_tail_unsupported',not eligible[0] and not eligible[72:].any())
    facts['native_real_attraction_duty_fraction']=float(eligible[:150].mean());facts['maximum_integrated_real_support_bonus_if_unitmatch']=float(eligible[:150].sum()*.02)
    edges=np.r_[0,np.flatnonzero(eligible[1:]!=eligible[:-1])+1,151];facts['contiguous_masks']=[{'begin':int(a),'end_exclusive':int(b),'eligible':bool(eligible[a])} for a,b in zip(edges[:-1],edges[1:])]
    result={'schema':'full151_H123_dynamic_cone_saved_review:1','status':'PASS_SAVED_ALGEBRA_NOT_NEW_FEASIBILITY_SOLVE' if all(checks.values()) else 'SAVED_REVIEW_FAIL','reviewer':pin(__file__),'LP_report':pin(r/'report.json'),'checks':checks,'passed':sum(checks.values()),'total':len(checks),'facts':facts,'wall_s':time.perf_counter()-start,
      'limits':['SolverINFEASIBLE labels are retained from savedHiGHS status2. This review does not regenerate Farkas dual certificates or solveLP.','FEASIBLE requires only the narrowmu.1 unitray cone primal witness with unbounded coefficients. It is not an actualnative-friction/contactforce/handreach/actuator certificate.','A model-specific23-frame support gate can be an interimablation. It cannot replace the full151 motion objective or declare the128unsupported frames released.'],
      'LP_calls':0,'native_queries':0,'Env_WM_GPU':False}
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({'status':result['status'],'passed':result['passed'],'total':result['total'],'report':pin(out/'report.json')}));return 0 if all(checks.values()) else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();raise SystemExit(main(a.output))
