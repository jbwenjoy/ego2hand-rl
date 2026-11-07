"""Saved arithmetic only; no model, FK, LP solver, optimizer or neural import."""
from pathlib import Path
import hashlib,json,math,sys
import numpy as np

OUT=Path(__file__).resolve().parent;ROOT=OUT.parent;P=ROOT.parent;RUN=ROOT/'support_refinement_v5_run1'
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def checked(x):
    if pin(x['path'])!=x:raise ValueError('Changed pin '+x['path'])
    return Path(x['path'])
def need(x,label):
    if not x:raise AssertionError(label)
def save(p,x):
    with p.open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False)
def dot(a,b):return math.fsum(float(x)*float(y) for x,y in zip(a,b))
def cross(a,b):return np.array([math.fsum([a[1]*b[2],-a[2]*b[1]]),math.fsum([a[2]*b[0],-a[0]*b[2]]),math.fsum([a[0]*b[1],-a[1]*b[0]])])
def close(a,b,tol=1e-10):return bool(np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0)<=tol)


def review_state(S,old):
    st=S['verification_state'];lp=S['support_LP'];op=st['support_operands']
    need(len(st['phone_geometry_rows'])==22 and len(st['self_geometry_rows'])==189,'all22/189 geometry')
    geometry=not st['unknown'];gaps=[]
    for row in st['phone_geometry_rows']:
        need(row['status']=='EXACT_NUMERICAL_IDEAL_CONVEX' and not row.get('not_exact_gap',False),'phone exact')
        lb=row['exact_gap_m']-row['guard_m'];ub=row['exact_gap_m']+row['guard_m']
        need(close(lb,row['lower_bound_m']) and close(ub,row['upper_bound_m']),'gap bound arithmetic')
        geometry=geometry and lb>=0
        if row['is_pad']:geometry=geometry and ub<=.0002;gaps.append(row['exact_gap_m'])
    for row in st['self_geometry_rows']:
        need(row['status'] in ['EXACT_NUMERICAL_IDEAL_CONVEX','DISJOINT_AABB_CERTIFICATE'],'self query status');geometry=geometry and row['lower_bound_m']>=0
    need(bool(geometry)==S['physical_geometry_pass'],'saved geometry gate')
    checked(S['selected_state'])
    with np.load(S['selected_state']['path'],allow_pickle=False) as z:need(np.array_equal(z['qpos27'],np.array(st['qpos27'])),'raw selected qpos')
    n=np.array(op['normals_force_on_object']);n/=np.linalg.norm(n,axis=1)[:,None];points=np.array(op['contact_points_world']);COM=np.array(op['object_COM']);J=np.array(op['J_robot_contact_3x20']);bias=np.array(op['robot_gravity_bias_tau']);g=np.array(op['gravity_world'])
    need(not np.any(st['qfrc_passive26_Nm'][:20]),'passive not silently omitted')
    for i,ct in enumerate(st['contacts']):
        f=ct['features'];p=ct['pairing']['closest_pair'];Ps=np.array(f['object_support_points_world_m']);Hs=np.array(f['hand_support_points_world_m'])
        pp=np.array([dot(p['object_weights'],Ps[:,j]) for j in range(3)]);hh=np.array([dot(p['hand_weights'],Hs[:,j]) for j in range(3)])
        need(ct['pairing']['status']=='UNIQUE_CLOSEST_PAIR' and p['rank']==p['unknown_count'],'actual unique pair')
        need(close(points[i],pp) and close(hh-pp,np.array(f['normal_world'])*f['signed_gap_m']),'paired point reconstruction')
        need(close(n[i],-np.array(f['normal_world'])) and close(J[i],np.array(ct['actual_hand_point_J3x26'])[:,:20]),'normal/J bound to paired hand point')
    ref=np.array([0.,1,0]);t1=np.array([ref-nn*dot(nn,ref) for nn in n]);t1/=np.linalg.norm(t1,axis=1)[:,None];t2=np.array([cross(nn,a) for nn,a in zip(n,t1)])
    rays=np.array([[nn+.7/math.sqrt(2)*(math.cos(k*math.pi/4)*a+math.sin(k*math.pi/4)*b) for k in range(8)] for nn,a,b in zip(n,t1,t2)])
    need(close(rays,lp['rays_force_world_per_normal_N']) and lp['normal_cap_N']==[3.,3.],'actual cone and3N cap')
    primary=lp['lambda_normal_N'] is not None;V=lp if primary else lp['force_recovery'];lam=np.array(V['lambda_normal_N'])
    forces=np.array([[math.fsum(lam[i,k]*rays[i,k,j] for k in range(8)) for j in range(3)] for i in range(2)])
    fr=np.array([math.fsum(forces[:,j])+op['mass']*g[j] for j in range(3)]);fn=np.array([math.fsum(x) for x in lam])
    mm=np.array([cross(p-COM,f) for p,f in zip(points,forces)]);moment=np.array([math.fsum(mm[:,j]) for j in range(3)])
    tau=np.array([bias[j]+math.fsum(J[i,k,j]*forces[i,k] for i in range(2) for k in range(3)) for j in range(20)])
    q=np.array(st['q20_rad']);offset=old['biasprm'][:,0]+old['biasprm'][:,1]*q
    lo=np.maximum.reduce([old['gain20']*old['ctrlrange'][:,0]+offset,old['actuator_forcerange'][:,0],old['joint_actfrcrange'][:,0]])
    hi=np.minimum.reduce([old['gain20']*old['ctrlrange'][:,1]+offset,old['actuator_forcerange'][:,1],old['joint_actfrcrange'][:,1]])
    for a,b,k in [(forces,V['force_world_N'],'forces'),(fr,V['force_balance_residual_world_N'],'force residual'),(fn,V['normal_force_N'],'normal'),(moment,V['moment_residual_world_Nm'],'moment'),(tau,V['tau_act_required_Nm'],'tau'),(lo,op['torque_lower'],'lower'),(hi,op['torque_upper'],'upper')]:need(close(a,b),k)
    hard=bool(lam.min()>=-1e-9 and fn.max()<=3+1e-9 and min(tau-lo)>=-1e-9 and min(hi-tau)>=-1e-9)
    force=bool(primary and hard and max(abs(fr))<=1e-8)
    need(force==lp['feasible_force_balance'],'force-hard status')
    if primary:need(close(max(abs(moment)),lp['moment_infinity_Nm']) and max(abs(moment))<=lp['solver_slack_objective_t_Nm']+1e-9,'primary moment')
    else:need(close(max(abs(fr)),V['force_infinity_N']) and not V['qualified'],'recovery separate')
    u=points[1]-points[0];length=float(np.linalg.norm(u));u/=length;r=COM-points[0];side=dot(r-u*dot(u,r),g-u*dot(u,g))
    need(close(side,S['selected_evaluation']['actual_below_line_dot_m2_s2']),'actual below side')
    reachable=False;ctrl_difference=None
    if primary:
        ctrl=np.array(S['proposed_ctrl20']);reconstructed=(tau-offset)/old['gain20']
        ctrl_difference=float(np.max(abs(ctrl-reconstructed)))
        need(close(ctrl,reconstructed,1e-12),'PD ctrl inverse')
        reachable=bool(np.isfinite(ctrl).all() and np.all(ctrl>=old['ctrlrange'][:,0]) and np.all(ctrl<=old['ctrlrange'][:,1]))
        need(reachable==S['proposed_ctrl_reachable_without_clamping'],'actual unmodified command range')
    eligible=bool(geometry and force and side>=0 and reachable and not S['unknown'])
    strict=bool(eligible and max(abs(moment))<=1e-4)
    need(strict==S['candidate_qualified_for_bounded_holding_diagnostic'],'producer legacy flag remains strict1e-4 proposal')
    return {'seed':S['seed'],'physical_geometry_pass':bool(geometry),'pad_gaps_m':gaps,'primary_force_hard_feasible':force,
      'primal_kind':'primary' if primary else 'recovery_only','force_residual_max_N':float(max(abs(fr))),'force_residual_world_N':fr.tolist(),
      'normal_force_N':fn.tolist(),'moment_residual_world_Nm':moment.tolist(),'moment_infinity_Nm':float(max(abs(moment))),
      'minimum_torque_slack_Nm':float(min(min(tau-lo),min(hi-tau))),'actual_COM_below_dot_m2_s2':side,
      'paired_line_length_m':length,'unclamped_producer_ctrl_reachable':reachable,'ctrl_inverse_max_difference_rad':ctrl_difference,
      'strict_support_proposal_pass':strict,'eligible_for_bounded_dynamic_diagnostic':eligible,
      'holding_qualified':False,'LP_optimality_independently_certified':False}


def main():
    need(not (OUT/'seed0_priority_report.json').exists(),'fresh priority report')
    rf=RUN/'report.json';need(pin(rf)['sha256']=='38881cf58f18f3d59771de5f78d80df201ed0b39ada22fd606462736a6df7ab8','expected actual report')
    r=read(rf);c=read(checked(r['protocol']))
    for v in [*c['inputs'].values(),*c['source_pins'].values()]:checked(v)
    need((RUN/'execution_source.py').read_bytes()==checked(c['source_pins']['runner']).read_bytes() and read(RUN/'protocol.json')==c,'executedsource/protocol')
    parent=P/'hand2_holding_v1/support_preparation_v1/actual_v1/report.json';pr=read(parent)
    with np.load(checked(pr['outputs']['support_operands_and_controls.npz']),allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
    S=next(s for s in r['seed_results'] if s['seed']==0);result=review_state(S,old)
    need(result['eligible_for_bounded_dynamic_diagnostic'] and not result['strict_support_proposal_pass'],'expected separate diagnostic/strict outcome')
    report={'schema':'v5_seed0_saved_diagnostic_gate_independent:1','status':'PASS_SAVED_SEED0_DIAGNOSTIC_ENROLLMENT_NOT_HOLDING',
      'source':pin(__file__),'producer_report':pin(rf),'protocol':r['protocol'],'selected_state':S['selected_state'],
      'prior_actual_PD_operands':pr['outputs']['support_operands_and_controls.npz'],'result':result,
      'eligible_for_bounded_dynamic_diagnostic':result['eligible_for_bounded_dynamic_diagnostic'],'strict_support_proposal_pass':False,
      'scope':'Existing nominal free-object simulator diagnostic enrollment only, not real-hardware safety or2s success. Saved matrices/force geometry inequality arithmetic independently recomputed; actual FK/Jacobians/convex distances remain pinned-source/prior-contract evidence. No new LP/FK/model. Large moment residual is explicitly retained. Producer legacy-named holding-diagnostic flag is actually its unchanged strict1e-4 proposal flag.',
      'model_FK_LP_solver_or_neural_imported':any(k in sys.modules for k in ['mujoco','torch','warp','mujoco_warp','scipy.optimize'])}
    save(OUT/'seed0_priority_report.json',report);print(json.dumps({'report':pin(OUT/'seed0_priority_report.json'),'result':result}))


if __name__=='__main__':main()
