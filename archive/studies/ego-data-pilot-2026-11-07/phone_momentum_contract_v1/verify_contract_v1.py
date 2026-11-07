"""One saved TRAIN join plus independent analytic/AD fixtures; no model/physics."""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import torch
import momentum_contract as core

ROOT=Path(__file__).resolve().parent
P=ROOT.parent

def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def checked(r):assert pin(r['path'])==r, 'changed input '+r['path'];return Path(r['path'])
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def tensor(a):return torch.as_tensor(a,dtype=torch.float64)
def vector_rms(a):return float(np.sqrt(np.mean(np.sum(np.asarray(a)**2,axis=-1))))

def independent_rotation(v):
    v=np.asarray(v);t=np.linalg.norm(v);k=np.array([[0.,-v[2],v[1]],[v[2],0.,-v[0]],[-v[1],v[0],0.]])
    return np.eye(3)+np.sin(t)/t*k+(1-np.cos(t))/t**2*k@k

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);a=ap.parse_args()
    pr=read(a.protocol)
    assert pr['schema']=='phone_momentum_saved_TRAIN_and_analytic_contract_protocol:1'
    assert pr['source']['verifier']==pin(__file__) and pr['source']['core']==pin(core.__file__)
    for rec in [*pr['source'].values(),*pr['inputs'].values()]:checked(rec)
    assert pr['neural_training_authorized'] is False and pr['native_physics_authorized'] is False
    out=Path(pr['output']);out.mkdir(exist_ok=False);start=time.perf_counter();checks={}
    pars=read(checked(pr['inputs']['physical_parameters']))
    m=pars['mass_kg'];I=np.array(pars['I_body_3x3']);g=np.array(pars['gravity_world_m_s2'])
    assert m>0 and np.all(np.linalg.eigvalsh(I)>0)
    # Independent NumPy construction: nontrivial source AND endpoint R, anisotropic I.
    R0=independent_rotation([.3,-.2,.4]);R1=independent_rotation([-.17,.31,-.27])
    x=np.zeros(58);x[43:52]=R0.ravel();x[52:55]=[.7,-.2,.3];x[55:58]=[1.3,-.6,.9]
    J=np.array([.013,-.021,.041,.0004,-.0007,.0002]);dt=.02
    I0=R0@I@R0.T;I1=R1@I@R1.T
    expected_v=x[52:55]+g*dt+R0@J[:3]/m
    expected_w=np.linalg.solve(I1,I0@x[55:58]+R0@J[3:])
    y=x.copy();y[43:52]=R1.ravel();y[52:55]=expected_v;y[55:58]=expected_w
    args=[tensor(x),tensor(y),tensor(J),m,tensor(I),tensor(g),tensor(dt)]
    residual=core.momentum_residual_world(*args)
    checks['independent_rotated_anisotropic_linear_balance']=float(residual['linear_world_Ns'].abs().max())<1e-14
    checks['independent_rotated_anisotropic_angular_balance']=float(residual['angular_world_Nms'].abs().max())<1e-17
    req=core.contact_impulse_required_source(tensor(x),tensor(y),m,tensor(I),tensor(g),tensor(dt))
    checks['endpoint_required_impulse_matches_independent_impulse']=np.max(abs(req.numpy()-J))<1e-14
    vp=core.velocities_from_source_impulse(tensor(x),tensor(R1),tensor(J),m,tensor(I),tensor(g),tensor(dt))
    checks['conditional_velocity_closure_matches_numpy']=np.max(abs(vp.numpy()-np.r_[expected_v,expected_w]))<1e-13
    # Covariance under a world rotation; fixed source-BODY impulse stays unchanged.
    G=independent_rotation([.48,.19,-.22]);xr=x.copy();yr=y.copy()
    for row,R in [(xr,R0),(yr,R1)]:
        row[40:43]=G@row[40:43];row[43:52]=(G@R).ravel()
        row[52:55]=G@row[52:55];row[55:58]=G@row[55:58]
    req_rot=core.contact_impulse_required_source(tensor(xr),tensor(yr),m,tensor(I),tensor(G@g),tensor(dt))
    checks['world_rotation_covariance_source_BODY_components']=np.max(abs(req_rot.numpy()-J))<1e-14
    # No contact + world gravity; angular momentum conserved across a supplied R1.
    ballistic=core.velocities_from_source_impulse(tensor(x),tensor(R1),torch.zeros(6,dtype=torch.float64),m,tensor(I),tensor(g),tensor(dt))
    checks['no_contact_linear_ballistic_closure']=np.max(abs(ballistic.numpy()[:3]-(x[52:55]+g*dt)))<1e-14
    checks['no_contact_angular_momentum_closure']=np.max(abs(I1@ballistic.numpy()[3:]-I0@x[55:58]))<1e-17
    def f(j):return core.velocities_from_source_impulse(tensor(x),tensor(R1),j,m,tensor(I),tensor(g),tensor(dt))
    jac=torch.autograd.functional.jacobian(f,tensor(J).requires_grad_(True)).detach().numpy()
    expected_jac=np.zeros((6,6));expected_jac[:3,:3]=R0/m;expected_jac[3:,3:]=np.linalg.solve(I1,R0)
    jacerr=float(np.max(abs(jac-expected_jac)))
    checks['analytic_impulse_Jacobian_matches_input_AD']=jacerr<1e-10
    # Saved join only TRAIN state/target arrays. The published label archive has all
    # 960 split rows, including already-exposed TEST; explicitly no unseen claim.
    with np.load(checked(pr['inputs']['train_npz']),allow_pickle=False) as z:
        train={k:z[k].copy() for k in ['row_id_i64','family_id_i64','training_eligible_bool','x58_F64','y58_F64']}
    with np.load(checked(pr['inputs']['published_all960_labels']),allow_pickle=False) as z:
        labels={k:z[k].copy() for k in ['row_id','complete_H_bool','duration_s','source_BODY_axes_moving_COM_impulse6',
            'source_BODY_fixed_origin_impulse6','instantaneous_BODY_integral6',
            'linear_residual_world_Ns','angular_residual_world_Nms']}
    ids=read(checked(pr['inputs']['published_row_identities']))
    train_ids=set(int(v['sample_id']) for v in ids if v['split']=='TRAIN')
    # Split strings are bound to original acquisition records, not inferred by range.
    if not train_ids:
        train_ids=set(int(v['sample_id']) for v in ids if v['split']=='train')
    mask=train['training_eligible_bool'];rids=train['row_id_i64'][mask]
    checks['only_original_TRAIN_state_rows']=set(map(int,train['row_id_i64']))==train_ids
    lookup={int(r):i for i,r in enumerate(labels['row_id'])};li=np.array([lookup[int(r)] for r in rids])
    checks['eligible684_complete_H_no_silent_partial_target']=len(rids)==684 and bool(np.all(labels['complete_H_bool'][li]))
    sx=train['x58_F64'][mask];ty=train['y58_F64'][mask];ji=labels['source_BODY_axes_moving_COM_impulse6'][li];h=labels['duration_s'][li]
    residual=core.momentum_residual_world(tensor(sx),tensor(ty),tensor(ji),m,tensor(I),tensor(g),tensor(h))
    lin=residual['linear_world_Ns'].numpy();ang=residual['angular_world_Nms'].numpy()
    le=float(np.max(abs(lin-labels['linear_residual_world_Ns'][li])));ae=float(np.max(abs(ang-labels['angular_residual_world_Nms'][li])))
    checks['TRAIN_linear_residual_matches_saved_world_export']=le<1e-14
    checks['TRAIN_angular_residual_matches_saved_world_export']=ae<1e-16
    vp=core.velocities_from_source_impulse(tensor(sx),tensor(ty[:,43:52].reshape(-1,3,3)),tensor(ji),m,tensor(I),tensor(g),tensor(h)).numpy()
    req=core.contact_impulse_required_source(tensor(sx),tensor(ty),m,tensor(I),tensor(g),tensor(h)).numpy()
    wrong={}
    for name,key in [('rotating_BODY_integral','instantaneous_BODY_integral6'),('fixed_source_COM_origin','source_BODY_fixed_origin_impulse6')]:
        bad=core.momentum_residual_world(tensor(sx),tensor(ty),tensor(labels[key][li]),m,tensor(I),tensor(g),tensor(h))
        wrong[name]=dict(linear_vectorRMS_Ns=vector_rms(bad['linear_world_Ns'].numpy()),angular_vectorRMS_Nms=vector_rms(bad['angular_world_Nms'].numpy()))
    checks['rotating_BODY_integral_not_interchangeable']=wrong['rotating_BODY_integral']['linear_vectorRMS_Ns']>10*vector_rms(lin)
    checks['fixed_source_COM_origin_not_interchangeable']=wrong['fixed_source_COM_origin']['angular_vectorRMS_Nms']>10*vector_rms(ang)
    np.savez_compressed(out/'TRAIN684_momentum_closure.npz',row_id=rids,contact_impulse_required_source_BODY_moving_COM6=req,
        linear_world_residual_Ns=lin,angular_world_residual_Nms=ang,conditional_velocity6=vp,
        analytic_impulse_Jacobian=jac,analytic_expected_Jacobian=expected_jac)
    report=dict(schema='phone_momentum_saved_TRAIN_and_analytic_contract_result:1',status='PASS' if all(checks.values()) else 'FAILED_CHECKS',
        protocol=pin(a.protocol),checks=checks,passed=sum(checks.values()),total=len(checks),rows=len(rids),
        linear_vectorRMS_Ns=vector_rms(lin),angular_vectorRMS_Nms=vector_rms(ang),
        saved_linear_match_max_abs_Ns=le,saved_angular_match_max_abs_Nms=ae,
        conditional_vCOM_vectorRMS_error_m_s=vector_rms(vp[:,:3]-ty[:,52:55]),
        conditional_omega_world_vectorRMS_error_rad_s=vector_rms(vp[:,3:]-ty[:,55:58]),
        analytic_Jacobian_max_abs_error=jacerr,wrong_frame_or_origin_diagnostics=wrong,
        counts=dict(original_TRAIN_NPZ_reads=1,published_all_split_derived_label_NPZ_reads=1,original_VAL_TEST_state_NPZ_reads=0,
            input_impulse_AD_Jacobians=1,NN_constructors=0,NN_parameter_updates=0,optimizer_steps=0,native_model_reads=0,physics_steps=0),
        published_label_archive_contains_already_exposed_TEST=True,unseen_generalization_claim=False,
        conditional_endpoint_orientation_is_independently_predicted_or_feasible=False,
        complete_position_orientation_transition_defined=False,causal_contact_paths_certified=False,
        physical_parameters_measured=False,training_loss_weights_selected=False,
        outputs={'TRAIN684_closure':pin(out/'TRAIN684_momentum_closure.npz')},elapsed_s=time.perf_counter()-start)
    write(out/'report.json',report)
    print(json.dumps(dict(status=report['status'],passed=report['passed'],total=report['total'],rows=report['rows'],report=pin(out/'report.json'))))
    assert all(checks.values()), 'Contract failures retained, do not retry in same output'

if __name__=='__main__':main()
