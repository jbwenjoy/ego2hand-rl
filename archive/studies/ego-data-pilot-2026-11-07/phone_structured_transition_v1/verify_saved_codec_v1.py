"""Physical target-codec arithmetic on committed causal data, no neural model."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
import structured_codec as codec
H=Path(__file__).resolve().parent;P=H.parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def checked(r):assert pin(r['path'])==r;return Path(r['path'])
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def tensor(a):return torch.as_tensor(a,dtype=torch.float64)
def rms(a):return float(np.sqrt(np.mean(np.sum(a*a,axis=-1))))
out=H/'completed_v1';protocol=H/'protocol_v1.json';assert not out.exists() and not protocol.exists()
pr=dict(schema='causal_pair_impulse_structured_codec_saved_protocol:1',sources={'verifier':pin(__file__),'codec':pin(codec.__file__)},
    inputs={'acquisition_report':pin(P/'phone_causal_train_acquisition_v1/acquired_v1/report.json'),
        'impulse_report':pin(P/'phone_causal_impulse_labels_v1/completed_v2/report.json'),
        'physical_parameters':pin(P/'phone_object_wrench_labels_v1/completed_v1/physical_parameters.json')},
    neural_training_or_native_authorized=False,scope='All2942 executed known-state response rows for algebraic codec validation; no model selection/normalizer fit or unseen claim.')
write(protocol,pr);out.mkdir();start=time.perf_counter();torch.set_num_threads(4)
acq=read(checked(pr['inputs']['acquisition_report']));imp=read(checked(pr['inputs']['impulse_report']));pars=read(checked(pr['inputs']['physical_parameters']))
with np.load(checked(imp['outputs']['all_labels']),allow_pickle=False) as z:
    J=z['impulse_source_BODY_moving_COM22x6'].copy();known=z['physical_labels_known'].copy();dur=z['actual_duration_s'].copy()
    linear=z['linear_momentum_residual_world_Ns'].copy();angular=z['angular_momentum_residual_world_Nms'].copy()
x=np.full((2952,58),np.nan);y=x.copy();order=[]
for rec in acq['arrays'].values():
    with np.load(checked(rec),allow_pickle=False) as z:
        ids=z['branch_id_i64'];x[ids]=z['x58_F64'];y[ids]=z['y58_at_H_F64'];order.extend(map(int,ids))
assert sorted(order)==list(range(2952)) and known.sum()==2942 and np.isnan(y[~known]).all()
heads=codec.encode_observed_numpy(x[known],y[known],J[known],dur[known],np.array(pars['gravity_world_m_s2']))
tx=tensor(x[known]);ty=tensor(y[known]);h=tensor(dur[known]);I=tensor(pars['I_body_3x3']);g=tensor(pars['gravity_world_m_s2']);mass=pars['mass_kg']
th={k:tensor(v) for k,v in heads.items()};pred=codec.decode_physical_heads(tx,th,h,mass,I,g).numpy()
checks={}
checks['all2942_executed_known_rows_only']=bool(len(pred)==2942 and np.isfinite(pred).all())
checks['joint40_observed_head_roundtrip']=float(np.max(abs(pred[:,:40]-y[known,:40])))<1e-14
checks['COM_head_continuum_baseline_roundtrip']=float(np.max(abs(pred[:,40:43]-y[known,40:43])))<1e-14
checks['world_left_SO3_head_roundtrip']=float(np.max(abs(pred[:,43:52]-y[known,43:52])))<1e-14
closure=codec.momentum.momentum_residual_world(tx,tensor(pred),th['pair_impulse_source_BODY_moving_COM22x6'].sum(-2),mass,I,g,h)
lc=closure['linear_world_Ns'].numpy();kc=closure['angular_world_Nms'].numpy()
checks['predicted_linear_momentum_closure']=float(np.max(abs(lc)))<1e-14
checks['predicted_angular_momentum_closure']=float(np.max(abs(kc)))<1e-16
checks['observed_velocity_defect_not_relabelled_zero']=bool(rms(linear[known])>0 and rms(angular[known])>0)
# The original per-pair observed labels are unchanged, not reconstructed from y.
checks['head_impulses_identical_to_independent_contact_export']=bool(heads['pair_impulse_source_BODY_moving_COM22x6'].tobytes()==J[known].tobytes())
one={k:v[0].clone() for k,v in th.items()};one['pair_impulse_source_BODY_moving_COM22x6']=torch.zeros((22,6),dtype=torch.float64)
base=codec.decode_physical_heads(tx[0],one,h[0],mass,I,g)
one2={k:v.clone() for k,v in one.items()};one2['pair_impulse_source_BODY_moving_COM22x6'][0,0]=.015625;one2['pair_impulse_source_BODY_moving_COM22x6'][1,0]=-.015625
same=codec.decode_physical_heads(tx[0],one2,h[0],mass,I,g)
checks['net_momentum_does_not_identify_pair_distribution']=bool(torch.equal(base,same))
# One pure head-Jacobian: each pair's impulse affects velocities by the same
# conditional R0/m and Iworld(R1)^-1 R0 blocks. This is not native action AD.
fixed={k:v.clone() for k,v in one.items()};z=torch.zeros((22,6),dtype=torch.float64,requires_grad=True)
def response(a):return codec.decode_physical_heads(tx[0],fixed|{'pair_impulse_source_BODY_moving_COM22x6':a},h[0],mass,I,g)[52:58]
jac=torch.autograd.functional.jacobian(response,z).detach().numpy();R0=x[known][0,43:52].reshape(3,3);R1=base.detach().numpy()[43:52].reshape(3,3)
expected=np.zeros((6,6));expected[:3,:3]=R0/mass;expected[3:,3:]=np.linalg.solve(R1@np.array(pars['I_body_3x3'])@R1.T,R0)
err=float(np.max(abs(jac-expected[:,None,:])));checks['pair_impulse_head_AD_matches_analytic_blocks']=err<1e-10
np.savez_compressed(out/'decoded_saved_targets.npz',branch_id=np.flatnonzero(known),decoded_state58=pred,observed_y58=y[known],head_Jacobian6x22x6=jac,
    original_linear_defect_world_Ns=linear[known],original_angular_defect_world_Nms=angular[known])
checks={k:bool(v) for k,v in checks.items()}
report=dict(schema='causal_pair_impulse_structured_codec_saved_result:1',status='PASS' if all(checks.values()) else 'FAILED_CHECKS',protocol=pin(protocol),
    checks=checks,passed=sum(checks.values()),total=len(checks),known_rows=2942,unexecuted_rows_kept_upstream=10,
    actual_vCOM_reconstruction_vectorRMS_m_s=rms(pred[:,52:55]-y[known,52:55]),actual_omega_reconstruction_vectorRMS_rad_s=rms(pred[:,55:58]-y[known,55:58]),
    original_linear_momentum_defect_vectorRMS_Ns=rms(linear[known]),original_angular_momentum_defect_vectorRMS_Nms=rms(angular[known]),
    decoded_closure_max_abs_linear_Ns=float(np.max(abs(lc))),decoded_closure_max_abs_angular_Nms=float(np.max(abs(kc))),
    pair_impulse_head_Jacobian_max_abs_error=err,
    counts=dict(derived_impulse_NPZ_reads=1,grouped_native_target_NPZ_reads=2,pure_Torch_decode_calls=4,pure_head_AD_Jacobians=1,NN_constructors=0,NN_parameter_updates=0,native_model_forward_steps=0),
    elapsed_s=time.perf_counter()-start,outputs={'codec_arrays':pin(out/'decoded_saved_targets.npz')},
    scope='Physical-output codec only. Original observed defects preserved; closure imposed on decoded prediction, not on labels. Independent pose heads are not a contact/time/rigid-motion feasibility certificate. Per-pair impulses require per-pair supervision; total momentum cannot determine contact correspondence. No neural model trained, normalizer selected, Actor or CHORD control benefit shown.')
write(out/'report.json',report);print(json.dumps(dict(report=pin(out/'report.json'),passed=report['passed'],total=report['total'])));assert all(checks.values())
