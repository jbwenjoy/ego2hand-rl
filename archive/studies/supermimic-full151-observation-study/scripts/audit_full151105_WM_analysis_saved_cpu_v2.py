"""Saved NumPy audit of one frozen105 analysis; never reloads a model."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import math
import os
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/full151105-A2-WM-physical-gradient-CPU-v1.json'
OUT=ROOT/'outputs/full151105_WM_analysis_saved_independent_v2'
EPS=(.002,.001);JOINTS=(0,4,8,12);PHASES=(0,49,99,149)
GROUPS={'q':(0,20,2e-5),'qd':(20,40,2e-4),'p':(40,43,2e-5),'R':(43,52,2e-5),'v':(52,55,2e-4),'omega':(55,58,2e-4)}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
    p=Path(p).resolve(strict=True)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def pin_matches(p):
    v=pin(p['path'])
    return v['sha256']==p['sha256'] and os.path.samefile(v['path'],p['path'])
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def jl(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def exact(a,b):return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)
def bits(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def close(a,b):
    if a is None or b is None:return a is b
    return bool(np.isclose(a,b,rtol=1e-12,atol=1e-12,equal_nan=False))
def ma(v):return float(np.abs(v).max()) if np.size(v) else 0.
def scalar64(x,p,R):
    x,p,R=np.asarray(x,float),np.asarray(p,float),np.asarray(R,float)
    pos=x[40:43];rot=x[43:52].reshape(3,3);e=.5*(np.linalg.norm(pos+.1*rot[:,0]-p-.1*R[:,0])+np.linalg.norm(pos-.1*rot[:,0]-p+.1*R[:,0]))
    return np.array([2*np.exp(-(np.linalg.norm(pos-p)/.05)**2),8*np.exp(-(e/.04)**2)])
def signature(s):
    raw=s['raw_wp_contact']
    return dict(raw_nacon=s['nacon'],raw_nefc=s['nefc'],CPU_contact_geom_active=sorted((tuple(c['geom']),c['efc_address']>=0) for c in s['contacts']),WP_contact_geom=raw['geom'],WP_contact_type=raw['type'],WP_contact_efc=raw['efc_address'],efc_type=s['raw_efc_type'])
def classify(minus,plus,span,err):
    delta=np.asarray(plus,np.float64)-np.asarray(minus,np.float64);s=delta/(2*np.asarray(EPS,float));floor=max(1e-5,10*max(span,err));rel=float(abs(s[0]-s[1])/max(abs(s[0]),abs(s[1]),1e-12));signal=abs(delta)>floor;stable=bool(signal.all() and np.sign(s[0])==np.sign(s[1]) and rel<=.5)
    return dict(label='STABLE_NONFLAT' if stable else 'FLAT_OR_BELOW_NOISE' if not signal.all() else 'NONFLAT_UNSTABLE',stable_nonflat=stable,signal_floor=floor,delta_score=delta.tolist(),slopes=s.tolist(),relative_disagreement=rel,both_above_noise=bool(signal.all()))

def main():
    OUT.mkdir(exist_ok=False);start=time.perf_counter();checks={};details={}
    def check(k,v):checks[k]=bool(v)
    c=read(CONFIG);run=Path(c['output']);physical=Path(c['physical_run']);r=read(run/'report.json');a=load(c['artifact_pins']['physical_arrays']['path']);w=load(run/'WM_predictions_gradients.npz');rows=read(run/'comparisons.json');classes=read(run/'classifications.json');bases=read(run/'baselines_independent.json');loader=read(run/'loader_receipt.json');metrics=read(run/'prediction_metrics.json');selfmetrics=read(run/'self15_metrics.json');diag=read(run/'contact_force_diagnostics.json')
    check('configc7f84_current_actual_output_route',pin(CONFIG)['sha256']=='c7f84bfbff5e3bd62f8f75f4f297917948cc208bc7d8b7f2796c96c0f76f7cbf' and r['config']==pin(CONFIG) and run.resolve()==ROOT/'outputs/full151105_A2_WM_physical_gradient_CPU_v1')
    check('source698_all_current_report_exact',len(c['source_pins'])==698 and r['source_pins']==c['source_pins'] and all(pin_matches(p) for p in c['source_pins'].values()))
    check('17physical_artifact_pins_current_report_exact',len(c['artifact_pins'])==17 and r['input_pins']==c['artifact_pins'] and all(pin(p['path'])==p for p in c['artifact_pins'].values()))
    check('analysis72b07_entry_loader_current_pins',any(p['sha256']=='72b07c31792f768fe8e6d5b60b3dd2c4002a6a3580295d8c7139e71f9aff2e58' for p in c['source_pins'].values()) and pin(c['loader']['path'])==c['loader'] and c['loader']['sha256']=='119fa7cebc0e9b0727b0160660b1a47822deee2a0e0e9fa5be69465ac5578052')
    check('output_schema_status_fixed_A2final2000',r['schema']=='full151105_physical_gradient_CPU_analysis:1' and r['status']=='COMPLETED_CPU_DIAGNOSTIC' and r['method']=='A2' and r['iteration']==2000)
    check('actual_checkpoint_health_current_and_source_bound',pin(c['checkpoint']['path'])==c['checkpoint']==r['checkpoint'] and c['checkpoint']['sha256']=='70d0a21d5b2f0238ab745264bf24db08515fef65f072cbe00fb6893a33512a0c' and pin(c['health']['path'])==c['health']==r['health'] and Path(c['checkpoint']['path']).resolve()==Path(c['training_run']).resolve()/'train/model_final.pt')
    saved=read(c['artifact_pins']['saved_review']['path']);check('actual1762_saved_engineering_proof_not_inventory_or_gradient',saved['schema']=='full151105_full_serial_saved_independent:1' and saved['engineering_qualified'] is True and saved['passed']==saved['total']==len(saved['checks']) and all(v is True for v in saved['checks'].values()) and saved['completed_branches']==152 and saved['baseline_count']==24 and saved['episode_step_returned']==1808 and saved['physical_gradient_qualified'] is False and saved['whole_hidden_state_qualified'] is False)
    check('source_counter_claim1loader2loads1batch8forward1AD',all(r[k]==n for k,n in {'strict_WM_loader_calls':1,'actual_torch_load_calls':2,'WM_forward_calls':1,'predicted_rows':8,'u_autograd_calls':1,'comparison_rows':64,'dependent_direction_groups':32,'physical_direction_branches':128,'baseline_branches':24}.items()))
    check('scope_no_new_physics_GPU_or_hidden_gradient',all(r[k] is False for k in ('physical_gradient_qualified','whole_graph_hidden_state_qualified','Env_created','GPU_used')) and r['physics_steps']==0 and r['classification_quality_gate'] is None)
    invkeys={'checkpoint_unchanged','WM_weights_unchanged','normalizers_unchanged','input_arrays_unchanged','input_tensors_unchanged','RNG_unchanged','CUDA_uninitialized','Warp_not_imported'}
    check('runtime_invariance_assertions_present_all_true_not_raw_replay',set(r['invariants'])==invkeys and all(v is True for v in r['invariants'].values()))
    check('loader_receipt_final_primary_context_and_105_scope',loader['schema']=='full151_105_world_model_CPU_evaluation_load:1' and loader['status']=='PASS_FROZEN_CPU_WORLD_MODEL_LOAD' and loader['checkpoint']==c['checkpoint'] and loader['health']==c['health'] and loader['loader']==c['loader'] and loader['method']=='A2' and loader['iteration']==2000 and loader['state_dim']==58 and loader['physical_xi_dim']==46 and loader['self_collision_logits_dim']==15 and loader['physical_context_worlds_bitexact']==128 and loader['parameter_requires_grad'] is False and loader['input_control_autograd_supported'] is True and loader['amp_enabled_for_CPU_diagnostic'] is False and loader['GPU_used'] is False and loader['Env_or_Warp_created'] is False and loader['physics_steps']==0)
    for k in ('training_config','fixed_physics_context','source_physical_context','bundle','health_auditor'):check('loader_'+k+'_pin_current',pin_matches(loader[k]))
    check('8prediction8gradient_F32finite_ABI',set(w)=={'predicted_x','self15_logits','gradient_u','predicted_score','truth_x','truth_score','snapshot_x','actualu'} and all(w[k].dtype==np.float32 and np.isfinite(w[k]).all() for k in w) and w['predicted_x'].shape==w['truth_x'].shape==w['snapshot_x'].shape==(8,58) and w['self15_logits'].shape==(8,15) and w['gradient_u'].shape==w['actualu'].shape==(8,20) and w['predicted_score'].shape==(8,))
    check('8inputs_and_truth_actualu_no_reprojection_exact',bits(w['snapshot_x'],a['snapshot_x']) and bits(w['actualu'],a['snapshot_u']) and bits(w['truth_x'],a['donor_next_x']) and bits(w['truth_score'],a['donor_object_score_k']))
    baseline_recomputed=[]
    for s in range(8):
        ids=np.arange(s*19,s*19+3);delta=abs(a['branch_x_next'][ids].astype(float)-a['donor_next_x'][s].astype(float));errors={k:ma(delta[:,lo:hi]) for k,(lo,hi,tol) in GROUPS.items()};scores=a['branch_object_score_k'][ids].astype(float);span=float(np.ptp(scores));err=ma(scores-float(a['donor_object_score_k'][s]));b=dict(snapshot_id=s,errors=errors,baseline_range=span,baseline_donor_error=err,signal_floor=max(1e-5,10*max(span,err)),passed=all(errors[k]<=tol for k,(_,_,tol) in GROUPS.items()) and err<=2e-5);baseline_recomputed.append(b)
        check('baseline'+str(s)+'_6channels_reward_noise_floor_exact',exact(b,bases[s]) and b['passed'])
    samples=jl(c['artifact_pins']['physical_samples']['path']);groups={}
    for sample in samples:groups.setdefault((sample['path'],sample['phase'],sample['branch_id']),[]).append(sample)
    check('all452intervals1808solverrows_retained',len(samples)==1808 and len(groups)==len(diag)==452 and all([s['substep'] for s in v]==[1,2,3,4] for v in groups.values()))
    for item,((path,k,bid),samplegroup) in zip(diag,groups.items()):
        sig=[signature(v) for v in samplegroup]
        check('contactdiag'+str(len([k for k in checks if k.startswith('contactdiag')]))+'_phase_signature_force_passthrough',item['path']==path and item['k']==k and item['branch_id']==bid and item['solver_preintegration_offset_ms']==[0,5,10,15] and item['postintegration_state_offset_ms']==[5,10,15,20] and exact(item['contact_signature'],sig) and item['contact_signature_changed_within_interval']==any(x!=sig[0] for x in sig[1:]) and exact(item['force_on_object_w_N'],[s['hand_object_force_on_object_w_N'] for s in samplegroup]) and exact(item['moment_about_COM_w_Nm'],[s['hand_object_moment_about_COM_w_Nm'] for s in samplegroup]) and item['direct_torque_included_once'] is True)
    expected_classes=[];expected_rows=[];projection_error=[]
    for s in range(8):
        for ji,joint in enumerate(JOINTS):
            ids=[(s*19+3+ei*8+ji*2,s*19+4+ei*8+ji*2) for ei in range(2)];minus=np.array([a['branch_object_score_k'][i] for i,_ in ids],np.float32);plus=np.array([a['branch_object_score_k'][i] for _,i in ids],np.float32);base=baseline_recomputed[s];cl=classify(minus,plus,base['baseline_range'],base['baseline_donor_error']);expected=dict(snapshot_id=s,path=('A0','A2')[s//4],k=PHASES[s%4],joint_index=joint,**cl);expected_classes.append(expected)
            check('group'+str(s)+'_'+str(joint)+'_twoeps_noise_classification_exact',exact(expected,classes[len(expected_classes)-1]))
            for ei,eps in enumerate(EPS):
                ci=(s*4+ji)*2+ei;row=rows[ci];mid,pid=ids[ei];m,p=a['direction__minus'][s,ji,ei],a['direction__plus'][s,ji,ei];center=a['snapshot_u'][s];d=(p.astype(float)-m.astype(float))/(2*eps);off=(p.astype(float)+m.astype(float))*.5-center.astype(float);proj=float(w['gradient_u'][s].astype(float)@d);nominal=float(w['gradient_u'][s,joint]);projection_error.append(abs(proj-row['WM_AD_effective']));path=('A0','A2')[s//4];k=PHASES[s%4];ms=groups[(path,k,mid)];ps=groups[(path,k,pid)];ds=groups[(path,k,None)];changedpm=any(not exact(signature(x),signature(y)) for x,y in zip(ms,ps));changedcenter=any(not exact(signature(x),signature(t)) for vv in (ms,ps) for x,t in zip(vv,ds));clipped=bool(a['direction__clipped'][s,ji,ei])
                check('comparison'+str(ci)+'_F64FD_actual_direction_projection_allrows',row['snapshot_id']==s and row['path']==path and row['k']==k and row['joint_index']==joint and row['epsilon_index']==ei and row['epsilon_rad']==eps and row['minus_row']==mid and row['plus_row']==pid and row['physical_slope']==cl['slopes'][ei] and row['WM_AD_effective']==proj and row['WM_AD_nominal']==nominal and exact(row['d_eff'],d.tolist()) and exact(row['center_offset'],off.tolist()) and bits(d,a['direction__d_eff'][s,ji,ei]) and bits(off,a['direction__center_offset'][s,ji,ei]) and row['clipped']==clipped and row['center_offset_nonzero']==bool(off.any()))
                check('comparison'+str(ci)+'_class_contact_flags_not_filter',row['physical_label']==cl['label'] and row['stable_nonflat']==cl['stable_nonflat'] and row['signal_floor']==cl['signal_floor'] and row['relative_epsilon_disagreement']==cl['relative_disagreement'] and row['contact_signature_change_plus_minus']==changedpm and row['contact_signature_change_vs_fresh_donor']==changedcenter and row['central_unique_derivative_qualified'] is False and row['units']=='controlled score/rad')
                expected_rows.append(dict(epsilon_index=ei,physical_slope=cl['slopes'][ei],WM_AD_effective=proj,clipped=clipped,stable_nonflat=cl['stable_nonflat']))
    check('64rows32groups_none_dropped',len(rows)==64 and len(classes)==32 and len(expected_rows)==64 and len(expected_classes)==32)
    counts={label:sum(c['label']==label for c in expected_classes) for label in ('STABLE_NONFLAT','NONFLAT_UNSTABLE','FLAT_OR_BELOW_NOISE')};check('classification_count_summary_recomputed',counts==r['classification_counts'])
    agreement={}
    for ei,eps in enumerate(EPS):
        selected=[row for row in expected_rows if row['epsilon_index']==ei and row['stable_nonflat']];x=np.array([v['physical_slope'] for v in selected]);y=np.array([v['WM_AD_effective'] for v in selected]);n=len(selected);g=dict(eligible=n,sign_correct=int((np.sign(x)==np.sign(y)).sum()),sign_fraction=None if n==0 else float((np.sign(x)==np.sign(y)).mean()),pearson=None if n<2 or np.ptp(x)==0 or np.ptp(y)==0 else float(np.corrcoef(x,y)[0,1]),mean_absolute_disagreement=None if n==0 else float(abs(x-y).mean()),clipped_eligible=sum(v['clipped'] for v in selected));agreement[str(eps)]=g
        check('agreement_eps'+str(eps)+'_no_absentclass_success',all(close(v,r['agreement'][str(eps)][k]) for k,v in g.items()))
    pred=w['predicted_x'].astype(float);truth=w['truth_x'].astype(float);delta=pred-truth;rot=pred[:,43:52].reshape(-1,3,3);truthR=truth[:,43:52].reshape(-1,3,3);angle=np.arccos(np.clip((np.einsum('bij,bij->b',rot,truthR)-1)*.5,-1,1));rmsnorm=lambda z:float(np.sqrt(np.mean(np.sum(z*z,axis=-1))))
    met=dict(q_component_RMSE_mrad=float(np.sqrt(np.mean(delta[:,:20]**2))*1000),qd_component_RMSE_rad_s=float(np.sqrt(np.mean(delta[:,20:40]**2))),p_vector_RMS_mm=rmsnorm(delta[:,40:43])*1000,R_geodesic_RMS_degrees=float(np.rad2deg(np.sqrt(np.mean(angle**2)))),v_vector_RMS_m_s=rmsnorm(delta[:,52:55]),omega_vector_RMS_rad_s=rmsnorm(delta[:,55:58]),score_RMS=float(np.sqrt(np.mean((w['predicted_score'].astype(float)-w['truth_score'].astype(float))**2))))
    check('8center_prediction_error_groups_independently_recomputed',all(close(v,metrics[k]) and close(v,r['prediction_metrics'][k]) for k,v in met.items()))
    scorediag=[]
    for s in range(8):
        terms=scalar64(w['predicted_x'][s],a['goal_p'][s],a['goal_R'][s]);difference=abs(float(w['predicted_score'][s])-float(terms.sum()));scorediag.append(difference);bound=float((8*np.finfo(np.float32).eps*np.maximum(1.,abs(terms))).sum()+np.finfo(np.float32).eps*max(1.,abs(terms.sum())))
        check('predicted_score'+str(s)+'_same20_formula_F64_diagnostic',difference<=bound)
    z=w['self15_logits'].astype(float);y=a['donor_self15'].astype(float);probs=np.exp(-np.logaddexp(0.,-z));predicted=probs>=.5;tp,fp,fn,tn=[int(x.sum()) for x in (predicted&(y==1),predicted&(y==0),(~predicted)&(y==1),(~predicted)&(y==0))];sm=dict(labels=int(y.size),positive_truth=int(y.sum()),positive_prediction=int(predicted.sum()),BCE=float((np.logaddexp(0.,z)-y*z).mean()),Brier=float(((probs-y)**2).mean()),TP=tp,FP=fp,FN=fn,TN=tn,precision=None if tp+fp==0 else tp/(tp+fp),recall=None if tp+fn==0 else tp/(tp+fn),F1=None if tp+fn==0 or 2*tp+fp+fn==0 else 2*tp/(2*tp+fp+fn))
    check('native_self15_120labels_metrics_not_HO',all(close(v,selfmetrics[k]) for k,v in sm.items()) and 'self-collision15' in selfmetrics['scope'] and selfmetrics['label_phase']=='last solver preintegration15ms; state prediction ispost20ms')
    outputpins=read(run/'output_pins.json');check('all_analysis_output_pins_current',all(pin(v['path'])==v and Path(v['path']).resolve().parent==run.resolve() for v in outputpins.values()) and set(outputpins)=={v.name for v in run.iterdir() if v.is_file()}-{'output_pins.json'})
    with (run/'comparisons.csv').open(newline='') as stream:csvrows=list(csv.DictReader(stream))
    check('CSV64rows_fixedorder_bound_to_JSON',len(csvrows)==64 and all(int(x['snapshot_id'])==j['snapshot_id'] and int(x['joint_index'])==j['joint_index'] and float(x['epsilon_rad'])==j['epsilon_rad'] and float(x['physical_slope'])==j['physical_slope'] and float(x['WM_AD_effective'])==j['WM_AD_effective'] for x,j in zip(csvrows,rows)))
    check('independent_audit_no_native_model_or_framework_import',not any(n.split('.')[0] in ('torch','warp','mujoco','mujoco_warp','mjlab','dex_supermimic','scipy') for n in sys.modules))
    details.update(classification_counts=counts,agreement=agreement,prediction_metrics=met,self15_metrics=sm,projection_maxabs_error=max(projection_error),predicted_score_F64_diagnostic_maxerror=max(scorediag),comparison_rows=64,dependent_groups=32,source_count=698,artifact_count=17)
    np.savez(OUT/'independent_numeric_arrays.npz',physical_slopes=np.asarray([v['physical_slope'] for v in expected_rows]),WM_AD_effective=np.asarray([v['WM_AD_effective'] for v in expected_rows]),center_state_errors=delta,classification_stable32=np.asarray([v['stable_nonflat'] for v in expected_classes]),native_self15_probability=probs)
    result={'schema':'full151105_WM_analysis_saved_independent:1','status':'PASS_SAVED_ARITHMETIC_AND_PROVENANCE_NOT_PHYSICAL_GRADIENT_QUALIFICATION' if all(checks.values()) else 'SAVED_ANALYSIS_AUDIT_FAILURE_RETAINED','engineering_analysis_reproduced':all(checks.values()),'passed':sum(checks.values()),'total':len(checks),'checks':checks,'auditor':pin(Path(__file__)),'config':pin(CONFIG),'analysis_report':pin(run/'report.json'),'analysis_arrays':pin(run/'WM_predictions_gradients.npz'),'physical_saved_review':c['artifact_pins']['saved_review'],'details':details,'numeric_arrays':pin(OUT/'independent_numeric_arrays.npz'),'physical_gradient_qualified':False,'whole_hidden_state_qualified':False,'HO_gradient_qualified':False,'limits':['All64physical secants and32groups are independently reconstructed from original FP32 captured GPU scores cast to F64 before subtraction. Effective directions use F64 differences of actual FP32 endpoints, with clipping and center-offset retained. No second EMA/C2 is applied in this audit or WM comparison.','WM predicted arrays/gradients are saved actual inference output. Projections, classes, scalar arithmetic, errors, self15 metrics and agreement summaries are independently recomputed without loading a WM, Actor, Env or model. The autograd derivative itself is not re-executed or newly qualified.','Weights/normalizers/RNG/input invariants and1strictloader/2torchloads/1batch8forward/1AD counts are runtime source-backed assertions; raw before/after tensors or RNG states are not serialized for independent replay. Current checkpoint/source/artifact/output hashes are independently verified.','Only24local baselines and visited inventories were empirically replayed. Graph executable caches and hidden scratch remain UNKNOWN. Small or signed secants may reflect finite-interval/nonsmooth/contact/quantization behavior and are not unique true physical Jacobians.','Stable/flat/unstable are frozen two-epsilon signal labels. Clipping/asymmetry is an orthogonal flag rather than silent filtering. Correlations and sign fractions are descriptive of dependent directions from one seed/source; no CI or method-benefit inference.','The main scalar is controlled post20 native-formula2+8 with fixed goal_k. Original derived15 RewardManager reward is a secondary diagnostic. Native self15 is not HO contact/force or a human label; no HO gradient qualification is inferred.'],'new_Env_model_WM_Actor_GPU':False,'wall_seconds':time.perf_counter()-start}
    result['reviewer_v1_failure_retained']={'source':pin(ROOT/'scripts/audit_full151105_WM_analysis_saved_cpu_v1.py'),'report':pin(ROOT/'outputs/full151105_WM_analysis_saved_independent_v1/report.json'),'correction':'Only Windows slash spelling normalized using samefile and strict SHA; numeric equations/tolerances/input files unchanged.'}
    (OUT/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'passed':result['passed'],'total':result['total'],'failed':[k for k,v in checks.items() if not v],'report':pin(OUT/'report.json')},indent=2));return 0 if all(checks.values()) else 1
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        if OUT.exists():(OUT/'failure.json').write_text(json.dumps({'error':repr(exc),'source':pin(Path(__file__)),'new_native_execution':False},indent=2)+'\n')
        raise
