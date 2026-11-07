"""Independent saved bonus algebra only: no Torch/objective/geometry/model import."""
from pathlib import Path
import hashlib
import json
import math
import os
import sys
import time
import zipfile
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/full151105_trained_bonus_CPU_v1'
CONFIG=ROOT/'configs/full151105-trained-bonus-CPU-v1.json'
OUT=ROOT/'outputs/full151105_trained_bonus_saved_independent_v1'
CHANNELS=('raw_P','raw_S','BP','CS','A2');PHASES=(0,49,99,149);EPS=(.002,.001);JOINTS=(0,4,8,12)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
    p=Path(p).resolve(strict=True);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def matches(v):return pin(v['path'])['sha256']==v['sha256'] and os.path.samefile(v['path'],Path(v['path']).resolve())
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def bits(a,b):
    a,b=np.asarray(a),np.asarray(b);return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def exact(a,b):return canonical(a)==canonical(b)
def close(a,b):return a is b if a is None or b is None else bool(np.isclose(a,b,rtol=1e-12,atol=1e-12))
def maxabs(a):return float(abs(a).max()) if np.size(a) else 0.
def f32bound(a,b):return bool(np.all(abs(np.asarray(a,float)-np.asarray(b,float))<=8*np.finfo(np.float32).eps*np.maximum(1.,abs(np.asarray(b,float)))))
def proxy_scores(cap,target):
    d=cap['sampled_signed_distance_m'];weights=np.exp(np.float32(-.5)*(np.maximum(d,np.float32(0))/np.float32(.001))**2)
    excess=np.maximum(-d-np.float32(.003),np.float32(0)).max(-1)/np.float32(.001);P=excess*excess
    current=cap['support'];t=np.asarray(target,np.float32).reshape(len(current),64);commanded=t>np.float32(.001);live=current>np.float32(.001);lo=np.maximum(np.float32(.9)*t-current,np.float32(0));hi=np.maximum(current-np.float32(1.1)*t,np.float32(0));value=np.where(commanded&live,np.exp(-(lo*lo+hi*hi)/np.float32(.1)),np.float32(0));S=value.sum(-1,dtype=np.float32)/np.maximum(commanded.sum(-1),1).astype(np.float32)
    return P,S,weights
def reconstructed_support(cap,basis,radius):
    # Pure algebra on captured COM points/normals. No geometry/SDF/FK query.
    B=np.asarray(basis,np.float32);r=np.float32(radius);answer=[]
    def normalize(x):return x/np.maximum(np.sqrt(np.sum(x*x,axis=-1,keepdims=True)),np.float32(1e-12))
    for points,normal,weight in zip(cap['projected_contact_com_o_m'],cap['inward_normal_o'],cap['soft_contact_weight']):
        n=normalize(normal);sign=np.where(n[:,2]>=0,np.float32(1),np.float32(-1));a=-np.float32(1)/(sign+n[:,2]);b=n[:,0]*n[:,1]*a;t1=normalize(np.stack((1+sign*n[:,0]*n[:,0]*a,sign*b,-sign*n[:,0]),-1));t2=normalize(np.stack((b,sign+n[:,1]*n[:,1]*a,-n[:,1]),-1))
        def project(v):return np.einsum('pj,nj->pn',v,B[:,:3])+np.einsum('pj,nj->pn',np.cross(points,v),B[:,3:])/r
        pn,p1,p2=project(n),project(t1),project(t2);best=None
        for edge in range(8):
            angle=2*math.pi*edge/8;v=np.float32(math.cos(angle))*p1+np.float32(math.sin(angle))*p2;best=v if best is None else np.maximum(best,v)
        support=np.maximum(pn,(pn+np.float32(.1)*best)/np.float32(math.sqrt(1.01)));answer.append(np.maximum((support*weight[:,None]).max(0),np.float32(0)))
    return np.asarray(answer,np.float32)
def noise(donor,baseline):
    b=np.asarray(baseline,float);span=float(np.ptp(b));err=float(abs(b-float(donor)).max());scale=max(1.,abs(float(donor)),float(abs(b).max()));return dict(range=span,donor_error=err,N=max(span,err),scale=scale,roundoff_proxy=8*2**-23*scale)
def floors(scores,s):
    ns={k:noise(scores[k][s],scores[k][8+19*s:8+19*s+3]) for k in CHANNELS};f={k:max(v,10*ns[k]['N'],ns[k]['roundoff_proxy']) for k,v in (('raw_P',1e-4),('raw_S',1e-5),('CS',1e-5))};f['BP']=.1*f['raw_P'];f['A2']=max(f['BP']+f['CS'],10*ns['A2']['N'],ns['A2']['roundoff_proxy']);return dict(floors=f,baseline_components=ns,BP_representation_consistent=ns['BP']['N']<=f['BP'],dt_weighted_floors={k:.02*f[k] for k in ('BP','CS','A2')},provenance='NEW_PROPOSED prereg9d48 representational guard, not accumulated-error theorem')
def classify(minus,plus,floor):
    numerator=np.asarray(plus,float)-np.asarray(minus,float);slopes=numerator/(2*np.asarray(EPS));signal=abs(numerator)>floor;rel=float(abs(slopes[0]-slopes[1])/max(abs(slopes[0]),abs(slopes[1]),1e-12));stable=bool(signal.all() and np.sign(slopes[0])==np.sign(slopes[1]) and rel<=.5);return dict(label='STABLE_NONFLAT' if stable else 'FLAT_OR_BELOW_NOISE' if not signal.all() else 'NONFLAT_UNSTABLE',stable_nonflat=stable,numerators=numerator.tolist(),slopes=slopes.tolist(),both_above_noise=bool(signal.all()),relative_epsilon_disagreement=rel,numerator_floor=float(floor))
def projected(slope,ad,floor,eps):
    df=floor/(2*eps);den=max(abs(slope),abs(ad),df);eligible=abs(slope)>df;return dict(absolute_error=float(abs(ad-slope)),normalized_error=float(abs(ad-slope)/den),denominator=float(den),physical_sign_eligible=bool(eligible),sign_agreement=None if not eligible else bool(ad!=0 and np.sign(ad)==np.sign(slope)),AD_over_physical=None if not eligible else float(ad/slope),undefined_ratio_reason=None if eligible else 'PHYSICAL_NOT_ABOVE_COMPONENT_FLOOR')
def main():
    OUT.mkdir(exist_ok=False);start=time.perf_counter();checks={};details={}
    def check(k,v):checks[k]=bool(v)
    c=read(CONFIG);r=read(RUN/'report.json');proto=read(c['protocol']['path']);inv=read(RUN/'source_inventory.json')
    check('actual_config0aef_report_and_copy_join',pin(CONFIG)['sha256']=='0aef37ffbe8b0d5f84dfcb2b623a5832e2dba79f325b925d4811790ebf1214d0' and r['config']==pin(CONFIG) and exact(c,read(RUN/'config.json')) and Path(c['output']).resolve()==RUN.resolve())
    check('prereg9d48_and_source9c34_execution_scope',c['protocol']['sha256']=='9d48ae798471638bb6bbc3e47d80112b185de68fd861e9eac5449286428413b7' and matches(c['protocol']) and proto['execution_authorized'] is False and exact(proto,read(RUN/'protocol.json')) and any(v['sha256']=='9c34ea62f2aab6c1492bc544e916ef5c6d77348452e282501a148047a9505fbd' for v in inv.values()))
    check('1711current_sources_exact_config_report',len(inv)==1711 and inv==c['source_pins']==r['source_pins'] and all(matches(v) for v in inv.values()))
    with zipfile.ZipFile(RUN/'execution_sources.zip') as z:
        names=[f'{i:04d}_{Path(v["path"]).name}' for i,v in enumerate(inv.values())];check('1711ZIP_order_and_bytes_exact',z.namelist()==names and all(hashlib.sha256(z.read(n)).hexdigest()==v['sha256'] for n,v in zip(names,inv.values())))
    inputs=proto['inputs'];check('all_original_inputs_current_and_report_exact',r['input_pins']==inputs and all(matches(v) for v in inputs.values()))
    ready=read(c['cpu_ready']['path']);proof=read(ready['cpu_report']['path']);check('preparation66_actual_source_pins_and_scope',matches(c['cpu_ready']) and matches(ready['cpu_report']) and proof['passed']==proof['total']==66 and all(v is True for v in proof['checks'].values()) and all(matches(v) for v in ready['source_pins'].values()))
    expected={'strict_WM_loader':1,'objective_factory':2,'physical_objective_forward':2,'WM_forward':1,'predicted_objective_forward':2,'u_VJP':4};check('sourcebacked_actual_attempted_returned_budget_all',set(r['counts'])==set(expected) and all(r['counts'][k]=={'attempted':n,'returned':n} for k,n in expected.items()) and r['actual_torch_load_calls']==2 and r['physical_scored_rows']==160 and r['WM_predicted_rows']==8 and r['comparison_rows']==320 and r['direction_groups_per_channel']==32)
    check('complete_shadow_not_GPU_training_gradient_or_physics',r['schema']=='full151105_trained_bonus_CPU_analysis:1' and r['status']=='COMPLETED_CPU_SHADOW_TRAINED_BONUS_SECANT_COMPARISON' and r['channels']==list(CHANNELS) and all(r[k] is False for k in ('whole_hidden_state_qualified','physical_gradient_qualified','GPUAMP_training_gradient_reproduced','Env_created','GPU_used')) and r['physics_steps']==0 and not (RUN/'failure.json').exists())
    invariant_keys={'input_arrays_unchanged','input_tensors_unchanged','all_module_weights_and_buffers_unchanged','normalizers_unchanged','RNG_unchanged','CUDA_uninitialized','Warp_not_imported','no_parameter_gradients'};check('sourcebacked_invariants_all_true_not_raw_weight_RNG_replay',set(r['invariants'])==invariant_keys and all(v is True for v in r['invariants'].values()))
    a=load(inputs['physical_arrays']['path']);w=load(RUN/'WM_predictions_gradients.npz');before=load(RUN/'WM_prediction_before_AD.npz');scores=load(RUN/'physical_scores.npz');predscores=load(RUN/'predicted_scores.npz');target=load(inputs['target']['path']);receipts=read(RUN/'loader_factory_receipts.json')
    phaseids=np.r_[np.arange(8),np.repeat(np.arange(8),19)];phases=np.asarray(PHASES,dtype=np.int64)[phaseids%4];pph=np.asarray(PHASES*2,dtype=np.int64);target64=target['support512'][...,::8].astype(np.float32)
    check('160order_next20_preactivation_k149_not150',bits(w['physical_phases'],phases) and bits(w['predicted_phases'],pph) and np.count_nonzero(phases==149)==40 and not (phases==150).any() and bits(w['snapshot_x'],a['snapshot_x']) and bits(w['actualu'],a['snapshot_u']))
    check('exact_frozen_target64_by_preactivation_phase',bits(w['target64_physical'],target64[phases]) and bits(w['target64_predicted'],target64[pph]) and (target['phase_weight'][phases]==1).all() and (target['phase_weight'][pph]==1).all())
    check('prediction_before_AD_and_afterAD_raw_identical',all(bits(before[k],w[k]) for k in ('predicted_x','self15_logits','snapshot_x','actualu')) and w['predicted_x'].shape==(8,58) and w['self15_logits'].shape==(8,15))
    check('5physical160_pred8_scores_gradient8x20_finite',set(scores)==set(predscores)==set(CHANNELS) and all(scores[k].dtype==predscores[k].dtype==np.float32 and scores[k].shape==(160,) and predscores[k].shape==(8,) and w['gradient_'+k].shape==(8,20) and w['gradient_'+k].dtype==np.float32 and all(np.isfinite(v).all() for v in (scores[k],predscores[k],w['gradient_'+k])) for k in CHANNELS))
    check('rawS_gradient_CS_alias_only_with_weights1',bits(w['gradient_raw_S'],w['gradient_CS']) and (target['phase_weight'][phases]==1).all())
    wmreceipt=receipts['WM'];check('strict_WM_receipt_actual_final_and_health',wmreceipt['checkpoint']==inputs['fixed_A2_checkpoint'] and wmreceipt['health']==inputs['fixed_A2_health'] and wmreceipt['method']=='A2' and wmreceipt['iteration']==2000 and wmreceipt['physical_xi_dim']==46 and wmreceipt['disturbance_dim']==174 and wmreceipt['GPU_used'] is False)
    check('actual_P_ONLY_A2_trained_spec_math64_not_contact_qualification',receipts['P_ONLY']['scoring']['support_coefficient']==0. and receipts['A2']['scoring']['support_coefficient']==1. and all(receipts[k]['basis']['training_directions']==64 and receipts[k]['dt_applied_inside_module'] is False and receipts[k]['caller_dt_s']==.02 and receipts[k]['frame_count']==151 and receipts[k]['human_contact_observation_known'] is False and receipts[k]['target_defined_from_policy_contacts'] is False for k in ('P_ONLY','A2')))
    basis=target['support_basis'][::8].astype(np.float32);radius=float(target['object_nominal_enclosing_radius_m']);capture_stats={};arithmetic_max={}
    for kind,n,ph,truth in (('physical',160,phases,scores),('predicted',8,pph,predscores)):
        pc=load(RUN/(kind+'_proxy_P_ONLY.npz'));ac=load(RUN/(kind+'_proxy_A2.npz'));capkeys={'support','presence','minimum_signed_distance_m','projected_contact_com_o_m','inward_normal_o','soft_contact_weight','sampled_signed_distance_m'};p=ac['sampled_signed_distance_m'].shape[1]
        check(kind+'__4hook_payload_shape_finite_and_modules_byte_equal',set(pc)==set(ac)==capkeys and all(bits(pc[k],ac[k]) and ac[k].dtype==np.float32 and np.isfinite(ac[k]).all() for k in capkeys) and ac['sampled_signed_distance_m'].shape==ac['soft_contact_weight'].shape==(n,p) and ac['support'].shape==(n,64) and ac['presence'].shape==(n,6) and ac['projected_contact_com_o_m'].shape==ac['inward_normal_o'].shape==(n,p,3) and ac['minimum_signed_distance_m'].shape==(n,))
        P,S,weights=proxy_scores(ac,target64[ph]);rebuild=reconstructed_support(ac,basis,radius)
        check(kind+'__P_from_saved_SDF_normalized_squared',f32bound(truth['raw_P'],P))
        check(kind+'__S_from_captured_support_and_target_toleranceband',f32bound(truth['raw_S'],S))
        check(kind+'__softweights_from_positive_SDF_bandwidth1mm',f32bound(ac['soft_contact_weight'],weights) and (ac['soft_contact_weight']>=0).all() and (ac['soft_contact_weight']<=1).all())
        check(kind+'__support_from_captured_COM_points_normals_mu1_unitrays',f32bound(ac['support'],rebuild))
        check(kind+'__minimumSDF_and_CS_BP_actual_channels',bits(ac['minimum_signed_distance_m'],ac['sampled_signed_distance_m'].min(-1)) and bits(truth['CS'],truth['raw_S']) and f32bound(truth['BP'],-np.float32(.1)*P) and f32bound(truth['A2'],truth['CS']+truth['BP']))
        diagnostics=read(RUN/(kind+'_proxy_diagnostics.json'));d=ac['sampled_signed_distance_m'];support=ac['support'];expected_diag={'rows':n,'sample_points_per_row':p,'basis':64,'minimum_sdf_m':d.min(-1).tolist(),'maximum_soft_weight':ac['soft_contact_weight'].max(-1).tolist(),'sample_count_abs_sdf_le_1mm':(abs(d)<=.001).sum(-1).tolist(),'sample_count_abs_sdf_le_3mm':(abs(d)<=.003).sum(-1).tolist(),'support_live_count_gt_1e_3':(support>1e-3).sum(-1).tolist()};check(kind+'__all_nearsurface_live_support_diagnostics',all(exact(v,diagnostics[k]) for k,v in expected_diag.items()))
        capture_stats[kind]={'rows':n,'points_per_row':p,'P_max':float(truth['raw_P'].max()),'S_min':float(truth['raw_S'].min()),'S_max':float(truth['raw_S'].max()),'zero_support_rows':int((support.max(-1)==0).sum()),'maxweight_min':float(ac['soft_contact_weight'].max(-1).min()),'support_reconstruction_maxerror':maxabs(ac['support'].astype(float)-rebuild.astype(float)),'softweight_reconstruction_maxerror':maxabs(ac['soft_contact_weight'].astype(float)-weights.astype(float))};arithmetic_max[kind]={'P':maxabs(truth['raw_P'].astype(float)-P.astype(float)),'S':maxabs(truth['raw_S'].astype(float)-S.astype(float))}
    arithmetic=read(RUN/'arithmetic_diagnostics.json');calc={'BP_plus_support_vs_A2_maxabs':float(abs(scores['BP']+scores['CS']-scores['A2']).max()),'gBP_vs_minus_point1gP_maxabs':float(abs(w['gradient_BP']+.1*w['gradient_raw_P']).max()),'gA2_vs_gCS_minus_point1gP_maxabs':float(abs(w['gradient_A2']-w['gradient_CS']+.1*w['gradient_raw_P']).max())};check('arithmetic_residuals_independent_not_primary_rewrites',all(close(v,arithmetic[k]) for k,v in calc.items()))
    rows=read(RUN/'comparisons.json');classes=read(RUN/'classifications.json');savednoise=read(RUN/'component_noise.json');expected_rows=[];class_counts={k:{} for k in CHANNELS};floor_data=[]
    check('320comparisons160classifications_no_rows_selected',len(rows)==320 and len(classes)==160 and len(savednoise)==8)
    for s in range(8):
        f=floors(scores,s);floor_data.append(f);check('state'+str(s)+'__baseline_component_noise_scale_floors_cancellation',exact({'snapshot_id':s,**f},savednoise[s]))
        for ch,channel in enumerate(CHANNELS):
            for j,joint in enumerate(JOINTS):
                slots=[(8+s*19+3+ei*8+j*2,8+s*19+4+ei*8+j*2) for ei in range(2)];cl=classify([scores[channel][m] for m,p in slots],[scores[channel][p] for m,p in slots],f['floors'][channel]);ci=(s*5+ch)*4+j;expectcl={'snapshot_id':s,'path':('A0','A2')[s//4],'k':PHASES[s%4],'channel':channel,'joint_index':joint,**cl};check('class'+str(ci)+'__both_eps_original_component_units',exact(expectcl,classes[ci]));class_counts[channel][cl['label']]=class_counts[channel].get(cl['label'],0)+1
                for ei,eps in enumerate(EPS):
                    ri=ci*2+ei;row=rows[ri];grad=w['gradient_'+channel][s].astype(float);deff=a['direction__d_eff'][s,j,ei];off=a['direction__center_offset'][s,j,ei];ad=float(grad@deff);slope=cl['slopes'][ei];consistent=channel!='BP' or f['BP_representation_consistent'];expected={'snapshot_id':s,'path':('A0','A2')[s//4],'k':PHASES[s%4],'channel':channel,'joint_index':joint,'epsilon_index':ei,'epsilon_rad':eps,'minus_physical_index':slots[ei][0],'plus_physical_index':slots[ei][1],'physical_slope':slope,'WM_AD_effective':ad,'WM_AD_nominal_secondary':float(grad[joint]),'clipped':bool(a['direction__clipped'][s,j,ei]),'d_eff':deff.tolist(),'center_offset':off.tolist(),'center_offset_maxabs':float(abs(off).max()),'physical_label':cl['label'],'stable_nonflat':cl['stable_nonflat'],'component_representation_consistent':consistent,'numerator_floor':f['floors'][channel],**projected(slope,ad,f['floors'][channel],eps),'comparison_eligible':consistent}
                    if not consistent:expected.update(sign_agreement=None,AD_over_physical=None,undefined_ratio_reason='BP_COMPONENT_REPRESENTATION_INCONSISTENT')
                    if channel in ('BP','CS','A2'):expected['dt_derived']={'physical_slope':.02*slope,'WM_AD_effective':.02*ad,'numerator_floor':.02*f['floors'][channel],'same_classification':True}
                    check('row'+str(ri)+'__F64_secant_d_eff_denominator_sign_ratio_dt_once',exact(expected,row));expected_rows.append(expected)
    stats=[]
    for channel in CHANNELS:
        for ei in range(2):
            allrows=[v for v in expected_rows if v['channel']==channel and v['epsilon_index']==ei]
            for subset,chosen in (('all32',allrows),('stable_secondary',[v for v in allrows if v['stable_nonflat'] and v['component_representation_consistent']]),('stable_unclipped_secondary',[v for v in allrows if v['stable_nonflat'] and not v['clipped'] and v['component_representation_consistent']])):
                n=len(chosen);stats.append({'channel':channel,'epsilon_rad':EPS[ei],'subset':subset,'rows':n,'sign_eligible':sum(v['physical_sign_eligible'] and v['comparison_eligible'] for v in chosen),'sign_agreement_count':sum(v['sign_agreement'] is True for v in chosen),'mean_absolute_error':None if not n else float(np.mean([v['absolute_error'] for v in chosen])),'RMS_absolute_error':None if not n else float(np.sqrt(np.mean([v['absolute_error']**2 for v in chosen]))),'clipped_rows':sum(v['clipped'] for v in chosen),'dependent_groups_not_independent_samples':True})
    check('30all32_stable_unclipped_stats_original_unit_scopes',exact(stats,read(RUN/'statistics.json')) and exact(stats,r['statistics']))
    outputpins=read(RUN/'output_pins.json');check('all_immutable_output_pins_exact',all(matches(v) and Path(v['path']).resolve().parent==RUN.resolve() for v in outputpins.values()) and set(outputpins)=={p.name for p in RUN.iterdir() if p.is_file()}-{'output_pins.json'})
    check('no_new_objective_geometry_WM_native_Torch_import',not any(k.split('.')[0] in ('torch','mujoco','warp','mujoco_warp','mjlab','dex_supermimic','scipy','full151_intent','frozen_contact_math') for k in sys.modules))
    details.update(capture_statistics=capture_stats,scalar_arithmetic_reconstruction_errors=arithmetic_max,component_classification_counts=class_counts,statistics=stats,arithmetic_diagnostics=calc,BP_representation_consistent=[f['BP_representation_consistent'] for f in floor_data],source_count=len(inv),physical_unique_rows=160,predicted_rows=8,physical_plus_predicted_captures=336)
    result={'schema':'full151105_trained_bonus_saved_independent:1','status':'PASS_SAVED_TRAINED_FUNCTION_ALGEBRA_NOT_PHYSICAL_BONUS_GRADIENT_QUALIFICATION' if all(checks.values()) else 'SAVED_BONUS_AUDIT_FAILURE_RETAINED','engineering_reproduced':all(checks.values()),'passed':sum(checks.values()),'total':len(checks),'checks':checks,'auditor':pin(Path(__file__)),'config':pin(CONFIG),'protocol':c['protocol'],'producer_report':pin(RUN/'report.json'),'details':details,'physical_gradient_qualified':False,'whole_hidden_state_qualified':False,'GPUAMP_training_gradient_reproduced':False,'HO_gradient_qualified':False,'limits':['Only saved NPZ/JSON/CSV/bytes were consumed. P,S,softweights and support algebra are independently rebuilt from captured outputs and frozen target64/basis; no SDF/geometry/FK/collision query or model/objective import was made. This verifies the frozen proxy arithmetic, not geometric distance truth, actual contact, force or a qualified predictor.','Four hook captures cover two repeated-functions on physical160 and predicted8 (336total row evaluations,168unique input states). P_ONLY retains eager A2 geometry; identical captures are not independent observations. Presence6 is shape/finite/byte-equality checked; per-point group-ID buffers were not serialized, so the six group reductions are not independently reconstructed.','Actual factory BP and A2 are retained as primary values. Sum/gradient component identities are diagnostics only; unscaled dimensionless normalized P/S and weighted BP/CS/A2 are kept separate. Derived .02 reward contributions and floors apply once.','All320rows/160dependent two-epsilon channel groups remain, including clipped/far/drop/offmanifold states. New prereg9d48 component floors and cancellation guards are unchanged; below-noise means at least one epsilon is below/equal its floor, not proof of physical flatness.','Saved WM gradients are projected in F64 without rerunning AD. Weights/buffers/normalizer/RNG invariants and budget counts are source-backed runtime assertions; before/after raw parameter/RNG states were not saved for independent replay. Current checkpoint/input/source/output hashes are independently bound.','This is CPU FP32 shadow arithmetic for functions used in training, not a reproduction of historical GPUAMP gradients or proof of original training benefit. Only the WM is learned; the contact functions are analytic geometry priors. Whole-hidden-state and true physical/HO-force gradient qualifications remain false.'],'new_model_objective_geometry_Env_physics_GPU':False,'wall_seconds':time.perf_counter()-start}
    (OUT/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'passed':result['passed'],'total':result['total'],'failed':[k for k,v in checks.items() if not v],'report':pin(OUT/'report.json')},indent=2));return 0 if all(checks.values()) else 1
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        if OUT.exists():(OUT/'failure.json').write_text(json.dumps({'error':repr(exc),'source':pin(Path(__file__)),'new_objective_WM_geometry_execution':False},indent=2)+'\n')
        raise
