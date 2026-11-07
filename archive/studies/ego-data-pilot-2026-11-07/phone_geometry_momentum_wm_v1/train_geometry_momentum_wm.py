"""Fresh geometry + per-pair momentum + finite-response phone WM.

Default is metadata/hash only. NN/data-array work is behind an explicitly
authorized CPU run. No native/Env/physics/Actor code is imported or called.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,os,random,sys,time,traceback
import numpy as np

HERE=Path(__file__).resolve().parent;P=HERE.parent
SCHEMA='phone_geometry_momentum_counterfactual_training_CPU_protocol:1'
ARTIFACT='phone_geometry_momentum_counterfactual_WM_checkpoint:1'
INPUT_SCHEMA='phone_raw80_current_relative_pose22x12_directional22_366:1'
OUTPUT_SCHEMA='phone_physical178_meanFn22_coverage_logits22:1'
ROLE='fresh_current_geometry_pair_momentum_counterfactual_phone_WM'
FIT='CAUSAL_FIT_KNOWN_TRAIN_STATE';CHECK='HELDOUT_ACTION_RESPONSE_CHECK_KNOWN_TRAIN_STATE'
INPUTS={'original_acquisition','original_review','causal_acquisition','causal_review','original_pair_report',
 'causal_pair_report','causal_pair_review','pair_role_packaging','physical_parameters','geometry_catalog',
 'geometry_proof','directional_proof','structured_proof'}
RECIPE=dict(seed=1731,cpu_threads=4,device='cpu',dtype='float32',physical_decode_dtype='float64',
 hidden=[128,128],activation='SiLU',updates=5000,checkpoint_updates=[250,1000,5000],
 stop_after_first_healthy_checkpoint=True,batch='all2118_absolute_rows_and_all1434_FIT_pairs_each_update',
 optimizer='AdamW',learning_rate=.001,weight_decay=0.,betas=[.9,.999],adam_epsilon=1e-8,foreach=False,
 clip_grad_norm=1.,input_std_floor=1e-6,geometry_chunk_rows=4,
 head_std_floors=dict(q_rad=.001,qd_rad_s=.01,COM_residual_m=.0001,world_dR_rad=.001,force_impulse_Ns=.0001,torque_impulse_Nms=.00001),
 decoded_velocity_std_floors=dict(vCOM_m_s=.01,omega_world_rad_s=.1),
 CF_action_reference_rad=1.,CF_denominator='actual_F64_selected_coordinate_du_no_epsilon',
 CF_scale='FIT1434_target_secant_population_std_with_same_physical_unit_floors_per_radian',
 physical_block_reduction='equal_mean_of_q20_qd20_p3_dR3_Jforce66_Jmoment66_block_MSE',
 force_reference_N=1.,force_loss='mean22rows_log1p_pred_N_over_1N_minus_log1p_target_N_over_1N_squared_no_std',
 coverage_loss='BCE_logits_mean_rows_and22pairs',
 lambda_absolute_heads=1.,lambda_decoded_velocity=1.,lambda_FIT_response=1.,lambda_meanFn=1.,lambda_coverage=1.,
 last_layer_initialization='zero_weights_normalized_physical_mean_Fn_TRAINmean_inverse_softplus_coverage_TRAINmean_logit',
 initial_meanFn_floor_N=.001,initial_coverage_probability_clip=.0001,
 validation='original114_complete_VAL_state58_only_no_normalizer_fit_no_model_selection',
 actionCHECK_TEST_fresh='hash_only_not_np_load_not_loss_not_normalizer')

def need(v,msg):
    if not v:raise ValueError(msg)
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(v):
    p=Path(v['path']);need(p.is_absolute() and pin(p)==v,'Current absolute pin: '+str(p));return p
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def clean(v):
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,dict):return {k:clean(a) for k,a in v.items()}
    if isinstance(v,(tuple,list)):return [clean(a) for a in v]
    if isinstance(v,(np.bool_,bool)):return bool(v)
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,np.floating):return float(v)
    return v
def atomic_json(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('x',encoding='utf8') as f:json.dump(clean(v),f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def loadmod(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def bits(a,b):
    a,b=np.asarray(a),np.asarray(b);return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def positive(r,schema):
    need(r['schema']==schema and r.get('status','').startswith('PASS') and all(v is True for v in r['checks'].values()),'Positive frozen parent: '+schema)

def metadata_preflight(protocol):
    p=read(protocol)
    need(p['schema']==SCHEMA and p['role']==ROLE and p['input_schema']==INPUT_SCHEMA and p['output_schema']==OUTPUT_SCHEMA,'New geometry/momentum namespace')
    need(type(p['CPU_run_authorized']) is bool and p['GPU_run_authorized'] is False and p['recipe']==RECIPE and set(p['inputs'])==INPUTS,'Exact CPU recipe/inputs')
    need(read(checked(p['recipe_pin']))['recipe']==RECIPE,'Frozen source recipe file')
    for v in [*p['sources'].values(),*p['inputs'].values()]:checked(v)
    need(checked(p['sources']['trainer']).samefile(__file__),'Own source identity')
    out=Path(p['output']);need(out.is_absolute() and out.parent.resolve()==HERE and not out.exists(),'Fresh local output')
    a={k:read(checked(v)) for k,v in p['inputs'].items()}
    original=a['original_acquisition'];causal=a['causal_acquisition'];pack=a['pair_role_packaging']
    need(original['schema']=='phone_episode_acquisition_result:1' and original['partition_qualified'] is True and original['cross_partition_alias_groups']==[],'Original grouped48 family data')
    positive(a['original_review'],'phone_episode_acquisition_saved_independent:1')
    need(a['original_review']['dataset_report']==p['inputs']['original_acquisition'],'Original review lineage')
    need(causal['schema']=='phone_causal_TRAIN_native_action_acquisition_result:1' and causal['all72_baselines_passed_before_any_side'] is True and causal['side_branches_returned']==2870,'All same-state baselines before sides')
    positive(a['causal_review'],'phone_causal_TRAIN_acquisition_author_saved_review:1')
    need(a['causal_review']['producer_report']==p['inputs']['causal_acquisition'],'Saved causal review lineage; author scope retained')
    positive(a['original_pair_report'],'original_TRAIN_pair22_saved_contact_export_result:1')
    positive(a['causal_pair_report'],'causal_phone22_saved_impulse_export_result:1')
    positive(a['causal_pair_review'],'causal_phone22_impulse_labels_saved_independent:1')
    need(a['causal_pair_review']['producer_report']==p['inputs']['causal_pair_report'],'Independent impulse review lineage')
    need(pack['schema']=='causal_pair22_target_role_packaging:1' and pack['producer_report']==p['inputs']['causal_pair_report'] and pack['summaries'][FIT]==dict(rows=1476,known=1470,unknown=6),'FIT-only label packaging')
    for v in [*original['outputs'].values(),*causal['arrays'].values(),*pack['outputs'].values(),*a['original_pair_report']['outputs'].values()]:checked(v)
    # CHECK and TEST are SHA-bound above but never opened with np.load below.
    positive(a['geometry_proof'],'phone_geometry_features_bounded_CPU_result:1')
    positive(a['directional_proof'],'phone_directional_gap_bounded_tensor_result:1')
    positive(a['structured_proof'],'causal_pair_impulse_structured_codec_saved_result:1')
    cp=read(checked(a['structured_proof']['protocol']))
    need(cp['sources']['codec']==p['sources']['structured_codec'] and cp['inputs']['impulse_report']==p['inputs']['causal_pair_report'] and cp['inputs']['physical_parameters']==p['inputs']['physical_parameters'],'Actual structured codec/labels/context lineage')
    meta=read(checked(a['geometry_catalog']['provider_metadata']))
    need(a['causal_pair_report']['pair_other_geom_ids']==[v['other_geom_id'] for v in meta['pairs']],'Canonical22 order shared by geometry/labels')
    need(a['geometry_catalog']['provider_source']==p['sources']['FK'] and a['directional_proof']['sources']['provider']==p['sources']['directional'],'Frozen geometry sources')
    physical=a['physical_parameters'];need(physical['mass_kg']==.2 and physical['gravity_world_m_s2']==[0.,0.,-9.81],'Pinned nominal simulation context, not measured force')
    if p.get('resume_checkpoint') is not None:
        checked(p['resume_checkpoint']);rh=read(checked(p['resume_health']))
        need(rh['schema']=='phone_geometry_momentum_checkpoint_health:1' and rh['checkpoint']==p['resume_checkpoint'] and all(v is True for v in rh['checks'].values()),'Resume only a separately pinned healthy own checkpoint')
    return p,a,out

def declaration():
    return dict(protocol_schema=SCHEMA,role=ROLE,artifact_schema=ARTIFACT,input_schema=INPUT_SCHEMA,output_schema=OUTPUT_SCHEMA,
        inputs=sorted(INPUTS),recipe=RECIPE,NN_parameter_tensors=6,input_dim=366,output_dim=222,
        first250_counts=dict(training_FW=250,backward=250,optimizer_steps=250,health_FW=2,validation_FW=1,NN_constructors=2,own_checkpoint_loads=1,native_FK_forward_steps=0),
        actual_execution=0,scope='Fresh supervised WM component baseline; no native/Actor/CHORD run or matched-budget method qualification')


def _npz(v,keys=None):
    with np.load(checked(v),allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if keys is None else keys)}

def prepare_data(p,a,codec):
    """No CHECK/TEST NPZ opens; retain every excluded original/FIT row ID."""
    o=_npz(a['original_acquisition']['outputs']['train']);s=_npz(a['causal_acquisition']['arrays'][FIT])
    ol=_npz(a['original_pair_report']['outputs']['labels']);sl=_npz(a['pair_role_packaging']['outputs'][FIT])
    oi={int(v):i for i,v in enumerate(ol['sample_id_i64'])};si={int(v):i for i,v in enumerate(sl['branch_id'])}
    keep=o['training_eligible_bool'];need(len(keep)==720 and int(keep.sum())==684,'Original684 complete plus36 retained partial')
    need(all(ol['physical_H_labels_known'][oi[int(v)]]==bool(k) for v,k in zip(o['row_id_i64'],keep)),'Original label mask/rowID match')
    rows=np.flatnonzero(keep);orig_index={int(o['row_id_i64'][i]):j for j,i in enumerate(rows)}
    contexts=read(checked(a['causal_acquisition']['inputs']['contexts']));side=np.flatnonzero(s['executed_bool']&(s['branch_id_i64']>=72))
    need(len(side)==1434 and s['training_eligible_bool'][side].all(),'Exactly1434 available complete FIT SIDES')
    baseline=np.flatnonzero(s['branch_id_i64']<72);need(len(baseline)==36,'36 baseline references, not extra absolute training rows')
    X=o['x58_F64'][rows];U=o['u20_F64'][rows];Y=o['y58_F64'][rows];h=o['native_dt_s_F64'][rows];n=o['n_substeps_i64'][rows]
    li=np.array([oi[int(v)] for v in o['row_id_i64'][rows]])
    J=ol['impulse_source_BODY_moving_COM22x6'][li];Fn=ol['mean_positive_Fn22_N'][li];coverage=ol['Fn_gt_0p1_coverage22'][li]
    need(bits(ol['pair_other_geom_ids'],sl['pair_other_geom_ids']),'Original/FIT label slot order')
    base_for_context={}
    for j in baseline:
        c=contexts[int(s['context_id_i64'][j])];old=orig_index[c['source_sample_id']];lb=si[int(s['branch_id_i64'][j])]
        need(bits(s['x58_F64'][j],X[old]) and bits(s['u20_F64'][j],U[old]) and bits(s['y58_at_H_F64'][j],Y[old]),'Baseline = existing original absolute sample')
        need(bits(sl['impulse_source_BODY_moving_COM22x6'][lb],J[old]) and bits(sl['mean_positive_Fn22_N'][lb],Fn[old]) and bits(sl['Fn_gt_0p1_coverage22'][lb],coverage[old]),'Replayed baseline labels identical, not double-weighted')
        base_for_context[c['context_id']]=old
    slidx=np.array([si[int(v)] for v in s['branch_id_i64'][side]])
    need(sl['physical_labels_known'][slidx].all(),'Actual measured simulation contact labels, no inferred impulse targets')
    X=np.concatenate([X,s['x58_F64'][side]]);U=np.concatenate([U,s['u20_F64'][side]]);Y=np.concatenate([Y,s['y58_at_H_F64'][side]])
    h=np.r_[h,s['native_h_s_F64'][side]];n=np.r_[n,s['n_substeps_i64'][side]]
    J=np.concatenate([J,sl['impulse_source_BODY_moving_COM22x6'][slidx]]);Fn=np.concatenate([Fn,sl['mean_positive_Fn22_N'][slidx]]);coverage=np.concatenate([coverage,sl['Fn_gt_0p1_coverage22'][slidx]])
    b=np.array([base_for_context[int(v)] for v in s['context_id_i64'][side]],dtype=np.int64);ss=np.arange(684,2118,dtype=np.int64)
    du20=U[ss]-U[b];joint=np.argmax(abs(du20),axis=1);du=du20[np.arange(1434),joint]
    need(np.all(abs(du)>=1e-6) and np.all(np.sum(du20!=0,axis=1)==1) and bits(X[ss],X[b]) and bits(h[ss],h[b]),'All1434 same-visible-input one-coordinate actual responses')
    need(np.all(U[ss].astype(np.float32)[np.arange(1434),joint]!=U[b].astype(np.float32)[np.arange(1434),joint]),'No encoded endpoint collapse accepted as finite model secant')
    physical=a['physical_parameters'];T=h*n;g=np.asarray(physical['gravity_world_m_s2'])
    encoded=codec.encode_observed_numpy(X,Y,J,T,g)
    head=np.concatenate([encoded['joint_delta40'],encoded['COM_ballistic_residual3_m'],encoded['world_rotation_delta3_rad'],J.reshape(2118,132)],axis=1)
    need(head.shape==(2118,178) and all(np.isfinite(v).all() for v in [X,U,Y,h,head,Fn,coverage]) and np.all(T==.02),'All complete finite FIT absolute labels')
    need(np.all(Fn>=0) and np.all((coverage>=0)&(coverage<=1)),'Nonnegative physical normal load and fraction labels')
    val=_npz(a['original_acquisition']['outputs']['validation']);vm=val['training_eligible_bool'];need(int(vm.sum())==114,'Original114 complete validation targets')
    receipt=dict(absolute_rows=2118,original_complete_rows=684,FIT_side_rows=1434,baseline_references_not_duplicated=36,
        absolute_original_row_ids=o['row_id_i64'][rows].tolist(),absolute_FIT_side_branch_ids=s['branch_id_i64'][side].tolist(),
        excluded_original_partial_row_ids=o['row_id_i64'][~keep].tolist(),excluded_FIT_unavailable_branch_ids=s['branch_id_i64'][~s['executed_bool']].tolist(),
        CF_side_absolute_index=ss.tolist(),CF_baseline_absolute_index=b.tolist(),CF_actual_selected_du_rad=du.tolist(),CF_joint_index=joint.tolist(),
        original_VAL_values_opened_for_fixed_health_not_fit=True,TEST_CHECK_NPZ_opens=0,
        scope='Known-state native action responses, original state aliases retained. No unseen-state/ego claim or fullhidden qualification.')
    return dict(x=X,u=U,y=Y,h=h,n=n,T=T,head=head,Fn=Fn,coverage=coverage,cf_side=ss,cf_base=b,du=du,
        val=dict(x=val['x58_F64'][vm],u=val['u20_F64'][vm],y=val['y58_F64'][vm],h=val['native_dt_s_F64'][vm],n=val['n_substeps_i64'][vm])),receipt


def physical_blocks():
    return [np.arange(0,20),np.arange(20,40),np.arange(40,43),np.arange(43,46),
        np.array([46+6*p+j for p in range(22) for j in range(3)]),np.array([46+6*p+j for p in range(22) for j in range(3,6)])]

def physical_floors():
    f=RECIPE['head_std_floors'];values=[f['q_rad'],f['qd_rad_s'],f['COM_residual_m'],f['world_dR_rad'],f['force_impulse_Ns'],f['torque_impulse_Nms']]
    out=np.zeros(178)
    for index,value in zip(physical_blocks(),values):out[index]=value
    return out

def current_features(x,u,h,n,geometry,torch):
    """Live differentiable current geometry. No target/force/contact labels."""
    pose=geometry.provider.poses(x);gap=geometry.features(x)
    need(gap['directional_gap_known'].all().item(),'No fake zero/unknown gap inputs')
    rel=torch.cat([pose['pair_relative_translation_BODY_COM_m'],pose['pair_relative_rotation_BODY9']],dim=-1).flatten(-2)
    raw=torch.cat([x,u,h[...,None].to(x.dtype),n[...,None].to(x.dtype)],dim=-1)
    result=torch.cat([raw,rel,gap['directional_gap_m']],dim=-1)
    need(result.shape[-1]==366 and torch.isfinite(result).all().item(),'Finite current observable366 input')
    return result

def precompute_features(data,geometry,torch):
    """Cache by identical current state bytes, not by future labels or actions."""
    X=data['x'];unique=[];ids=[];lookup={}
    for x in X:
        key=x.tobytes()
        if key not in lookup:lookup[key]=len(unique);unique.append(x)
        ids.append(lookup[key])
    relgap=[];chunk=RECIPE['geometry_chunk_rows']
    with torch.no_grad():
        for i in range(0,len(unique),chunk):
            x=torch.tensor(np.array(unique[i:i+chunk]),dtype=torch.float64);u=torch.zeros((len(x),20),dtype=torch.float64)
            v=current_features(x,u,torch.full((len(x),),.00015625,dtype=torch.float64),torch.full((len(x),),128,dtype=torch.int64),geometry,torch)
            relgap.append(v[:,80:].numpy())
    G=np.concatenate(relgap)[ids];raw=np.column_stack([X,data['u'],data['h'],data['n']])
    return np.column_stack([raw,G]),dict(unique_current_states=len(unique),rows=len(X),pure_geometry_FK_calls=2*math.ceil(len(unique)/chunk),
        geometry_is_current_only=True,caching_does_not_change_training_row_weight=True)

def make_model(torch):
    class GeometryMomentumNet(torch.nn.Module):
        def __init__(self):
            super().__init__();self.layers=torch.nn.Sequential(torch.nn.Linear(366,128),torch.nn.SiLU(),torch.nn.Linear(128,128),torch.nn.SiLU(),torch.nn.Linear(128,222))
        def forward(self,normalized_features366):return self.layers(normalized_features366)
        def forward_from_raw(self,x,u,h,n,norms,geometry):
            f=current_features(x,u,h,n,geometry,torch)
            return self(((f-norms['input_mean'])/norms['input_std']).to(torch.float32))
    return GeometryMomentumNet()

def heads_from_physical(v):
    return dict(joint_delta40=v[:,:40],COM_ballistic_residual3_m=v[:,40:43],world_rotation_delta3_rad=v[:,43:46],pair_impulse_source_BODY_moving_COM22x6=v[:,46:].reshape(-1,22,6))

def block_mse(error,blocks,torch):return torch.stack([error[:,index].square().mean() for index in blocks]).mean()

def losses(raw,d,norm,codec,physical,torch):
    head=raw[:,:178].double()*norm['head_std']+norm['head_mean']
    pred=codec.decode_physical_heads(d['x'],heads_from_physical(head),d['T'],physical['mass'],physical['I'],physical['g'])
    Lhead=block_mse((head-d['head'])/norm['head_std'],physical_blocks(),torch)
    verr=(pred[:,52:58]-d['y'][:,52:58])/norm['velocity_std'];Lv=(verr[:,:3].square().mean()+verr[:,3:].square().mean())*.5
    sec=(head[d['cf_side']]-head[d['cf_base']])/d['du'][:,None]
    Lcf=block_mse((sec-d['cf_target'])/norm['CF_std'],physical_blocks(),torch)
    fn=torch.nn.functional.softplus(raw[:,178:200])*RECIPE['force_reference_N']
    Lfn=(torch.log1p(fn/RECIPE['force_reference_N'])-torch.log1p(d['Fn']/RECIPE['force_reference_N'])).square().mean()
    Lcov=torch.nn.functional.binary_cross_entropy_with_logits(raw[:,200:222],d['coverage'])
    total=RECIPE['lambda_absolute_heads']*Lhead+RECIPE['lambda_decoded_velocity']*Lv+RECIPE['lambda_FIT_response']*Lcf+RECIPE['lambda_meanFn']*Lfn+RECIPE['lambda_coverage']*Lcov
    return total,dict(absolute_heads=Lhead,decoded_velocity=Lv,FIT_head_secants=Lcf,meanFn_log1p=Lfn,coverage_BCE=Lcov,total=total),head,pred


def tensor_digest(values,torch):
    h=hashlib.sha256()
    for name,v in sorted(values.items()):
        t=v.detach().cpu().contiguous();h.update((name+str(t.dtype)+str(tuple(t.shape))).encode());h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def rng(torch):return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state().clone())
def tree_equal(a,b,torch):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),b.detach().cpu().contiguous().reshape(-1).view(torch.uint8))
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and bits(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(tree_equal(a[k],b[k],torch) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(tree_equal(x,y,torch) for x,y in zip(a,b))
    return a==b
def atomic_checkpoint(path,payload,torch):
    temp=path.with_suffix('.tmp')
    with temp.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def checkpoint_health(model,opt,checkpoint,probe_features,grads,norm,data_tensors,fit_digest,physical,torch,iteration):
    """Own serialized readback only; raw RNG values, not pickle digests."""
    before=rng(torch)
    with torch.no_grad():expected=model(probe_features).clone()
    loaded=torch.load(checkpoint,map_location='cpu',weights_only=False)
    need(loaded['schema']==ARTIFACT and loaded['iteration']==iteration and loaded['tag']==f'AFTER_OPTIMIZER_{iteration}' and loaded['gradient_snapshot']['tag']==f'PRE_OPTIMIZER_{iteration}' and loaded['gradient_snapshot']['weights_after_updates']==iteration-1,'Own serialized checkpoint/gradient phase')
    with torch.random.fork_rng(devices=[]):restored=make_model(torch)
    restored.load_state_dict(loaded['model_state'],strict=True)
    with torch.no_grad():actual=restored(probe_features)
    lp=loaded['log_prefix']
    with Path(lp['path']).open('rb') as f:prefix=f.read(lp['bytes'])
    prefix_rows=[json.loads(line) for line in prefix.splitlines()]
    names=[n for n,_ in model.named_parameters()];checks=dict(exact_serialized_prediction=tree_equal(expected,actual,torch),
        all6_parameters_finite=len(names)==6 and all(torch.isfinite(v).all().item() for v in model.parameters()),
        all_pre_post_clip_gradients_finite=all(torch.isfinite(v).all().item() for group in grads.values() for v in group.values()),
        Adam6_states_stepI=len(opt.state)==6 and all(float(v['step'])==iteration and torch.isfinite(v['exp_avg']).all().item() and torch.isfinite(v['exp_avg_sq']).all().item() for v in opt.state.values()),
        checkpoint_weights_rawbytes=tensor_digest(model.state_dict(),torch)==tensor_digest(loaded['model_state'],torch),
        serialized_Adam_matches_raw_current=tree_equal(loaded['optimizer_state'],opt.state_dict(),torch),
        serialized_gradient_snapshot_rawbytes=tree_equal(loaded['gradient_snapshot']['pre_clip'],grads['pre'],torch) and tree_equal(loaded['gradient_snapshot']['post_clip'],grads['post'],torch),
        train_only_normalizers_rawbytes=tree_equal(loaded['normalizers'],norm,torch),
        all_normalizers_finite_positive_std=all(torch.isfinite(v).all().item() for v in norm.values()) and all(torch.all(norm[k]>0).item() for k in ['input_std','head_std','CF_std','velocity_std']),
        physical_context_rawbytes=tree_equal(loaded['physical_context'],physical,torch),
        FIT_inputs_targets_unchanged=tensor_digest(data_tensors,torch)==fit_digest==loaded['fit_input_target_digest'],
        saved_RNG_matches_this_run_current=tree_equal(loaded['RNG'],before,torch),health_RNG_raw_unchanged=tree_equal(before,rng(torch),torch),CPU_only=not torch.cuda.is_initialized())
    checks['serialized_log_prefix_bytes_and_iterations']=(len(prefix)==lp['bytes'] and hashlib.sha256(prefix).hexdigest()==lp['sha256'] and len(prefix_rows)==lp['records'] and [v['iteration'] for v in prefix_rows]==list(range(lp['global_iteration_range'][0],lp['global_iteration_range'][1]+1)) and lp['global_iteration_range'][1]==iteration)
    need(all(checks.values()),'Checkpoint health failure; preserve and stop')
    health=dict(schema='phone_geometry_momentum_checkpoint_health:1',iteration=iteration,checkpoint=pin(checkpoint),checks=checks,passed=sum(checks.values()),total=len(checks),
        gradients_phase=f'PRE_OPTIMIZER_{iteration}; weights AFTER_OPTIMIZER_{iteration}',own_CPU_loads=1,extra_NN_constructors=1,extra_NN_forward_calls=2,
        raw_RNG_comparison=True,no_native_physics_or_training_replay=True)
    evidence=dict(before_reload_raw222_FP32=expected.numpy(),after_reload_raw222_FP32=actual.numpy(),Torch_RNG_before=before['torch'].numpy(),Torch_RNG_after=torch.get_rng_state().numpy())
    return health,evidence


def run_training(protocol):
    p,a,out=metadata_preflight(protocol);need(p['CPU_run_authorized'] is True,'Root must freeze an authorized run protocol')
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch
    torch.set_num_threads(4);need(not torch.cuda.is_initialized(),'CPU only')
    random.seed(RECIPE['seed']);np.random.seed(RECIPE['seed']);torch.manual_seed(RECIPE['seed'])
    out.mkdir();atomic_json(out/'protocol.json',p);started=time.perf_counter();iteration=0;phase='dataset'
    counts=dict(training_FW=0,backward=0,optimizer_steps=0,validation_FW=0,health_FW=0,NN_constructors=0,own_checkpoint_loads=0,native_FK_forward_steps=0)
    try:
        codec=loadmod('geom_momentum_structured',checked(p['sources']['structured_codec']))
        gapmodule=loadmod('geom_momentum_current_gap',checked(p['sources']['directional']))
        geometry=gapmodule.DirectionalGapProvider(checked(p['inputs']['geometry_catalog']))
        data,receipt=prepare_data(p,a,codec);atomic_json(out/'data_selection.json',receipt)
        F,cache=precompute_features(data,geometry,torch);VF,vcache=precompute_features(data['val'],geometry,torch)
        floors=physical_floors();sec=(data['head'][data['cf_side']]-data['head'][data['cf_base']])/data['du'][:,None]
        norm_np=dict(input_mean=F.mean(0),input_std=np.maximum(F.std(0),RECIPE['input_std_floor']),head_mean=data['head'].mean(0),head_std=np.maximum(data['head'].std(0),floors),CF_std=np.maximum(sec.std(0),floors),
            velocity_std=np.maximum((data['y'][:,52:58]-data['x'][:,52:58]).std(0),[.01,.01,.01,.1,.1,.1]))
        norm={k:torch.tensor(v,dtype=torch.float64) for k,v in norm_np.items()}
        feature=torch.tensor((F-norm_np['input_mean'])/norm_np['input_std'],dtype=torch.float32)
        valfeature=torch.tensor((VF-norm_np['input_mean'])/norm_np['input_std'],dtype=torch.float32)
        d={k:torch.tensor(data[k],dtype=torch.float64) for k in ['x','y','T','head','du']}
        d.update(cf_side=torch.tensor(data['cf_side']),cf_base=torch.tensor(data['cf_base']),cf_target=torch.tensor(sec,dtype=torch.float64),Fn=torch.tensor(data['Fn'],dtype=torch.float32),coverage=torch.tensor(data['coverage'],dtype=torch.float32))
        phys=a['physical_parameters'];physical=dict(mass=torch.tensor(phys['mass_kg'],dtype=torch.float64),I=torch.tensor(phys['I_body_3x3'],dtype=torch.float64),g=torch.tensor(phys['gravity_world_m_s2'],dtype=torch.float64))
        with torch.no_grad():
            label_decoded=codec.decode_physical_heads(d['x'],heads_from_physical(d['head']),d['T'],physical['mass'],physical['I'],physical['g'])
            label_decode_defect=(label_decoded-d['y']).numpy()
        model=make_model(torch);counts['NN_constructors']+=1
        with torch.no_grad():
            last=model.layers[-1];last.weight.zero_();last.bias.zero_()
            meanFn=np.maximum(data['Fn'].mean(0)/RECIPE['force_reference_N'],RECIPE['initial_meanFn_floor_N'])
            last.bias[178:200].copy_(torch.tensor(meanFn+np.log(-np.expm1(-meanFn)),dtype=torch.float32))
            prob=np.clip(data['coverage'].mean(0),RECIPE['initial_coverage_probability_clip'],1-RECIPE['initial_coverage_probability_clip'])
            last.bias[200:222].copy_(torch.tensor(np.log(prob/(1-prob)),dtype=torch.float32))
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.,betas=(.9,.999),eps=1e-8,foreach=False)
        names=[n for n,_ in model.named_parameters()];initial_rng=rng(torch);start_update=1
        digest_tensors=dict(d,normalized_features=feature);fit_digest=tensor_digest(digest_tensors,torch)
        initial_path=out/'fresh_initial_state.pt'
        atomic_checkpoint(initial_path,dict(schema='phone_geometry_momentum_WM_initial:1',recipe=RECIPE,protocol=pin(protocol),inputs=p['inputs'],sources=p['sources'],
            model_state={k:v.detach().clone() for k,v in model.state_dict().items()},ordered_parameter_names=names,normalizers=norm,physical_context=physical,RNG=initial_rng,
            fit_input_target_digest=fit_digest,iteration=0,counts=counts.copy()),torch)
        if p.get('resume_checkpoint') is not None:
            ck=torch.load(checked(p['resume_checkpoint']),map_location='cpu',weights_only=False);counts['own_checkpoint_loads']+=1
            parent_protocol=read(checked(ck['protocol']))
            need(ck['schema']==ARTIFACT and ck['recipe']==RECIPE and ck['inputs']==p['inputs'] and ck['sources']==p['sources'] and ck['iteration'] in [250,1000] and parent_protocol['inputs']==p['inputs'] and parent_protocol['recipe']==RECIPE,'Explicit same fixed recipe/data/source resume lineage')
            need(tensor_digest(ck['normalizers'],torch)==tensor_digest(norm,torch),'Resume recomputed TRAIN-only normalizers identical')
            need(ck['fit_input_target_digest']==fit_digest,'Resume uses identical FIT inputs/targets/pairs')
            model.load_state_dict(ck['model_state']);opt.load_state_dict(ck['optimizer_state']);start_update=ck['iteration']+1
            random.setstate(ck['RNG']['python']);np.random.set_state(ck['RNG']['numpy']);torch.set_rng_state(ck['RNG']['torch'])
            counts['training_FW']=counts['backward']=counts['optimizer_steps']=ck['iteration']
        atomic_json(out/'normalizer_scope.json',dict(rows=2118,CF_pairs=1434,fit_only=True,normalizer_digest=tensor_digest(norm,torch),geometry_cache=cache,validation_geometry_cache=vcache,TEST_CHECK_np_load=0))
        np.savez_compressed(out/'prepared_FIT_targets_and_features.npz',features366_F64=F,features366_FP32_normalized=feature.numpy(),heads178_F64=data['head'],CF_target178_per_rad_F64=sec,CF_side=data['cf_side'],CF_base=data['cf_base'],CF_actual_du_F64=data['du'],observed_label_decode_defect58_F64=label_decode_defect,**norm_np)
        setup_s=time.perf_counter()-started;update_times=[];training_log_IO_s=0.
        with (out/'updates.jsonl').open('x',encoding='utf8') as log:
            for iteration in range(start_update,5001):
                phase=f'update_{iteration}';tick=time.perf_counter();opt.zero_grad(set_to_none=True)
                raw=model(feature);counts['training_FW']+=1
                total,scalars,head,pred=losses(raw,d,norm,codec,physical,torch)
                need(torch.isfinite(total).item(),'Finite loss, no retry');total.backward();counts['backward']+=1
                need(all(v.grad is not None and torch.isfinite(v.grad).all().item() for v in model.parameters()),'All named gradients finite')
                pre={n:v.grad.detach().clone() for n,v in model.named_parameters()}
                norm_before=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=False)
                post={n:v.grad.detach().clone() for n,v in model.named_parameters()};opt.step();counts['optimizer_steps']+=1
                need(all(torch.isfinite(v).all().item() for v in model.parameters()),'Finite parameters after update')
                row=dict(iteration=iteration,losses={k:float(v.detach()) for k,v in scalars.items()},clip_pre_norm=float(norm_before),update_seconds=time.perf_counter()-tick)
                update_times.append(row['update_seconds']);io_start=time.perf_counter();log.write(json.dumps(row)+'\n');log.flush();training_log_IO_s+=time.perf_counter()-io_start
                if iteration in [250,1000,5000]:
                    checkpoint_phase_start=time.perf_counter()
                    os.fsync(log.fileno());path=out/f'checkpoint_{iteration:04d}.pt'
                    prefix=(out/'updates.jsonl').read_bytes();need(len(prefix)==log.tell(),'Exact flushed log prefix bytes')
                    payload=dict(schema=ARTIFACT,iteration=iteration,tag=f'AFTER_OPTIMIZER_{iteration}',recipe=RECIPE,protocol=pin(protocol),inputs=p['inputs'],sources=p['sources'],input_schema=INPUT_SCHEMA,output_schema=OUTPUT_SCHEMA,
                        model_state={k:v.detach().clone() for k,v in model.state_dict().items()},optimizer_state=opt.state_dict(),ordered_parameter_names=names,normalizers=norm,physical_context=physical,
                        gradient_snapshot=dict(tag=f'PRE_OPTIMIZER_{iteration}',weights_after_updates=iteration-1,pre_clip=pre,post_clip=post,clip_returned_norm=norm_before.detach()),
                        loss_snapshot=row['losses'],RNG=rng(torch),initial_RNG_this_run=initial_rng,counts_at_save_before_health=counts.copy(),data_selection=receipt,normalizer_scope='2118FIT absolute/1434FIT response only',
                        fit_input_target_digest=fit_digest,fresh_initial_checkpoint=pin(initial_path),resume_parent_checkpoint=p.get('resume_checkpoint'),
                        log_prefix=dict(path=str(out/'updates.jsonl'),bytes=len(prefix),sha256=hashlib.sha256(prefix).hexdigest(),records=iteration-start_update+1,global_iteration_range=[start_update,iteration]),training_model_not_native_checkpoint=True)
                    atomic_checkpoint(path,payload,torch)
                    phase=f'health_{iteration}'
                    health,evidence=checkpoint_health(model,opt,path,feature[:8],dict(pre=pre,post=post),norm,digest_tensors,fit_digest,physical,torch,iteration)
                    proofpath=out/f'health_readback_{iteration:04d}.npz';np.savez_compressed(proofpath,**evidence);health['saved_readback_evidence']=pin(proofpath)
                    counts['own_checkpoint_loads']+=1;counts['NN_constructors']+=1;counts['health_FW']+=2
                    with torch.no_grad():
                        vr=model(valfeature);counts['validation_FW']+=1
                        vh=vr[:,:178].double()*norm['head_std']+norm['head_mean'];vx=torch.tensor(data['val']['x'],dtype=torch.float64)
                        vy=codec.decode_physical_heads(vx,heads_from_physical(vh),torch.tensor(data['val']['h']*data['val']['n']),physical['mass'],physical['I'],physical['g'])
                        truth=torch.tensor(data['val']['y'],dtype=torch.float64);error=vy-truth
                        angles=np.array([np.linalg.norm(codec.scalar.log_world(rp@rt.T)) for rp,rt in zip(vy[:,43:52].numpy().reshape(-1,3,3),data['val']['y'][:,43:52].reshape(-1,3,3))])
                        health['validation_state_descriptor']=dict(rows=114,COM_vector_RMS_mm=float(torch.sqrt(error[:,40:43].square().sum(1).mean())*1000),SO3_geodesic_RMS_deg=float(np.sqrt(np.mean(angles**2))*180/np.pi),vCOM_vector_RMS_m_s=float(torch.sqrt(error[:,52:55].square().sum(1).mean())),omega_world_vector_RMS_rad_s=float(torch.sqrt(error[:,55:58].square().sum(1).mean())),scope='Fixed monitor only; no checkpoint/weight selection, no VAL pair labels opened')
                    elapsed=time.perf_counter()-started;remaining=5000-iteration
                    mean_all=float(np.mean(update_times));mean_recent=float(np.mean(update_times[-100:]));overhead=time.perf_counter()-checkpoint_phase_start
                    future_checks=sum(v>iteration for v in RECIPE['checkpoint_updates'])
                    resume_setup=setup_s if remaining else 0.
                    health.update(elapsed_s=elapsed,setup_s=setup_s,training_update_seconds_sum=float(sum(update_times)),training_log_IO_s=training_log_IO_s,checkpoint_health_validation_IO_s=overhead,
                        update_seconds_mean_all=mean_all,update_seconds_mean_recent100=mean_recent,
                        estimated_remaining_s=mean_recent*remaining,ETA_scope='Remaining training updates at observed last100 update_seconds only; excludes cache/setup/health/IO',
                        estimated_resume_setup_s=resume_setup,
                        estimated_resume_end_to_end_s_rate_scenarios=[resume_setup+min(mean_all,mean_recent)*remaining+future_checks*overhead,resume_setup+max(mean_all,mean_recent)*remaining+future_checks*overhead],
                        resume_ETA_scope='Two observed-rate scenarios plus one observed setup rebuild and future checkpoint overhead; heuristic, not confidence bounds or a precise wall-time guarantee',
                        counts=counts.copy(),first_checkpoint_pausepoint=iteration==250,process_stops_here=iteration==250 and RECIPE['stop_after_first_healthy_checkpoint'])
                    atomic_json(out/f'health_{iteration:04d}.json',health)
                    print(json.dumps(dict(iteration=iteration,checkpoint=str(path),health_passed=health['passed'],health_total=health['total'],ETA_s=health['estimated_remaining_s'])),flush=True)
                    if iteration==250 and RECIPE['stop_after_first_healthy_checkpoint']:
                        atomic_json(out/'result.json',dict(schema='phone_geometry_momentum_training_result:1',status='STOPPED_AT_FIRST_HEALTHY_250_RESUMABLE_TARGET5000_NOT_COMPLETE',iteration=iteration,checkpoint=pin(path),health=pin(out/f'health_{iteration:04d}.json'),counts=counts,ETA_s=health['estimated_remaining_s'],protocol=pin(protocol),no_Actor_CHORD_or_ego_qualification=True));return 0
        atomic_json(out/'result.json',dict(schema='phone_geometry_momentum_training_result:1',status='COMPLETED_FIXED5000_UPDATES_NOT_POLICY_QUALIFICATION',iteration=5000,checkpoint=pin(out/'checkpoint_5000.pt'),health=pin(out/'health_5000.json'),counts=counts,protocol=pin(protocol)))
        return 0
    except BaseException as e:
        atomic_json(out/'failure.json',dict(status='FAILED_PRESERVED_NO_RETRY',phase=phase,iteration=iteration,error=repr(e),traceback=traceback.format_exc(),counts=counts));raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run',action='store_true');arg=ap.parse_args()
    if arg.run:raise SystemExit(run_training(arg.protocol))
    p,a,o=metadata_preflight(arg.protocol)
    print(json.dumps(dict(status='METADATA_ONLY_NO_TORCH_NPZ_NN_NATIVE',input_dim=366,output_dim=222,absolute_FIT_rows=2118,CF_FIT_pairs=1434,CPU_run_authorized=p['CPU_run_authorized'],pausepoint_update=250,resumable_target_update=5000)))
