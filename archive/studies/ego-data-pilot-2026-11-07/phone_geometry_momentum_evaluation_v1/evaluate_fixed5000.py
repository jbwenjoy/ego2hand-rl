"""Fixed5000 known-state action-response evaluation. Default metadata only.

One own CPU checkpoint load, one frozen NN ctor and two inference forwards.
No fit, optimizer, AD, native or TEST/FIT NPZ opens. CHECK is first used here
for model evaluation, not represented as unseen states or independent ego.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,os,random,sys,time,traceback
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent
SCHEMA='phone_geometry_momentum_fixed5000_evaluation_CPU_protocol:1'
CHECK='HELDOUT_ACTION_RESPONSE_CHECK_KNOWN_TRAIN_STATE'
TRAINER_SHA='224a0c9f47e196c626e862aa52d46402a454a6d16f0b497d3720ae74e25ca539'
INPUTS={'training_result','training_health','training_terminal','original_acquisition','causal_acquisition',
 'causal_review','pair_role_packaging','causal_pair_report','causal_pair_review','geometry_catalog'}
RULES=dict(primary_iteration=5000,checkpoint_selection='fixed_before_CHECK_no_selection',cpu_threads=4,
 CHECK_plan_rows=1476,CHECK_known_endpoints=1472,CHECK_baselines=36,CHECK_side_records=1440,CHECK_known_sides=1436,
 central_pair_records=720,VAL_state_rows=114,TEST_NPZ_opens=0,TRAIN_FIT_NPZ_opens=0,
 primary_denominator='actual_F64_u_side_j_minus_u_baseline_j_rad',
 raw_FP32_denominator_descriptor='F64(cast_FP32_u_side_j)_minus_F64(cast_FP32_u_baseline_j)_rad',
 secondary_denominator='F64(normalized_input_FP32_side_j_minus_base_j)_times_checkpoint_input_std_j_rad',
 du_strata=['REQUESTED_0p01_REPRESENTATION','BOUND_REDUCED_GE_1e_6','UNAVAILABLE'],
 native_signal_strata=['EXACT_F64_ZERO','NONZERO_F64_FP32_ENDPOINT_COLLAPSED','FP32_ENDPOINT_RESOLVED'],
 precision_scope='Per-pair/component endpoint ULP magnitude descriptors, not native noise/accuracy/stability floors',
 baselines=['constant_world_velocity_constant_joint_velocity','continuous_ballistic_constant_world_omega'],
 pair_contact_baselines='explicit_zero_impulse_zero_meanFn_zero_coverage_prior_not_inferred_support',
 no_best_case_axis_checkpoint_or_weight_selection=True)

def need(x,msg):
    if not x:raise ValueError(msg)
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(v):need(Path(v['path']).is_absolute() and pin(v['path'])==v,'Current absolute file pin');return Path(v['path'])
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def clean(v):
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,(bool,np.bool_)):return bool(v)
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    return v
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(clean(v),f,indent=2,allow_nan=False)
def loadmod(name,p):
    sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def npz(v):
    with np.load(checked(v),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def bits(a,b):a,b=np.asarray(a),np.asarray(b);return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()

def metadata_preflight(path):
    p=read(path);need(p['schema']==SCHEMA and p['rules']==RULES and set(p['inputs'])==INPUTS,'Fixed evaluation protocol/inputs')
    need(type(p['CPU_run_authorized']) is bool and p['GPU_run_authorized'] is False,'CPU-only flag')
    for v in [*p['inputs'].values(),*p['sources'].values()]:checked(v)
    need(p['sources']['evaluator']==pin(__file__) and p['sources']['trainer']['sha256']==TRAINER_SHA,'Exact eval and frozen trained architecture')
    a={k:read(checked(v)) for k,v in p['inputs'].items()}
    r,health,t=a['training_result'],a['training_health'],a['training_terminal']
    need(r['schema']=='phone_geometry_momentum_training_result:1' and r['status']=='COMPLETED_FIXED5000_UPDATES_NOT_POLICY_QUALIFICATION' and r['iteration']==5000,'Final5000 only, no250/1000 selection')
    need(r['health']==p['inputs']['training_health'] and health['iteration']==5000 and health['checkpoint']==r['checkpoint'] and all(v is True for v in health['checks'].values()),'Final health lineage')
    need(all(health['counts'][k]==5000 for k in ['training_FW','backward','optimizer_steps']),'Completed cumulative5000 training counts')
    need(t['returncode']==0 and t['protocol']==r['protocol'] and t['source']==p['sources']['trainer'],'Final process/source closed')
    checked(r['checkpoint']);checked(r['protocol'])
    trainp=read(r['protocol']['path']);need(trainp['sources']['trainer']==p['sources']['trainer'] and trainp['inputs']['original_acquisition']==p['inputs']['original_acquisition'] and trainp['inputs']['causal_acquisition']==p['inputs']['causal_acquisition'],'Same fitted data/context')
    need(trainp['recipe']['updates']==5000 and trainp['recipe']['stop_after_first_healthy_checkpoint'] is True,'Unchanged fixed recipe')
    need(a['pair_role_packaging']['producer_report']==p['inputs']['causal_pair_report'] and a['causal_pair_review']['producer_report']==p['inputs']['causal_pair_report'],'Qualified CHECK labels lineage')
    need(a['causal_review']['producer_report']==p['inputs']['causal_acquisition'] and all(v is True for v in a['causal_review']['checks'].values()),'Saved state/branch grouping review')
    need(a['pair_role_packaging']['summaries'][CHECK]==dict(rows=1476,known=1472,unknown=4),'All CHECK rows retained')
    for v in [*a['original_acquisition']['outputs'].values(),*a['causal_acquisition']['arrays'].values(),*a['pair_role_packaging']['outputs'].values()]:checked(v)
    out=Path(p['output']);need(out.is_absolute() and out.parent.resolve()==H and not out.exists(),'Fresh exact evaluation output')
    return p,a,trainp,out

def load_evaluation_data(a):
    d=npz(a['causal_acquisition']['arrays'][CHECK]);labels=npz(a['pair_role_packaging']['outputs'][CHECK])
    records=read(checked(a['causal_acquisition']['records']));contexts=read(checked(a['causal_acquisition']['inputs']['contexts']))
    ids={int(v):i for i,v in enumerate(labels['branch_id'])};li=np.array([ids[int(v)] for v in d['branch_id_i64']])
    known=d['training_eligible_bool'];need(len(known)==1476 and int(known.sum())==1472 and bits(known,labels['physical_labels_known'][li]),'All1476 CHECK exact labels/masks')
    base={};side=[];central={};rows=[]
    for i,bid in enumerate(d['branch_id_i64']):
        r=records[int(bid)];c=contexts[int(d['context_id_i64'][i])]
        need(r['partition']==c['partition']==CHECK and r['original_family_id']==c['original_family_id'],'KnownTRAIN family role, no reassignment')
        rows.append(dict(branch_id=int(bid),context_id=c['context_id'],family_id=c['original_family_id'],source_branch=c['source_branch'],native_h_s=c['native_h_s'],kind=r['kind'],known=bool(known[i]),plan=r['plan']))
        if r['kind']=='BASELINE':base[c['context_id']]=i
        else:
            j=r['plan']['joint_index'];sign=r['plan']['sign'];central.setdefault((c['context_id'],j),{})[sign]=i;side.append(i)
    need(len(base)==36 and len(side)==1440 and len(central)==720,'All baselines/20axes/signs')
    bidx=np.array([base[rows[i]['context_id']] for i in side]);sidx=np.array(side);valid=known[sidx]&known[bidx]
    du20=d['u20_F64'][sidx]-d['u20_F64'][bidx];joint=np.array([rows[i]['plan']['joint_index'] for i in side]);du=du20[np.arange(1440),joint]
    u32s=d['u20_F64'][sidx].astype(np.float32);u32b=d['u20_F64'][bidx].astype(np.float32)
    du_encoded=u32s.astype(np.float64)[np.arange(1440),joint]-u32b.astype(np.float64)[np.arange(1440),joint]
    du_rounded=(u32s-u32b)[np.arange(1440),joint]
    need(int(valid.sum())==1436 and np.all(abs(du[valid])>=1e-6) and np.all(du_encoded[valid]!=0),'All eligible finite actual steps, unavailable retained')
    need(bits(d['x58_F64'][sidx],d['x58_F64'][bidx]),'Same visible current state58; not fullhidden certificate')
    off=du20.copy();off[np.arange(1440),joint]=0;need(np.all(off==0),'All19 offaxis coordinates unchanged')
    val=npz(a['original_acquisition']['outputs']['validation']);vk=val['training_eligible_bool'];need(int(vk.sum())==114,'Fixed114 VAL monitor')
    return dict(check=d,known=known,rows=rows,J=labels['impulse_source_BODY_moving_COM22x6'][li],Fn=labels['mean_positive_Fn22_N'][li],cov=labels['Fn_gt_0p1_coverage22'][li],
        pair_ids=labels['pair_other_geom_ids'],side=sidx,baseline=bidx,side_known=valid,joint=joint,du=du,du_encoded=du_encoded,du_rounded=du_rounded,
        central=central,val={k:val[k][vk] for k in ['x58_F64','u20_F64','y58_F64','native_dt_s_F64','n_substeps_i64','contact_summary_F64','row_id_i64']},VAL_original_excluded_partial_ids=val['row_id_i64'][~vk].tolist())

def flat_head(h):return np.concatenate([h['joint_delta40'],h['COM_ballistic_residual3_m'],h['world_rotation_delta3_rad'],h['pair_impulse_source_BODY_moving_COM22x6'].reshape(len(h['joint_delta40']),132)],axis=1)
def basic(error):
    e=np.asarray(error,dtype=np.float64)
    if not len(e):return dict(count=0,MAE=None,RMS=None,max_abs=None)
    return dict(count=len(e),MAE=float(np.mean(abs(e))),RMS=float(np.sqrt(np.mean(e*e))),max_abs=float(np.max(abs(e))))
def state_stats(pred,target,scalar):
    e=pred-target;angles=np.array([np.linalg.norm(scalar.log_world(p@t.T)) for p,t in zip(pred[:,43:52].reshape(-1,3,3),target[:,43:52].reshape(-1,3,3))])
    return dict(rows=len(e),q_rad=basic(e[:,:20]),qd_rad_s=basic(e[:,20:40]),COM_vector_RMS_mm=float(np.sqrt(np.mean(np.sum(e[:,40:43]**2,axis=1)))*1000),SO3_geodesic_RMS_deg=float(np.sqrt(np.mean(angles**2))*180/np.pi),
        raw_R9_dimensionless=basic(e[:,43:52]),vCOM_vector_RMS_m_s=float(np.sqrt(np.mean(np.sum(e[:,52:55]**2,axis=1)))),omega_world_vector_RMS_rad_s=float(np.sqrt(np.mean(np.sum(e[:,55:58]**2,axis=1)))))
def pair_stats(predJ,trueJ,predFn,trueFn,predCov,trueCov):
    return [dict(pair_index=j,force_impulse_Ns=basic(predJ[:,j,:3]-trueJ[:,j,:3]),moment_impulse_Nms=basic(predJ[:,j,3:]-trueJ[:,j,3:]),meanFn_N=basic(predFn[:,j]-trueFn[:,j]),
        meanFn_log1p_1N=basic(np.log1p(predFn[:,j])-np.log1p(trueFn[:,j])),coverage_Brier=float(np.mean((predCov[:,j]-trueCov[:,j])**2))) for j in range(22)]
def signal_classes(a,b):
    delta=a-b;exact=np.all(delta==0,axis=-1);collapsed=np.all(a.astype(np.float32)==b.astype(np.float32),axis=-1)
    return np.where(exact,0,np.where(collapsed,1,2)).astype(np.int8)
def ulp_ratio(a,b):
    # Representation magnitude descriptor, not a measured native-noise bound.
    ulp=np.maximum(abs(np.spacing(a.astype(np.float32)).astype(np.float64)),abs(np.spacing(b.astype(np.float32)).astype(np.float64)))
    return abs(a-b)/ulp


def execute(protocol):
    protocol=Path(protocol).resolve(strict=True);p,a,trainp,out=metadata_preflight(protocol);need(p['CPU_run_authorized'] is True,'Root-only actual evaluation')
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch
    torch.set_num_threads(4);need(not torch.cuda.is_initialized(),'CPU-only')
    out.mkdir();write(out/'protocol.json',p);began=time.perf_counter();phase='load_data'
    counts=dict(own_checkpoint_CPU_loads=0,NN_constructors=0,NN_forward_calls=0,NN_rows=0,structured_decode_calls=0,NN_AD_optimizer_training_native_GPU=0,CHECK_data_NPZ_reads=0,CHECK_label_NPZ_reads=0,VAL_NPZ_reads=0,TRAIN_FIT_TEST_NPZ_reads=0)
    try:
        data=load_evaluation_data(a);counts.update(CHECK_data_NPZ_reads=1,CHECK_label_NPZ_reads=1,VAL_NPZ_reads=1)
        trainer=loadmod('frozen_geom_momentum_trainer',checked(p['sources']['trainer']));codec=loadmod('fixed_eval_structured_codec',checked(p['sources']['structured_codec']))
        gap=loadmod('fixed_eval_directional',checked(p['sources']['directional']));geometry=gap.DirectionalGapProvider(checked(p['inputs']['geometry_catalog']))
        phase='load_own5000';ck=torch.load(checked(a['training_result']['checkpoint']),map_location='cpu',weights_only=False);counts['own_checkpoint_CPU_loads']+=1
        need(ck['schema']==trainer.ARTIFACT and ck['iteration']==5000 and ck['recipe']==trainp['recipe'] and ck['sources']==trainp['sources'] and ck['inputs']==trainp['inputs'],'Exact fixed trained component/dataset/recipe')
        norm=ck['normalizers'];physical=ck['physical_context'];rng0=trainer.rng(torch)
        with torch.random.fork_rng(devices=[]):model=trainer.make_model(torch)
        counts['NN_constructors']+=1;model.load_state_dict(ck['model_state'],strict=True);model.eval();model.requires_grad_(False)
        before_weights=trainer.tensor_digest(model.state_dict(),torch);before_norm=trainer.tensor_digest(norm,torch)
        known=data['known'];z=data['check'];idx=np.flatnonzero(known)
        checkinput=dict(x=z['x58_F64'][idx],u=z['u20_F64'][idx],h=z['native_h_s_F64'][idx],n=z['n_substeps_i64'][idx])
        v=data['val'];valinput=dict(x=v['x58_F64'],u=v['u20_F64'],h=v['native_dt_s_F64'],n=v['n_substeps_i64'])
        predictions={};feature_scopes={}
        with torch.inference_mode():
            for name,inputs in [('CHECK',checkinput),('VAL',valinput)]:
                phase='predict_'+name;F,scope=trainer.precompute_features(inputs,geometry,torch);feature_scopes[name]=scope
                x=torch.tensor(inputs['x'],dtype=torch.float64);T=torch.tensor(inputs['h']*inputs['n'],dtype=torch.float64)
                normalized_input=((torch.tensor(F,dtype=torch.float64)-norm['input_mean'])/norm['input_std']).float()
                raw=model(normalized_input);counts['NN_forward_calls']+=1;counts['NN_rows']+=len(x)
                head=raw[:,:178].double()*norm['head_std']+norm['head_mean']
                pred=codec.decode_physical_heads(x,trainer.heads_from_physical(head),T,physical['mass'],physical['I'],physical['g']);counts['structured_decode_calls']+=1
                Fn=torch.nn.functional.softplus(raw[:,178:200]);cov=torch.sigmoid(raw[:,200:222].double())
                need(all(torch.isfinite(y).all().item() for y in [raw,head,pred,Fn,cov]),'All requested predictions finite; no row filtering')
                predictions[name]=dict(raw222_FP32=raw.numpy(),head178_F64=head.numpy(),prediction58_F64=pred.numpy(),meanFn22_N=Fn.double().numpy(),coverage_probability22_F64=cov.numpy(),normalized_input366_FP32=normalized_input.numpy())
        need(counts['NN_forward_calls']==2 and counts['NN_rows']==1586,'Exactly two declared batches')
        unchanged=dict(weights_rawbytes=before_weights==trainer.tensor_digest(model.state_dict(),torch),normalizers_rawbytes=before_norm==trainer.tensor_digest(norm,torch),raw_RNG=trainer.tree_equal(rng0,trainer.rng(torch),torch),parameters_frozen_no_grad=all(not q.requires_grad and q.grad is None for q in model.parameters()),CPU_no_CUDA=not torch.cuda.is_initialized())
        need(all(unchanged.values()),'Inference immutability/RNG check')
        C={key:np.full((1476,)+value.shape[1:],np.nan,dtype=value.dtype) for key,value in predictions['CHECK'].items()}
        for key,value in predictions['CHECK'].items():C[key][idx]=value
        T=z['native_h_s_F64']*z['n_substeps_i64'];g=physical['g'].numpy()
        true_head=np.full((1476,178),np.nan);true_head[idx]=flat_head(codec.encode_observed_numpy(z['x58_F64'][idx],z['y58_at_H_F64'][idx],data['J'][idx],T[idx],g))
        model_delta=np.full((1476,52),np.nan)
        model_delta[idx]=np.array([codec.scalar.delta52(x,y) for x,y in zip(z['x58_F64'][idx],C['prediction58_F64'][idx])])
        native_delta=z['target_delta52_F64']
        Jhat=C['head178_F64'][:,46:].reshape(-1,22,6)
        # Baselines are explicitly kinematic priors, not hidden native solves.
        baseline_predictions={}
        for name,gravity in [('CV',np.zeros(3)),('ballistic',g)]:
            x=z['x58_F64'][idx];tt=T[idx];y=x.copy();y[:,:20]+=x[:,20:40]*tt[:,None];y[:,40:43]+=x[:,52:55]*tt[:,None]+.5*gravity*tt[:,None]**2;y[:,52:55]+=gravity*tt[:,None]
            y[:,43:52]=np.array([codec.scalar.exp_world(w*t)@R for w,t,R in zip(x[:,55:58],tt,x[:,43:52].reshape(-1,3,3))]).reshape(-1,9)
            h=flat_head(codec.encode_observed_numpy(x,y,np.zeros((len(x),22,6)),tt,g))
            baseline_predictions[name]=dict(prediction58=y,head178=h)
        cohort_masks={'ALL_CHECK_ENDPOINTS':known,'NEW_ACTION_SIDES':known&(z['branch_id_i64']>=72),'ORIGINAL_BASELINE_REFERENCES':known&(z['branch_id_i64']<72)}
        for family in sorted({r['family_id'] for r in data['rows']}):cohort_masks[f'family_{family}']=known&np.array([r['family_id']==family for r in data['rows']])
        for h in sorted(set(z['native_h_s_F64'])):cohort_masks[f'dt_{h:g}']=known&(z['native_h_s_F64']==h)
        for branch in sorted({r['source_branch'] for r in data['rows']}):cohort_masks[branch]=known&np.array([r['source_branch']==branch for r in data['rows']])
        endpoint=[]
        for name,mask in cohort_masks.items():
            ii=np.flatnonzero(mask);need(len(ii)>0,'Predeclared nonempty endpoint cohort')
            pred_i=np.searchsorted(idx,ii)
            for role in ['model5000','CV','ballistic']:
                py=C['prediction58_F64'][ii] if role=='model5000' else baseline_predictions[role]['prediction58'][pred_i]
                ph=C['head178_F64'][ii] if role=='model5000' else baseline_predictions[role]['head178'][pred_i]
                pj=Jhat[ii] if role=='model5000' else np.zeros((len(ii),22,6));pf=C['meanFn22_N'][ii] if role=='model5000' else np.zeros((len(ii),22));pc=C['coverage_probability22_F64'][ii] if role=='model5000' else np.zeros((len(ii),22))
                endpoint.append(dict(cohort=name,role=role,state=state_stats(py,z['y58_at_H_F64'][ii],codec.scalar),head178_blocks=[basic((ph-true_head[ii])[:,b]) for b in trainer.physical_blocks()],
                    per_pair22=pair_stats(pj,data['J'][ii],pf,data['Fn'][ii],pc,data['cov'][ii]),baseline_scope=None if role=='model5000' else 'Constantqd and worldomega pose prior, zero-contact J/Fn/coverage assumption; CVstate is not momentum-balanced under gravity, ballistic is not full torque-free anisotropic integration'))
        s,b=data['side'],data['baseline'];valid=data['side_known'];du=data['du'];du32=data['du_encoded']
        jcol=58+data['joint'];ii=np.arange(1440)
        normalized_du=C['normalized_input366_FP32'][s].astype(np.float64)[ii,jcol]-C['normalized_input366_FP32'][b].astype(np.float64)[ii,jcol]
        actual_NN_physical_du=normalized_du*norm['input_std'].numpy()[jcol]
        normalized_offaxis=C['normalized_input366_FP32'][s].copy();normalized_base=C['normalized_input366_FP32'][b].copy()
        normalized_offaxis[ii,jcol]=0;normalized_base[ii,jcol]=0
        need(bits(normalized_offaxis[valid],normalized_base[valid]),'Actual NN endpoints differ only in declared encoded control feature')
        representation_allowance=8*np.finfo(np.float64).eps*(abs(z['u20_F64'][s][ii,data['joint']])+abs(z['u20_F64'][b][ii,data['joint']])+.01)
        full_step=valid&(abs(abs(du)-.01)<=representation_allowance)
        du_class=np.where(~valid,2,np.where(full_step,0,1)).astype(np.int8)
        native_response=true_head[s]-true_head[b];model_response=C['head178_F64'][s]-C['head178_F64'][b]
        native_slope=np.full_like(native_response,np.nan);model_slope=native_slope.copy();model_slope_encoded=native_slope.copy()
        native_slope[valid]=native_response[valid]/du[valid,None];model_slope[valid]=model_response[valid]/du[valid,None]
        encoded_defined=valid&(actual_NN_physical_du!=0);model_slope_encoded[encoded_defined]=model_response[encoded_defined]/actual_NN_physical_du[encoded_defined,None]
        native_J_response=data['J'][s]-data['J'][b];model_J_response=Jhat[s]-Jhat[b]
        native_delta_response=native_delta[s]-native_delta[b];model_delta_response=model_delta[s]-model_delta[b]
        object_class=np.full(1440,-1,dtype=np.int8);object_class[valid]=signal_classes(native_delta[s[valid],40:52],native_delta[b[valid],40:52])
        classes=np.full((1440,22),-1,dtype=np.int8);classes[valid]=signal_classes(data['J'][s[valid]],data['J'][b[valid]])
        ulp=np.full((1440,22,6),np.nan);ulp[valid]=ulp_ratio(data['J'][s[valid]],data['J'][b[valid]])
        secant=[]
        side_masks={'ALL_KNOWN_SIDES':valid}
        for j in range(20):side_masks[f'joint_{j}']=valid&(data['joint']==j)
        for h in sorted(set(z['native_h_s_F64'])):side_masks[f'dt_{h:g}']=valid&(z['native_h_s_F64'][s]==h)
        for cl,name in enumerate(RULES['du_strata'][:2]):side_masks[name]=valid&(du_class==cl)
        for name,mask in side_masks.items():
            secant.append(dict(cohort=name,defined_count=int(mask.sum()),native_response_blocks=[basic(native_response[mask][:,ix]) for ix in trainer.physical_blocks()],model_response_error_blocks=[basic((model_response-native_response)[mask][:,ix]) for ix in trainer.physical_blocks()],actual_du_slope_error_blocks=[basic((model_slope-native_slope)[mask][:,ix]) for ix in trainer.physical_blocks()]))
        signals=[]
        for j in range(20):
            for pair in range(22):
                for cl,label in enumerate(RULES['native_signal_strata']):
                    mask=valid&(data['joint']==j)&(classes[:,pair]==cl)
                    signals.append(dict(joint=j,pair_index=pair,signal_class=label,count=int(mask.sum()),native_delta_force_Ns=basic(native_J_response[mask,pair,:3]),native_delta_moment_Nms=basic(native_J_response[mask,pair,3:]),model_delta_force_Ns=basic(model_J_response[mask,pair,:3]),model_delta_moment_Nms=basic(model_J_response[mask,pair,3:]),response_error_force_Ns=basic((model_J_response-native_J_response)[mask,pair,:3]),response_error_moment_Nms=basic((model_J_response-native_J_response)[mask,pair,3:]),native_delta_endpoint_FP32_ULP=basic(ulp[mask,pair])))
        object_signals=[]
        for j in range(20):
            for cl,label in enumerate(RULES['native_signal_strata']):
                mask=valid&(data['joint']==j)&(object_class==cl)
                object_signals.append(dict(joint=j,signal_class=label,count=int(mask.sum()),
                    native=[basic(native_delta_response[mask,k:k+3]) for k in [40,43,46,49]],model=[basic(model_delta_response[mask,k:k+3]) for k in [40,43,46,49]],
                    response_error=[basic((model_delta_response-native_delta_response)[mask,k:k+3]) for k in [40,43,46,49]],units=['m','rad','m/s','rad/s']))
        central=[]
        for (ctx,j),ends in sorted(data['central'].items()):
            minus,plus=ends[-1],ends[1];defined=bool(known[minus] and known[plus]);den=float(z['u20_F64'][plus,j]-z['u20_F64'][minus,j])
            central.append(dict(context_id=ctx,joint=j,minus_row=minus,plus_row=plus,defined=defined,actual_denominator_rad=den,
                native_slope178=(true_head[plus]-true_head[minus])/den if defined else np.full(178,np.nan),model_slope178=(C['head178_F64'][plus]-C['head178_F64'][minus])/den if defined else np.full(178,np.nan)))
        val=predictions['VAL'];pv=v['contact_summary_F64'][:,:2];padcols=[int(np.flatnonzero(data['pair_ids']==g)[0]) for g in [15,28]]
        valreport=dict(state=state_stats(val['prediction58_F64'],v['y58_F64'],codec.scalar),original_complete_row_ids=v['row_id_i64'],original_excluded_partial_row_ids=data['VAL_original_excluded_partial_ids'],designated_pad_coverage_Brier=float(np.mean((val['coverage_probability22_F64'][:,padcols]-pv)**2)),pair_J22=None,meanFn22=None,missing_label_scope='OriginalVAL has state58 and two padcoverage/min/lastPRE force, not meanFn22/impulse22; no phase-mismatched substitute or label derivation')
        np.savez_compressed(out/'CHECK_all1476_predictions_and_targets.npz',**C,known=known,true_head178_F64=true_head,true_J22x6_F64=data['J'],true_meanFn22_N=data['Fn'],true_coverage22=data['cov'],branch_id=z['branch_id_i64'],current_x58=z['x58_F64'],actual_u20=z['u20_F64'],model_physical_delta52=model_delta,native_physical_delta52=native_delta)
        np.savez_compressed(out/'CHECK_all1440_side_response_arrays.npz',side_row=s,baseline_row=b,defined=valid,joint=data['joint'],actual_du_F64=du,rawphysical_FP32_endpoint_du_F64=du32,rounded_FP32_subtraction=data['du_rounded'],actual_normalized_NN_FP32_du_physical_F64=actual_NN_physical_du,encoded_denominator_defined=encoded_defined,du_stratum=du_class,native_response178=native_response,model_response178=model_response,native_slope178_actualdu=native_slope,model_slope178_actualdu=model_slope,model_slope178_actual_NN_encodeddu=model_slope_encoded,native_J_response=native_J_response,model_J_response=model_J_response,perpair_native_signal_class=classes,native_delta_endpoint_FP32_ULP=ulp,native_delta52_response=native_delta_response,model_delta52_response=model_delta_response,native_object_signal_class=object_class,native_meanFn_response22=data['Fn'][s]-data['Fn'][b],model_meanFn_response22=C['meanFn22_N'][s]-C['meanFn22_N'][b],native_coverage_response22=data['cov'][s]-data['cov'][b],model_coverage_response22=C['coverage_probability22_F64'][s]-C['coverage_probability22_F64'][b])
        np.savez_compressed(out/'VAL114_predictions.npz',**val,row_id=v['row_id_i64'],source_x58=v['x58_F64'],actual_u20=v['u20_F64'],target_y58=v['y58_F64'],native_h_s=v['native_dt_s_F64'],n_substeps=v['n_substeps_i64'])
        write(out/'endpoint_statistics.json',endpoint);write(out/'side_response_statistics.json',secant);write(out/'all20x22_native_signal_strata.json',signals);write(out/'all20_object_native_signal_strata.json',object_signals);write(out/'all720_central_pair_records.json',central);write(out/'all1476_identities.json',data['rows']);write(out/'VAL_state_monitor.json',valreport)
        result=dict(schema='phone_geometry_momentum_fixed5000_evaluation_result:1',status='COMPLETE_FIXED5000_KNOWN_STATE_ACTION_CHECK_NOT_POLICY_QUALIFICATION',protocol=pin(protocol),checkpoint=a['training_result']['checkpoint'],counts=counts,immutability=unchanged,feature_cache_scopes=feature_scopes,
            endpoint_rows=1476,known_endpoints=1472,unknown_unavailable=4,side_records=1440,known_sides=1436,central_pair_records=720,defined_central_pairs=sum(v['defined'] for v in central),
            model_role='fixed5000_geometry_momentum_counterfactual_baseline',normalizers_refit=False,checkpoint_or_weight_selected_using_CHECK=False,TRUE_Jacobian_or_ego_CHORD_qualification=False,
            native_signal_rule=RULES['native_signal_strata'],nearzero_scope='Raw magnitudes and FP32 endpoint-ULP ratios retained. No invented dynamics noise floor; exact observed zero/encoded collapse are descriptive, not wholeH path certificates.',
            elapsed_s=time.perf_counter()-began,outputs={q.name:pin(q) for q in out.iterdir() if q.is_file()},
            scope='Same-lineage originalTRAIN visible states with held-out new action responses. All20 axes/pairs/zero/weak and unavailable records retained. No fit/native/AD/TEST or freshfamily evaluation. CV/ballistic are explicit priors; final checkpoint was fixed before model CHECK evaluation. No Actor/task gain claim.')
        write(out/'report.json',result);print(json.dumps(dict(status=result['status'],counts=counts,report=str(out/'report.json'))));return 0
    except BaseException as e:
        write(out/'failure.json',dict(status='FAILED_PRESERVED_NO_RETRY',phase=phase,error=repr(e),traceback=traceback.format_exc(),counts=counts));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run',action='store_true');args=ap.parse_args()
    if args.run:raise SystemExit(execute(args.protocol))
    p,a,tp,out=metadata_preflight(args.protocol);print(json.dumps(dict(status='CPUFALSE_METADATA_NO_TORCH_NPZ_MODEL',CPU_run_authorized=p['CPU_run_authorized'],primary_checkpoint=5000,planned_NN_calls=2,planned_NN_rows=1586,TRAIN_TEST_FIT_np_load=0,output=str(out))))
