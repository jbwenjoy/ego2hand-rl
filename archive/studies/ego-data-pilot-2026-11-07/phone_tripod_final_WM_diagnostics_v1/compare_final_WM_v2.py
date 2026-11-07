"""Fixed offline-final WM diagnostics. Default: JSON/hash only; never fit.

Actual execution is separately authorized by Root after paired1000/WM1100.
Old CHECK/fresh warm predictions are saved references, not extra forwards.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, sys, time, traceback
import numpy as np

HERE=Path(__file__).resolve().parent
SCHEMA='phone_tripod_fixed_final_WM_diagnostics_CPU_protocol:2'
RESULT='phone_tripod_fixed_final_WM_diagnostics_result:2'
CHECK='HELDOUT_ACTION_RESPONSE_CHECK_KNOWN_TRAIN_STATE'
INPUTS={'final_training_protocol','final_result','final_health','final_terminal',
        'warm_result','warm_health','tripod_dataset','CHECK_report','fresh_report'}
PENDING={'final_training_protocol','final_result','final_health','final_terminal'}
RULES=dict(final_paired_round=1000,final_shared_WM_updates=1100,warm_iteration=5000,
    checkpoint_selection=False,normalizer_or_scale_fitting=False,cpu_threads=4,
    MON_rows=52,CHECK_rows=1476,CHECK_known=1472,CHECK_sides=1440,CHECK_defined_sides=1436,
    fresh_rows=240,fresh_known=228,data_NPZ_reads=5,frozen_geometry_asset_NPZ_reads=2,checkpoint_CPU_loads=2,WM_constructors=2,
    NN_forward_calls=4,NN_rows=1804,structured_decode_calls=4,
    Actor_Critic_constructors=0,AD_optimizer_native_GPU_calls=0,
    old_input='reuse_saved_normalized_input366_FP32_after_raw_normalizer_and_context_equality',
    MON_input='precompute_current_geometry_once_then_same_input_for_both_models',
    roles=['offline_final1000','warm5000'])
FROZEN={'offline_trainer':'a1bcf9f820e00d7da4ac515dd81d45d39a2c59abd07c85319d1346db384b2439',
 'WM_trainer':'224a0c9f47e196c626e862aa52d46402a454a6d16f0b497d3720ae74e25ca539',
 'structured_codec':'057abf6e653f2dbd25b3812dc8de007539d17968389c57c2633dbf0ed6811a6a',
 'frozen_metrics':'32f565f0d120272114692505d5ada9455fc2f857b9225ee6b1a3e2b82ca8a302'}

def need(v,msg):
    if not v: raise ValueError(msg)
def pin(path):
    path=Path(path).resolve(strict=True);b=path.read_bytes()
    return dict(path=str(path),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(value):
    need(Path(value['path']).is_absolute() and pin(value['path'])==value,'Current absolute pin')
    return Path(value['path'])
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def loadmod(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(clean(value),f,indent=2,allow_nan=False)
def bits(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def positive_health(h):
    c=h.get('checks');return isinstance(c,dict) and bool(c) and all(v is True for v in c.values()) and h['passed']==h['total']==len(c)

def metadata_preflight(protocol_path):
    p=read(protocol_path)
    need(p['schema']==SCHEMA and p['rules']==RULES and set(p['inputs'])==INPUTS,'Exact diagnostic namespace/rules/inputs')
    need(type(p['CPU_run_authorized']) is bool and p['GPU_run_authorized'] is False,'Explicit CPU-only authority')
    for v in p['sources'].values():checked(v)
    need(p['sources']['evaluator']==pin(__file__),'Own source bound')
    for k,sha in FROZEN.items():need(p['sources'][k]['sha256']==sha,'Frozen source '+k)
    missing=sorted(k for k in PENDING if p['inputs'][k] is None)
    need(not any(p['inputs'][k] is None for k in INPUTS-PENDING),'Non-final inputs required')
    if missing:need(p['CPU_run_authorized'] is False,'No execution while final inputs pending')
    a={k:read(checked(v)) for k,v in p['inputs'].items() if v is not None}
    wr,wh=a['warm_result'],a['warm_health']
    need(wr['schema']=='phone_geometry_momentum_training_result:1' and wr['iteration']==5000,'Fixed warm5000')
    need(wr['health']==p['inputs']['warm_health'] and wh['checkpoint']==wr['checkpoint'] and wh['iteration']==5000 and positive_health(wh),'Warm health')
    checked(wr['checkpoint'])
    cr,fr,ds=a['CHECK_report'],a['fresh_report'],a['tripod_dataset']
    need(cr['schema']=='phone_geometry_momentum_fixed5000_evaluation_result:1' and cr['known_endpoints']==1472 and cr['known_sides']==1436,'Frozen CHECK report')
    need(fr['schema']=='phone_fresh12_fixed5000_state_evaluation_result:1' and fr['predicted_complete_rows']==228 and fr['partial_unknown_rows']==12,'Frozen fresh report')
    need(cr['checkpoint']==fr['checkpoint']==wr['checkpoint'],'Saved warm predictions same checkpoint')
    need(ds['schema']=='phone_tripod_replay_training_dataset:1' and ds['engineering_qualified'] is True and all(v is True for v in ds['checks'].values()),'Qualified new dataset')
    cp=read(checked(cr['protocol']));freshp=read(checked(fr['protocol']));ca=read(checked(cp['inputs']['causal_acquisition']))
    need(cp['inputs']['geometry_catalog']==freshp['inputs']['geometry_catalog'],'Saved CHECK/fresh same geometry catalog')
    data_pins=dict(MON=ds['partition_arrays']['monitor'],CHECK_saved=cr['outputs']['CHECK_all1476_predictions_and_targets.npz'],
        CHECK_raw=ca['arrays'][CHECK],CHECK_sides=cr['outputs']['CHECK_all1440_side_response_arrays.npz'],
        fresh_saved=fr['outputs']['all240_predictions_and_targets.npz'])
    for v in data_pins.values():checked(v)
    if not missing:
        fp,r,h,t=[a[k] for k in ['final_training_protocol','final_result','final_health','final_terminal']]
        need(fp['schema']=='phone_tripod_sharedWM_matched_supermimic_protocol:1' and fp['sources']['trainer']==p['sources']['offline_trainer'],'Final offline trainer')
        need(r['schema']=='phone_tripod_matched_supermimic_training_result:1' and r['status']=='COMPLETED_FIXED_OFFLINE_MATCHED1000_NOT_NATIVE_POLICY_QUALIFICATION' and r['paired_round']==1000 and r['shared_WM_updates']==1100,'Fixed final1000 only')
        need(h['schema']=='phone_tripod_matched_learning_checkpoint_health:1' and h['paired_round']==1000 and h['shared_WM_updates']==1100 and h['checkpoint']==r['checkpoint'] and positive_health(h),'Final health same checkpoint')
        need(t['returncode']==0 and t['protocol']==p['inputs']['final_training_protocol'] and t['source']==p['sources']['offline_trainer'],'Final terminal/protocol')
        need(fp['inputs']['dataset']==p['inputs']['tripod_dataset'] and fp['inputs']['WM_result']==p['inputs']['warm_result'],'Same new data/warm parent')
        for k in ['WM_trainer','structured_codec','directional']:need(fp['sources'][k]==p['sources'][k],'Final source join '+k)
        need(fp['inputs']['geometry_catalog']==cp['inputs']['geometry_catalog']==freshp['inputs']['geometry_catalog'],'Final and both saved normalized-feature catalogs exactly equal')
        checked(r['checkpoint']);checked(fp['inputs']['geometry_catalog'])
    out=Path(p['output']);need(out.is_absolute() and out.parent.resolve()==HERE and not out.exists(),'Fresh local output')
    return p,a,data_pins,out,missing

def npz(value):
    with np.load(checked(value),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def cohorts(known,h,family=None):
    out={'ALL':known}
    for value in sorted(set(h)):out['dt_'+format(float(value),'.8g')]=known&(h==value)
    if family is not None:
        for value in sorted(set(family)):out['family_'+str(int(value))]=known&(family==value)
    return out
def expand(values,known):
    out={k:np.full((len(known),)+v.shape[1:],np.nan,dtype=v.dtype) for k,v in values.items()}
    for k,v in values.items():out[k][known]=v
    return out
def prediction(model,normalized,x,T,norm,physical,trainer,codec,torch,counts):
    raw=model(torch.tensor(normalized,dtype=torch.float32));counts['NN_forward_calls']+=1;counts['NN_rows']+=len(x)
    head=raw[:,:178].double()*norm['head_std']+norm['head_mean']
    y=codec.decode_physical_heads(torch.tensor(x,dtype=torch.float64),trainer.heads_from_physical(head),torch.tensor(T,dtype=torch.float64),physical['mass'],physical['I'],physical['g']);counts['structured_decode_calls']+=1
    fn=torch.nn.functional.softplus(raw[:,178:200]);cov=torch.sigmoid(raw[:,200:222].double())
    need(all(torch.isfinite(v).all().item() for v in [raw,head,y,fn,cov]),'Every requested output finite')
    return dict(raw222_FP32=raw.numpy(),head178_F64=head.numpy(),prediction58_F64=y.numpy(),meanFn22_N=fn.double().numpy(),coverage_probability22_F64=cov.numpy(),normalized_input366_FP32=np.asarray(normalized).copy())

def response_statistics(C,old,side,metrics,trainer,codec,native_h):
    s,b=side['side_row'],side['baseline_row'];valid=side['defined'];du=side['actual_du_F64']
    need(len(s)==1440 and int(valid.sum())==1436 and np.all(C['known'][s[valid]]) and np.all(C['known'][b[valid]]),'All original side masks')
    need(np.isfinite(du[valid]).all() and np.all(du[valid]!=0),'Actual finite nonzero action denominators')
    native=side['native_response178'];need(bits(native[valid],(old['true_head178_F64'][s]-old['true_head178_F64'][b])[valid]),'Native response receipt')
    delta=np.full((1476,52),np.nan)
    idx=np.flatnonzero(C['known']);delta[idx]=np.array([codec.scalar.delta52(x,y) for x,y in zip(C['current_x58'][idx],C['prediction58_F64'][idx])])
    head_response=C['head178_F64'][s]-C['head178_F64'][b];delta_response=delta[s]-delta[b]
    need(bits(side['native_delta52_response'][valid],(old['native_physical_delta52'][s]-old['native_physical_delta52'][b])[valid]),'Native delta52 response receipt')
    arrays=dict(side_row=s,baseline_row=b,joint=side['joint'],defined=valid,actual_du_F64=du,
        head_response178=head_response,native_response178=native,physical_delta52_response=delta_response,
        native_delta52_response=side['native_delta52_response'],native_object_signal_class=side['native_object_signal_class'])
    masks={'ALL_KNOWN_SIDES':valid}
    for j in range(20):masks['joint_'+str(j)]=valid&(side['joint']==j)
    for h in sorted(set(native_h)):masks['dt_'+format(float(h),'.8g')]=valid&(native_h[s]==h)
    if 'du_stratum' in side:
        for v in [0,1]:masks['du_stratum_'+str(v)]=valid&(side['du_stratum']==v)
    rows=[]
    zero=valid&np.all(side['native_delta52_response'][:,40:43]==0,axis=1)
    for name,m in masks.items():
        r=dict(cohort=name,rows=int(m.sum()),head_response_error_blocks=[metrics.basic((head_response-native)[m][:,ix]) for ix in trainer.physical_blocks()],
            actual_du_secant_error_blocks=[metrics.basic(((head_response[m]-native[m])/du[m,None])[:,ix]) for ix in trainer.physical_blocks()],
            COM_native_exactzero_sides=int((m&zero).sum()),COM_native_nonzero_sides=int((m&~zero).sum()))
        z=m&zero
        r['COM_phantom_response_m']=metrics.basic(delta_response[z,40:43])
        r['COM_phantom_response_vector_norm_m']=metrics.basic(np.linalg.norm(delta_response[z,40:43],axis=1))
        r['COM_phantom_actualdu_secant_m_per_rad']=metrics.basic(delta_response[z,40:43]/du[z,None])
        rows.append(r)
    return arrays,rows

def execute(protocol_path):
    protocol_path=Path(protocol_path).resolve(strict=True);p,a,data_pins,out,missing=metadata_preflight(protocol_path)
    need(not missing and p['CPU_run_authorized'] is True,'Completed final inputs and Root CPU authority')
    need('torch' not in sys.modules,'Fresh CPU process before Torch import');os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch
    torch.set_num_threads(4);torch.set_default_dtype(torch.float32);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out.mkdir();write(out/'protocol.json',p);start=time.perf_counter();phase='load_saved_arrays'
    counts=dict(data_NPZ_reads=0,frozen_geometry_asset_NPZ_reads=0,checkpoint_CPU_loads=0,WM_constructors=0,NN_forward_calls=0,NN_rows=0,structured_decode_calls=0,Actor_Critic_constructors=0,AD_optimizer_native_GPU_calls=0)
    try:
        z={}
        for key,value in data_pins.items():z[key]=npz(value);counts['data_NPZ_reads']+=1
        trainer=loadmod('diagnostic_WM_trainer',checked(p['sources']['WM_trainer']));codec=loadmod('diagnostic_codec',checked(p['sources']['structured_codec']));metrics=loadmod('diagnostic_frozen_metrics',checked(p['sources']['frozen_metrics']))
        rng0=trainer.rng(torch)
        final=torch.load(checked(a['final_result']['checkpoint']),map_location='cpu',weights_only=False);counts['checkpoint_CPU_loads']+=1
        warm=torch.load(checked(a['warm_result']['checkpoint']),map_location='cpu',weights_only=False);counts['checkpoint_CPU_loads']+=1
        fp=a['final_training_protocol'];wp=read(checked(a['warm_result']['protocol']))
        need(final['schema']=='phone_tripod_sharedWM_A1A2_training_checkpoint:1' and final['paired_round']==1000 and final['shared_WM_updates']==1100,'Final own namespace/counters')
        need(final['protocol']==p['inputs']['final_training_protocol'] and final['inputs']==fp['inputs'] and final['sources']==fp['sources'] and final['recipe']==fp['recipe'],'Final payload authority')
        need(warm['schema']==trainer.ARTIFACT and warm['iteration']==5000 and warm['inputs']==wp['inputs'] and warm['sources']==wp['sources'] and warm['recipe']==wp['recipe'],'Warm payload authority')
        norm,physical=warm['normalizers'],warm['physical_context']
        need(trainer.tree_equal(final['WM_normalizers'],norm,torch) and trainer.tree_equal(final['physical_context'],physical,torch),'Final/warm context and normalizers raw equal before old normalized input reuse')
        models={}
        for role,state in [('offline_final1000',final['shared_WM']),('warm5000',warm['model_state'])]:
            with torch.random.fork_rng(devices=[]):m=trainer.make_model(torch)
            counts['WM_constructors']+=1;m.load_state_dict(state,strict=True);m.eval();m.requires_grad_(False);models[role]=m
        weights={k:trainer.tensor_digest(v.state_dict(),torch) for k,v in models.items()};norm0=trainer.tensor_digest(norm,torch);ctx0=trainer.tensor_digest(physical,torch)
        mon={k:z['MON'][v] for k,v in a['tripod_dataset']['field_map'].items()}
        need(len(mon['x58'])==52 and mon['WM_eligible'].dtype==np.bool_ and mon['WM_eligible'].all() and mon['complete_H'].all(),'All MON52 complete, no selection')
        gap=loadmod('diagnostic_directional',checked(p['sources']['directional']));geometry=gap.DirectionalGapProvider(checked(fp['inputs']['geometry_catalog']));counts['frozen_geometry_asset_NPZ_reads']+=2
        pair_ids=list(geometry.provider.pair_geom_ids)
        need(pair_ids==sorted(set(pair_ids)) and pair_ids==a['tripod_dataset']['canonical_pair_geom_ids'],'Canonical provider pair ordering')
        padcols=[pair_ids.index(g) for g in [15,28]]
        need(padcols[0]<padcols[1] and [pair_ids[j] for j in padcols]==[15,28],'Original two-pad coverage order')
        F,feature_scope=trainer.precompute_features(dict(x=mon['x58'],u=mon['u20'],h=mon['h'],n=mon['n']),geometry,torch)
        moninput=((torch.tensor(F,dtype=torch.float64)-norm['input_mean'])/norm['input_std']).float().numpy()
        C,W,S=z['CHECK_saved'],z['fresh_saved'],z['CHECK_sides'];raw=z['CHECK_raw']
        need(len(C['known'])==1476 and C['known'].dtype==np.bool_ and C['known'].sum()==1472 and bits(C['known'],raw['training_eligible_bool']),'CHECK1476/1472 original masks')
        need(bits(C['branch_id'],raw['branch_id_i64']) and bits(C['current_x58'],raw['x58_F64']) and bits(C['actual_u20'],raw['u20_F64']),'CHECK original row joins')
        need(len(W['known'])==240 and W['known'].dtype==np.bool_ and W['known'].sum()==228 and np.isnan(W['target_y58'][~W['known']]).all(),'Fresh240/228/12 partial preserved')
        for d in [C,W]:
            need(d['normalized_input366_FP32'].dtype==np.float32 and d['normalized_input366_FP32'].shape==(len(d['known']),366),'Saved actual normalized input')
            need(np.isnan(d['normalized_input366_FP32'][~d['known']]).all(),'Unknown normalized input remains NaN')
        predictions={}
        phase='four_fixed_forward_calls'
        with torch.inference_mode():
            for role in RULES['roles']:
                predictions[('MON',role)]=prediction(models[role],moninput,mon['x58'],mon['h']*mon['n'],norm,physical,trainer,codec,torch,counts)
            for name,d,h,n in [('CHECK',C,raw['native_h_s_F64'],raw['n_substeps_i64']),('fresh',W,W['native_h_s'],W['n_substeps'])]:
                k=d['known'];values=prediction(models['offline_final1000'],d['normalized_input366_FP32'][k],d['current_x58'][k],(h*n)[k],norm,physical,trainer,codec,torch,counts)
                predictions[(name,'offline_final1000')]=expand(values,k)
                keys=['raw222_FP32','head178_F64','prediction58_F64','coverage_probability22_F64','normalized_input366_FP32']
                saved={v:d[v] for v in keys};saved['meanFn22_N']=d['meanFn22_N'] if name=='CHECK' else d['meanFn22_prediction_N']
                predictions[(name,'warm5000')]=saved
        expected={k:RULES[k] for k in counts};need(counts==expected,'Exact 5dataNPZ+2frozenGeometryNPZ/2load/2ctor/4FW/1804row/4decode budget')
        imm=dict(weights_raw=all(weights[k]==trainer.tensor_digest(v.state_dict(),torch) for k,v in models.items()),normalizers_raw=norm0==trainer.tensor_digest(norm,torch),physical_context_raw=ctx0==trainer.tensor_digest(physical,torch),raw_RNG=trainer.tree_equal(rng0,trainer.rng(torch),torch),parameters_frozen_no_grad=all(not q.requires_grad and q.grad is None for m in models.values() for q in m.parameters()),no_CUDA=not torch.cuda.is_initialized())
        need(all(imm.values()),'Inference immutable/no AD/GPU')
        phase='descriptive_saved_statistics';stats=[];responses=[]
        for name in ['MON','CHECK','fresh']:
            if name=='MON':known=np.ones(52,dtype=bool);h=mon['h'];family=mon['family_id'];x=mon['x58'];target=mon['y58']
            elif name=='CHECK':known=C['known'];h=raw['native_h_s_F64'];family=None;x=C['current_x58'];target=raw['y58_at_H_F64']
            else:known=W['known'];h=W['native_h_s'];family=W['family_id'];x=W['current_x58'];target=W['target_y58']
            masks=cohorts(known,h,family)
            if name=='CHECK':masks.update(NEW_ACTION_SIDES=known&(C['branch_id']>=72),ORIGINAL_BASELINES=known&(C['branch_id']<72))
            for role in RULES['roles']:
                pred=predictions[(name,role)]
                np.savez_compressed(out/(name+'__'+role+'.npz'),**pred,known=known,current_x58=x,target_y58=target,native_h_s=h)
                for label,mask in masks.items():
                    ii=np.flatnonzero(mask);r=dict(dataset=name,cohort=label,role=role,rows=len(ii),state=metrics.state_stats(pred['prediction58_F64'][ii],target[ii],codec.scalar),label_scope='state and canonical22' if name!='fresh' else 'state and original two-pad coverage only')
                    if name!='fresh':
                        J=pred['head178_F64'][:,46:].reshape(-1,22,6);trueJ=mon['impulse22x6'] if name=='MON' else C['true_J22x6_F64'];fn=mon['meanFn22'] if name=='MON' else C['true_meanFn22_N'];cov=mon['coverage22'] if name=='MON' else C['true_coverage22']
                        r['per_pair22']=metrics.pair_stats(J[ii],trueJ[ii],pred['meanFn22_N'][ii],fn[ii],pred['coverage_probability22_F64'][ii],cov[ii])
                    else:
                        r.update(two_pad_coverage_Brier=float(np.mean((pred['coverage_probability22_F64'][ii][:,padcols]-W['true_two_pad_coverage'][ii])**2)),pair_J22_truth=None,meanFn22_truth=None)
                    stats.append(r)
                if name=='CHECK':
                    joined=dict(pred,known=known,current_x58=x);arr,r=response_statistics(joined,C,S,metrics,trainer,codec,h)
                    np.savez_compressed(out/('CHECK_sides__'+role+'.npz'),**arr)
                    responses.extend(dict(role=role,**v) for v in r)
        np.savez_compressed(out/'MON52_truth_and_input.npz',normalized_input366_FP32=moninput,**{k:np.asarray(v) for k,v in mon.items()})
        write(out/'endpoint_statistics.json',stats);write(out/'response_statistics.json',responses)
        write(out/'normalizer_context_equality.json',dict(norm_digest=norm0,physical_context_digest=ctx0,final_and_warm_raw_equal=True,normalizer_refit=False,MON_feature_cache=feature_scope))
        outputs={f.name:pin(f) for f in out.iterdir() if f.is_file()}
        result=dict(schema=RESULT,status='COMPLETE_FIXED_OFFLINE_FINAL_WM_DIAGNOSTICS_NOT_POLICY_OR_GENERALIZATION_QUALIFICATION',protocol=pin(protocol_path),final_checkpoint=a['final_result']['checkpoint'],warm_checkpoint=a['warm_result']['checkpoint'],counts=counts,checks=imm,passed=sum(imm.values()),total=len(imm),outputs=outputs,data_pins=data_pins,known=dict(MON=52,CHECK=1472,fresh=228),unknown=dict(CHECK=4,fresh=12),elapsed_s=time.perf_counter()-start,scope='Fixed training progression comparison, not best-checkpoint selection. CHECK uses known TRAIN states/new actions; fresh shares known source poses. Side responses are finite-action endpoint responses, not infinitesimal Jacobians/contact-force derivatives. No aggregate cross-unit winner; fresh J22/meanFn truth absent. Warm CHECK/fresh predictions are saved references, not repeated inference.')
        write(out/'report.json',result);return result
    except Exception as e:
        write(out/'failure.json',dict(phase=phase,error=repr(e),traceback=traceback.format_exc(),counts=counts,automatic_retry=False));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run',action='store_true');args=ap.parse_args()
    if args.run:print(json.dumps(execute(args.protocol)['counts']))
    else:
        p,a,d,out,missing=metadata_preflight(args.protocol)
        print(json.dumps(dict(schema=SCHEMA,status='CPUFALSE_METADATA_ONLY_PENDING_FINAL' if missing else 'CPUFALSE_METADATA_ONLY_READY',missing=missing,data_pins=d,output=str(out),NN_native_calls=0)))
