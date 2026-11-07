"""Fresh12 state-only fixed5000 inference. Default JSON/hash only, no fitting."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys,time,traceback
import numpy as np
H=Path(__file__).resolve().parent
SCHEMA='phone_fresh12_fixed5000_state_evaluation_CPU_protocol:1'
INPUTS={'training_result','training_health','training_terminal','fresh_acquisition','fresh_saved_review','geometry_catalog'}
RULES=dict(checkpoint_iteration=5000,checkpoint_selection='fixed_before_fresh_predictions_no_selection',all_rows=240,known_rows=228,partial_rows=12,CPU_checkpoint_loads=1,NN_constructors=1,NN_forward_calls=1,NN_rows=228,AD_optimizer_native_calls=0,normalizer_refit=False,TRAIN_CHECK_VAL_TEST_target_array_opens=0)
TRAINER_SHA='224a0c9f47e196c626e862aa52d46402a454a6d16f0b497d3720ae74e25ca539'
CHECK_EVAL_SHA='32f565f0d120272114692505d5ada9455fc2f857b9225ee6b1a3e2b82ca8a302'
def need(v,msg):
    if not v:raise ValueError(msg)
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(v):need(Path(v['path']).is_absolute() and pin(v['path'])==v,'Current absolute pin');return Path(v['path'])
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def loadmod(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def metadata_preflight(path):
    p=read(path);need(p['schema']==SCHEMA and p['rules']==RULES and set(p['inputs'])==INPUTS,'Fresh state-only namespace/rules')
    need(type(p['CPU_run_authorized']) is bool and p['GPU_run_authorized'] is False,'CPU only')
    for v in [*p['inputs'].values(),*p['sources'].values()]:checked(v)
    need(p['sources']['evaluator']==pin(__file__) and p['sources']['trainer']['sha256']==TRAINER_SHA and p['sources']['frozen_CHECK_evaluator']['sha256']==CHECK_EVAL_SHA,'Frozen model and metric interfaces')
    a={k:read(checked(v)) for k,v in p['inputs'].items()}
    r,h,t=a['training_result'],a['training_health'],a['training_terminal']
    need(r['schema']=='phone_geometry_momentum_training_result:1' and r['iteration']==5000 and r['status']=='COMPLETED_FIXED5000_UPDATES_NOT_POLICY_QUALIFICATION','Fixed final5000 only')
    need(r['health']==p['inputs']['training_health'] and h['iteration']==5000 and h['checkpoint']==r['checkpoint'] and all(v is True for v in h['checks'].values()),'Final health lineage')
    need(t['returncode']==0 and t['protocol']==r['protocol'] and t['source']==p['sources']['trainer'],'Closed final process')
    checked(r['checkpoint']);trainp=read(checked(r['protocol']))
    need(trainp['sources']['trainer']==p['sources']['trainer'] and trainp['inputs']['geometry_catalog']==p['inputs']['geometry_catalog'],'Same frozen geometry/model context')
    fresh=a['fresh_acquisition'];review=a['fresh_saved_review']
    need(fresh['schema']=='phone_new12_controlled_final_acquisition_result:1' and fresh['rows']==240 and fresh['families']==12 and fresh['episode_count']==24,'All preregistered fresh controlled families')
    need(review['schema']=='phone_fresh12_acquisition_saved_review:1' and review['producer_report']==p['inputs']['fresh_acquisition'] and review['engineering_qualified'] is True and all(v is True for v in review['checks'].values()),'Author saved-integrity scope retained')
    need(fresh['raw_identity_aliases_absent'] is True and fresh['cross_original_input_action_alias_groups']==0 and fresh['no_deduplication_or_reassignment'] is True,'No raw receipt/action alias or selection')
    for v in [*fresh['outputs'].values(),fresh['sample_identities'],fresh['family_plan']]:checked(v)
    need(fresh['family_plan']['sha256']=='bb547651f8b6137d00d15ab63d16ec29c313bc3365a0c9b05f9a8d9ea38e7600','Plan fixed before acquisition')
    out=Path(p['output']);need(out.is_absolute() and out.parent.resolve()==H and not out.exists(),'Fresh local output')
    return p,a,trainp,out

def execute(protocol_path):
    protocol_path=Path(protocol_path).resolve(strict=True);p,a,trainp,out=metadata_preflight(protocol_path)
    need(p['CPU_run_authorized'] is True,'Root-only actual inference')
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch
    torch.set_num_threads(4);need(not torch.cuda.is_initialized(),'CPU only')
    out.mkdir();write(out/'protocol.json',p);started=time.perf_counter()
    counts=dict(fresh_NPZ_reads=0,checkpoint_CPU_loads=0,NN_constructors=0,NN_forward_calls=0,NN_rows=0,structured_decode_calls=0,AD_optimizer_training_native_GPU=0,TRAIN_CHECK_VAL_TEST_target_NPZ_reads=0)
    try:
        helper=loadmod('fresh_fixed_CHECK_metrics',checked(p['sources']['frozen_CHECK_evaluator']))
        trainer=loadmod('fresh_fixed_geometry_trainer',checked(p['sources']['trainer']))
        codec=loadmod('fresh_fixed_structured_codec',checked(p['sources']['structured_codec']))
        gap=loadmod('fresh_fixed_directional',checked(p['sources']['directional']));geometry=gap.DirectionalGapProvider(checked(p['inputs']['geometry_catalog']))
        fresh=a['fresh_acquisition'];z=helper.npz(fresh['outputs']['fresh_final']);counts['fresh_NPZ_reads']+=1
        ids=read(checked(fresh['sample_identities']));known=z['training_eligible_bool'];idx=np.flatnonzero(known)
        need(len(known)==len(ids)==240 and known.dtype==np.bool_ and len(idx)==228 and int((~z['complete_H_bool']).sum())==12,'All240 retained,228 known12 partial')
        need(np.isnan(z['y58_F64'][~known]).all() and np.isfinite(z['y58_F64'][known]).all(),'Unknown H targets remain NaN')
        need(np.array_equal(z['row_id_i64'],np.arange(240)) and set(z['family_id_i64'])==set(range(1000,1012)),'Exact row/family identities')
        for i,d in enumerate(ids):
            need(d['sample_id']==i and d['family_id']==int(z['family_id_i64'][i]) and d['training_eligible']==bool(known[i]),'All raw row joins')
        ck=torch.load(checked(a['training_result']['checkpoint']),map_location='cpu',weights_only=False);counts['checkpoint_CPU_loads']+=1
        need(ck['schema']==trainer.ARTIFACT and ck['iteration']==5000 and ck['sources']==trainp['sources'] and ck['inputs']==trainp['inputs'] and ck['recipe']==trainp['recipe'],'Exact trained own component')
        norm,physical=ck['normalizers'],ck['physical_context'];rng0=trainer.rng(torch)
        with torch.random.fork_rng(devices=[]):model=trainer.make_model(torch)
        counts['NN_constructors']+=1;model.load_state_dict(ck['model_state'],strict=True);model.eval();model.requires_grad_(False)
        w0=trainer.tensor_digest(model.state_dict(),torch);n0=trainer.tensor_digest(norm,torch);c0=trainer.tensor_digest(physical,torch)
        inputs=dict(x=z['x58_F64'][idx],u=z['u20_F64'][idx],h=z['native_dt_s_F64'][idx],n=z['n_substeps_i64'][idx])
        with torch.inference_mode():
            F,scope=trainer.precompute_features(inputs,geometry,torch)
            normalized=((torch.tensor(F,dtype=torch.float64)-norm['input_mean'])/norm['input_std']).float()
            raw=model(normalized);counts['NN_forward_calls']+=1;counts['NN_rows']+=len(idx)
            head=raw[:,:178].double()*norm['head_std']+norm['head_mean']
            x=torch.tensor(inputs['x'],dtype=torch.float64);T=torch.tensor(inputs['h']*inputs['n'],dtype=torch.float64)
            pred=codec.decode_physical_heads(x,trainer.heads_from_physical(head),T,physical['mass'],physical['I'],physical['g']);counts['structured_decode_calls']+=1
            fn=torch.nn.functional.softplus(raw[:,178:200]);cov=torch.sigmoid(raw[:,200:222].double())
            need(all(torch.isfinite(v).all().item() for v in [raw,head,pred,fn,cov]),'No row filtering for nonfinite model output')
            values=dict(raw222_FP32=raw.numpy(),head178_F64=head.numpy(),prediction58_F64=pred.numpy(),meanFn22_prediction_N=fn.double().numpy(),coverage_probability22_F64=cov.numpy(),normalized_input366_FP32=normalized.numpy())
        unchanged=dict(weights_rawbytes=w0==trainer.tensor_digest(model.state_dict(),torch),normalizers_rawbytes=n0==trainer.tensor_digest(norm,torch),physical_context_rawbytes=c0==trainer.tensor_digest(physical,torch),RNG_values=trainer.tree_equal(rng0,trainer.rng(torch),torch),parameters_frozen_gradNone=all(not q.requires_grad and q.grad is None for q in model.parameters()),CPU_no_CUDA=not torch.cuda.is_initialized())
        need(all(unchanged.values()),'Inference immutable')
        full={k:np.full((240,)+v.shape[1:],np.nan,dtype=v.dtype) for k,v in values.items()}
        for k,v in values.items():full[k][idx]=v
        g=physical['g'].numpy();Tn=inputs['h']*inputs['n'];x=inputs['x'];priors={}
        # Literal same CV/ballistic state prior as frozen CHECK32f; no native solve.
        for role,gravity in [('CV',np.zeros(3)),('ballistic',g)]:
            y=x.copy();y[:,:20]+=x[:,20:40]*Tn[:,None];y[:,40:43]+=x[:,52:55]*Tn[:,None]+.5*gravity*Tn[:,None]**2;y[:,52:55]+=gravity*Tn[:,None]
            y[:,43:52]=np.array([codec.scalar.exp_world(w*t)@R for w,t,R in zip(x[:,55:58],Tn,x[:,43:52].reshape(-1,3,3))]).reshape(-1,9)
            priors[role]=y
        masks={'ALL_COMPLETE_FRESH_ENDPOINTS':known}
        for fid in range(1000,1012):masks['family_'+str(fid)]=known&(z['family_id_i64']==fid)
        for h in [.0003125,.00015625]:masks['dt_'+str(h)]=known&(z['native_dt_s_F64']==h)
        for branch in sorted({d['source_branch'] for d in ids}):masks[branch]=known&np.array([d['source_branch']==branch for d in ids])
        stats=[];pair_ids=geometry.provider.pair_geom_ids;padcols=[pair_ids.index(gid) for gid in [15,28]]
        for name,mask in masks.items():
            ii=np.flatnonzero(mask);need(len(ii)>0,'All fixed cohorts nonempty');pos=np.searchsorted(idx,ii)
            for role in ['model5000','CV','ballistic']:
                py=full['prediction58_F64'][ii] if role=='model5000' else priors[role][pos]
                pc=full['coverage_probability22_F64'][ii][:,padcols] if role=='model5000' else np.zeros((len(ii),2))
                stats.append(dict(cohort=name,role=role,state=helper.state_stats(py,z['y58_F64'][ii],codec.scalar),two_pad_coverage_Brier=float(np.mean((pc-z['contact_summary_F64'][ii,:2])**2)),pair_impulse22_truth=None,meanFn22_truth=None,baseline_scope=None if role=='model5000' else 'Explicit constantqd/worldomega pose prior and zero-contact coverage prior; no native/contact feasibility certificate.'))
        np.savez_compressed(out/'all240_predictions_and_targets.npz',**full,known=known,complete=z['complete_H_bool'],row_id=z['row_id_i64'],family_id=z['family_id_i64'],episode_id=z['episode_id_i64'],interval=z['interval_index_i64'],current_x58=z['x58_F64'],actual_u20=z['u20_F64'],native_h_s=z['native_dt_s_F64'],n_substeps=z['n_substeps_i64'],target_y58=z['y58_F64'],true_two_pad_coverage=z['contact_summary_F64'][:,:2])
        with (out/'all240_identities.json').open('xb') as f:f.write(checked(fresh['sample_identities']).read_bytes())
        write(out/'endpoint_statistics.json',stats)
        result=dict(schema='phone_fresh12_fixed5000_state_evaluation_result:1',status='COMPLETE_FIXED5000_FRESH_CONTROLLED_FAMILIES_NOT_EGO_OR_POLICY',protocol=pin(protocol_path),checkpoint=a['training_result']['checkpoint'],fresh_acquisition=p['inputs']['fresh_acquisition'],counts=counts,rows=240,predicted_complete_rows=228,partial_unknown_rows=12,excluded_row_ids=z['row_id_i64'][~known].tolist(),immutability=unchanged,feature_cache_scope=scope,normalizers_refit=False,checkpoint_selection=False,labels_scope='Only state58 and two original designated pad coverage are available. Predicted meanFn22/J22 are retained without invented truth; lastPRE/minFn is not meanFn.',scope='Fixed checkpoint/new controlled q-actions at same known two poses. Partial/aborts retained; no held independent ego scene/trajectory, holding or Actor qualification.',elapsed_s=time.perf_counter()-started,outputs={q.name:pin(q) for q in out.iterdir() if q.is_file()})
        write(out/'report.json',result);print(json.dumps(dict(report=pin(out/'report.json'),counts=counts)));return 0
    except BaseException as e:
        write(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),counts=counts,automatic_retry=False));raise
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run',action='store_true');args=ap.parse_args()
    if args.run:raise SystemExit(execute(args.protocol))
    p,a,tp,out=metadata_preflight(args.protocol);print(json.dumps(dict(status='CPUFALSE_METADATA_ONLY_NO_TORCH_NPZ_MODEL',rules=RULES,output=str(out))))
