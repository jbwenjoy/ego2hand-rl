"""Independent saved arithmetic; never import or forward a learned model."""
from __future__ import annotations
import datetime,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/current104_physical_v2_WM_CPU_v1'
OUT=ROOT/'outputs/current104_physical_v2_WM_CPU_saved_independent_v2'
METHODS=('A1L','P_ONLY','A2');PATHS=('A0','A1L','P_ONLY','A2');PHASES=(0,4,8,12);EPS=(.002,.001)


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def pin(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return {'path':str(p.resolve()),'sha256':h.hexdigest()}
def each(x):
    if isinstance(x,dict):
        if set(x)=={'path','sha256'}:yield x
        else:
            for v in x.values():yield from each(v)
    elif isinstance(x,list):
        for v in x:yield from each(v)
def same(a,b):
    if a is None or b is None:return a is b
    if isinstance(a,(bool,str)) or isinstance(b,(bool,str)):return type(a) is type(b) and a==b
    if isinstance(a,dict) or isinstance(b,dict):return isinstance(a,dict) and isinstance(b,dict) and set(a)==set(b) and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) or isinstance(b,(list,tuple)):return isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return bool(np.isclose(a,b,atol=1e-12,rtol=1e-10))
def json_scalar(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError('Unsupported metadata scalar '+type(value).__name__)


def write(name,value):
    with (OUT/name).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False,default=json_scalar);f.write('\n')
def reward(terms):return np.add(terms[:,0],terms[:,1],dtype=np.float32).astype(np.float64)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    report=read(RUN/'report.json');plan=read(RUN/'runplan.json');proposal=read(plan['proposal']['path']);manifest=read(plan['physical_manifest']['path'])
    a=load(manifest['physical_arrays']['path']);d=load(proposal['inputs']['direction_preparation']['targets_and_vectors']['path']);w=load(RUN/'WM_predictions_gradients.npz')
    recorded=read(RUN/'comparisons.json');sec_saved=read(RUN/'physical_secants.json');group_saved=read(RUN/'descriptive_agreement.json');base_saved=read(RUN/'baseline_recomputed.json')
    checks={};pk=reward(a['branch_object_terms_k']);pn=reward(a['branch_object_terms_kplus1']);dk=reward(a['donor_object_terms_k'])
    noises=[];secants=[];physical={};classcounts={};base_rows=[]
    for s in range(16):
        ids=np.arange(s*35,s*35+3);b=pk[ids];noise=float(max(np.max(b)-np.min(b),np.max(np.abs(b-dk[s]))));noises.append(noise)
        base_rows.append({'snapshot_id':s,'noise':noise,'signal_floor':max(1e-5,10*noise),'baseline_rewards':b.tolist(),'fresh_donor_reward':float(dk[s])})
        for j in range(8):
            ids=[];signals=[];slopes=[]
            for e,eps in enumerate(EPS):
                mi=35*s+3+16*e+2*j;pl=mi+1;ids.append([mi,pl]);signal=float(pk[pl]-pk[mi]);signals.append(signal);slopes.append(signal/(2*eps))
                physical[s,j,e]={'slope':slopes[-1],'secondary_slope':float((pn[pl]-pn[mi])/(2*eps)),'minus':mi,'plus':pl}
            floor=max(1e-5,10*noise);nonflat=all(abs(v)>floor for v in signals)
            agree=bool(slopes[0]*slopes[1]>0);disagree=abs(slopes[0]-slopes[1])/max(abs(slopes[0]),abs(slopes[1]),1e-12)
            stable=bool(nonflat and agree and disagree<=.5);label='STABLE_NONFLAT' if stable else 'NONFLAT_UNSTABLE' if nonflat else 'FLAT_OR_BELOW_NOISE'
            classcounts[label]=classcounts.get(label,0)+1
            secants.append({'snapshot_id':s,'path':PATHS[s//4],'k':PHASES[s%4],'j':j,'complete':True,'classification':label,'signals':signals,'slopes':slopes,'signal_floor':floor,'two_epsilon_sign_agrees':agree,'relative_slope_disagreement':disagree,'stable_nonflat':stable,'branch_minus_plus_ids':ids})
    checks['all256secants_and128classifications_match']=same(secants,sec_saved)
    checks['noise_and_baseline_reward_rows_match']=all(same(row['noise'],base_saved['snapshots'][s]['noise']) and same(row['baseline_rewards'],base_saved['snapshots'][s]['baseline3_rewards']) and same(row['fresh_donor_reward'],base_saved['snapshots'][s]['fresh_donor_reward']) for s,row in enumerate(base_rows))
    checks['classification93_30_5']=classcounts=={'STABLE_NONFLAT':93,'NONFLAT_UNSTABLE':30,'FLAT_OR_BELOW_NOISE':5}
    comparisons=[]
    for method in METHODS:
        g=w[method+'__gradient_refk'].astype(np.float64);gn=w[method+'__gradient_refkplus1'].astype(np.float64)
        for z in secants:
            s,j=z['snapshot_id'],z['j']
            for e,eps in enumerate(EPS):
                item=physical[s,j,e];eff=d['effective_vectors'][s,j,e]
                ad=float(np.sum(g[s]*eff,dtype=np.float64));nominal=float(np.sum(g[s]*d['vectors'][s,j],dtype=np.float64))
                comparisons.append({'model':method,'path':z['path'],'k':z['k'],'snapshot_id':s,'j':j,'epsilon_rad':eps,'status':'COMPLETED','classification':z['classification'],'stable_nonflat':z['stable_nonflat'],'physical_slope':item['slope'],'AD_effective_FP32':ad,'AD_nominal_secondary':nominal,'AD_effective_kplus1_secondary':float(np.sum(gn[s]*eff,dtype=np.float64)),'physical_slope_kplus1_secondary':item['secondary_slope'],'effective_direction_norm':float(np.sqrt(np.sum(eff*eff))), 'midpoint_shift_L2_rad':float(np.linalg.norm((d['plus'][s,j,e].astype(float)+d['minus'][s,j,e].astype(float))/2-d['centers'][s])),'sign_match':bool(np.sign(ad)==np.sign(item['slope'])) if z['stable_nonflat'] else None})
    checks['all768_dots_secondary_effective_quantization_rows_match']=same(comparisons,recorded)
    groups=[]
    for method in METHODS:
        for eps in EPS:
            for path in (None,*PATHS):
                for k in (None,*PHASES):
                    subset=[z for z in comparisons if z['model']==method and z['epsilon_rad']==eps and (path is None or z['path']==path) and (k is None or z['k']==k)]
                    eligible=[z for z in subset if z['stable_nonflat']]
                    x=np.asarray([z['AD_effective_FP32'] for z in eligible]);y=np.asarray([z['physical_slope'] for z in eligible])
                    corr=float(np.corrcoef(x,y)[0,1]) if len(x)>=3 and np.std(x)>1e-12 and np.std(y)>1e-12 else None
                    groups.append({'model':method,'epsilon_rad':eps,'path':path or 'ALL_DEPENDENT_PATHS','k':k,'planned_rows':len(subset),'eligible_rows':len(eligible),'sign_agreement':float(np.mean(np.sign(x)==np.sign(y))) if len(x) else None,'pearson_r':corr})
    checks['all150_group_statistics_denominators_match']=same(groups,group_saved)
    calibrations={}
    labels=a['donor_self15'].astype(float);checks['all240native_self15_labels_zero']=labels.shape==(16,15) and np.all(labels==0)
    for method in METHODS:
        z=w[method+'__self15_logits'].astype(float);prob=np.exp(-np.logaddexp(0,-z));bce=float(-np.log1p(-prob).mean());brier=float((prob*prob).mean())
        saved=read(RUN/('self15_'+method+'.json'));pred=prob>=.5
        bins=[]
        for i in range(10):
            mask=(prob>=i/10)&(prob<((i+1)/10) if i<9 else prob<=1);n=int(mask.sum())
            bins.append({'bin':i,'count':n,'mean_probability':float(prob[mask].mean()) if n else None,'positive_fraction':0.0 if n else None})
        calc={'BCE':bce,'Brier':brier,'accuracy_at_probability_half':float((~pred).mean()),'true_positive_slots':0,'predicted_positive_slots':int(pred.sum()),'TP':0,'FP':int(pred.sum()),'FN':0,'precision':0.0 if pred.sum() else None,'recall':None,'F1':0.0 if pred.sum() else None,'calibration_bins':bins,'ECE10':sum(x['count']*x['mean_probability'] for x in bins if x['count'])/240}
        checks[method+'_self15_all0_calibration_matches']=same(calc,{k:saved[k] for k in calc})
        calibrations[method]=calc
    response=read(RUN/'physical_response_deltas.json');unique={};consistent=True
    for z in response:
        key=(z['snapshot_id'],z['row_id']);value={k:v for k,v in z.items() if k!='model'}
        if key in unique and not same(unique[key],value):consistent=False
        unique[key]=value
    checks['1680physicaldeltas_are_three_labels_of560_unique']=len(response)==1680 and len(unique)==560 and consistent
    qmodels={z['method']:z for z in proposal['inputs']['models']};loaders={};allpins=list(each(plan))+list(each(proposal))+list(each(manifest))+list(each(report))
    for model in report['models']:
        method=model['method'];spec=qmodels[method];loader=read(model['loader_receipt']['path']);loaders[method]=loader;allpins+=list(each(loader))
        expected={'state_dim':58,'physical_xi_dim':46,'disturbance_dim':174,'dynamics_dim':278,'WM_total_input_dim':298,'delta_dim':52,'self_collision_logits_dim':15,'native_task_context_dim':368,'iteration':2000}
        checks[method+'_loader_provenance_original_final_xi46_ABI']=loader['checkpoint']==spec['checkpoint'] and loader['fixed_physics_context']==spec['actual_physical_context'] and loader['loader']==plan['source_pins']['WM_loader'] and loader['method']==method and all(loader[k]==v for k,v in expected.items()) and loader['amp_enabled_for_CPU_diagnostic'] is False and loader['parameter_requires_grad'] is False and loader['GPU_used'] is False
        ctx=load(spec['actual_physical_context']['path']);source=load(loader['source_physical_context']['path']);xi=ctx['reader_context'];wanted=np.broadcast_to(source['reader_context'][:1],xi.shape)
        checks[method+'_saved_physical_xi128x46_bitexact']=xi.dtype==np.float32 and xi.shape==(128,46) and np.array_equal(xi.view(np.uint32),wanted.copy().view(np.uint32))
        checks[method+'_runtime_assertions_labeled_not_remeasured']=model['weights_and_norm_unchanged'] is True and model['input_unchanged'] is True and len(model['state_before_after_sha256'])==64
        norm=np.linalg.norm(w[method+'__gradient_refk'].astype(float),axis=-1);masked=np.linalg.norm(w[method+'__gradient_refk'].astype(float)*d['free_mask'],axis=-1)
        checks[method+'_saved_gradient_full_masked_norms_match']=same(norm.tolist(),model['gradient_full_norm_by_snapshot']) and same(masked.tolist(),model['gradient_masked_norm_by_snapshot'])
    current_bad=[v for v in allpins if pin(v['path'])['sha256']!=v['sha256']]
    checks['all_current_source_data_loader_model_file_pins_match']=not current_bad
    checks['seven_consumer_source_archives_match']=all(pin(RUN/(key+'__'+Path(v['path']).name))['sha256']==v['sha256'] for key,v in plan['source_pins'].items())
    checks['original_final_model_files_exact']=all(pin(s['checkpoint']['path'])==s['checkpoint'] for s in qmodels.values())
    checks['authorized_plan71_actual_saved']=plan['CPU_run_authorized'] is True and report['runplan']['sha256']=='71a64e1e0611007a615100c46db5c9197922f212c25ef9d104a84a246da60679' and read(report['runplan']['path'])==plan
    authfile=ROOT/'outputs/current104_physical_v2_WM_CPU_authorization_v1.json';auth=read(authfile);base=read(auth['base_runplan']['path'])
    checks['CPU_authorization_bytebound_flag_only']=pin(authfile)['sha256']=='c2bc2177295db68a51b2dc752a1ace6f61ad6b400564ed876727eaa607525b67' and {**plan,'CPU_run_authorized':False}==base and auth['physical_manifest']==plan['physical_manifest']
    checks['runtime_counts_and_scope_match_saved_table']=report['status']=='COMPLETED_CPU_SHARED_PHYSICAL_SECANT_COMPARISON' and report['WM_forwards']==3 and report['comparison_rows']==len(comparisons)==768 and report['all3_WM_completed'] is True and report['physical_gradient_qualified'] is False and report['RNG_unchanged'] is True and report['normalization_updated'] is False and report['GPU_used'] is False and report['physics_steps_added']==0
    checks['saved_predictions_gradient_layout_finite']=all(w[m+'__'+key].shape==shape and w[m+'__'+key].dtype==np.float32 and np.isfinite(w[m+'__'+key]).all() for m in METHODS for key,shape in [('prediction',(16,58)),('self15_logits',(16,15)),('gradient_refk',(16,20)),('gradient_refkplus1',(16,20))])
    checks['no_model_or_framework_execution_by_audit']=not any(m in sys.modules for m in ('torch','warp','mujoco','mujoco_warp'))
    write('recomputed_physical_secants.json',secants);write('recomputed_reward_noise.json',base_rows);write('recomputed_comparisons.json',comparisons);write('recomputed_group_statistics.json',groups);write('recomputed_negative_self15_calibration.json',calibrations)
    allrows=[z for z in groups if z['path']=='ALL_DEPENDENT_PATHS' and z['k'] is None]
    result={'schema':'current104_WM_physical_secants_saved_independent_audit:1','status':'PASS_SAVED_ARITHMETIC_AND_PROVENANCE' if all(checks.values()) else 'AUDIT_DISCREPANCY','checks':checks,'passed':sum(checks.values()),'total':len(checks),'auditor':pin(__file__),'consumer_report':pin(RUN/'report.json'),'authorized_runplan':report['runplan'],'physical_manifest':plan['physical_manifest'],'checked_pin_occurrences':len(allpins),'pin_mismatches':current_bad,'counts':{'physical_snapshots':16,'physical_branches':560,'physical_central_pairs':256,'directions':128,'stable':93,'unstable':30,'flat':5,'WM_batched_forwards_recorded':3,'saved_nominal_predictions':48,'saved_gradient_batches':6,'comparison_rows':768,'physical_delta_rows_labeled3times':1680,'unique_physical_delta_rows':560},'all_paths_per_epsilon_summary':allrows,'negative_only_self15':calibrations,'runtime_state_evidence_scope':'Only saved loader/state_dict-digest flags and pinned implementation assertions support in-memory parameter/normalizer/input/RNG preservation. No raw before/after RNG or in-memory WM state was saved for this auditor to recompare. Wrapperstate digest includes physical_xi/prefixed names and is not compared to loader rawWM-weight digest or wholecheckpoint-file SHA. Original checkpoint bytes are independently hashed only.','no_repeat':{'WM_loads':0,'WM_forwards':0,'Env_created':False,'physics_steps':0,'GPU_used':False},'scientific_limits':['One seen source/trainingseed0 with16 dependent states and8 directions per state, not N128/256/768/1680 independent observations.','93 stable directions are a predeclared numerical subset;30unstable and5flat retained and not counted correct.','Interiorcoordinate subspace only, face-normal/one-sided directions and full action Jacobian not tested.','Self15 truth is allnegative; calibration describes negative confidence only, not positive-event sensitivity or hand-object/contact-force competence.','Local one-step current-refk objective directional evidence, not multi-step imagination/Actor improvement/full151/shared/ego/hardware qualification.'],'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write('report.json',result)
    print(json.dumps({'status':result['status'],'passed':result['passed'],'total':result['total'],'failed':[k for k,v in checks.items() if not v],'summary':allrows,'report':pin(OUT/'report.json')},indent=2,default=json_scalar))
    if result['status']!='PASS_SAVED_ARITHMETIC_AND_PROVENANCE':raise SystemExit(1)


if __name__=='__main__':main()
