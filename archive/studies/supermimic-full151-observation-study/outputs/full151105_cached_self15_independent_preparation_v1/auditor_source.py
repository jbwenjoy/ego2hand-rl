"""Independent cached15 JSON join/scalar metrics. No model/native imports."""
import argparse,csv,hashlib,importlib.util,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'scripts/audit_boundary_components_final_saved_cpu_v1.py';BASE_SHA='5417e20e9642019d9cfded7b9dba0fdcc5739dcdc424d56a206a406a4bf3d94d'
s=importlib.util.spec_from_file_location('cached15_independent_saved_base',BASE)
if hashlib.sha256(BASE.read_bytes()).hexdigest()!=BASE_SHA:raise ValueError('Frozen helper changed')
V=importlib.util.module_from_spec(s);s.loader.exec_module(V)
M=V.M;A=V.A;pin=V.pin;sha=V.sha;read=V.read;checked=V.checked;same=V.same;dump=V.dump;require=V.require;bits=V.bits
ROLES=V.ROLES
CORE=ROOT/'src/full151_gradient/boundary_self15_cached_saved105_v1.py';CORE_SHA='97b74e0173bb6d95b55c59d924db7c5502c83cf610907c6c3aa357d54808da5e'
CONTRACT=ROOT/'docs/full151_boundary_final_self15_evaluation_source_contract_v1.md';CONTRACT_SHA='a535112324eeddb9113ebf47b2d68e379c3ddf5f191604eea498e756a30254f5'
ALIASES=dict(zip(ROLES,('baseline_forward152','control_forward152','method_forward152')))

def metadata(path):
    p=read(path);require(p['schema']=='full151105_cached_self15_label_evaluation_CPU_protocol:1' and p['role']=='all152_cached_native15_sensor_labels_saved_only','New cached15 namespace')
    require(p['CPU_run_authorized'] is True and p['GPU_run_authorized'] is p['training_authorized'] is p['automatic_retry'] is False and p['accuracy_gate'] is None,'Authorized saved-only no threshold/retry')
    require(sha(CORE)==CORE_SHA and sha(CONTRACT)==CONTRACT_SHA and same(p['source_contract'],pin(CONTRACT)),'Frozen source/phase contract')
    i=p['inputs'];require(set(i)=={'native_eval_protocol','native_eval_report','baseline_forward152','control_forward152','method_forward152','native_eval_group152','target_protocol','target_report','target_group152','reward_snapshots','sensor_mapping_A0','sensor_mapping_A2','physical_saved_review'},'Exact13 input aliases')
    for x in [*i.values(),*p['source_pins'].values()]:
        require(Path(x['path']).name!='physical_arrays.npz','This audit must not read physical arrays');checked(x)
    parent=read(checked(i['native_eval_report']));A.U.allchecks(parent)
    require(parent['status']=='COMPLETED_FIXED_INITIAL_AND_TWO_AFTER1000_EVAL_NOT_CONFIRMATION' and parent['counts']['native_FW_returned']==3 and same(parent['protocol'],i['native_eval_protocol']),'Complete parent predictions')
    for role,alias in ALIASES.items():require(same(parent['outputs'][role+'_forward152.npz'],i[alias]),'Exact same parent logits')
    physical=read(checked(i['physical_saved_review']));A.U.allchecks(physical);require(physical['schema']=='full151105_full_serial_saved_independent:1' and Path(physical['run_dir']).samefile(checked(i['reward_snapshots']).parent),'Prior source evidence only, no physical_arrays reads')
    ready=read(ROOT/'outputs/full151105_cached_self15_saved_preparation_CPU_v1/READY.json');proof=read(checked(ready['cpu_report']));A.U.allchecks(proof);require(proof['passed']==5 and ready['source_pins']==proof['source_pins'],'Source-bound Tiny5 not rerun')
    for x in ready['source_pins'].values():checked(x)
    return p

def mapped15(raw):
    require(type(raw) is list and len(raw)==1 and type(raw[0]) is list and len(raw[0])==25,'Raw found25 schema')
    for x in raw[0]:require(type(x) in (bool,int,float) and math.isfinite(x) and x>=0 and int(x)==x,'Nonnegative integer-valued sensor counts')
    # Derive each finger's middle2/middle3 then distal4|tip slots by stride5.
    return np.asarray([raw[0][5*f+j]>0 for f in range(5) for j in (1,2)]+[(raw[0][5*f+3]>0 or raw[0][5*f+4]>0) for f in range(5)],dtype='f4')

def score(logits,target):
    V.typed(logits,(len(target),15),'f4','native saved logits15');V.typed(target,logits.shape,'f4','cached label15');require(np.isin(target,(0,1)).all(),'Binary labels')
    p=M.probability(logits);y=target.astype('f8');bce=M.bce_elements(logits,target.astype('?'))
    pred=np.asarray([[float(v)>0 for v in row] for row in logits]);truth=target.astype('?');ties=logits==0
    intensity=np.asarray([math.fsum(float(v) for v in row) for row in p]);actual=np.asarray([math.fsum(float(v) for v in row) for row in y]);error=intensity-actual
    return {'logits15_FP32':logits.copy(),'probability15_F64':p,'BCE15_F64':bce,'Brier15_F64':(p-y)**2,'decision15_bool':pred,'truth15_bool':truth,'zero_logit_tie15_bool':ties,'pred_intensity_F64':intensity,'target_intensity_F64':actual,'intensity_error_F64':error,'pred_proxy_penalty_F64':-float(1/50)*intensity,'target_proxy_penalty_F64':-float(1/50)*actual,'proxy_penalty_error_F64':-float(1/50)*error}

def describe(a):
    v=np.asarray(a,dtype='f8');return {'mean':M.fmean(v),'MAE':M.fmean(abs(v)),'RMS':math.sqrt(M.fmean(v*v)),'maxabs':float(abs(v).max())}

def summary(s,ids):
    p,y,tie=s['decision15_bool'][ids],s['truth15_bool'][ids],s['zero_logit_tie15_bool'][ids];m={'TP':p&y,'FP':p&~y,'FN':~p&y,'TN':~p&~y,'zero_logit_tie':tie,'tie_positive_target':tie&y,'tie_negative_target':tie&~y}
    b=s['BCE15_F64'][ids];br=s['Brier15_F64'][ids]
    return {'rows':len(ids),'decisions':15*len(ids),'BCE_mean_entry_F64':M.fmean(b),'BCE_sum15_mean_rows_F64':math.fsum(math.fsum(float(v) for v in row) for row in b)/len(ids),'Brier_mean_entry_F64':M.fmean(br),'per_feature_BCE_F64':[M.fmean(b[:,j]) for j in range(15)],'per_feature_Brier_F64':[M.fmean(br[:,j]) for j in range(15)],'confusion_total':{k:int(v.sum()) for k,v in m.items()},'confusion_per_feature':{k:[int(v[:,j].sum()) for j in range(15)] for k,v in m.items()},'pred_intensity_mean':M.fmean(s['pred_intensity_F64'][ids]),'target_intensity_mean':M.fmean(s['target_intensity_F64'][ids]),'intensity_error':describe(s['intensity_error_F64'][ids]),'Pen_proxy_penalty_error_F64':describe(s['proxy_penalty_error_F64'][ids])}

def audit(protocol,run,out):
    out.mkdir(exist_ok=False);auditor=pin(__file__);(out/'executed_auditor.py').write_bytes(Path(__file__).read_bytes());dump(out/'equations_before_actual_read.json',{'auditor':auditor,'math':pin(V.MATH),'actual_reward_rows_logits_metrics_read':False,'formulas':'stride5 raw found25→middle10+distalORtip5; branch sigmoid/stableBCE/math.fsum; threshold>0 tiesnegative; dt_F64=1/50','F64_atol':2e-14,'F64_rtol':2e-12})
    p=metadata(protocol);i=p['inputs'];r=read(run/'report.json');outputs=read(run/'output_pins.json');checks={}
    def gate(k,v):checks[k]=bool(v)
    for x in outputs.values():checked(x)
    require(r['status']=='COMPLETED_ALL152_CACHED_SELF15_LABEL_METRICS_NOT_GEOMETRY_ACCURACY' and same(r['protocol'],pin(protocol)) and r['input_pins']==i,'One completed actual saved producer')
    sourceok=True
    for name,x in r['source_pins'].items():
        checked(x)
        if Path(x['path']).suffix=='.py':sourceok &= sha(run/('executed_'+name+'.py'))==x['sha256']
    gate('fixed_source_protocol_inputs_outputpins_and_executed_sources',sourceok and read(run/'protocol.json')==p and all(Path(x['path']).parent.samefile(run) for x in outputs.values()))
    plan=read(checked(i['target_group152']));require(plan==read(checked(i['native_eval_group152'])) and len(plan)==152,'Same all152 source/copy identities')
    expected=V.schedule()
    for n,(g,q) in enumerate(zip(plan,expected)):
        require(all(type(g[k]) is type(v) and g[k]==v for k,v in q.items()) and g['branch_id']==n and g['snapshot_id']==n//19 and g['physical_rows']==[4*q['group_id']+t for t in range(4)] and g['terminal_observation']==g['k']+1,'Exact branch order/physical rows/terminal identity')
    map_expected={'middle':[5*f+j for f in range(5) for j in (1,2)],'distal':[5*f+3 for f in range(5)],'tips':[5*f+4 for f in range(5)]}
    for path in ('A0','A2'):
        m=read(checked(i['sensor_mapping_'+path]));require(all(m[k]==v for k,v in map_expected.items()) and m['native_sensor']=='finger_collision' and m['state_phase']=='postintegration20ms' and m['sensor_phase']=='lastsolver preintegration15ms before Managerforward','Per-path actual fixed sensor-space mapping')
    reward=[json.loads(line) for line in checked(i['reward_snapshots']).read_text(encoding='utf-8').splitlines() if line.strip()];require(len(reward)==452,'Full cached source452 records');lookup={}
    for row in reward:
        key=(row['path'],row['k'],row['branch_id']);require(key not in lookup,'Unique raw reward source key');lookup[key]=row
    target=[];joined=[]
    for g in plan:
        row=lookup[(g['path'],g['k'],g['branch_id'])]
        require(row['mode']==g['mode'] and row['source_frame']==200+g['k'] and row['terminal_observation']==g['k']+1 and row['valid_next_policy_decision'] is (g['k']<149) and row['no_u150'] is True and row['label_time']=='lastsolver_preintegration~kdt+.015' and row['state_time']=='postintegration_kdt+.020' and row['capture']=='after_original_reward_compute_before_Manager_forward','Cached15/state20 phase contract')
        y=mapped15(row['raw_found']);c=row['contact'];require(type(c) is list and len(c)==15 and all(type(v) in (int,float) and math.isfinite(v) and v in (0,1) for v in c) and bits(y,np.asarray(c,dtype='f4')),'Raw found→original contact15 bytes independently exact')
        target.append(y);joined.append({**g,'contact15':c,'raw_found25':row['raw_found'][0],'label_time':row['label_time'],'state_time':row['state_time'],'source_frame':row['source_frame'],'valid_next_policy_decision':row['valid_next_policy_decision'],'no_u150':True})
    target=np.asarray(target,dtype='f4');gate('all152_raw_found25_contact15_exact_join_and_preserved_timing',joined==read(run/'joined152_cached_sensor_labels.json'))
    saved=V.load(run/'all152_three_role_cached_self15_metrics.npz');require(bits(saved['cached_self15_target_FP32'],target),'All2280 shared cached label values')
    groups={name:np.flatnonzero(mask).tolist() for name,mask in V.cohort_masks(expected)};require([len(v) for v in groups.values()]==[152,76,76]+[19]*8+[24,128],'Exact13 dependent cohorts')
    allstats={};rows=[];numeric=True;maxdiff={}
    for role,alias in ALIASES.items():
        with np.load(checked(i[alias]),allow_pickle=False) as f:logits=f['self15_logits_FP32'].copy()
        s=score(logits,target);maxdiff[role]={}
        for k,v in s.items():
            actual=saved[role+'__'+k];numeric &= bits(actual,v) if v.dtype in (np.dtype('f4'),np.dtype('?')) else A.close(actual,v)
            if v.dtype==np.dtype('f8'):maxdiff[role][k]=float(np.max(abs(actual-v)))
        allstats[role]={name:summary(s,ids) for name,ids in groups.items()}
        for bid,g in enumerate(plan):
            z=summary(s,[bid]);rows.append({'role':role,'row_id':bid,'snapshot_id':g['snapshot_id'],'path':g['path'],'k':g['k'],'mode':g['mode'],'group_id':g['group_id'],'BCE_mean15_F64':z['BCE_mean_entry_F64'],'Brier_mean15_F64':z['Brier_mean_entry_F64'],**z['confusion_total'],'pred_intensity_F64':float(s['pred_intensity_F64'][bid]),'target_intensity_F64':float(s['target_intensity_F64'][bid]),'Pen_proxy_error_F64':float(s['proxy_penalty_error_F64'][bid])})
    gate('all3x2280_logits_probabilities_BCE_Brier_bits_intensity_and_proxy',numeric)
    gate('all39cohorts_perfeature_confusion_intensity_penalty_summaries',V.equivalent(read(run/'three_roles_13cohorts.json'),allstats) and r['cohort_row_counts']=={k:len(v) for k,v in groups.items()})
    originalrows=read(run/'all456_role_rows.json');csvrows=list(csv.DictReader((run/'all456_role_rows.csv').open(encoding='utf-8-sig')));require(len(rows)==len(originalrows)==len(csvrows)==456,'All456 role rows')
    csvok=True
    for wanted,c in zip(rows,csvrows):
        require(set(c)==set(wanted),'Complete row CSV columns')
        for k,v in wanted.items():csvok &= c[k]==v if isinstance(v,str) else V.equivalent(int(c[k]) if type(v) is int else float(c[k]),v)
    gate('all456_JSON_CSV_complete_order_identity_metrics',V.equivalent(originalrows,rows) and csvok)
    features=[f'finger{f}_link{j}' for f in range(1,6) for j in (2,3)]+[f'finger{f}_link4_OR_tip' for f in range(1,6)]
    gate('count_feature_phase_scopes_preserved_no_geometry_or_accuracy_upgrade',r['counts']=={'forward_NPZ_attempted':3,'forward_NPZ_returned':3,'selected_labels':152,'weights_models_NN_FW_AD_MJB_Env_physics_GPU':0} and r['feature_names']==features and r['reward_dt_F64']==float(1/50) and r['geometry_physics_Actor_ego_or_wholehidden_qualified'] is False)
    for x in [*i.values(),*p['source_pins'].values(),*outputs.values()]:checked(x)
    gate('reviewer_no_Torch_models_native_physics_or_source_mutation',pin(__file__)==auditor and not any(k.split('.')[0] in ('torch','mujoco','warp','dex_supermimic','mjlab') for k in sys.modules))
    dump(out/'independent_39cohort_statistics.json',allstats);dump(out/'maximum_element_differences.json',maxdiff)
    result={'schema':'full151105_cached_self15_saved_independent:1','status':'PASS_ALL152_ALL3ROLES_CACHED15_METRICS_NOT_GEOMETRY' if all(checks.values()) else 'FAILED_CACHED15_SAVED_REVIEW','engineering_qualified':all(checks.values()),'checks':checks,'passed':sum(checks.values()),'total':len(checks),'auditor':auditor,'protocol':pin(protocol),'producer_report':pin(run/'report.json'),'source_contract':pin(CONTRACT),'run_dir':str(run),'independent_statistics':pin(out/'independent_39cohort_statistics.json'),'maximum_differences':pin(out/'maximum_element_differences.json'),'counts':{'original_reward_records':452,'selected_original_branch_labels':152,'shared_cached_label_entries':2280,'role_logit_entries':6840,'cohort_role_combinations':39,'role_rows':456},'limits':['Labels are original lastsolver~15ms sensor caches. They are not new POST20 geometry, force, human or global-clearance truth.','physical_arrays↔JSON correspondence is delegated to frozen writer/prior1762 source proof; this audit never opens physical_arrays.npz.','No additional self15 training supervision, no independent trajectory/generalization, causal or policy benefit claim.','F64 saved BCE/sigmoid/Pen proxy is separate from native FP32 reward arithmetic and full reward. Threshold strictly logit>0, zero ties negative; no label filtering.'],'reviewer_Torch_weights_models_FW_AD_MJB_Env_physics_GPU':0,'geometry_physics_Actor_ego_or_wholehidden_qualified':False}
    dump(out/'report.json',result);print(json.dumps({'report':pin(out/'report.json'),'passed':result['passed'],'total':result['total']},indent=2));return 0 if all(checks.values()) else 2

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run-dir',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--run',action='store_true');a=ap.parse_args();metadata(a.protocol)
    if not a.run:print(json.dumps({'status':'PASS_SOURCE_METADATA_ONLY_NO_ACTUAL_LABELS_LOGITS','auditor':pin(__file__)}));return 0
    require(a.run_dir is not None and a.output is not None,'Saved paths required')
    try:return audit(a.protocol,a.run_dir.resolve(strict=True),a.output.resolve())
    except Exception as e:
        if a.output.exists() and not(a.output/'failure.json').exists():dump(a.output/'failure.json',{'error':repr(e),'auditor':pin(__file__),'no_automatic_retry':True})
        raise
if __name__=='__main__':raise SystemExit(main())
