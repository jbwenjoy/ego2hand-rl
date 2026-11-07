"""Independent saved task-score projection audit, NumPy/stdlib only.

Default reads hash/JSON metadata. --run requires Root's actual terminal notice.
Captured AD values are checked for shape/finite/projection arithmetic, never
recomputed or certified as a physical, Actor, HO-force or hidden-state Jacobian.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE_PATH=ROOT/'scripts/audit_WM_component_confirmation_saved_cpu_v1.py'
BASE_SHA='c13cf42a0f73adfab89313fdc2cf992109d7659860de2690e233eb5bb2f8a58a'
MATH_PATH=ROOT/'src/full151_gradient/component_task_gradient_independent_math_v1.py'
MATH_SHA='950efb098c171c3e7e178f4b8fbd611eef2b818401ff7e9a1008d37ccb211668'
CORE_PATH=ROOT/'src/full151_gradient/component_task_gradient105_v1.py'
CORE_SHA='bc7d310a1b70f84cf257b973596ae5e01ad9e0cd3a1abe1e4e0a32e26b1a0921'
READY=ROOT/'outputs/full151105_component_task_gradient_preparation_CPU_v1/READY.json'
READY_SHA='d3324ff18024074c5efab815ac70a406169c49a37c75fbe247aece5b57b5918d'
ARMS=('pointwise','pointwise_plus_common_pre_response')
PHASES=(24,74,124,148);JOINTS=(0,4,8,12);EPS=(.002,.001)
FLOOR={'minimum_score':1e-5,'empirical_multiplier':10.,'two_eps_relative_disagreement_max':.5,'relative_slope_denominator_floor':1e-12,
 'classification_slope_denominator':'2*requested_epsilon_rad','classification_scope':'Shared physical32groups, both epsilon rows retained for both models; descriptive only'}


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def pin(p):return {'path':str(Path(p).resolve(strict=True)),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def require(v,s):
    if not v:raise ValueError(s)


def module(p,expected):
    require(sha(p)==expected,'Frozen independent helper '+str(p))
    s=importlib.util.spec_from_file_location('saved_independent_'+Path(p).stem,p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


BASE=module(BASE_PATH,BASE_SHA)
checked=BASE.checked;same=BASE.same;load=BASE.load;bits=BASE.bits;dump=BASE.dump


def typed(a,shape,dtype,label):
    require(a.shape==shape and a.dtype==np.dtype(dtype) and np.isfinite(a).all(),'Finite typed '+label)
    return a


def raw_tree_equal(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,dict):return set(a)==set(b) and all(raw_tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(raw_tree_equal(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return struct.pack('d',a)==struct.pack('d',b)
    return a==b


def subset(wanted,actual):
    if isinstance(wanted,dict):return isinstance(actual,dict) and all(k in actual and subset(v,actual[k]) for k,v in wanted.items())
    if isinstance(wanted,list):return isinstance(actual,list) and len(wanted)==len(actual) and all(subset(x,y) for x,y in zip(wanted,actual))
    return type(wanted) is type(actual) and wanted==actual


def allchecks(r):
    require(type(r.get('checks')) is dict and r['checks'] and all(v is True for v in r['checks'].values()),'Actual all-true checks')
    require(type(r.get('passed')) is type(r.get('total')) is int and r['passed']==r['total']==len(r['checks']),'Actual check counts')


def metadata(protocol_path):
    p=read(protocol_path)
    require(p['schema']=='full151105_component_task_gradient_CPU_protocol:1' and p['role']=='exploratory_not_new_blind_confirmation','Exploratory namespace')
    require(p['arms']==list(ARMS) and p['test_phases']==list(PHASES) and type(p['checkpoint_iteration']) is int and p['checkpoint_iteration']==1000,'Fixed two final1000/eight centers')
    require(type(p['cpu_threads']) is int and p['cpu_threads']==4 and p['include_endpoint_model_secants'] is False,'Fixed CPU4/no endpoint forwards')
    for k in ('execution_authorized','CPU_run_authorized','GPU_run_authorized','training_authorized','Env_or_physics_authorized'):require(type(p[k]) is bool,'Explicit authorization bool '+k)
    require(p['GPU_run_authorized'] is p['training_authorized'] is p['Env_or_physics_authorized'] is False,'CPU only/no new training/physics')
    require(p['physical_floor_rule']==FLOOR and p['comparison_modes']==['actual_coordinate_secant','legacy_nominal_epsilon_effective_direction'],'Frozen physical floor and separate units')
    i=p['inputs'];require(len(i)==19 and set(i)==set(read(checked(i['parent_confirmation_protocol']))['inputs'])|{'parent_confirmation_protocol','parent_confirmation_report','parent_confirmation_saved_review'},'Exact19 semantic aliases')
    parent=BASE.metadata(checked(i['parent_confirmation_protocol']))
    require(p['architecture']==parent['architecture'] and all(type(p['architecture'][k]) is type(v) for k,v in parent['architecture'].items()),'Fixed CPU FP32 architecture types')
    require(p['output_prefix']=='full151105_component_task_gradient_CPU_','Fresh task-gradient namespace')
    require(all(same(i[k],v) for k,v in parent['inputs'].items()),'Unchanged parent16 inputs')
    for v in [*i.values(),*p['source_pins'].values()]:checked(v)
    saved=read(checked(i['parent_confirmation_saved_review']));allchecks(saved);report=read(checked(i['parent_confirmation_report']))
    require(saved['schema']=='full151105_WM_component_confirmation_saved_independent:1' and saved['engineering_qualified'] is True,'Original confirmation independent24')
    require(i['parent_confirmation_report']['sha256']=='82b88d40d26aa0a940b09777a40e741c62dfecf2c2e14ba5bbbe51040b100908','No new component selection')
    require(same(saved['protocol'],i['parent_confirmation_protocol']) and same(saved['producer_report'],i['parent_confirmation_report']) and saved['decision']==report['decision'],'Parent original decision/source join')
    require(sha(READY)==READY_SHA and sha(CORE_PATH)==CORE_SHA,'Current frozen task-gradient source and focused proof')
    ready=read(READY);proof=read(checked(ready['cpu_report']));allchecks(proof)
    require(ready['source_pins']==proof['source_pins'] and ready['source_pins']['consumer']['sha256']==CORE_SHA,'Actual current-source focused CPU proof')
    for v in ready['source_pins'].values():checked(v)
    require(any(same(pin(CORE_PATH),v) for v in p['source_pins'].values()),'Protocol binds active core')
    return p,report['decision']


def score_terms64(x,p,R):
    x,p,R=(np.asarray(v,np.float64) for v in (x,p,R))
    center=x[:,40:43];shaft=x[:,43:52].reshape(-1,3,3)[:,:,0]
    left=center+.1*shaft-p-.1*R[:,:,0];right=center-.1*shaft-p+.1*R[:,:,0]
    endpoint=.5*(np.linalg.norm(left,axis=1)+np.linalg.norm(right,axis=1))
    return np.column_stack((2*np.exp(-(np.linalg.norm(center-p,axis=1)/.05)**2),8*np.exp(-(endpoint/.04)**2)))


def baseline_rows(a):
    rows=[]
    blocks=(('q',0,20,2e-5),('qd',20,40,2e-4),('p',40,43,2e-5),('R',43,52,2e-5),('v',52,55,2e-4),('omega',55,58,2e-4))
    for s in range(8):
        e=np.abs(a['branch_x_next'][s*19:s*19+3].astype('f8')-a['donor_next_x'][s].astype('f8'))
        errors={k:float(e[:,lo:hi].max()) for k,lo,hi,_ in blocks}
        scores=a['branch_object_score_k'][s*19:s*19+3].astype('f8');span=float(scores.max()-scores.min())
        err=float(np.abs(scores-float(a['donor_object_score_k'][s])).max())
        rows.append({'snapshot_id':s,'errors':errors,'baseline_range':span,'baseline_donor_error':err,
            'signal_floor':max(1e-5,10*max(span,err)),'passed':all(errors[k]<=tol for k,_,_,tol in blocks) and err<=2e-5})
    return rows


def physical_rows(a,pairs,math):
    groups=[];rows=[]
    for s in range(8):
        for j,q in enumerate(JOINTS):
            ids=[s*19+3+8*ei+2*j for ei in range(2)]
            m=a['branch_object_score_k'][ids];pp=a['branch_object_score_k'][np.asarray(ids)+1]
            cl=math.physical_classification(m,pp,a['branch_object_score_k'][s*19:s*19+3],a['donor_object_score_k'][s])
            groups.append({'group_id':s*4+j,'snapshot_id':s,'path':('A0','A2')[s//4],'k':PHASES[s%4],'j':j,'joint_index':q,
              'label':cl['label'],'stable_nonflat':cl['stable_nonflat'],'signal_floor':cl['signal_floor'],'delta_score':cl['delta_score'].tolist(),
              'slopes':cl['legacy_nominal_slopes'].tolist(),'relative_disagreement':cl['relative_epsilon_disagreement'],'both_above_noise':bool(cl['signal_above_floor'].all())})
            for ei,eps in enumerate(EPS):
                source=pairs[(s*4+j)*2+ei];mi,pl=source['minus_branch_id'],source['plus_branch_id']
                delta=a['branch_u'][pl].astype('f8')-a['branch_u'][mi].astype('f8');den=float(delta[q]);require(den>0,'Any nonpositive actual denominator blocks whole run')
                dr=float(np.float64(a['branch_object_score_k'][pl])-np.float64(a['branch_object_score_k'][mi]))
                rows.append({**source,'group_id':s*4+j,'primary_GPU_score_minus_FP32':float(a['branch_object_score_k'][mi]),'primary_GPU_score_plus_FP32':float(a['branch_object_score_k'][pl]),
                  'delta_score_F64':dr,'actual_delta_u20_rad_F64':delta.tolist(),'actual_coordinate_denominator_rad':den,'actual_coordinate_defined':True,
                  'physical_actual_coordinate_secant':dr/den,'physical_legacy_nominal_epsilon_slope':dr/(2*eps),
                  'asymmetric_about_baseline':bool(np.any(a['direction__center_offset'][s,j,ei]!=0)),
                  'physical_group_label':cl['label'],'stable_nonflat_shared_physical':cl['stable_nonflat'],'signal_floor_score':cl['signal_floor'],
                  'relative_epsilon_disagreement':cl['relative_epsilon_disagreement'],
                  'primary_GPU_terms_minus_FP32':a['branch_object_terms_k'][mi].tolist(),'primary_GPU_terms_plus_FP32':a['branch_object_terms_k'][pl].tolist(),
                  'CPU_secondary_terms_minus_FP32':a['branch_object_terms_k_CPU_secondary'][mi].tolist(),'CPU_secondary_terms_plus_FP32':a['branch_object_terms_k_CPU_secondary'][pl].tolist(),
                  'true_Jacobian_qualified':False})
    return groups,rows


def comparison_summary(rows):
    def describe(v,pk,mk):
        if not v:return {'reported_rows':0,'MAE':None,'RMSE':None,'sign_matches':None}
        p=np.array([x[pk] for x in v],dtype='f8');m=np.array([x[mk] for x in v],dtype='f8')
        return {'reported_rows':len(v),'MAE':float(abs(p-m).mean()),'RMSE':float(np.sqrt(np.mean((p-m)**2))),'sign_matches':int((np.sign(p)==np.sign(m)).sum())}
    return {'all64_rows_retained':len(rows),'actual_defined_rows':len(rows),'actual_undefined_rows':0,
      'all_defined_actual_coordinate':describe(rows,'physical_actual_coordinate_secant','WM_AD_actual_delta_projected_per_coordinate'),
      'all64_legacy':describe(rows,'physical_legacy_nominal_epsilon_slope','WM_AD_legacy_effective_direction'),
      'stable_physical_groups_per_epsilon':{str(eps):describe([r for r in rows if r['epsilon_index']==ei and r['stable_nonflat_shared_physical']],
         'physical_legacy_nominal_epsilon_slope','WM_AD_legacy_effective_direction') for ei,eps in enumerate(EPS)}}


def csv_matches(path,expected):
    with Path(path).open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    if len(rows)!=len(expected):return False
    for r,w in zip(rows,expected):
        for k,v in w.items():
            if k not in r:return False
            s=r[k]
            if isinstance(v,(dict,list)):ok=subset(v,json.loads(s))
            elif v is None:ok=s==''
            elif type(v) is bool:ok=s==str(v)
            elif type(v) is int:ok=int(s)==v
            elif type(v) is float:ok=float(s)==v
            else:ok=s==v
            if not ok:return False
    return True


def tensor_digest_arrays(values):
    h=hashlib.sha256()
    for k,a in sorted(values.items()):
        typed(a,a.shape,'f4',k);h.update(k.encode());h.update(str(('torch.float32',tuple(a.shape))).encode());h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def raw_rng(run):
    a,b=load(run/'before_RNG_arrays.npz'),load(run/'after_RNG_arrays.npz')
    x,y=read(run/'before_RNG_python_numpy.json'),read(run/'after_RNG_python_numpy.json')
    for z in (a,b):
        require(set(z)=={'numpy_MT','torch_CPU'},'Three raw CPU RNG groups')
        require(z['numpy_MT'].dtype==np.uint32 and z['numpy_MT'].shape==(624,) and z['torch_CPU'].dtype==np.uint8 and z['torch_CPU'].ndim==1 and z['torch_CPU'].size>0,'Raw RNG vector types/shapes')
    def header(z):
        p,n=z['python'],z['numpy_header']
        return (type(p) is list and len(p)==3 and type(p[0]) is int and p[0]==3 and type(p[1]) is list and len(p[1])==625 and all(type(v) is int for v in p[1])
          and (p[2] is None or type(p[2]) is float and np.isfinite(p[2])) and type(n) is list and len(n)==4 and n[0]=='MT19937' and type(n[1]) is type(n[2]) is int and type(n[3]) is float and np.isfinite(n[3]))
    return {'raw_header_schema_valid':bool(header(x) and header(y)),'Python_values_types_floatbits_equal':raw_tree_equal(x['python'],y['python']),
      'NumPy_header_values_types_floatbits_equal':raw_tree_equal(x['numpy_header'],y['numpy_header']),'NumPy_MT_bytes_equal':bits(a['numpy_MT'],b['numpy_MT']),
      'Torch_CPU_bytes_equal':bits(a['torch_CPU'],b['torch_CPU']),'CUDA_initialized_both_false':x['CUDA_initialized'] is y['CUDA_initialized'] is False}


def audit(protocol_path,run,out):
    out.mkdir(exist_ok=False);auditor=pin(__file__);(out/'executed_auditor.py').write_bytes(Path(__file__).read_bytes())
    dump(out/'equations_before_actual_saved_read.json',{'auditor':auditor,'math':pin(MATH_PATH),'prior_independent_saved_helper':pin(BASE_PATH),'actual_task_gradient_values_read':False,
      'scope':'All32/64, F64 before score/control differences and projection; actual and nominal denominators separate; original physical-only floor; no AD/model replay.'})
    p,parent_decision=metadata(protocol_path);i=p['inputs'];r=read(run/'report.json');sources=read(run/'source_inventory.json');math=module(MATH_PATH,MATH_SHA)
    checks={};details={}
    def gate(k,v):checks[k]=bool(v)
    gate('actual_authorized_protocol_19aliases_and_original_parent_decision',p['execution_authorized'] is p['CPU_run_authorized'] is True and same(r['protocol'],pin(protocol_path))
      and raw_tree_equal(p,read(run/'protocol.json')) and raw_tree_equal(r['input_pins'],i) and r['parent_ea22_decision_unchanged']==parent_decision)
    for v in sources.values():checked(v)
    gate('current_unique_source_closure_frozen_core',any(same(v,pin(CORE_PATH)) for v in sources.values()) and len({(checked(v).stat().st_dev,checked(v).stat().st_ino) for v in sources.values()})==len(sources))
    with zipfile.ZipFile(run/'execution_sources.zip') as z:zipok=len(z.namelist())==len(sources) and all(hashlib.sha256(z.read(k)).hexdigest()==v['sha256'] for k,v in sources.items())
    gate('source_ZIP_immutable_bytes_exact',zipok)
    outputpins=read(run/'output_pins.json')
    for v in outputpins.values():checked(v)
    gate('all_immutable_output_pins_exact',True)
    a=load(checked(i['test_full_arrays']));_,identities,pairs=BASE.expected_input(a,read(checked(i['test_branch_index'])),read(checked(i['test_snapshot_records'])))
    gate('all152_original_order_actual_control_bits_and64_nonzero_pairs',subset(identities,read(run/'all152_identities.json')) and len(pairs)==64 and all(x['actual_du_rad']>0 for x in pairs))
    d=load(run/'physical_task_inputs.npz')
    fields={k:k for k in ('snapshot_x','snapshot_u','goal_p','goal_R','branch_x_next','branch_u','branch_object_score_k','branch_object_terms_k','branch_object_terms_k_CPU_secondary','donor_object_score_k')}
    fields.update(direction_plus='direction__plus',direction_minus='direction__minus',legacy_d_eff='direction__d_eff',center_offset_rad='direction__center_offset',clipped='direction__clipped')
    gate('saved_task_arrays_all_source_states_goals_primary_scores_actual_u_exact',all(bits(d[k],a[v]) for k,v in fields.items()))
    for k,shape in [('goal_p',(8,3)),('goal_R',(8,3,3)),('branch_object_score_k',(152,)),('branch_object_terms_k',(152,2)),('branch_object_terms_k_CPU_secondary',(152,2)),('donor_object_score_k',(8,))]:typed(a[k],shape,'f4',k)
    expected=score_terms64(a['branch_x_next'],a['goal_p'][np.arange(152)//19],a['goal_R'][np.arange(152)//19]);bound=8*np.finfo(np.float32).eps*np.maximum(1.,abs(expected));delta=a['branch_object_terms_k'].astype('f8')-expected
    gate('all152_primary_GPU_terms_original_samepost20_8eps_arithmetic',bool((abs(delta)<=bound).all()) and bits(a['branch_object_terms_k'].sum(-1,dtype='f4'),a['branch_object_score_k'])
      and bits(d['physical_terms_expected_F64'],expected) and bits(d['physical_terms_delta_F64'],delta) and bits(d['physical_terms_original_bound_F64'],bound))
    gate('CPU_secondary_is_separate_saved_diagnostic_only',bits(d['CPU_secondary_score_sum_FP32'],a['branch_object_terms_k_CPU_secondary'].sum(-1,dtype='f4')))
    bases=baseline_rows(a);gate('all8_original_three_baseline_sixchannels_and_score_gates',all(x['passed'] for x in bases) and raw_tree_equal(bases,read(run/'baseline8.json')))
    groups,physical=physical_rows(a,pairs,math)
    gate('all32_shared_physical_floor_twoepsilon_classifications_and_CSV',subset(groups,read(run/'physical_groups32.json')) and csv_matches(run/'physical_groups32.csv',groups))
    gate('all64_physical_chords_actual_vs_legacy_units_and_CSV',subset(physical,read(run/'physical_rows64.json')) and csv_matches(run/'physical_rows64.csv',physical))
    receipt=read(checked(i['initial_validation']));typed(d['physical_xi46'],(46,),'f4','xi46');typed(d['frozen_s52'],(52,),'f4','s52');require((d['frozen_s52']>0).all(),'Positive original scale52')
    confirmation_run=checked(i['parent_confirmation_report']).parent
    gate('fixed_physical_xi46_initial_digest_and_frozen_scale52_parent_capture',tensor_digest_arrays({'xi':d['physical_xi46']})==receipt['xi46_sha256']
      and all(bits(d['frozen_s52'],load(confirmation_run/(arm+'_evaluation_arrays.npz'))['scale52_FP32']) for arm in ARMS))
    summaries={};center_errors={};arrays_out={}
    for arm in ARMS:
        e=load(run/(arm+'_center_gradients.npz'))
        for k,shape in [('predicted_x8_FP32',(8,58)),('self15_logits_FP32',(8,15)),('score_terms8_FP32',(8,2)),('score8_FP32',(8,)),('gradient_u8x20_FP32',(8,20))]:typed(e[k],shape,'f4',arm+'/'+k)
        gate(arm+'_finite_captured8predictions_self15_and_input_AD_no_reload',all(bits(e[k],d[v]) for k,v in [('snapshot_x8_FP32','snapshot_x'),('snapshot_u8_FP32','snapshot_u'),('goal_p8_FP32','goal_p'),('goal_R8_FP32','goal_R')]))
        terms64=score_terms64(e['predicted_x8_FP32'],d['goal_p'],d['goal_R']);termerr=e['score_terms8_FP32'].astype('f8')-terms64;termbound=8*np.finfo('f4').eps*np.maximum(1.,abs(terms64))
        gate(arm+'_captured_score_samepost20_goalk_formula_and_FP32_subset_sum',bool((abs(termerr)<=termbound).all()) and bits(e['score_terms8_FP32'].sum(-1,dtype='f4'),e['score8_FP32']))
        rows=[]
        for s in range(8):
            for j,q in enumerate(JOINTS):
                ids=np.array([s*19+3+8*ei+2*j for ei in range(2)])
                calc=math.projection_conventions(e['gradient_u8x20_FP32'][s],a['snapshot_u'][s],a['branch_u'][ids],a['branch_u'][ids+1],a['branch_object_score_k'][ids],a['branch_object_score_k'][ids+1],q)
                for ei in range(2):
                    row=dict(physical[(s*4+j)*2+ei]);actual=float(calc['model_actual_coordinate_projection'][ei]);legacy=float(calc['model_legacy_effective_projection'][ei])
                    row.update(WM_AD_coordinate_at_baseline_u0=float(calc['model_nominal_coordinate_gradient']),WM_AD_actual_delta_projected_per_coordinate=actual,WM_AD_legacy_effective_direction=legacy,
                       actual_secant_minus_WM_AD=row['physical_actual_coordinate_secant']-actual,legacy_slope_minus_WM_AD_effective=row['physical_legacy_nominal_epsilon_slope']-legacy)
                    rows.append(row)
        expected_arrays={'actual_defined_mask64':np.ones(64,bool),'stable_shared_physical_mask64':np.array([x['stable_nonflat_shared_physical'] for x in rows],bool),
          'physical_actual_secant64_F64':np.array([x['physical_actual_coordinate_secant'] for x in rows],dtype='f8'),
          'WM_AD_actual_projected64_F64':np.array([x['WM_AD_actual_delta_projected_per_coordinate'] for x in rows],dtype='f8'),
          'physical_legacy_slope64_F64':np.array([x['physical_legacy_nominal_epsilon_slope'] for x in rows],dtype='f8'),
          'WM_AD_legacy_effective64_F64':np.array([x['WM_AD_legacy_effective_direction'] for x in rows],dtype='f8')}
        gate(arm+'_all64_F64_actual_and_legacy_projections_arrays_JSON_CSV',all(bits(v,e[k]) for k,v in expected_arrays.items()) and subset(rows,read(run/(arm+'_comparison64.json'))) and csv_matches(run/(arm+'_comparison64.csv'),rows))
        summaries[arm]=comparison_summary(rows)
        gate(arm+'_all64_and_shared32_twoeps_summaries_no_model_filter',subset(summaries[arm],r['comparison_summaries'][arm]))
        center_errors[arm]={'term_max_absolute_F64_discrepancy':float(abs(termerr).max()),'original_same20_arithmetic_gate':True}
        np.savez_compressed(out/(arm+'_independent_projections64.npz'),**expected_arrays,score_terms_expected_F64=terms64,score_term_difference_F64=termerr,score_term_original_bound_F64=termbound)
    counts={key:sum(g['label']==key for g in groups) for key in sorted({g['label'] for g in groups})}
    gate('all32_shared_group_counts_and64rows_perarm_retained',r['physical_group_counts']==counts and r['coverage']=={'snapshots':8,'shared_physical_direction_groups':32,'epsilon_rows_per_arm':64} and r['physical_floor_rule']==FLOOR)
    rng=raw_rng(run);gate('raw_Python_NumPy_TorchCPU_RNG_values_headers_and_vectors_equal',all(rng.values()) and r['raw_RNG_saved_before_after'] is True)
    memory={key:d[value] for key,value in [('x','snapshot_x'),('u0','snapshot_u'),('goal_p','goal_p'),('goal_R','goal_R'),('xi','physical_xi46'),('scale','frozen_s52')]}
    gate('captured_input_memory_digest_matches_source_before_after',tensor_digest_arrays(memory)==r['input_memory_digests']['before']==r['input_memory_digests']['after'])
    gate('weight_norm_source_assertions_bound_not_independent_AD_or_weight_replay',all(v is True for v in r['invariants']['weights_norm_config_unchanged'].values())
      and set(r['weight_digests'])==set(ARMS) and all(v['before']==v['after'] for v in r['weight_digests'].values()) and all(v is True for k,v in r['invariants'].items() if k!='weights_norm_config_unchanged'))
    wanted_counts={'trusted_component_torch_loads':3,'native_WM_constructors':2,'WM_forwards':2,'center_predictions':16,'uVJP_calls':2,'original119f_loads':0,'optimizer_Actor_MJB_Env_physics_GPU':0}
    gate('source_successful_path_3loads_2constructors_2B8FW_2inputVJP',r['counts']==wanted_counts)
    gate('exploratory_status_no_branchPRE_hidden_physical_HO_Actor_or_newconfirmation',r['schema']=='full151105_component_task_gradient_CPU_analysis:1'
      and r['status']=='COMPLETED_EXPLORATORY_TWO_COMPONENT_TASK_SCORE_GRADIENT_NOT_CONFIRMATION' and r['role']==p['role']
      and all(r[k] is False for k in ('branch_PRE_full58_directly_observed','whole_hidden_state_qualified','physical_gradient_qualified','HO_force_gradient_qualified','Actor_or_CHORD_benefit_qualified','new_blind_confirmation')))
    for v in [*i.values(),*sources.values(),*outputpins.values()]:checked(v)
    gate('reviewer_NumPy_only_no_Torch_nativeLog_AD_model_or_physics',not any(k.split('.')[0] in ('torch','mujoco','warp','mujoco_warp','mjlab','dex_supermimic') for k in sys.modules) and pin(__file__)==auditor and sha(MATH_PATH)==MATH_SHA)
    details.update(raw_RNG=rng,original_three_baselines=bases,center_score_arithmetic=center_errors,source_unique_files=len(sources),output_pins=len(outputpins),
      derivative_scope='Saved gradient values finite and projected independently; no model/AD replay or proof of derivative correctness. Weights/norm/config equality is a source-backed digest relation.',
      counts_scope='Successful producer counters/source path, not a reviewer API interceptor.',
      physical_scope='CPU model derivative versus saved GPU finite chords; two eps per dependent direction group. Clipping orthogonal; no whole hidden state, true physical/HO/Actor Jacobian or historical AMP claim.')
    result={'schema':'full151105_component_task_gradient_saved_independent:1','status':'PASS_SAVED_ALL32_ALL64_TASK_GRADIENT_ARITHMETIC' if all(checks.values()) else 'FAILED_SAVED_TASK_GRADIENT_REVIEW',
      'engineering_qualified':all(checks.values()),'passed':sum(checks.values()),'total':len(checks),'checks':checks,'auditor':auditor,'independent_math':pin(MATH_PATH),
      'producer_READY':pin(READY),'protocol':pin(protocol_path),'producer_report':pin(run/'report.json'),'run_dir':str(run.resolve()),'input_pins':i,
      'physical_group_counts':counts,'comparison_summaries':summaries,'parent_ea22_decision_unchanged':parent_decision,'details':details,
      'whole_hidden_state_qualified':False,'physical_gradient_qualified':False,'HO_force_gradient_qualified':False,'Actor_or_CHORD_benefit_qualified':False,'new_blind_confirmation':False,
      'reviewer_model_checkpoint_loads_FW_AD_MJB_Env_physics_GPU':0}
    dump(out/'report.json',result);print(json.dumps({'report':pin(out/'report.json'),'passed':result['passed'],'total':result['total'],'physical_group_counts':counts,'comparison_summaries':summaries},indent=2))
    return 0 if all(checks.values()) else 2


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run-dir',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--run',action='store_true');a=ap.parse_args()
    metadata(a.protocol)
    if not a.run:
        print(json.dumps({'status':'PASS_SOURCE_JSON_HASH_ONLY_NO_TASK_VALUES','auditor':pin(__file__),'math':pin(MATH_PATH),'producer_READY':pin(READY),'actual_taskscore_predictions_gradients_read':0}));return 0
    require(a.run_dir is not None and a.output is not None,'Fresh actual run/output required')
    try:return audit(a.protocol,a.run_dir.resolve(strict=True),a.output.resolve())
    except Exception as error:
        if a.output.exists() and not(a.output/'failure.json').exists():dump(a.output/'failure.json',{'error':repr(error),'auditor':pin(__file__),'saved_only':True,'producer_outputs_unchanged':True})
        raise


if __name__=='__main__':raise SystemExit(main())
