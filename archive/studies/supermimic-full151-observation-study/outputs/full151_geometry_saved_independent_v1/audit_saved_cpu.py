"""Saved native geometric contacts audit. No model/geometry/solver execution."""
from pathlib import Path
import collections,hashlib,json,re,sys,traceback
import numpy as np

ROOT=Path('D:/workspace/supermimic-full151-observation-study');RUN=ROOT/'outputs/full151_failed_candidate_geometry_v1';OUT=Path(__file__).parent
J=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin=lambda p:{'path':str(p),'sha256':sha(p)}
def load(p,keys=None):
    with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in (a.files if keys is None else keys)}
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def pins(x):
    if isinstance(x,dict):
        if set(x)=={'path','sha256'}:yield x
        else:
            for v in x.values():yield from pins(v)
    elif isinstance(x,list):
        for v in x:yield from pins(v)
checks={};details={};exceptions={}
def C(k,v):
    if k in checks:raise ValueError('Duplicate check '+k)
    checks[k]=bool(v)
def section(name):
    def deco(fn):
        try:fn()
        except Exception:checks[name+'_exception_free']=False;exceptions[name]=traceback.format_exc()
    return deco
receipt=J(RUN/'receipt.json');protocol=J(receipt['protocol']['path']);meta=J(RUN/'query_metadata.json')
catalog=load(RUN/'catalog.npz');arrays=load(RUN/'sampled_native_geometry.npz');summary=load(RUN/'generated_depth_summary.npz')
rows=[json.loads(s) for s in (RUN/'native_contacts.jsonl').read_text().splitlines()]
pairs=catalog['pairs'];kind=catalog['pair_kind'];names=catalog['geom_names'];times=np.arange(601)*.005
rebuilt_dist=np.full((601,273),np.nan);rebuilt_mask=np.zeros((601,273),bool);rebuilt_counts=np.zeros((601,273),np.int32);rebuilt_ties=np.zeros((601,273),np.int32)
selected_raw_index=np.full((601,273),-1,np.int32)

@section('identity')
def identity():
    allpins=list(pins(protocol));parent=J(protocol['inputs']['failed_receipt']['path']);parentdir=Path(protocol['inputs']['failed_receipt']['path']).parent
    C('exact_actual_protocol_pin',sha(receipt['protocol']['path'])==receipt['protocol']['sha256']=='f147bb33b661427a0ddcd6f6b982079b0d1e01058703a15e71e59b17bf95f0c5')
    C('all_current_protocol_source_input_pins_exact',all(sha(p['path'])==p['sha256'] for p in allpins))
    C('all_saved_artifact_pins_exact',all(sha(p['path'])==p['sha256'] for p in receipt['artifacts']))
    C('five_source_before_copies_exact',len(protocol['sources'])==5 and all(sha(RUN/'source_before'/(k+Path(p['path']).suffix))==p['sha256'] for k,p in protocol['sources'].items()))
    C('parent_FAIL_no_READY_still_unchanged',parent['status']==receipt['parent_candidate_status']=='CPU_CANDIDATE_GATES_FAIL_NO_QUALIFICATION' and not (parentdir/'READY.json').exists())
    C('no_geometry_READY_or_qualification',not (RUN/'READY.json').exists() and receipt['physics_qualified'] is False and receipt['contact_target_ready'] is False and receipt['training_authorized'] is False)
    C('reported_geometry_only_scope',receipt['status']=='COMPLETED_SAMPLED_GEOMETRY_DIAGNOSTIC' and receipt['CUDA_VISIBLE_DEVICES']=='-1' and protocol['queries']==601 and protocol['physics'] is False and protocol['training'] is False and protocol['GPU'] is False)
    C('all_runtime_call_receipts_geometry_only',receipt['actual_calls']['allowed']=={'mj_kinematics':602,'mj_comPos':602,'mj_collision':601} and receipt['actual_calls']['sample_count']==601 and all(v==0 for v in receipt['actual_calls']['forbidden'].values()) and receipt['checks']['clock_velocity_unchanged'] is True)
    C('original_native_query_options_retained',meta['query_options']=={'ccd_tolerance':1e-6,'ccd_iterations':50,'disableflags':0,'enableflags':0,'scope':'Original MJB collision numerical settings; kinematics/comPos/mj_collision only'})
    C('frozen_native_geometry_and_model_context_identity',meta['module_sha256']=='c964d5b300baac57a5353afd3b5aa4cfb381eca1da03ceef712691dd10376c72' and meta['model_sha256']==protocol['inputs']['model']['sha256']=='24fd7bf2658e1a6795e636eff1f1bc0a6f65f14baf4ed88a85dd2ffe125c33dd' and meta['context_sha256']==protocol['inputs']['actual_context']['sha256']=='5edd125973015c034d45e47092786b7fac105a06af52864582afa08d8660b89a')
    details['source_scope']={'protocol_pin_occurrences':len(allpins),'runtime_call_evidence':'Saved instrumented counts and source-bound query mutation guards; reviewer does not rerun native APIs or independently observe a live MjData.','CPU_elapsed_s':receipt['wall_s']}

@section('catalog')
def check_catalog():
    hand=catalog['hand_ids'];obj=catalog['object_ids']
    expected_self=[(int(a),int(b)) for j,a in enumerate(hand) for b in hand[j+1:]]
    expected_ho=[tuple(sorted((int(a),int(b)))) for a in hand for b in obj]
    C('273_potential_pair_catalogue_combinatorics_exact',pairs.shape==(273,2) and np.array_equal(pairs,np.array(expected_self+expected_ho)) and len(set(map(tuple,pairs)))==273)
    C('21_enabledhand_3object_no_mount',len(hand)==21 and len(obj)==3 and len(set(hand)&set(obj))==0 and all(re.fullmatch(r'robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)',str(names[h])) for h in hand))
    C('five_disabled_link1_geoms_excluded',all(x not in set(names[hand]) for x in meta['disabled_hand_names']) and len(meta['disabled_hand_names'])==5)
    C('210self63HO_kind_order_exact',np.array_equal(kind,np.array(['hand_self']*210+['hand_object']*63)) and meta['hand_self_potential_pairs']==210 and meta['hand_object_mask_compatible_pairs']==63)
    C('object_component_names_exact',set(names[obj])=={'object/cube','object/A_bottom','object/A_top'})
    C('catalogue_geometry_metadata_counts_consistent',meta['hand_geometry_enabled_count']==21 and meta['object_geometry_enabled_count']==3 and meta['native_parent_filter_enabled'] is True and meta['physical_reference_qualified'] is False)

@section('raw_records')
def raw_records():
    C('exact601_unique_raw_sample_indices_times',len(rows)==601 and [r['sample'] for r in rows]==list(range(601)) and np.array_equal(np.array([r['time_s'] for r in rows]),times))
    C('all601_saved_query_mask',arrays['queried'].shape==(601,) and arrays['queried'].all())
    fieldsok=True;nativecounts=True;raw_order=True;normalmax=0.;allraw=0;uncatalogued=0
    for i,r in enumerate(rows):
        buckets=collections.defaultdict(list);indices=[]
        nativecounts &= r['native_ncon']-r['uncatalogued']==len(r['records']) and r['uncatalogued']>=0
        allraw+=len(r['records']);uncatalogued+=r['uncatalogued']
        for v in r['records']:
            if len(v)!=7:fieldsok=False;continue
            pair_index,native_index,a,b,d,position,normal=v
            fieldsok &= 0<=pair_index<273 and tuple(sorted((a,b)))==tuple(pairs[pair_index]) and 0<=native_index<r['native_ncon']
            vals=np.r_[d,position,normal];fieldsok &= np.isfinite(vals).all() and np.shape(position)==(3,) and np.shape(normal)==(3,)
            err=abs(np.linalg.norm(normal)-1);normalmax=max(normalmax,float(err));fieldsok &= err<=meta['normal_unit_tolerance']
            indices.append(native_index);buckets[pair_index].append((float(d),native_index))
        raw_order &= indices==sorted(indices) and len(set(indices))==len(indices)
        for pair_index,entries in buckets.items():
            distances=np.array([x[0] for x in entries]);selected=int(np.argmin(distances));minimum=float(distances[selected])
            rebuilt_dist[i,pair_index]=minimum;rebuilt_mask[i,pair_index]=True;rebuilt_counts[i,pair_index]=len(entries)
            rebuilt_ties[i,pair_index]=int(np.count_nonzero(np.abs(distances-minimum)<=1e-9));selected_raw_index[i,pair_index]=entries[selected][1]
    C('all_raw_record_pair_geometry_indices_finite_unitnormals',fieldsok)
    C('all_native_ncon_uncatalogued_accounting_exact',nativecounts)
    C('raw_native_contact_order_unique_firstargmin_semantics',raw_order)
    C('generated_mask_rebuilt_exact',exact(rebuilt_mask,arrays['generated_mask']))
    C('all_generated_selected_deepest_distances_exact',np.array_equal(rebuilt_dist[rebuilt_mask],arrays['selected_distance_m'][rebuilt_mask]))
    C('all_absent_pair_distances_remain_NaN',np.isnan(rebuilt_dist[~rebuilt_mask]).all() and np.isnan(arrays['selected_distance_m'][~rebuilt_mask]).all())
    C('all_contact_count_by_pair_exact',exact(rebuilt_counts,arrays['contact_count']))
    C('all_1nm_diagnostic_tie_counts_exact',exact(rebuilt_ties,arrays['tie_count']))
    C('absent_count_tie_arrays_zero_not_measured_gap',not rebuilt_counts[~rebuilt_mask].any() and not rebuilt_ties[~rebuilt_mask].any())
    details['raw_records']={'catalogued_contact_records':allraw,'uncatalogued_native_contact_records':uncatalogued,'largest_normal_norm_error':normalmax,'selected_distance_rule':'Exact minimum, first native-index-order winner on exact equality. Tie count uses the original1e-9m diagnostic tolerance only.','no_force_claim':'Rows contain geometric normal/position/dist only; neither solved contact forces nor observed human contacts.'}

@section('depth_statistics')
def depth_statistics():
    depth=np.where(rebuilt_mask,np.maximum(-rebuilt_dist,0),0)
    bycat={k:depth[:,kind==k].max(axis=1) for k in ['hand_object','hand_self']}
    presence={k:rebuilt_mask[:,kind==k].any(axis=1) for k in ['hand_object','hand_self']}
    tieframes=np.any(rebuilt_ties>1,axis=1)
    recomputed={'maximum_generated_hand_object_depth_m':float(bycat['hand_object'].max()),'maximum_generated_hand_self_depth_m':float(bycat['hand_self'].max()),
                'sample_counts_over_original3mm':{k:int((bycat[k]>.003).sum()) for k in bycat},'sample_counts_without_generated_hand_object':int((~presence['hand_object']).sum()),
                'sample_counts_with_tied_pairs':int(tieframes.sum()),'sampled_depth3mm_pass':bool(all(np.all(v<=.003) for v in bycat))}
    C('summary_rebuilt_raw_all_declared_depth_counts',all(receipt['summary'][k]==v for k,v in recomputed.items()))
    C('saved_depth_summary_all601_exact',exact(summary['time_s'],times) and exact(summary['hand_object_depth_m'],bycat['hand_object']) and exact(summary['hand_self_depth_m'],bycat['hand_self']))
    worst=np.unravel_index(int(np.argmax(depth)),depth.shape);i,j=map(int,worst)
    expected={'sample':i,'time_s':float(times[i]),'pair_index':j,'pair_kind':str(kind[j]),'geom_names':[str(names[v]) for v in pairs[j]],'depth_m':float(depth[i,j]),'ties':int(rebuilt_ties[i,j])}
    C('global_worst_pair_time_depth_tie_exact',expected==receipt['worst_generated_overlap'])
    C('independent_geometric_sample_gate_FAIL_retained',recomputed['sampled_depth3mm_pass'] is False)
    percat={}
    for k in bycat:
        ids=np.flatnonzero(kind==k);flat=np.unravel_index(np.argmax(depth[:,ids]),(601,len(ids)));ii=int(flat[0]);jj=int(ids[flat[1]])
        percat[k]={'max_depth_m':float(depth[ii,jj]),'max_sample':ii,'time_s':float(times[ii]),'pair_index':jj,'geom_names':[str(names[v]) for v in pairs[jj]],
                   'samples_gt3mm':int((bycat[k]>.003).sum()),'samples_without_generated_pairs_unknown_gap':int((~presence[k]).sum()),
                   'absent_sample_indices':np.flatnonzero(~presence[k]).tolist()}
    details['summary']=recomputed;details['worst_generated_overlap']=expected;details['categories']=percat
    details['sample_ranges']=[{'local_samples_half_open':[a,b],'time_first_last_s':[float(times[a]),float(times[b-1])],**{k:{'max_generated_depth_m':float(v[a:b].max()),'samples_gt3mm':int((v[a:b]>.003).sum()),'samples_absent_unknown':int((~presence[k][a:b]).sum())} for k,v in bycat.items()}} for a,b in [(0,61),(61,601),(0,601)]]
    np.savez_compressed(OUT/'independent_reconstruction.npz',time_s=times,rebuilt_generated_mask=rebuilt_mask,rebuilt_selected_distance_m=rebuilt_dist,rebuilt_contact_count=rebuilt_counts,rebuilt_tie_count=rebuilt_ties,selected_native_contact_index=selected_raw_index,
                        generated_hand_object_depth_storage_sentinel=bycat['hand_object'],generated_hand_self_depth_storage_sentinel=bycat['hand_self'],hand_object_category_present=presence['hand_object'],hand_self_category_present=presence['hand_self'])

@section('input_and_macro_points')
def input_and_macro_points():
    ref=load(protocol['inputs']['reference']['path'],['planned_time601_s','planned_q601_rad','mapped_object_pose_w','control_points_rad'])
    parentinit=load(Path(protocol['inputs']['failed_receipt']['path']).parent/'init_requested.npz')
    C('initial_q0_FP32_preserved_source_no_rebase',exact(ref['planned_q601_rad'][0].astype(np.float32),parentinit['q_init_requested_FP32_rad']))
    C('initial_object0_FP32_preserved_source',exact(ref['mapped_object_pose_w'][0].astype(np.float32),parentinit['object_pose_init_requested_FP32_w']))
    C('all601_reference_input_times_andfinite_q_unchanged',exact(ref['planned_time601_s'],times) and ref['planned_q601_rad'].shape==(601,20) and np.isfinite(ref['planned_q601_rad']).all() and receipt['summary']['q_control_input_max_abs_rad']==float(np.abs(ref['planned_q601_rad']).max()))
    C('saved_macro_points21_shape_finite_only',arrays['fk_points_w'].shape==(601,21,3) and np.isfinite(arrays['fk_points_w']).all())
    details['macro_point_scope']='Saved fk_points_w are the frozen retarget_palmroot RobotKinematics macro21 points, not native BODY21 link_pos_w. This audit does not recompute FK or assert BODY21/native ABI identity.'
    details['input_state_scope']='Prescribed source64 q/object pose; query time/qvel0 is a geometric scratch-state guard. The original designed nonzero reset velocities and real target controls are not simulated or replaced.'

passed=sum(checks.values());total=len(checks)
result={'schema':'full151_failed_candidate_geometry_saved_independent_audit:1','status':'PASS_SAVED_AUDIT_GEOMETRY_AND_PARENT_REMAIN_FAILED' if passed==total and not exceptions else 'REVIEW_FAILURE_PRESERVED',
        'engineering_review_passed':passed==total and not exceptions,'passed':passed,'total':total,'checks':checks,'exceptions':exceptions,
        'original_receipt':pin(RUN/'receipt.json'),'review_source':pin(__file__),'details':details,
        'parent_candidate_qualified':False,'sampled_geometry_depth_gate_pass':False,'physical_reference_qualified':False,'contact_target_ready':False,'training_authorized':False,
        'reviewer_solver_FK_collision_Env_WM_physics_GPU_calls':0,
        'boundaries':['No native geometry APIs were re-executed; all reconstructions use saved raw records. Source-bound runtime counters are not independently rerun.',
                      'Ungenerated native pairs retain NaN distances. A zero in the category-depth storage array is a masked generated-overlap sentinel, never a positive-gap or no-contact certificate.',
                      '601samples do not certify intervals between samples. Generated overlap failures do not quantify dynamic grasp success or establish observed human contact/force labels.',
                      'The original CPU linear candidate remains FAIL/noREADY and cannot be promoted by diagnostic engineering completion.']}
(OUT/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({'status':result['status'],'passed':passed,'total':total,'failed':[k for k,v in checks.items() if not v],'exceptions':exceptions,'report':pin(OUT/'report.json')},indent=2))
sys.exit(0 if result['engineering_review_passed'] else 1)
