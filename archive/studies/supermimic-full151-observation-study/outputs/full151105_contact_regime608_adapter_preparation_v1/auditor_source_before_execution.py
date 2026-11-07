"""Independent saved contact-regime adapter; NumPy/stdlib only.

Default is source/hash/JSON metadata only. Actual 608/contact/response arrays
are read only with --run after Root terminal authorization. Generated rows are
never force-filtered; no new world force or COM torque is calculated.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MATH=ROOT/'src/full151_gradient/contact_regime_saved_independent_math_v1.py'
MATH_SHA='f14405e569323000769290b513e09c742539ca6bda156c6d486505f591ff7e8a'
PROTOCOL_SHA='ffbfcaca6e5b0950e9f70ae3aa457fadc69be16c34a2412cb4a6aa5de374baa7'
CORE_SHA='f5ac0417125759b859516de743d0c3b6ad6c2f0acd2f6361e6f6ec396b96cd0e'
PRODUCER_READY=ROOT/'outputs/full151105_contact_regime608_preparation_CPU_v1/READY.json'
PHASES=(0,49,99,149)


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def pin(p):return {'path':str(Path(p).resolve(strict=True)),'sha256':sha(p)}


def checked(p):
    if type(p) is not dict or set(p)!={'path','sha256'} or not Path(p['path']).is_absolute():raise ValueError('Exact absolute pin')
    q=Path(p['path']).resolve(strict=True)
    if sha(q)!=p['sha256']:raise ValueError('Pin mismatch '+str(q))
    return q


def same(a,b):return a['sha256']==b['sha256'] and checked(a).samefile(checked(b))


def dump(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as f:return [json.loads(line) for line in f if line.strip()]


def bits(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def strict_json(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,dict):return set(a)==set(b) and all(strict_json(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(strict_json(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return np.float64(a).tobytes()==np.float64(b).tobytes()
    return a==b


def independent_math():
    if sha(MATH)!=MATH_SHA:raise ValueError('Frozen independent math changed')
    spec=importlib.util.spec_from_file_location('independent_contact_regime',MATH);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def positive_review(r,expected):
    return type(r.get('passed')) is int and r['passed']==r.get('total')==expected==len(r.get('checks',{})) and all(v is True for v in r['checks'].values())


def metadata(protocol_path):
    """No NPZ or physical JSONL read; do not inspect a future run directory."""
    if sha(protocol_path)!=PROTOCOL_SHA:raise ValueError('Frozen contact-regime prereg mismatch')
    p=read(protocol_path)
    if p['schema']!='full151105_saved_contact_regime608_preregistration:1' or p['execution_authorized'] is not False:raise ValueError('Source-only prereg scope')
    for value in [*p['inputs'].values(),*p['source_pins'].values()]:checked(value)
    old=read(checked(p['inputs']['response_saved_independent22']))
    dense=read(checked(p['inputs']['dense_saved_independent35']))
    physical=read(checked(p['inputs']['physical_saved_independent1762']))
    if not(positive_review(old,22) and positive_review(dense,35) and positive_review(physical,1762)):raise ValueError('Actual parent saved reviews required')
    for r,prefix in [(old,'response'),(dense,'dense'),(physical,'physical')]:
        if not(same(r['config'],p['inputs'][prefix+'_config']) and same(r['producer_report'],p['inputs'][prefix+'_report']) and Path(r['run_dir']).samefile(checked(p['inputs'][prefix+'_report']).parent)):raise ValueError('Parent source identity '+prefix)
    if p['source_pins']['independent_contact_math']['sha256']!=MATH_SHA:raise ValueError('Independent math binding')
    ready=read(checked(p['inputs']['independent_math_READY']));proof=read(checked(ready['cpu_report']))
    if not positive_review(proof,7) or proof['actual_full608_or_contact_longtable_reads']!=0 or proof['new_world_force_COM_moment_calculations']!=0:raise ValueError('Synthetic source scope')
    tested=read(PRODUCER_READY);testreport=read(checked(tested['cpu_report']))
    if not positive_review(testreport,9) or tested['source_pins']!=testreport['source_pins'] or tested['source_pins']['consumer']['sha256']!=CORE_SHA or not same(tested['protocol'],pin(protocol_path)):raise ValueError('Current producer9 source proof')
    for value in tested['source_pins'].values():checked(value)
    for key in ('actual_saved_contact_regime_runs','actual_physical_samples_read','actual_NPZ_values_decoded','new_model_FW_AD_MJB_FK_Env_GPU_physics'):
        if testreport[key]!=0:raise ValueError('Synthetic proof scope')
    return p


def summaries(rows,math):
    return math.generated_summary(list(rows))


def independent_contact_rows(samples,branch_index,response_raw,response_metadata,catalog,math):
    """Future actual adapter calculation; not called by default metadata mode."""
    if len(samples)!=1808 or len(branch_index)!=152 or len(response_metadata)!=152:raise ValueError('Complete original1808/152 source')
    sets=math.catalog_sets(catalog['native_ngeom'],catalog['native_object_body_id'],catalog['native_geom_catalog'])
    if sets['hand']!={x['id'] for x in catalog['native_hand_geoms']} or sets['object']!={x['id'] for x in catalog['native_object_geoms']}:raise ValueError('Catalog explicit/native rule sets differ')
    selected=[];allcontacts=[];seen=set()
    for source_index,s in enumerate(samples):
        if s['branch_id'] is None:continue
        bid=math.integer(s['branch_id'],'branch ID',0);sub=math.integer(s['substep'],'substep',1)
        if bid>=152 or sub>4 or (bid,sub) in seen:raise ValueError('Duplicate/range608 row')
        seen.add((bid,sub));sid,offset=divmod(bid,19);path=('A0','A2')[sid//4];k=PHASES[sid%4];mode='baseline' if offset<3 else 'direction'
        branch,rm=branch_index[bid],response_metadata[bid]
        if any(s[key]!=val for key,val in [('path',path),('phase',k),('mode',mode),('source_frame',200+k),('canonical_pre_tick',4*k+sub-1),('canonical_post_tick',4*k+sub)]):raise ValueError('Sample phase/order binding')
        if any(branch.get(key)!=val for key,val in [('row_id',bid),('snapshot_id',sid),('path',path),('k',k),('type',mode)]):raise ValueError('Original branch source identity')
        if any(rm.get(key)!=val for key,val in [('row_id',bid),('snapshot_id',sid),('path',path),('k',k),('type',mode)]):raise ValueError('Response branch source identity')
        if not bits(np.asarray(s['ctrl'],dtype='f4'),response_raw['actual_u152_FP32'][sid,offset]):raise ValueError('All4 held control bytes vs response source')
        n=math.integer(s['nacon'],'nacon',0);nefc=math.integer(s['nefc'],'nefc',0);contacts=s['contacts'];raw=s['raw_wp_contact']
        if len(contacts)!=n or set(raw)!={'geom','type','worldid','efc_address'} or any(len(v)!=n for v in raw.values()):raise ValueError('All original contact rows/counts mandatory')
        math.finite(s['raw_efc_force'],(nefc,),'raw efc force');et=np.asarray(s['raw_efc_type'])
        if et.shape!=(nefc,) or (nefc and et.dtype.kind not in 'iu'):raise ValueError('Raw efc type array')
        # Empty [] is accepted only with nacon0; malformed nonempty shapes fail.
        for value,shape,label in [(s['object_solver_pose'],(7,),'solver pose'),(s['object_solver_COM'],(3,),'solver COM'),
            (s['hand_object_force_on_object_w_N'],(3,),'SOURCE world HO force'),(s['hand_object_moment_about_COM_w_Nm'],(3,),'SOURCE COM moment')]:math.finite(value,shape,label)
        if s['direct_torque_included_once'] is not True:raise ValueError('Original directtorque scope')
        current=[]
        for ci,c in enumerate(contacts):
            wp={key:raw[key][ci] for key in raw};r=math.inspect_contact(ci,c,wp,nefc,sets)
            r.update(sample_id=len(selected),source_physical_row_index=source_index,branch_id=bid,snapshot_id=sid,path=path,k=k,substep=sub,
                mode=mode,original_contact_position_world_m=c['pos'],original_contact_frame_rows=c['frame'],
                native_geom_types=[sets['catalog'][g]['type'] for g in r['ordered_geom_pair']],
                source_helper_active_by_original_formula=bool(c['efc_address']>=0 and c['force6'][0]>math.FN_THRESHOLD_N))
            del r['solver_phase']
            current.append(r);allcontacts.append(r)
        selected.append({'sample_id':len(selected),'source_physical_row_index':source_index,'branch_id':bid,'snapshot_id':sid,'path':path,'k':k,'mode':mode,
            'substep':sub,'canonical_pre_tick':s['canonical_pre_tick'],'canonical_post_tick':s['canonical_post_tick'],'summary':summaries(current,math),
            'original_contact_count':n,'source_sample':s,'contact_start':len(allcontacts)-len(current),'contact_count':len(current)})
    expected=[(b,ss) for b in range(152) for ss in range(1,5)]
    if [(r['branch_id'],r['substep']) for r in selected]!=expected or len(seen)!=608:raise ValueError('All608 canonical branch/substep rows required, no subset')
    return selected,allcontacts,sets


def independent_pair_descriptors(raw,pairs,math):
    if len(pairs)!=64:raise ValueError('All64 source epsilon pairs')
    result=[]
    for s in range(8):
        for j,q in enumerate((0,4,8,12)):
            for ei,eps in enumerate((.002,.001)):
                i=len(result);source=pairs[i];off=3+8*ei+2*j
                for key,value in [('pair_id',i),('snapshot_id',s),('j',j),('joint_index',q),('epsilon_index',ei),('requested_epsilon_rad',eps),('minus_branch_id',s*19+off),('plus_branch_id',s*19+off+1)]:
                    if type(source.get(key)) is not type(value) or source[key]!=value:raise ValueError('Original pair order metadata')
                d=math.zero_descriptors(raw['actual_u152_FP32'][s,off],raw['actual_u152_FP32'][s,off+1],raw['target_y152_FP32'][s,off],raw['target_y152_FP32'][s,off+1],
                    raw['predicted_y152_FP32'][s,off],raw['predicted_y152_FP32'][s,off+1],q)
                result.append({'pair_id':i,'source_pair':source,'descriptors':d,'minus_sample_ids':[4*(s*19+off)+z for z in range(4)],'plus_sample_ids':[4*(s*19+off+1)+z for z in range(4)]})
    return result


def clean(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    return v


def encoded(a,b):
    if a.dtype!=np.float32 or b.dtype!=np.float32 or a.shape!=(58,) or b.shape!=(58,):raise ValueError('Original state58 FP32 encodings')
    zero=a.astype('f8')-b.astype('f8')==0;be=a.view('u4')==b.view('u4')
    return {'numeric_zero58':zero.tolist(),'bit_equal58':be.tolist(),'p_all_numeric_zero':bool(zero[40:43].all()),'R_all_numeric_zero':bool(zero[43:52].all()),
       'p_all_bit_equal':bool(be[40:43].all()),'R_all_bit_equal':bool(be[43:52].all())}


def profiles(samples):
    result={}
    for cat in ('all','hand_object','hand_self','other','object_nonhand','hand_other'):
        result[cat]={'generated_count_by_substep':[sum(s['categories'][cat]['generated_candidate_count'] for s in samples if s['substep']==i) for i in range(1,5)],
          'observed_candidate_depth_by_substep_m':[max([0.]+[s['categories'][cat]['observed_candidate_depth_m'] for s in samples if s['substep']==i]) for i in range(1,5)],
          'secondary_positive_force_count_by_substep':[sum(s['categories'][cat]['secondary_positive_force_count'] for s in samples if s['substep']==i) for i in range(1,5)],
          'global_clearance':'UNKNOWN'}
    return result


def expected_tables(ind_samples,contacts,branches,snapshots,response_metadata,raw,dec,pairs,math):
    if len(snapshots)!=8:raise ValueError('All8 original snapshots')
    pair_data=independent_pair_descriptors(raw,pairs,math);pair_table=[];links={i:[] for i in range(152)}
    for item in pair_data:
        src,d=item['source_pair'],item['descriptors'];s=src['snapshot_id'];mi,pi=src['minus_branch_id'],src['plus_branch_id']
        row={**src,'actual_delta_u20_rad_F64':d['actual_delta_u20_rad_F64'].tolist(),'zero_actual_coordinate_delta':d['zero_actual_coordinate_delta'],
          'zero_actual_all20_delta':d['zero_actual_all20_delta'],'control_endpoints_all_bits_equal':d['control_endpoints_all_bits_equal'],
          'both_absolute_controls_all_numeric_zero':d['both_absolute_controls_all_numeric_zero'],
          'physical_encoding':encoded(raw['target_y152_FP32'][s,pi%19],raw['target_y152_FP32'][s,mi%19]),
          'model_encoding':encoded(raw['predicted_y152_FP32'][s,pi%19],raw['predicted_y152_FP32'][s,mi%19]),
          'minus_sample_ids':item['minus_sample_ids'],'plus_sample_ids':item['plus_sample_ids']}
        # Reconcile explicit independent58 zero masks with the table projection.
        for label,key in [('physical','target_y152_FP32'),('model','predicted_y152_FP32')]:
            if row[label+'_encoding']['numeric_zero58']!=d[label+'_encoded_numeric_zero58'].tolist() or row[label+'_encoding']['bit_equal58']!=d[label+'_endpoint_bits_equal58'].tolist():raise ValueError('Independent encoded projection')
        pair_table.append(row);links[mi].append(src['pair_id']);links[pi].append(src['pair_id'])
    response=[]
    for bid,m in enumerate(response_metadata):
        s,off=divmod(bid,19)
        for k in ('row_id','snapshot_id','path','k','type','j','joint_index','epsilon_index','epsilon_rad','sign','baseline_repeat'):
            if type(m[k]) is not type(branches[bid][k]) or m[k]!=branches[bid][k]:raise ValueError('Original branch/response fields')
        if off<3:links[bid]=[r['pair_id'] for r in pairs if r['snapshot_id']==s]
        if snapshots[s]['path']!=m['path'] or snapshots[s]['k']!=m['k']:raise ValueError('Snapshot source join')
        u=raw['actual_u152_FP32'][s,off];u0=raw['original_snapshot_u_FP32'][s]
        du=u.astype('f8')-u0.astype('f8')
        pr=raw['predicted_y152_FP32'][s,off].astype('f8')-dec['baseline_predicted_mean_F64'][s]
        tr=raw['target_y152_FP32'][s,off].astype('f8')-dec['baseline_target_mean_F64'][s]
        if off>=3 and not(np.array_equal(pr,dec['centered_model_response_F64'][s,off-3]) and np.array_equal(tr,dec['centered_physical_response_F64'][s,off-3])):raise ValueError('Original response22 centered arrays')
        response.append({**m,'actual_u20_FP32':u.tolist(),'actual_control_offset_from_original_u0_rad_F64':du.tolist(),
          'zero_control_change_all20_numeric':bool((du==0).all()),'control_same_bits_original_u0':bits(u,u0),'absolute_control_all_numeric_zero':bool((u==0).all()),
          'centered_physical_response58_F64':tr.tolist(),'centered_model_response58_F64':pr.tolist(),
          'physical_p_centered_all_numeric_zero':bool((tr[40:43]==0).all()),'physical_R_centered_all_numeric_zero':bool((tr[43:52]==0).all()),
          'model_p_centered_all_numeric_zero':bool((pr[40:43]==0).all()),'model_R_centered_all_numeric_zero':bool((pr[43:52]==0).all()),
          'physical_endpoint_vs_first_baseline_encoding':encoded(raw['target_y152_FP32'][s,off],raw['target_y152_FP32'][s,0]),
          'model_endpoint_vs_first_baseline_encoding':encoded(raw['predicted_y152_FP32'][s,off],raw['predicted_y152_FP32'][s,0]),'related_pair_ids':links[bid]})
    sample_table=[]
    copy_fields=('pre_time','post_time','pre_qpos','pre_qvel','post_qpos','post_qvel','canonical_pre_tick','canonical_post_tick','source_frame','phase_contract',
       'measured_hand_object','hand_object_force_on_object_w_N','hand_object_moment_about_COM_w_Nm','direct_torque_included_once','object_solver_COM','object_solver_pose','actuator_force','external_force_max')
    for item in ind_samples:
        s=item['source_sample'];ident={k:item[k] for k in ('sample_id','source_physical_row_index','branch_id','snapshot_id','path','k','mode','substep')}
        sample_table.append({**ident,**{k:s[k] for k in copy_fields},'ctrl20_FP32':s['ctrl'],'nacon':s['nacon'],'nefc':s['nefc'],
          'contact_long_start':item['contact_start'],'contact_long_count':item['contact_count'],'categories':item['summary'],'related_pair_ids':links[item['branch_id']],
          'source_wrench_scope':'copied original aggregate; no new world force or COM torque arithmetic','independent_pyramid_decode_qualified':False})
    branch_table=[]
    for bid,b in enumerate(branches):
        ss=sample_table[4*bid:4*bid+4]
        cr=[r for r in contacts if r['branch_id']==bid]
        branch_table.append({'branch_id':bid,'original_branch':b,'response_descriptor':response[bid],'sample_ids':[s['sample_id'] for s in ss],
          'related_pair_ids':links[bid],'category_four_substep_profile':profiles(ss),'categories_over_four_PRE':summaries(cr,math)})
    snapshot_table=[]
    for sid,source in enumerate(snapshots):
        ss=sample_table[sid*76:(sid+1)*76]
        snapshot_table.append({'snapshot_id':sid,'path':source['path'],'k':source['k'],'branch_ids':list(range(sid*19,(sid+1)*19)),
          'sample_ids':[r['sample_id'] for r in ss],'related_pair_ids':[r['pair_id'] for r in pairs if r['snapshot_id']==sid],
          'category_four_substep_profile_summed_over_19_branches':profiles(ss),
          'profile_scope':'19 separate restored intervals; not a continuous 76-step trajectory'})
    return {'sample608':sample_table,'contact_long':contacts,'branch152':branch_table,'snapshot8':snapshot_table,'response_pairs64':pair_table,'response_branch152':response}


def CSV_matches(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);got=list(reader)
    if set(reader.fieldnames or [])!=set(keys) or len(got)!=len(rows):return False
    for actual,wanted in zip(got,rows):
        for key,value in wanted.items():
            if isinstance(value,(dict,list)):
                if not strict_json(json.loads(actual[key]),value):return False
            elif actual[key]!=('' if value is None else str(value)):return False
    return True


def audit(protocol_path,run,out):
    out.mkdir(exist_ok=False);auditor=pin(__file__)
    (out/'executed_auditor_source.py').write_bytes(Path(__file__).read_bytes())
    dump(out/'equations_before_actual_regime_read.json',{'auditor':auditor,'math':pin(MATH),'protocol':pin(protocol_path),'actual_regime_rows_read':False,
       'rules':'All608/ci/152/8/64, generated before force eligibility, original type&1/raw slots/strictFn; no new wrench/COM/state/model.'})
    proto=metadata(protocol_path);inputs=proto['inputs'];math=independent_math();checks={};details={}
    def gate(k,v):checks[k]=bool(v)
    r=read(run/'report.json');inventory=read(run/'source_inventory.json');parents=read(run/'parent_source_map_references.json')
    gate('protocol_role_input_source_semantic_snapshots',same(r['protocol'],pin(protocol_path)) and r['role']==proto['role']
       and strict_json(proto,read(run/'protocol.json')) and strict_json(inputs,r['input_pins']) and strict_json(inputs,read(run/'input_pins_before.json')) and strict_json(inputs,read(run/'input_pins_after.json')))
    ownready=read(PRODUCER_READY);proof=read(checked(ownready['cpu_report']))
    gate('current_producer9_frozen_proof_and_independent7_separate_scope',same(r['cpu_ready'],pin(PRODUCER_READY)) and same(r['cpu_report'],ownready['cpu_report']) and positive_review(proof,9)
       and r['source_pins']['consumer']['sha256']==CORE_SHA and same(r['source_pins']['consumer'],ownready['source_pins']['consumer']))
    expected_sources=[*r['source_pins'].values()];pm={}
    for alias in ('physical_config','dense_config','response_config'):
        pm[alias]={'config':inputs[alias],'source_map_field':'source_pins'};expected_sources.extend(read(checked(inputs[alias]))['source_pins'].values())
    def token(p):
        path=checked(p);st=path.stat();return (st.st_dev,st.st_ino,p['sha256'])
    want={token(p) for p in expected_sources};actual=[token(p) for p in inventory]
    gate('all_unique_actualfile_source_closure_parent_maps_and_current_SHA',len(actual)==len(set(actual))==r['source_unique_file_count'] and set(actual)==want and strict_json(pm,parents))
    with zipfile.ZipFile(run/'source_archive.zip') as z:
        zipok=len(z.namelist())==len(inventory) and all(hashlib.sha256(z.read('%04d_%s'%(i,Path(p['path']).name))).hexdigest()==p['sha256'] for i,p in enumerate(inventory))
    gate('all_current_immutable_sources_and_archive_exact',zipok)
    outputs=read(run/'output_pins.json')
    for p in outputs.values():checked(p)
    gate('all_original_output_pins_exact',True)
    for alias,keys in [('dense_output_pins',('geometry_catalog',)),('response_output_pins',('response_raw152','response_metadata152','response_decomposition','response_pairs64','response_guard_summary'))]:
        manifest=read(checked(inputs[alias]));gate(alias+'_actual_aliases',all(any(same(inputs[k],p) for p in manifest.values()) for k in keys))
    sourceguard=read(checked(inputs['response_guard_summary']))
    gate('all_original22_visible_control_guards_remain_strict_pass',all(v['all_passed'] is True and v['failed_elements']==0 and v['first_failed_indices']==[] for v in sourceguard.values()))
    catalog=read(checked(inputs['geometry_catalog']));branches=read(checked(inputs['branch_index']));snapshots=read(checked(inputs['snapshot_records']))
    raw=load(checked(inputs['response_raw152']));dec=load(checked(inputs['response_decomposition']));pairs=read(checked(inputs['response_pairs64']));rm=read(checked(inputs['response_metadata152']))
    source_samples=jsonl(checked(inputs['physical_samples']))
    ind_samples,contacts,sets=independent_contact_rows(source_samples,branches,raw,rm,catalog,math)
    tables=expected_tables(ind_samples,contacts,branches,snapshots,rm,raw,dec,pairs,math)
    actual_tables={name:(jsonl(run/(name+'.jsonl')) if name=='contact_long' else read(run/(name+'.json'))) for name in tables}
    gate('all608_original_sourceindex_phase_ctrl4steps_and_rawsample_fields',strict_json(tables['sample608'],actual_tables['sample608']))
    gate('every_original_contactindex_nativeID_category_rawforce_slots_masks',strict_json(tables['contact_long'],actual_tables['contact_long']))
    gate('all152_branch_fourstep_profiles_categories_and_response_join',strict_json(tables['branch152'],actual_tables['branch152']))
    gate('all8_snapshot19branch_profiles_complete_not_trajectories',strict_json(tables['snapshot8'],actual_tables['snapshot8']))
    gate('all64_actualdelta_encodedflat_pair_records_and_sample_links',strict_json(tables['response_pairs64'],actual_tables['response_pairs64']))
    gate('all152_zero_control_centered_and_rawencoding_descriptors',strict_json(tables['response_branch152'],actual_tables['response_branch152']))
    gate('all_five_CSV_tables_semantically_exact_complete',all(CSV_matches(run/(k+'.csv'),v) for k,v in tables.items() if k!='response_branch152'))
    gate('catalog73_nativegeom31body_hand21_object3_source_copied',strict_json(catalog,read(run/'geometry_catalog_source.json')) and len(sets['catalog'])==73 and len(sets['hand'])==21 and len(sets['object'])==3
       and len(catalog['native_body_catalog'])==31 and r['catalog_counts']=={'native_ngeom':73,'native_nbody':31,'enabled_hand':21,'object':3})
    helper=True;rawcopy=True
    for expected,actual in zip(ind_samples,actual_tables['sample608']):
        src=expected['source_sample'];rs=contacts[expected['contact_start']:expected['contact_start']+expected['contact_count']];ho=[x for x in rs if x['category']=='hand_object']
        helper&=type(src['measured_hand_object']['active']) is list and src['measured_hand_object']['active']==[x['source_helper_active_by_original_formula'] for x in ho]
        for k in ('pre_qpos','pre_qvel','post_qpos','post_qvel','actuator_force','object_solver_COM','object_solver_pose','hand_object_force_on_object_w_N','hand_object_moment_about_COM_w_Nm'):
            rawcopy&=bits(np.asarray(src[k],dtype='f8'),np.asarray(actual[k],dtype='f8'))
        rawcopy&=src['external_force_max']==actual['external_force_max']==0 and src['direct_torque_included_once'] is actual['direct_torque_included_once'] is True
    gate('source_helper_HO_active_order_retained_not_raw_forcevalidity',helper)
    gate('original_worldwrench_COM_pose_rawstates_actuator_force_copied_bits_no_recompute',rawcopy)
    count={'physical_source_rows':1808,'sample608':608,'contact_long':len(contacts),'branch152':152,'snapshot8':8,'response_pairs64':64,'baseline_PRE_rows':96,'direction_PRE_rows':512}
    gate('all608_152_8_64_original_counts_and_original_contact_sum',r['counts']==count and len(contacts)==sum(s['nacon'] for s in source_samples if s['branch_id'] is not None))
    gate('source_runtime_costscope_and_no_new_qualifications',r['schema']=='full151105_saved_contact_regime608_CPU_analysis:1' and r['status']==proto['outputs']['success_status']
       and r['new_model_FW_AD_MJB_FK_Env_GPU_physics']==r['new_world_force_COM_torque_calculations']==0
       and r['invariants']=={'input_source_hashes_unchanged':True} and all(r[k] is False for k in proto['outputs']['flags']))
    for p in [*inventory,*inputs.values(),*outputs.values()]:checked(p)
    gate('source_inputs_outputs_and_frozen_independent_math_unchanged',pin(__file__)==auditor and sha(MATH)==MATH_SHA)
    gate('reviewer_no_Torch_native_model_MJB_FK_or_physics',not any(k in sys.modules for k in ('torch','warp','mujoco','mujoco_warp','dex_supermimic','mjlab','scipy')))
    details.update(generated_categories={c:sum(x['category']==c for x in contacts) for c in ('hand_object','hand_self','other')},
       force_eligible_contacts=sum(x['force_evidence_eligible'] for x in contacts),secondary_positive_force_contacts=sum(x['secondary_positive_force'] for x in contacts),
       force_ineligible_contacts=sum(not x['force_evidence_eligible'] for x in contacts),valid_empty_candidate_samples=sum(x['original_contact_count']==0 for x in ind_samples),
       global_clearance='UNKNOWN for all sample/categories including absent and positive-distance candidates',
       raw_wrench_scope='Original sample source aggregate bytes copied; no new rotation, force sum or COM moment performed.')
    result={'schema':'full151105_contact_regime608_saved_independent:1','status':'PASS_SAVED_ALL608_CONTACT_REGIME_NOT_PHYSICAL_CERTIFICATE' if all(checks.values()) else 'FAILED_SAVED_CONTACT_REGIME_REVIEW',
       'engineering_qualified':all(checks.values()),'passed':sum(checks.values()),'total':len(checks),'checks':checks,'auditor':auditor,'math':pin(MATH),'protocol':pin(protocol_path),
       'producer_report':pin(run/'report.json'),'run_dir':str(run.resolve()),'counts':count,'source_unique_file_count':len(inventory),'details':details,
       **proto['outputs']['flags'],'limitations':['All608 and all data-dependent contact rows retained before force qualification; no response, old9, force or depth selection.',
          'Original indices are sample-local observations, not a persistent contact feature or full geometry/clearance certificate.',
          'Secondary positive Fn uses source decoded local force plus original raw type/world/slots; no contact-dimension or pyramid re-decode.',
          'Source helper activity and source net world wrench/COM moment retained; no new world force or torque calculation.',
          'Zero executed control difference, absolute zero action, numeric response zeros and bitwise endpoint identity remain distinct.',
          'All fixed-visible58 responses remain dependent saved CPU-shadow descriptors, not causal/physical-gradient or whole-hidden-state qualification.'],
       'new_models_FW_AD_MJB_FK_Env_physics_GPU':0,'new_worldforce_COMmoment_calculations':0}
    dump(out/'report.json',result);print(json.dumps({'report':pin(out/'report.json'),'passed':result['passed'],'total':result['total'],'counts':count,'details':details},indent=2))
    return 0 if all(checks.values()) else 2


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--run-dir',type=Path);p.add_argument('--output',type=Path);p.add_argument('--run',action='store_true');a=p.parse_args()
    proto=metadata(a.protocol)
    if not a.run:
        print(json.dumps({'status':'PASS_SOURCE_HASH_JSON_METADATA_ONLY_ADAPTER','auditor':pin(__file__),'math':pin(MATH),'protocol':pin(a.protocol),
            'source_inputs':len(proto['inputs']),'actual608_or_contact_rows_read':0,'NPZ_opened':0,'model_FW_AD_MJB_Env_physics_GPU':0},indent=2));return 0
    if a.run_dir is None or a.output is None:raise ValueError('Actual audit requires explicit source run and fresh review output')
    try:return audit(a.protocol,a.run_dir,a.output)
    except Exception as e:
        if a.output.exists() and not (a.output/'failure.json').exists():dump(a.output/'failure.json',{'error':repr(e),'auditor':pin(__file__),'saved_only':True,'original_data_unchanged':True})
        raise


if __name__=='__main__':raise SystemExit(main())
