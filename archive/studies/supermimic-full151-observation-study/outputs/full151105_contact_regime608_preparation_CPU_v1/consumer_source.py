"""Saved generated-contact bookkeeping only. No native/model/geometry execution.

Every branch PRE and original contact index survives classification, including
empty and force-ineligible records. Stored world wrenches are copied, never
recomputed. A generated-candidate gap is not a global clearance certificate.
"""
from __future__ import annotations
import csv, hashlib, json, re, sys, time, zipfile
from pathlib import Path
import numpy as np

ROOT=Path('D:/workspace/supermimic-full151-observation-study')
PREREG=ROOT/'protocols/full151105_saved_contact_regime608_CPU_prereg_v1.json'
PREREG_SHA='ffbfcaca6e5b0950e9f70ae3aa457fadc69be16c34a2412cb4a6aa5de374baa7'
OWN={'consumer':ROOT/'src/full151_gradient/contact_regime608105_v1.py',
     'CLI':ROOT/'scripts/diagnose_full151_contact_regime608_saved_cpu_v1.py',
     'checker':ROOT/'scripts/check_full151_contact_regime608_saved_cpu_v1.py'}
PREP=ROOT/'outputs/full151105_contact_regime608_preparation_CPU_v1'
ROLE='all608_saved_branch_PRE_contact_regime'
HAND=r'robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)'
CATEGORIES=('all','hand_object','hand_self','other','object_nonhand','hand_other')
FLAGS={k:False for k in ('physical_gradient_qualified','HO_force_gradient_qualified','whole_hidden_state_qualified','global_clearance_qualified','training_causal_effect_qualified')}

def require(v,msg):
    if not v: raise ValueError(msg)

def integer(v,name,minimum=None):
    require(not isinstance(v,(bool,np.bool_)) and isinstance(v,(int,np.integer)),name+' strict integer')
    v=int(v);require(minimum is None or v>=minimum,name+' range');return v

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def pin(p):
    p=Path(p).resolve(strict=True);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return {'path':str(p),'sha256':h.hexdigest()}

def checked(v):
    require(type(v) is dict and set(v)=={'path','sha256'} and type(v['path']) is str and Path(v['path']).is_absolute(),'absolute exact pin')
    require(type(v['sha256']) is str and re.fullmatch('[0-9a-f]{64}',v['sha256']) is not None,'SHA256')
    p=Path(v['path']).resolve(strict=True);require(pin(p)['sha256']==v['sha256'],'Changed pin: '+str(p));return p

def same(a,b):return a['sha256']==b['sha256'] and checked(a).samefile(checked(b))

def canonical_sources(values):
    result={};paths={}
    for value in values:
        literal=(value['path'],value['sha256'])
        if literal in paths:continue
        path=checked(value);st=path.stat();key=(st.st_dev,st.st_ino,value['sha256'])
        if key in result:require(path.samefile(result[key]['path']),'inode/source identity')
        else:result[key]=value
        paths[literal]=path
    return list(result.values())

def clean(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    return v

def write(out,name,v):
    with (out/name).open('x',encoding='utf-8') as f:json.dump(clean(v),f,indent=2,allow_nan=False)

def tables(out,name,rows,jsonl=False):
    if jsonl:
        with (out/(name+'.jsonl')).open('x',encoding='utf-8') as f:
            for r in rows:f.write(json.dumps(clean(r),allow_nan=False)+'\n')
    else:write(out,name+'.json',rows)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (out/(name+'.csv')).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(clean(v),allow_nan=False) if isinstance(v,(dict,list,tuple,np.ndarray)) else v for k,v in r.items()})

def finite(v,shape,label):
    a=np.asarray(v);require(a.shape==shape and a.dtype.kind in 'fiu' and bool(np.isfinite(a).all()),'finite '+label+' '+str(shape));return a

def bits_equal(a,b):return bool(np.array_equal(np.asarray(a).view(np.uint32),np.asarray(b).view(np.uint32)))

def catalog_sets(c):
    n=integer(c['native_ngeom'],'ngeom',1);obj=integer(c['native_object_body_id'],'object body',0)
    table={};hand=set();objects=set()
    for g in c['native_geom_catalog']:
        i=integer(g['id'],'geom ID',0);require(i<n and i not in table,'unique native geom');table[i]=g
        require(type(g['name']) is str,'geom name')
        if re.fullmatch(HAND,g['name']) and (integer(g['contype'],'contype',0) or integer(g['conaffinity'],'conaffinity',0)):hand.add(i)
        if integer(g['native_body_id'],'body ID',0)==obj:objects.add(i)
    require(set(table)==set(range(n)) and hand and objects and not hand&objects,'complete native catalog')
    return table,hand,objects

def inspect_contact(ci,c,r,nefc,sets):
    table,hand,obj=sets;ci=integer(ci,'original contact index',0);nefc=integer(nefc,'nefc',0)
    pair=np.asarray(c['geom']);rp=np.asarray(r['geom'])
    require(pair.shape==(2,) and pair.dtype.kind in 'iu' and rp.shape==(2,) and rp.dtype.kind in 'iu' and np.array_equal(pair,rp),'ordered CPU/raw native geom pair')
    a,b=map(int,pair);require(a in table and b in table,'native geom range')
    hh=a in hand and b in hand;hasH=a in hand or b in hand;hasO=a in obj or b in obj
    cat='hand_object' if hasH and hasO else 'hand_self' if hh else 'other'
    typ=integer(r['type'],'raw contact type',0);world=integer(r['worldid'],'raw world',0);host=integer(c['efc_address'],'host address')
    slots=np.asarray(r['efc_address']);require(slots.ndim==1 and slots.size>0 and slots.dtype.kind in 'iu','original integer raw slots')
    legal=(slots==-1)|((slots>=0)&(slots<nefc));active=slots>=0
    f=finite(c['force6'],(6,),'original decoded force6');pos=finite(c['pos'],(3,),'contact pos');frame=finite(c['frame'],(3,3),'contact frame')
    d=float(c['dist']);require(np.isfinite(d),'finite generated distance')
    reasons=[]
    if not typ&1:reasons.append('NO_CONSTRAINT_TYPE_BIT')
    if world!=0:reasons.append('NONZERO_WORLD')
    if not active.any():reasons.append('NO_ACTIVE_RAW_EFC_SLOT')
    if not legal.all():reasons.append('RAW_EFC_SLOT_OUT_OF_RANGE')
    if host<0:reasons.append('NEGATIVE_HOST_EFC_ADDRESS')
    eligible=not reasons
    return {'original_contact_index':ci,'ordered_geom_pair':[a,b],'category':cat,'object_nonhand':hasO and not hasH,
      'hand_other':hasH and not hasO and not hh,'both_object':a in obj and b in obj,
      'native_geom_names':[table[x]['name'] for x in (a,b)],'native_body_ids':[table[x]['native_body_id'] for x in (a,b)],
      'native_geom_types':[table[x]['type'] for x in (a,b)],'generated_candidate':True,
      'dist_m':d,'observed_candidate_depth_m':max(0.,-d),'original_contact_position_world_m':pos.tolist(),'original_contact_frame_rows':frame.tolist(),
      'raw_WP_type':typ,'constraint_type_bit':bool(typ&1),'sensor_type_bit':bool(typ&2),'raw_worldid':world,
      'raw_efc_slots':slots.tolist(),'raw_efc_slot_legal_mask':legal.tolist(),'raw_active_slot_mask':active.tolist(),
      'host_efc_address_reordered':host,'decoded_normal_force_N':float(f[0]),'decoded_force6_contact_coordinates':f.tolist(),
      'source_helper_active_by_original_formula':bool(host>=0 and f[0]>1e-6),
      'force_evidence_eligible':eligible,'secondary_positive_force':bool(eligible and f[0]>1e-6),
      'force_ineligible_reasons':reasons,'independent_pyramid_decode_qualified':False,'global_clearance':'UNKNOWN'}

def summary(rows):
    rows=list(rows);result={}
    for cat in CATEGORIES:
        rs=[r for r in rows if cat=='all' or r['category']==cat or (cat in ('object_nonhand','hand_other') and r[cat])]
        ds=[r['dist_m'] for r in rs]
        result[cat]={'generated_candidate_count':len(rs),'minimum_generated_distance_m':min(ds) if ds else None,
          'observed_candidate_depth_m':max([0.]+[-x for x in ds]),'generated_candidate_absent':not bool(rs),
          'secondary_positive_force_count':sum(r['secondary_positive_force'] for r in rs),'global_clearance':'UNKNOWN'}
    return result

def sample_contacts(sample,sets):
    n=integer(sample['nacon'],'nacon',0);ne=integer(sample['nefc'],'nefc',0);cs=sample['contacts'];rw=sample['raw_wp_contact']
    require(type(cs) is list and len(cs)==n,'contacts count')
    finite(sample['raw_efc_force'],(ne,),'raw efc force');ef=np.asarray(sample['raw_efc_type']);require(ef.shape==(ne,) and (ne==0 or ef.dtype.kind in 'iu'),'raw efc type')
    if n==0:
        require(all(isinstance(rw[k],list) and rw[k]==[] for k in ('geom','type','worldid','efc_address')),'known empty raw contact arrays');return []
    arrays={k:np.asarray(rw[k]) for k in ('geom','type','worldid','efc_address')}
    require(arrays['geom'].shape==(n,2) and arrays['type'].shape==arrays['worldid'].shape==(n,) and arrays['efc_address'].ndim==2 and arrays['efc_address'].shape[0]==n and arrays['efc_address'].shape[1]>0,'raw contact shapes')
    require(all(a.dtype.kind in 'iu' for a in arrays.values()),'integer raw contact arrays')
    return [inspect_contact(ci,c,{k:a[ci] for k,a in arrays.items()},ne,sets) for ci,c in enumerate(cs)]

def encoded(a,b):
    a=finite(a,(58,),'original endpoint');b=finite(b,(58,),'original endpoint');require(a.dtype==b.dtype==np.float32,'FP32 endpoints')
    numeric=(a.astype('f8')-b.astype('f8'))==0;be=a.view('u4')==b.view('u4')
    return {'numeric_zero58':numeric.tolist(),'bit_equal58':be.tolist(),
       'p_all_numeric_zero':bool(numeric[40:43].all()),'R_all_numeric_zero':bool(numeric[43:52].all()),
       'p_all_bit_equal':bool(be[40:43].all()),'R_all_bit_equal':bool(be[43:52].all())}

def declaration():return {'role':ROLE,'default':'JSON/hash only; no output or saved array/contact derivation','actual_model_FW_AD_MJB_FK_Env_GPU_physics':0,'CLI':'--protocol PATH [--output FRESH] [--run]'}

def metadata_preflight(protocol_path,output=None):
    require(pin(protocol_path)['sha256']==PREREG_SHA and Path(protocol_path).resolve().samefile(PREREG.resolve()),'fixed prereg')
    p=read(protocol_path);require(p['schema']=='full151105_saved_contact_regime608_preregistration:1' and p['role']==ROLE,'protocol schema/role')
    for value in list(p['inputs'].values())+list(p['source_pins'].values()):checked(value)
    I=p['inputs']
    parent_specs=(('physical_saved_independent1762','physical_config','physical_report'),('dense_saved_independent35','dense_config','dense_report'),('response_saved_independent22','response_config','response_report'))
    for alias,ca,ra in parent_specs:
        h=read(checked(I[alias]));c=h['checks'];require(type(c) is dict and c and all(type(v) is bool and v for v in c.values()) and type(h['passed']) is int and h['passed']==h['total']==len(c),'strict positive parent '+alias)
        require(same(h['config'],I[ca]) and same(h['producer_report'],I[ra]),'parent actual config/report '+alias)
        require(Path(h['run_dir']).resolve().samefile(checked(I[ra]).parent),'parent output identity')
    for alias in ('dense_output_pins','response_output_pins'):
        manifest=read(checked(I[alias]));require(type(manifest) is dict,'output pins map')
        for v in manifest.values():checked(v)
    for manifest_alias,aliases in [('dense_output_pins',('geometry_catalog',)),('response_output_pins',('response_raw152','response_metadata152','response_decomposition','response_pairs64','response_guard_summary'))]:
        values=list(read(checked(I[manifest_alias])).values())
        for a in aliases:require(any(same(I[a],v) for v in values),'output membership '+a)
    guards=read(checked(I['response_guard_summary']));require(guards and all(type(v) is dict and v['all_passed'] is True and type(v['failed_elements']) is int and v['failed_elements']==0 and v['first_failed_indices']==[] for v in guards.values()),'original structured response guards')
    if output is not None:
        out=Path(output).resolve();require(out.parent==ROOT/'outputs' and out.name.startswith('full151105_saved_contact_regime608_CPU_') and not out.exists(),'fresh output scope')
    require(not any(k in sys.modules for k in ('torch','warp','mujoco')),'no heavy modules')
    return {'status':'PASS_JSON_HASH_METADATA_NO_CONTACT_DERIVATION','protocol':pin(protocol_path),'input_pins':I,'source_pins':p['source_pins'],'NPZ_values_decoded':0,'contact_regime_runs':0}

def _profile(sample_rows):
    return {cat:{'generated_count_by_substep':[sum(s['categories'][cat]['generated_candidate_count'] for s in sample_rows if s['substep']==i) for i in range(1,5)],
      'observed_candidate_depth_by_substep_m':[max([0.]+[s['categories'][cat]['observed_candidate_depth_m'] for s in sample_rows if s['substep']==i]) for i in range(1,5)],
      'secondary_positive_force_count_by_substep':[sum(s['categories'][cat]['secondary_positive_force_count'] for s in sample_rows if s['substep']==i) for i in range(1,5)],'global_clearance':'UNKNOWN'} for cat in CATEGORIES}

def _load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def run_analysis(protocol_path,output):
    started=time.perf_counter();meta=metadata_preflight(protocol_path,output);p=read(protocol_path);I=p['inputs'];out=Path(output).resolve();out.mkdir()
    write(out,'protocol.json',p);write(out,'input_pins_before.json',I)
    source={**p['source_pins'],**{k:pin(v) for k,v in OWN.items()}}
    ready=read(PREP/'READY.json');require(ready['cpu_checks']['passed']==ready['cpu_checks']['total'] and ready['cpu_checks']['total']>0,'focused CPU ready')
    proof=read(checked(ready['cpu_report']));require(proof['checks'] and all(type(v) is bool and v for v in proof['checks'].values()) and type(proof['passed']) is int and proof['passed']==proof['total']==len(proof['checks'])==ready['cpu_checks']['total'],'focused CPU checks')
    for k,v in OWN.items():require(same(ready['source_pins'][k],pin(v)),'current CPU-tested source '+k)
    source.update(CPU_READY=pin(PREP/'READY.json'),CPU_contract_report=ready['cpu_report'])
    all_sources=list(source.values());parent_maps={}
    for alias in ('physical_config','dense_config','response_config'):
        parent_maps[alias]={'config':I[alias],'source_map_field':'source_pins'}
        all_sources.extend(read(checked(I[alias]))['source_pins'].values())
    canonical=canonical_sources(all_sources);write(out,'source_inventory.json',canonical);write(out,'parent_source_map_references.json',parent_maps)
    with zipfile.ZipFile(out/'source_archive.zip','x',zipfile.ZIP_DEFLATED) as z:
        for i,v in enumerate(canonical):z.write(v['path'],f'{i:04d}_{Path(v["path"]).name}')
    try:
        catalog=read(checked(I['geometry_catalog']));sets=catalog_sets(catalog)
        require(len(sets[0])==73 and len(sets[1])==21 and len(sets[2])==3 and len(catalog['native_body_catalog'])==31,'original catalog counts')
        branches=read(checked(I['branch_index']));snapshots=read(checked(I['snapshot_records']));bm=read(checked(I['response_metadata152']));pairs=read(checked(I['response_pairs64']))
        require(len(branches)==len(bm)==152 and len(snapshots)==8 and len(pairs)==64,'complete original identity tables')
        raw=_load_npz(checked(I['response_raw152']));dec=_load_npz(checked(I['response_decomposition']))
        u=raw['actual_u152_FP32'];pred=raw['predicted_y152_FP32'];target=raw['target_y152_FP32'];u0=raw['original_snapshot_u_FP32']
        for a,shape in ((u,(8,19,20)),(u0,(8,20)),(pred,(8,19,58)),(target,(8,19,58))):require(a.dtype==np.float32 and a.shape==shape and np.isfinite(a).all(),'original response raw ABI')
        response=[];pair_out=[];links={i:[] for i in range(152)}
        for bid,(b,m) in enumerate(zip(branches,bm)):
            s,off=divmod(bid,19);require(b['row_id']==m['row_id']==bid and b['snapshot_id']==m['snapshot_id']==s,'branch IDs')
            require(b['path']==('A0','A2')[s//4] and b['k']==(0,49,99,149)[s%4] and b['type']==('baseline' if off<3 else 'direction'),'canonical snapshot/mode order')
            for k in ('path','k','type','j','joint_index','epsilon_index','epsilon_rad','sign','baseline_repeat'):require(b[k]==m[k],'original response identity '+k)
            require(snapshots[s]['path']==b['path'] and snapshots[s]['k']==b['k'],'snapshot join')
            require(bits_equal(np.asarray(b['actualu'],dtype='f4'),u[s,off]),'branch control source')
            du=u[s,off].astype('f8')-u0[s].astype('f8');pr=pred[s,off].astype('f8')-dec['baseline_predicted_mean_F64'][s];tr=target[s,off].astype('f8')-dec['baseline_target_mean_F64'][s]
            if off>=3:
                require(np.array_equal(pr,dec['centered_model_response_F64'][s,off-3]) and np.array_equal(tr,dec['centered_physical_response_F64'][s,off-3]),'saved centered response source')
            response.append({**m,'actual_u20_FP32':u[s,off].tolist(),'actual_control_offset_from_original_u0_rad_F64':du.tolist(),
             'zero_control_change_all20_numeric':bool((du==0).all()),'control_same_bits_original_u0':bits_equal(u[s,off],u0[s]),'absolute_control_all_numeric_zero':bool((u[s,off]==0).all()),
             'centered_physical_response58_F64':tr.tolist(),'centered_model_response58_F64':pr.tolist(),
             'physical_p_centered_all_numeric_zero':bool((tr[40:43]==0).all()),'physical_R_centered_all_numeric_zero':bool((tr[43:52]==0).all()),
             'model_p_centered_all_numeric_zero':bool((pr[40:43]==0).all()),'model_R_centered_all_numeric_zero':bool((pr[43:52]==0).all()),
             'physical_endpoint_vs_first_baseline_encoding':encoded(target[s,off],target[s,0]),'model_endpoint_vs_first_baseline_encoding':encoded(pred[s,off],pred[s,0])})
        for pid,record in enumerate(pairs):
            require(record['pair_id']==pid,'pair identity');s=record['snapshot_id'];mi=record['minus_branch_id'];pl=record['plus_branch_id'];j=record['joint_index']
            require(0<=mi<pl<152 and mi//19==pl//19==s and bm[mi]['sign']==-1 and bm[pl]['sign']==1,'ordered endpoint pair')
            for k in ('snapshot_id','path','k','j','joint_index','epsilon_index'):require(record[k]==bm[mi][k]==bm[pl][k],'pair metadata '+k)
            a=u[s,mi%19];b=u[s,pl%19];du=b.astype('f8')-a.astype('f8');require(float(du[j])==record['actual_coordinate_denominator_rad'],'saved pair denominator')
            row={**record,'actual_delta_u20_rad_F64':du.tolist(),'zero_actual_coordinate_delta':bool(du[j]==0),'zero_actual_all20_delta':bool((du==0).all()),
              'control_endpoints_all_bits_equal':bits_equal(a,b),'both_absolute_controls_all_numeric_zero':bool((a==0).all() and (b==0).all()),
              'physical_encoding':encoded(target[s,pl%19],target[s,mi%19]),'model_encoding':encoded(pred[s,pl%19],pred[s,mi%19]),
              'minus_sample_ids':list(range(mi*4,mi*4+4)),'plus_sample_ids':list(range(pl*4,pl*4+4))}
            pair_out.append(row);links[mi].append(pid);links[pl].append(pid)
        for bid in range(152):
            if bid%19<3:links[bid]=[r['pair_id'] for r in pairs if r['snapshot_id']==bid//19]
            response[bid]['related_pair_ids']=links[bid]
        samples=[];contacts=[];seen=set();original_lines=0
        copy_fields=('pre_time','post_time','pre_qpos','pre_qvel','post_qpos','post_qvel','canonical_pre_tick','canonical_post_tick','source_frame','phase_contract','measured_hand_object','hand_object_force_on_object_w_N','hand_object_moment_about_COM_w_Nm','direct_torque_included_once','object_solver_COM','object_solver_pose','actuator_force','external_force_max')
        with checked(I['physical_samples']).open(encoding='utf-8') as f:
            for li,line in enumerate(f):
                x=json.loads(line);original_lines+=1
                if x['branch_id'] is None:continue
                bid=integer(x['branch_id'],'branch ID',0);sub=integer(x['substep'],'substep',1);require(bid<152 and sub<=4,'branch/substep bounds')
                require((bid,sub) not in seen and bid*4+sub-1==len(samples),'complete original branch sample order');seen.add((bid,sub));b=branches[bid]
                require(x['path']==b['path'] and x['phase']==b['k'] and x['mode']==b['type'] and x['source_frame']==200+b['k'],'sample branch phase/type')
                require(x['canonical_pre_tick']==4*b['k']+sub-1 and x['canonical_post_tick']==4*b['k']+sub,'canonical solver tick')
                require(bits_equal(np.asarray(x['ctrl'],dtype='f4'),u[bid//19,bid%19]),'four substep held control bits')
                for k,sh in (('pre_qpos',(27,)),('pre_qvel',(26,)),('post_qpos',(27,)),('post_qvel',(26,)),('object_solver_COM',(3,)),('object_solver_pose',(7,)),('hand_object_force_on_object_w_N',(3,)),('hand_object_moment_about_COM_w_Nm',(3,)),('actuator_force',(20,))):finite(x[k],sh,k)
                require(type(x['external_force_max']) in (int,float) and x['external_force_max']==0.,'source external force remains zero')
                rs=sample_contacts(x,sets);sid=len(samples);start=len(contacts)
                active=x['measured_hand_object']['active'];ho=[r for r in rs if r['category']=='hand_object']
                require(type(active) is list and len(active)==len(ho) and all(type(v) is bool for v in active),'original helper HO active shape')
                require(active==[r['source_helper_active_by_original_formula'] for r in ho],'original helper HO activity/order')
                require(x['direct_torque_included_once'] is True,'source direct torque scope')
                for r,v in zip(ho,active):r['source_measured_hand_object_active']=v
                ident={'sample_id':sid,'source_physical_row_index':li,'branch_id':bid,'snapshot_id':bid//19,'path':x['path'],'k':b['k'],'mode':x['mode'],'substep':sub}
                contacts.extend({**ident,**r} for r in rs)
                samples.append({**ident,**{k:x[k] for k in copy_fields},'ctrl20_FP32':x['ctrl'],'nacon':x['nacon'],'nefc':x['nefc'],
                   'contact_long_start':start,'contact_long_count':len(rs),'categories':summary(rs),'related_pair_ids':links[bid],
                   'source_wrench_scope':'copied original aggregate; no new world force or COM torque arithmetic','independent_pyramid_decode_qualified':False})
        require(original_lines==1808 and len(samples)==608 and len(seen)==608,'all physical lines/branch PRE count')
        branch_out=[]
        for bid,b in enumerate(branches):
            ss=samples[4*bid:4*bid+4];require([s['substep'] for s in ss]==[1,2,3,4],'branch four PRE samples')
            branch_out.append({'branch_id':bid,'original_branch':b,'response_descriptor':response[bid],'sample_ids':[s['sample_id'] for s in ss],
              'related_pair_ids':links[bid],'category_four_substep_profile':_profile(ss),'categories_over_four_PRE':summary(contacts[r['contact_long_start']+ci] for r in ss for ci in range(r['contact_long_count']))})
        snapshot_out=[]
        for s in range(8):
            ss=samples[s*76:(s+1)*76];snapshot_out.append({'snapshot_id':s,'path':snapshots[s]['path'],'k':snapshots[s]['k'],'branch_ids':list(range(s*19,s*19+19)),
              'sample_ids':[r['sample_id'] for r in ss],'related_pair_ids':[r['pair_id'] for r in pairs if r['snapshot_id']==s],
              'categories_over_all_19_intervals':summary(contacts[r['contact_long_start']+ci] for r in ss for ci in range(r['contact_long_count'])),
              'category_four_substep_profile_summed_over_19_branches':_profile(ss),'profile_scope':'19 separate restored intervals; not a continuous 76-step trajectory'})
        tables(out,'sample608',samples);tables(out,'contact_long',contacts,True);tables(out,'branch152',branch_out);tables(out,'snapshot8',snapshot_out);tables(out,'response_pairs64',pair_out)
        write(out,'response_branch152.json',response);write(out,'geometry_catalog_source.json',catalog)
        for v in list(I.values())+canonical:checked(v)
        write(out,'input_pins_after.json',I)
        report={'schema':p['outputs']['report_schema'],'status':p['outputs']['success_status'],'role':ROLE,'protocol':pin(protocol_path),'input_pins':I,'source_pins':source,
          'counts':{'physical_source_rows':1808,'sample608':608,'contact_long':len(contacts),'branch152':152,'snapshot8':8,'response_pairs64':64,'baseline_PRE_rows':96,'direction_PRE_rows':512},
          'catalog_counts':{'native_ngeom':73,'native_nbody':31,'enabled_hand':21,'object':3},'source_unique_file_count':len(canonical),'cpu_ready':source['CPU_READY'],'cpu_report':source['CPU_contract_report'],'invariants':{'input_source_hashes_unchanged':True},
          'new_model_FW_AD_MJB_FK_Env_GPU_physics':0,'new_world_force_COM_torque_calculations':0,**FLAGS,
          'limitations':['Generated geometry only; missing candidates do not certify clearance.','Original decoded contact force is retained; no complete independent contact-dimension/pyramid decoding.','Copied world wrench/COM/pose are source-backed aggregates, not recomputed.','Same visible58/control-response descriptors do not establish full hidden-state identity or a physical Jacobian.','All snapshots/branches/pairs retained; repeated samples are dependent.'],
          'elapsed_seconds':time.perf_counter()-started}
        write(out,'report.json',report);write(out,'output_pins.json',{q.name:pin(q) for q in out.iterdir() if q.is_file()});return report
    except BaseException as e:
        write(out,'failure.json',{'status':p['outputs']['failure_status'],'exception_type':type(e).__name__,'exception':str(e),'repair_or_retry':False,**FLAGS});raise
