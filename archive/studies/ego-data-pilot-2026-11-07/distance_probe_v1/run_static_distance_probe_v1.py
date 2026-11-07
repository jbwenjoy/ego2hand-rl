"""Predeclared static observer probes. No integration, optimizer, Torch or GPU.

Public mj_geomDistance does not expose GJK/EPA convergence status. All outputs,
zeros, censoring, errors and settings are retained; no mode is auto-selected.
"""
from pathlib import Path
import copy, csv, hashlib, importlib.util, json, math, os, time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parent


def pin(p):
    p=Path(p).resolve(strict=True)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def clean(v):
    if isinstance(v,float) and not math.isfinite(v):return None
    if isinstance(v,list):return [clean(x) for x in v]
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    return v


def write(p,obj):
    with Path(p).open('x',encoding='utf8') as f:json.dump(clean(obj),f,indent=2,allow_nan=False)


def model_digest(model,np):
    h=hashlib.sha256();count=0
    for name in sorted(dir(model)):
        try:value=getattr(model,name)
        except (AttributeError,TypeError):continue
        if isinstance(value,np.ndarray):
            h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(value.shape).encode());h.update(np.ascontiguousarray(value).tobytes());count+=1
    return h.hexdigest(),count


def main():
    import numpy as np
    import mujoco
    started=time.perf_counter()
    if (ROOT/'report.json').exists():raise ValueError('Fresh probe only')
    protocol_path=ROOT/'probe_protocol_v1.json'
    assert pin(protocol_path)['sha256']=='b19d36144e821282ca856eeefb48cc87b5683b6b2e8c92bc68fb0d634ffd5916'
    protocol=json.loads(protocol_path.read_text(encoding='utf8'))
    for p in protocol['inputs'].values():assert pin(p['path'])==p
    for s in json.loads(Path(protocol['inputs']['source_manifest']['path']).read_text(encoding='utf8')):
        assert pin(s['path'])['sha256']==s['sha256']
    loader_spec=importlib.util.spec_from_file_location('exact_fixed_hand2_loader',protocol['inputs']['loader']['path']);loader=importlib.util.module_from_spec(loader_spec);loader_spec.loader.exec_module(loader)
    original=loader.load_model();before,narrays=model_digest(original,np)
    initial=loader.initial_state();Rp=initial['T_world_from_palm'][:3,:3];pp=initial['T_world_from_palm'][:3,3]
    retarget=json.loads(Path(protocol['inputs']['retarget_report']['path']).read_text(encoding='utf8'))
    oldlog=[json.loads(s) for s in Path(protocol['inputs']['retarget_log']['path']).read_text(encoding='utf8').splitlines()]
    exact_old_tiny=next(x for x in oldlog if x['seed_index']==2 and x['evaluation']==23)['x26']
    nativebit=int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    assert nativebit==131072 and int(original.opt.disableflags)&nativebit==0
    assert original.opt.ccd_iterations==35 and original.opt.ccd_tolerance==1e-6
    phone=original.geom(protocol['pairs']['phone']).id;pid=original.body('right_palm_link').id
    targets=[original.geom(g['name']).id for g in protocol['pairs']['targets']]
    assert targets==[g['original_geom_id'] for g in protocol['pairs']['targets']]
    pairmeta=[{'phone_geom_id':phone,'target_geom_id':g,'target_name':original.geom(g).name,'phone_type':int(original.geom_type[phone]),'target_type':int(original.geom_type[g]),'supported_dispatch_source':'engine_collision_driver.c41–51: mesh-mesh and capsule-mesh both mjc_Convex','source_supported_pair':True} for g in targets]
    seeds={}
    for seed in protocol['seeds']:
        with np.load(protocol['inputs'][f'seed{seed}_state']['path'],allow_pickle=False) as a:seeds[seed]=a['qpos27'].copy()
    mode_meta=[];records=[];warnings=[];active={};qposes=[];witnesses=[];dvalues=[];deltas=[]
    counts={'MJB_loads':1,'owned_model_copies':0,'MjData_instances':0,'mj_kinematics':0,'mj_geomDistance_attempted':0,'mj_geomDistance_returned':0,'mj_step':0,'optimization_training_GPU':0}
    old_warning=mujoco.get_mju_user_warning() if hasattr(mujoco,'get_mju_user_warning') else None
    def warning_callback(message):warnings.append({'context':dict(active),'message':str(message)})
    mujoco.set_mju_user_warning(warning_callback)
    try:
        for setting in protocol['modes']:
            model=copy.copy(original);counts['owned_model_copies']+=1
            model.opt.ccd_iterations=setting['ccd_iterations'];model.opt.ccd_tolerance=setting['ccd_tolerance']
            model.opt.disableflags=(int(model.opt.disableflags)&~nativebit) if setting['nativeccd_enabled'] else (int(model.opt.disableflags)|nativebit)
            assert model_digest(model,np)[0]==before
            data=mujoco.MjData(model);counts['MjData_instances']+=1
            meta={'mode':setting['name'],'ccd_iterations':int(model.opt.ccd_iterations),'ccd_tolerance':float(model.opt.ccd_tolerance),'disableflags':int(model.opt.disableflags),'enableflags':int(model.opt.enableflags),'nativeccd_enabled':not bool(int(model.opt.disableflags)&nativebit),'multiCCD_enabled':not bool(int(model.opt.disableflags)&int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)),'compiled_model_arrays_equal_original_before':True}
            for seed in protocol['seeds']:
                base=seeds[seed];x26=retarget['all_seed_results'][seed]['last_complete']['x26']
                for case in protocol['cases']:
                    qpos=base.copy();delta=np.asarray(case['delta_m'],np.float64)
                    if case['frame']=='world':qpos[20:23]+=delta
                    else:
                        local=np.asarray(x26[20:23],np.float64)+delta
                        if seed==2:local=np.asarray(exact_old_tiny[20:23],np.float64)
                        qpos[20:23]=pp+Rp@local
                    assert np.array_equal(qpos[:20].view(np.uint8),base[:20].view(np.uint8)) and np.array_equal(qpos[23:].view(np.uint8),base[23:].view(np.uint8))
                    actual_delta=qpos[20:23]-base[20:23]
                    data.qpos[:]=qpos;data.qvel[:]=0;data.ctrl[:]=0;data.time=0
                    mujoco.mj_kinematics(model,data);counts['mj_kinematics']+=1
                    assert np.array_equal(data.xpos[pid],pp) and np.array_equal(data.xmat[pid].reshape(3,3),Rp)
                    for repeat in range(protocol['identical_input_repeats']):
                        for pair_index,g in enumerate(targets):
                            for order in protocol['orders']:
                                a,b=(phone,g) if order=='phone_to_target' else (g,phone)
                                active={'mode':setting['name'],'seed':seed,'case':case['name'],'repeat':repeat,'target':original.geom(g).name,'order':order}
                                segment=np.full(6,12345.6789,np.float64);value=float('nan');error=None
                                counts['mj_geomDistance_attempted']+=1
                                try:
                                    value=float(mujoco.mj_geomDistance(model,data,a,b,protocol['distmax_m'],segment));counts['mj_geomDistance_returned']+=1
                                except Exception as e:error=f'{type(e).__name__}: {e}'
                                finite=bool(math.isfinite(value) and np.isfinite(segment).all());seglen=float(np.linalg.norm(segment[3:]-segment[:3])) if np.isfinite(segment).all() else float('nan')
                                row={'row_id':len(records),**active,'target_geom_id':g,'source_supported_pair':True,'actual_world_delta_m':actual_delta.tolist(),'translation_norm_m':float(np.linalg.norm(actual_delta)),'distance_m':value,'fromto_world_m':segment.tolist(),'finite':finite,'error':error,'distance_is_zero_UNKNOWN':bool(value==0),'witness_all_zero':bool(np.array_equal(segment,np.zeros(6))),'distance_is_distmax_censored':bool(value==protocol['distmax_m']),'segment_length_m':seglen,'segment_length_minus_abs_distance_m':seglen-abs(value),'native_convergence_status_exposed':False,'physics_time':float(data.time)}
                                records.append(row);qposes.append(qpos.copy());witnesses.append(segment.copy());dvalues.append(value);deltas.append(actual_delta.copy())
            meta['compiled_model_arrays_equal_original_after']=model_digest(model,np)[0]==before;mode_meta.append(meta)
    finally:mujoco.set_mju_user_warning(old_warning)
    assert model_digest(original,np)[0]==before
    bykey={(r['mode'],r['seed'],r['case'],r['repeat'],r['target'],r['order']):r for r in records}
    comparisons=[];allow=protocol['necessary_bound_screen']['allowance_m']
    for row in records:
        if row['case']=='base':continue
        base=bykey[(row['mode'],row['seed'],'base',row['repeat'],row['target'],row['order'])]
        change=abs(row['distance_m']-base['distance_m']);strict_excess=change-row['translation_norm_m'];ok=row['finite'] and base['finite']
        comparisons.append({'row_id':row['row_id'],'base_row_id':base['row_id'],'mode':row['mode'],'seed':row['seed'],'case':row['case'],'target':row['target'],'order':row['order'],'repeat':row['repeat'],'distance_change_m':change,'translation_norm_m':row['translation_norm_m'],'strict_excess_m':strict_excess,'fixed_allowance_m':allow,'engineering_bound_excess_m':strict_excess-allow,'engineering_bound_violated':bool(ok and strict_excess>allow),'comparable_finite':ok,'censored_operand':row['distance_is_distmax_censored'] or base['distance_is_distmax_censored']})
    np.savez_compressed(ROOT/'all_probe_rows.npz',qpos27_F64=np.asarray(qposes),distance_F64=np.asarray(dvalues),fromto_F64=np.asarray(witnesses),actual_world_delta_F64=np.asarray(deltas),finite_mask=np.array([r['finite'] for r in records],bool),row_id=np.arange(len(records)))
    write(ROOT/'all_probe_rows.json',records);write(ROOT/'all_lipschitz_comparisons.json',comparisons)
    for name,table in [('probe_rows.csv',records),('lipschitz_comparisons.csv',comparisons)]:
        with (ROOT/name).open('x',newline='',encoding='utf8') as f:
            fields=[k for k,v in table[0].items() if not isinstance(v,(dict,list))];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows([clean(t) for t in table])
    summaries=[]
    for m in mode_meta:
        rr=[r for r in records if r['mode']==m['mode']];cc=[c for c in comparisons if c['mode']==m['mode']]
        repeat_mismatches=0;order_diffs=[]
        for row in rr:
            base=bykey[(row['mode'],row['seed'],row['case'],0,row['target'],row['order'])]
            if row['distance_m']!=base['distance_m'] or row['fromto_world_m']!=base['fromto_world_m']:repeat_mismatches+=1
            if row['order']=='phone_to_target':order_diffs.append(abs(row['distance_m']-bykey[(row['mode'],row['seed'],row['case'],row['repeat'],row['target'],'target_to_phone')]['distance_m']))
        worst=max((x for x in cc if x['comparable_finite']),key=lambda x:x['strict_excess_m'])
        summaries.append({**m,'rows':len(rr),'finite_rows':sum(r['finite'] for r in rr),'exceptions':sum(r['error'] is not None for r in rr),'zero_distance_rows_UNKNOWN':sum(r['distance_is_zero_UNKNOWN'] for r in rr),'zero_distance_and_zero_witness_rows':sum(r['distance_is_zero_UNKNOWN'] and r['witness_all_zero'] for r in rr),'distmax_censored_rows':sum(r['distance_is_distmax_censored'] for r in rr),'lipschitz_comparisons':len(cc),'engineering_bound_violations':sum(c['engineering_bound_violated'] for c in cc),'maximum_strict_excess_m':worst['strict_excess_m'],'worst_case':worst,'identical_repeat_output_mismatches':repeat_mismatches,'max_reversed_order_distance_difference_m':max(order_diffs),'max_witness_segment_distance_error_m':max(abs(r['segment_length_minus_abs_distance_m']) for r in rr if r['finite'])})
    package=Path(mujoco.__file__).parent
    report={'schema':'hand2_phone_static_distance_probe_report:1','status':'COMPLETED_ALL_PREDECLARED_STATIC_OBSERVERS_NOT_CONTACT_CERTIFICATE','protocol':pin(protocol_path),'producer':pin(__file__),'installed_mujoco_version':mujoco.__version__,'installed_binary_pins':{'dll':pin(package/'mujoco.dll'),'functions_binding':pin(package/'_functions.cp312-win_amd64.pyd')},'matching_official_source_manifest':pin(ROOT/'source_manifest.json'),'official_source_build_equivalence_to_wheel':'NOT_INDEPENDENTLY_PROVEN; matchingversion source used to explain API branches','original_model':protocol['inputs']['model'],'original_model_array_digest':before,'original_model_array_count':narrays,'original_model_and_parameters_unchanged':True,'pair_metadata':pairmeta,'mode_summaries':summaries,'native_warnings':warnings,'counts':counts,'elapsed_seconds':time.perf_counter()-started,'source_semantics':{'public_status':'engine_support.c519–565 returns distance only, not GJK/EPA status; initializes fromto0. Unsupported dispatch returns distmax.','native_zero':'engine_collision_gjk.c2386–2424 can assume touching0 before EPA;1495–1500 face-failure can setnx0/dist0. The publicAPI does not distinguish these branches from an actual near-touch.','witness':'engine_support.c508–512 copies witness only ifstatus.nx>0. Zero segment is therefore not a reliable contact point/normal.','legacy':'Official3.11 XMLreference8423–8427 discourages legacy distances as inaccurate; thismode is diagnosticcontrol only.'},'interpretation':'Necessary observer consistency screens only. No result authorizes signed-distance thresholds, grasp, preload, force, dynamics or mode selection. Unique cause of any oldjump remainsconditional unless separately established.'}
    assert counts['mj_kinematics']==126 and counts['mj_geomDistance_returned']==2268
    report['outputs']={p.name:pin(p) for p in sorted(ROOT.iterdir()) if p.is_file() and p.name!='report.json'}
    write(ROOT/'report.json',report)
    print(json.dumps({'report':pin(ROOT/'report.json'),'seconds':report['elapsed_seconds'],'counts':counts,'modes':[{'mode':s['mode'],'zeros':s['zero_distance_rows_UNKNOWN'],'violations':s['engineering_bound_violations'],'max_excess_mm':s['maximum_strict_excess_m']*1000} for s in summaries]}))


if __name__=='__main__':main()
