"""New CPU-only same-solver-phase sampled-penalty observability consumer.

Default is JSON/hash only. Actual saved-state conversion, passive MJB mapping
and streamed original P_ONLY calls require Root's separate authorized config.
No WM, Actor, optimizer, native FK/query, environment or physics is used.
"""
from __future__ import annotations
import csv
import inspect
import json
import os
from pathlib import Path
import pickle
import random
import re
import sys
import time
from types import SimpleNamespace
import zipfile
import numpy as np
from . import trained_bonus_analysis105_v1 as common

ROOT=common.ROOT
PREREG=ROOT/'protocols/full151105_solver_phase_penalty_observability_CPU_prereg_v2.json'
PREREG_SHA='1ec18db963bc2931920bedce3f2bf43e345544798ee4ac1e0f856ac119cea13c'
SCHEMA='full151105_solver_phase_penalty_CPU_run:1'
ROLE='same_solver_phase_penalty_observability'
KEYS={'schema','role','protocol','output','CPU_run_authorized','cpu_threads','source_pins','cpu_ready'}
OWN={'consumer':ROOT/'src/full151_gradient/penalty_observability105_v1.py',
     'CLI':ROOT/'scripts/diagnose_solver_phase_penalty_cpu_v1.py',
     'checker':ROOT/'scripts/check_solver_phase_penalty_cpu_v1.py'}
require=common.require;pin=common.pin;checked=common.checked;read=common.read;same_pin=common.same_pin
write=common._write;npwrite=common._npwrite


def validate_header(c):
    require(type(c) is dict and set(c)==KEYS,'Exact dense penalty plan keys required')
    require(c['schema']==SCHEMA and c['role']==ROLE,'Wrong dense schema/role')
    require(type(c['CPU_run_authorized']) is bool,'Native bool authorization required')
    require(type(c['cpu_threads']) is int and c['cpu_threads']==1,'CPUthreads1 fixed')
    require(type(c['source_pins']) is dict and c['source_pins'],'Nonempty flat source table required')
    require(type(c['output']) is str and Path(c['output']).is_absolute(),'Absolute output required')
    out=Path(c['output']).resolve()
    require(out.parent==ROOT/'outputs' and out.name.startswith('full151105_solver_phase_penalty_CPU_') and not out.exists(),'Fresh scoped output, no overwrite')


def required_sources(p):
    result=dict(p['source_pins'])
    original=read(checked(p['inputs']['physical_config']))
    for k,v in original['source_pins'].items():result['physical__'+k]=v
    prior=read(checked(p['inputs']['previous_bonus_report']))
    prior_config=read(checked(prior['config']))
    for k,v in prior_config['source_pins'].items():result['bonus__'+k]=v
    result.update({k:pin(v) for k,v in OWN.items()})
    result['borrowed_no_score_helpers']=pin(common.__file__)
    return result


def declaration():
    require('torch' not in sys.modules and 'warp' not in sys.modules,'Default noTorch/Warp required')
    require(pin(PREREG)['sha256']==PREREG_SHA,'Prereg changed')
    return {'schema':'full151105_dense_penalty_CPU_declaration:1','status':'DECLARED_NOT_CONVERTED_NOT_SCORED',
       'runplan_schema':SCHEMA,'role':ROLE,'keys':sorted(KEYS),'protocol':pin(PREREG),'source_pins':{k:pin(v) for k,v in OWN.items()},
       'Torch_imported':False,'Warp_imported':False,'NPZ_opened':0,'MJB_loads':0,'state_conversions':0,'objective_constructs':0,'proxy_forwards':0,'WM_Actor_Env_GPU':0}


def metadata_preflight(config_path):
    require('torch' not in sys.modules and 'warp' not in sys.modules,'Fresh default metadata process only')
    c=read(config_path);validate_header(c)
    require(same_pin(c['protocol'],{'path':str(PREREG),'sha256':PREREG_SHA}),'Wrong fixed prereg')
    p=read(checked(c['protocol']));require(p['schema']=='full151105_penalty_solver_phase_observability_CPU_preregistration:1' and p['revision']==2 and p['CPU_run_authorized'] is False,'Wrong source-only protocol')
    for v in p['inputs'].values():checked(v)
    have={(str(checked(v)).casefold(),v['sha256']) for v in c['source_pins'].values()}
    needed=required_sources(p)
    require(all((str(checked(v)).casefold(),v['sha256']) in have for v in needed.values()),'Transitive source closure missing')
    ready=read(checked(c['cpu_ready']));require(ready.get('schema')=='full151105_dense_penalty_CPU_contract_READY:1' and ready.get('status')=='PASS_SYNTHETIC_NATIVE_STATE_AND_HOOK_ONLY','Wrong narrow CPU READY')
    proof=read(checked(ready['cpu_report']));common.all_checks(proof)
    require(proof['schema']=='full151105_dense_penalty_CPU_contract_proof:1' and same_pin(ready['protocol'],c['protocol']),'READY/prereg mismatch')
    for k,v in OWN.items():require(same_pin(ready['source_pins'][k],pin(v)) and same_pin(proof['source_pins'][k],pin(v)),'Executed source mismatch: '+k)
    require(proof['scope']['actual_physical_state_conversions']==proof['scope']['actual_MJB_loads']==proof['scope']['actual_objective_constructs']==proof['scope']['actual_full05_data_reads']==0,'Fixture scope elevated')
    saved=read(checked(p['inputs']['physical_saved_independent1762']));common.all_checks(saved)
    require(saved['schema']=='full151105_full_serial_saved_independent:1' and saved['completed_branches']==152 and saved['baseline_count']==24 and saved['episode_step_returned']==1808 and saved['physical_gradient_qualified'] is False,'Physical completion mismatch')
    require(same_pin(saved['config'],p['inputs']['physical_config']),'Actual physical config mismatch')
    spec=read(checked(p['inputs']['objective_spec']));require(spec['schema']=='full151_declared_H123_objective_spec:2' and spec['CPU_authorized'] is True,'Exact CPU-capable trained factory spec required')
    require('torch' not in sys.modules and 'warp' not in sys.modules,'Metadata imported backend')
    return {'config':c,'config_pin':pin(config_path),'protocol':p,'status':'PASS_HASH_JSON_ONLY_NOT_CONVERTED_NOT_SCORED','Torch_imported':False,'Warp_imported':False,'NPZ_opened':0,'state_conversions':0,'objective_constructs':0}


def arithmetic_check(actual,expected,name):
    a,b=np.asarray(actual,np.float64),np.asarray(expected,np.float64)
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),name+' finite shape mismatch')
    bound=8*2**-23*np.maximum(1.,np.maximum(abs(a),abs(b)))
    require(np.all(abs(a-b)<=bound),name+' exceeds frozen8eps32 bound')
    return float(abs(a-b).max()) if a.size else 0.


def validate_mapping(m):
    require(type(m) is dict and set(m)=={'hand_qpos','hand_dof','object_qpos_start','object_dof_start'},'Exact state mapping keys required')
    for k,limit in (('hand_qpos',27),('hand_dof',26)):
        a=np.asarray(m[k]);require(a.shape==(20,) and a.dtype.kind in 'iu' and len(set(a.tolist()))==20 and ((0<=a)&(a<limit)).all(),'Unique typed20 indices required')
    for k,n in (('object_qpos_start',20),('object_dof_start',20)):
        require(type(m[k]) is int and 0<=m[k]<=n,'Freejoint index must be nativeint')
    require(not set(m['hand_qpos']).intersection(range(m['object_qpos_start'],m['object_qpos_start']+7)) and not set(m['hand_dof']).intersection(range(m['object_dof_start'],m['object_dof_start']+6)),'Hand/freejoint index overlap')


def native_state58(qpos,qvel,palm_pos,palm_rot,mapping,reader_pin):
    """Call the EXACT unbound native state method on CPU views, no Reader init."""
    validate_mapping(mapping)
    q,v,pp,R=map(np.asarray,(qpos,qvel,palm_pos,palm_rot))
    require(q.ndim==2,'Native qpos batch rank required');n=len(q)
    require(q.shape==(n,27) and v.shape==(n,26) and pp.shape==(n,3) and R.shape==(n,3,3) and n>0,'Native27/26/palm batch ABI required')
    require(all(a.dtype==np.float32 and np.isfinite(a).all() for a in (q,v,pp,R)),'Exact finiteFP32 inputs required')
    oq=mapping['object_qpos_start'];norm=np.linalg.norm(q[:,oq+3:oq+7].astype(np.float64),axis=-1)
    require(np.all(abs(norm-1)<=1e-5),'Malformed object quaternion; no silent repair')
    arithmetic_check(np.swapaxes(R,-1,-2).astype(float)@R.astype(float),np.broadcast_to(np.eye(3),(n,3,3)),'recorded palm rotation')
    require(np.all(np.linalg.det(R.astype(float))>0),'Improper recorded palm rotation')
    import torch
    from dex_supermimic.rl.dex_supermimic.sim import SimReader
    require(Path(inspect.getsourcefile(SimReader.state)).samefile(checked(reader_pin)),'Foreign native state implementation')
    data=SimpleNamespace(qpos=torch.from_numpy(q.copy()),qvel=torch.from_numpy(v.copy()))
    obj=SimpleNamespace(indexing=SimpleNamespace(free_joint_q_adr=torch.arange(oq,oq+7),free_joint_v_adr=torch.arange(mapping['object_dof_start'],mapping['object_dof_start']+6)))
    robot=SimpleNamespace(indexing=SimpleNamespace(joint_q_adr=torch.from_numpy(np.asarray(mapping['hand_qpos'],np.int64)),joint_v_adr=torch.from_numpy(np.asarray(mapping['hand_dof'],np.int64))))
    fake=SimpleNamespace(env=SimpleNamespace(sim=SimpleNamespace(data=data)),obj=obj,robot=robot,palm_pos=torch.from_numpy(pp.copy()),palm_rot=torch.from_numpy(R.copy()))
    result=SimReader.state(fake)
    require(result.shape==(n,58) and result.device.type=='cpu' and result.dtype==torch.float32 and bool(torch.isfinite(result).all()),'Bad native state result')
    return result


def validate_x58(actual,reference,name):
    a,b=np.asarray(actual),np.asarray(reference)
    require(a.shape==b.shape and a.ndim==2 and a.shape[1]==58 and a.dtype==b.dtype==np.float32,'State58 FP32 reference required')
    require(a[:,:40].tobytes()==b[:,:40].tobytes(),name+' q/qd not bitexact')
    return arithmetic_check(a[:,40:],b[:,40:],name+' p/R/v/omega')


def pair_coverage(contacts,hand,objects,covered,min_by_geom=None):
    """Primary exact-deepest tie set; all violating pairs are SECONDARY."""
    pairs=[]
    for c in contacts:
        geom=c['geom'];require(type(geom) is list and len(geom)==2 and all(type(g) is int for g in geom),'Typed native geom pair required')
        distance=c['dist'];require(type(distance) in (int,float) and np.isfinite(distance),'Finite native distance required')
        a,b=geom
        if (a in hand and b in objects) or (b in hand and a in objects):
            pairs.append({'geom':geom,'hand_geom':a if a in hand else b,'depth_m':max(0.,-float(distance))})
    depth=max((r['depth_m'] for r in pairs),default=0.)
    deepest=[r for r in pairs if r['depth_m']==depth] if pairs else []
    violating=[r for r in pairs if r['depth_m']>.003]
    def flags(items):
        ids=sorted({r['hand_geom'] for r in items})
        absent=[g for g in ids if g not in covered]
        insensitive=[] if min_by_geom is None else [g for g in ids if g in covered and min_by_geom[g]>=-.003]
        return {'hand_geom_ids':ids,'geom_set_gap_ids':absent,'covered_proxy_undersensitivity_ids':insensitive,'overlapping_not_exclusive':True}
    return {'native_generated_HO_depth_m':depth,'native_generated_HO_count':len(pairs),'native_HO_depth_gt3mm':depth>.003,
      'all_exact_tied_deepest_pairs':deepest,'all_violating_pairs_SECONDARY':violating,
      'primary_deepest_tie_coverage':flags(deepest if depth>.003 else []),'secondary_all_violating_coverage':flags(violating)}


def group_rows(samples,rewards):
    require(len(samples)==1808 and len(rewards)==452,'Complete1808/452 required')
    groups={};reward_map={}
    for reward in rewards:
        key=(reward['path'],reward['k'],reward['branch_id']);require(key not in reward_map,'Duplicate reward group');reward_map[key]=reward
    expected_clock=np.empty(601,np.float32);expected_clock[0]=0
    for i in range(600):expected_clock[i+1]=np.float32(expected_clock[i]+np.float32(.005))
    for index,row in enumerate(samples):
        k=row['phase'];sub=row['substep'];bid=row['branch_id']
        require(row['path'] in ('A0','A2') and type(k) is int and 0<=k<150 and type(sub) is int and 1<=sub<=4 and (bid is None or type(bid) is int and 0<=bid<152),'Bad sample identity')
        require(type(row['source_frame']) is int and row['source_frame']==200+k,'Wrong sourceframe identity')
        require(type(row['canonical_pre_tick']) is int and type(row['canonical_post_tick']) is int,'Typed canonical ticks required')
        require(row['canonical_pre_tick']==4*k+sub-1 and row['canonical_post_tick']==4*k+sub,'Wrong solver ticks')
        for t,tick in (('pre_time',4*k+sub-1),('post_time',4*k+sub)):
            require(float(expected_clock[tick])==row[t],'Recorded time is not exact sourceFP32 recurrence')
        require(row['phase_contract']=='solver quantities preintegration; q/qvel/time postintegration; before scene.update and Manager forward','Wrong raw phase contract')
        require(row['external_force_max']==0,'Exogenous force data invalid')
        key=(row['path'],k,bid);groups.setdefault(key,[]).append(index)
    require(len(groups)==452 and set(groups)==set(reward_map),'Reward/sample groups differ')
    kinds={'donor':0,'baseline':0,'direction':0}
    for key,indices in groups.items():
        require(indices==list(range(indices[0],indices[0]+4)) and [samples[i]['substep'] for i in indices]==[1,2,3,4],'Four contiguous substeps required')
        rows=[samples[i] for i in indices]
        require(all(r['mode']==rows[0]['mode'] for r in rows),'Group mode changed')
        require(rows[0]['mode'] in kinds,'Unknown actual row mode');kinds[rows[0]['mode']]+=4
        for a,b in zip(rows,rows[1:]):
            for q,v in (('post_qpos','pre_qpos'),('post_qvel','pre_qvel')):require(np.array_equal(np.asarray(a[q],np.float32),np.asarray(b[v],np.float32)),'Within-group pre/post discontinuity')
    require(kinds=={'donor':1200,'baseline':96,'direction':512},'Row coverage changed')
    return list(groups.items()),reward_map


def native_contact_abi(row,ngeom):
    """JSON empty arrays lose their second dimension; normalize ONLY knownempty."""
    n=row['nacon'];nefc=row['nefc'];raw=row['raw_wp_contact'];contacts=row['contacts']
    require(type(n) is int and 0<=n<=192 and type(nefc) is int and 0<=nefc<=4096,'Invalid actual capacity row')
    require(type(raw) is dict and set(raw)=={'geom','type','worldid','efc_address'} and type(contacts) is list,'Exact saved contact schema required')
    if n==0:
        require(all(type(raw[k]) is list and not raw[k] for k in raw) and contacts==[],'Knownempty must not contain missing ornonempty arrays')
        return np.empty((0,2),np.int64),np.empty(0,np.int64)
    world,geom,typ=(np.asarray(raw[k]) for k in ('worldid','geom','type'))
    require(world.shape==typ.shape==(n,) and geom.shape==(n,2) and all(v.dtype.kind in 'iu' for v in (world,geom,typ)),'Typed world/type/geom ABI required')
    require(np.all(world==0) and ((0<=geom)&(geom<ngeom)).all() and (typ>=0).all(),'Unknown world/geom/type')
    require(len(contacts)==n and all(type(c['geom']) is list and len(c['geom'])==2 and all(type(g) is int for g in c['geom']) for c in contacts),'Contact rows/IDs malformed')
    require(np.array_equal(geom,np.asarray([c['geom'] for c in contacts],np.int64)),'CPU/WP contact order mismatch')
    return geom,typ


def exact_fp32(value,shape,name):
    source=np.asarray(value)
    require(source.shape==shape and source.dtype.kind in 'fiu' and np.isfinite(source).all(),name+' finite numericshape')
    out=source.astype(np.float32)
    require(np.array_equal(out.astype(np.float64),source.astype(np.float64)),name+' not originally representableFP32; no silentrepair')
    return out


def _mapping(model):
    require((model.nq,model.nv)==(27,26) and model.nplugin==0,'Fixed passive model ABI required')
    import mujoco
    require(model.opt.integrator==mujoco.mjtIntegrator.mjINT_IMPLICITFAST,'Wrong integrator phase semantics')
    ids=np.array([model.joint(f'robot/right_finger{f}_joint{j}').id for f in range(1,6) for j in range(1,5)])
    oj=model.joint('object/cube_freejoint').id;body=model.body('object/cube').id
    require(model.jnt_type[oj]==mujoco.mjtJoint.mjJNT_FREE and model.body_parentid[body]==0 and model.body_jntnum[body]==1 and np.array_equal(model.body_ipos[body],np.zeros(3)),'World freebody/COM0 required')
    palm=model.body('robot/right_palm_link').id;parent=int(model.body_parentid[palm])
    require(model.body(parent).name=='robot/mocap_base' and model.body_parentid[parent]==0 and model.body_mocapid[parent]==0 and model.body_jntnum[palm]==0 and np.array_equal(model.body_pos[palm],np.zeros(3)) and np.array_equal(model.body_quat[palm],np.array([1,0,0,0])),'Fixed identity palm child required')
    m={'hand_qpos':model.jnt_qposadr[ids].copy(),'hand_dof':model.jnt_dofadr[ids].copy(),'object_qpos_start':int(model.jnt_qposadr[oj]),'object_dof_start':int(model.jnt_dofadr[oj])}
    validate_mapping(m);return m


def _jsonl(p):
    with Path(p).open(encoding='utf-8') as f:return [json.loads(s) for s in f if s.strip()]


def run_analysis(config_path):
    started=time.perf_counter();p=metadata_preflight(config_path);c=p['config']
    require(c['CPU_run_authorized'] is True and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1','Explicit fresh RootCPU authorization required')
    out=Path(c['output']);out.mkdir();phase='archive';counts={k:{'attempted':0,'returned':0} for k in ('passive_mapping_MJB','native_state_conversion','objective_factory','objective_forward')};completed=0;invariants={}
    def call(k,fn,*args,**kwargs):counts[k]['attempted']+=1;v=fn(*args,**kwargs);counts[k]['returned']+=1;return v
    try:
        write(out,'config.json',c);write(out,'protocol.json',p['protocol']);write(out,'source_inventory.json',c['source_pins'])
        with zipfile.ZipFile(out/'execution_sources.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for i,v in enumerate(c['source_pins'].values()):z.write(checked(v),'%04d_%s'%(i,Path(v['path']).name))
        phase='saved_metadata';inputs=p['protocol']['inputs']
        samples=_jsonl(checked(inputs['physical_samples']));rewards=_jsonl(checked(inputs['reward_snapshots']));groups,reward_map=group_rows(samples,rewards)
        from . import analysis105_v1 as old
        run=checked(inputs['physical_samples']).parent;saved=read(checked(inputs['physical_saved_independent1762']))
        evidence={str(checked(v)).casefold():v for v in saved['evidence'].values()}
        def ev(path):return checked(evidence[str(Path(path).resolve(strict=True)).casefold()])
        old._validate_capacity(read(ev(run/'capacity_ledgers.json')))
        watch=Path(str(run)+'_watchdog')
        old._validate_lifecycle({'physical_run':str(run),'artifact_pins':{'producer_config':inputs['physical_config']}},
          {'watch_state':read(ev(watch/'state.json')),'engineering_identity':read(ev(watch/'engineering_identity.json')),'post_terminal_CIM':read(ev(watch/'post_terminal_CIM.json')),'runtime':read(ev(run/'runtime.json'))})
        phase='passive_constants_and_native_converter'
        import torch
        require(not torch.cuda.is_initialized() and 'warp' not in sys.modules,'Fresh noCUDA/noWarp CPU process required')
        torch.set_num_threads(1)
        def rng():return common.hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state()),protocol=5)+torch.get_rng_state().numpy().tobytes()).hexdigest()
        rng_before=rng()
        import mujoco
        from full151_native.bundle import load_bundle
        bundle=load_bundle(checked(inputs['full151_bundle']),expected_sha256=inputs['full151_bundle']['sha256'])
        from full151_native.bundle import FIELDS
        for path in ('A0','A2'):
            with np.load(checked(inputs[path+'__expected_context_fields.npz']),allow_pickle=False) as z:expected={k:z[k].copy() for k in z.files}
            with np.load(checked(inputs[path+'__actual_context_readbacks.npz']),allow_pickle=False) as z:actual={k:z[k].copy() for k in z.files}
            require(set(expected)==set(FIELDS) and set(actual)=={stage+'__'+f for stage in ('after_copy','after_set_const_0') for f in FIELDS},'Actual8 sourcefield set differs')
            for f in FIELDS:
                wanted=np.asarray(bundle['context'][f])[0]
                require(np.array_equal(expected[f],wanted),'Saved expected actual8 differs: '+f)
                for stage in ('after_copy','after_set_const_0'):
                    got=actual[stage+'__'+f]
                    require(got.ndim==wanted.ndim+1 and got.shape[1:]==wanted.shape and np.array_equal(got,np.broadcast_to(wanted.astype(got.dtype),got.shape)),'Actual8 double readback differs: '+path+'/'+stage+'/'+f)
        model=call('passive_mapping_MJB',mujoco.MjModel.from_binary_path,str(checked(inputs['nominal_model'])))
        mapping=_mapping(model);oq=mapping['object_qpos_start']
        # Use the recorded actual Reader matrices for each group, not inferred mocap.
        q=exact_fp32([r['pre_qpos'] for r in samples],(1808,27),'allpreqpos');v=exact_fp32([r['pre_qvel'] for r in samples],(1808,26),'allpreqvel')
        postq=[];postv=[];pp=np.empty((1808,3),np.float32);R=np.empty((1808,3,3),np.float32);gpp=[];gR=[];reference=[];lastids=[]
        source_palm={}
        native=np.load(checked(inputs['native_data']),allow_pickle=False)
        try:wrist=native['wrist_pose_w'].astype(np.float32)
        finally:native.close()
        require(wrist.shape==(151,7) and np.array_equal(wrist,np.broadcast_to(wrist[:1],wrist.shape)),'Fixed native151 wrist changed')
        from dex_supermimic.rl.dex_supermimic.rotations import quat_to_mat
        expected_R=quat_to_mat(torch.from_numpy(wrist[:1,3:].copy())).numpy()[0]
        for key,indices in groups:
            rw=reward_map[key];pos=exact_fp32(rw['state20_reader_palm_pos'],(1,3),'recordedReaderpalm')[0];rot=exact_fp32(rw['state20_reader_palm_rot'],(1,3,3),'recordedReaderR')[0]
            arithmetic_check(pos,wrist[0,:3],'Reader/reference palm position');arithmetic_check(rot,expected_R,'Reader/reference palm rotation')
            if key[0] in source_palm:require(np.array_equal(source_palm[key[0]][0],pos) and np.array_equal(source_palm[key[0]][1],rot),'Reader frame changed within path')
            else:source_palm[key[0]]=(pos.copy(),rot.copy())
            pp[indices]=pos;R[indices]=rot;gpp.append(pos);gR.append(rot)
            last=indices[-1];lastids.append(last);postq.append(samples[last]['post_qpos']);postv.append(samples[last]['post_qvel']);reference.append(rw['physics'])
        for path,(pos,rot) in source_palm.items():
            init=read(checked(inputs[path+'__initial_readback.json']));reset=read(checked(inputs[path+'__runtime_reset_readback.json']))
            require(np.array_equal(np.asarray(init['mocap_pos'],np.float32).reshape(3),wrist[0,:3]) and np.array_equal(np.asarray(init['mocap_quat'],np.float32).reshape(4),wrist[0,3:]),'Initial fixed parent differs')
            require(np.array_equal(np.asarray(reset['mocap_pos'],np.float32).reshape(3),wrist[0,:3]) and np.array_equal(np.asarray(reset['mocap_quat'],np.float32).reshape(4),wrist[0,3:]),'Reset fixed parent differs')
        postq=exact_fp32(postq,(452,27),'endpointqpos');postv=exact_fp32(postv,(452,26),'endpointqvel')
        allq=np.concatenate((q,postq),axis=0);allv=np.concatenate((v,postv),axis=0)
        allpp=np.concatenate((pp,np.asarray(gpp,np.float32)),axis=0);allR=np.concatenate((R,np.asarray(gR,np.float32)),axis=0)
        require(len(allq)==2260,'All1808pre+452post conversions required')
        converted=call('native_state_conversion',native_state58,allq,allv,allpp,allR,mapping,p['protocol']['source_pins']['native_reader']).numpy().copy()
        gates={'all452_post20_endpoint_maxerror':validate_x58(converted[1808:],np.asarray(reference,np.float32).reshape(452,58),'all452post20')}
        with np.load(checked(inputs['physical_arrays']),allow_pickle=False) as z:anchor=z['snapshot_x'].copy()
        anchors=[];wanted=[]
        for key,indices in groups:
            if key[2] is not None or key[1] in (0,49,99,149):
                sid=('A0','A2').index(key[0])*4+(0,49,99,149).index(key[1]);anchors.append(converted[indices[0]]);wanted.append(anchor[sid])
        require(len(anchors)==160,'All152branch+8donor leading anchors required')
        gates['all160_leading_anchor_maxerror']=validate_x58(np.asarray(anchors),np.asarray(wanted),'all160leading')
        solver_pose=np.asarray([s['object_solver_pose'] for s in samples],np.float32)
        require(solver_pose.shape==(1808,7) and np.isfinite(solver_pose).all() and np.all(abs(np.linalg.norm(solver_pose[:,3:].astype(float),axis=-1)-1)<=1e-5),'Finite solverpose required')
        gates['all1808_solver_pre_position_maxerror']=arithmetic_check(solver_pose[:,:3],q[:,oq:oq+3],'all1808solver/pre position')
        gates['all1808_solver_pre_R_maxerror']=arithmetic_check(quat_to_mat(torch.from_numpy(solver_pose[:,3:].copy())).numpy(),quat_to_mat(torch.from_numpy(q[:,oq+3:oq+7].copy())).numpy(),'all1808solver/pre R')
        gates.update(all_converter_gates_passed=True,frame_scope='452actualReader frames + fixed-palm source/topology; NOT1808 directmocap readings',proxy_calls_before_gates=0)
        write(out,'converter_qualification.json',gates)
        npwrite(out,'converted_state58.npz',{'pre_x':converted[:1808],'post20_x':converted[1808:],'post20_reference':np.asarray(reference,np.float32).reshape(452,58),
            'pre_qpos':q,'pre_qvel':v,'post20_qpos':postq,'post20_qvel':postv,'object_solver_pose':solver_pose,
            'palm_pos':pp,'palm_rot':R,'group_last_indices':np.asarray(lastids,np.int64),**{k:np.asarray(v) for k,v in mapping.items()}})
        # Native depth is a generated-pair metric, never positive-force filtering.
        hand={i for i in range(model.ngeom) if re.fullmatch(r'robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)',model.geom(i).name) and (model.geom_contype[i] or model.geom_conaffinity[i])}
        obj=model.body('object/cube').id;objects=set(np.flatnonzero(model.geom_bodyid==obj).tolist())
        native_depth=[]
        for s in samples:
            geom,typ=native_contact_abi(s,model.ngeom)
            info=pair_coverage(s['contacts'],hand,objects,set())
            info['primary_deepest_tie_coverage']=None;info['secondary_all_violating_coverage']=None
            info['proxy_coverage_status']='NOT_EVALUATED_BEFORE_FACTORY'
            require(info['native_generated_HO_depth_m']==s['measured_hand_object']['penetration_by_category_m']['hand_object'],'HO depth/category saved mismatch')
            native_depth.append(info)
        write(out,'native_depth_before_proxy.json',native_depth)
        phase='one_actual_P_ONLY_factory'
        from full151_intent.binding_v3 import build_penalty_only_objective
        objective=call('objective_factory',build_penalty_only_objective,read(checked(inputs['objective_spec'])),device='cpu');base=objective.base
        require(not objective.training and all(not t.requires_grad for t in objective.parameters()),'Frozen CPU objective required')
        before_buffers=common._tensor_digest(objective.state_dict())
        oldmeta=read(checked(inputs['previous_bonus_factory_geometry_manifest']))['P_ONLY']['geometry']
        require(common.canonical(base.geometry_proxy.metadata)==common.canonical(oldmeta),'Existing factory/sample geometry manifest differs')
        sample_geom=base.geometry_proxy.sample_geom_ids.detach().numpy().copy();unique=sorted(set(sample_geom.tolist()))
        require(sample_geom.shape==(1344,) and len(unique)==21,'Exact existing1344/21 sampling required')
        native_body_ids=list(map(int,base.geometry_proxy.fk.body_ids))
        write(out,'geometry_ID_mapping.json',{'FK_local_body_table':[{'local_FK_index':i,'native_body_id':b,'native_body_name':model.body(b).name} for i,b in enumerate(native_body_ids)],
            'sample_body_ids_semantics':'local_FK_index, not nativebody/mocapID','sample_geom_ids_semantics':'nativeMJB geomID',
            'native_hand_geoms':[{'id':g,'name':model.geom(g).name,'native_body_id':int(model.geom_bodyid[g])} for g in sorted(hand)],
            'native_object_geoms':[{'id':g,'name':model.geom(g).name,'native_body_id':int(model.geom_bodyid[g])} for g in sorted(objects)],
            'actual_proxy_hand_geometry_records':base.geometry_proxy.metadata['hand_collision_geoms'],'actual_proxy_object_geometry_records':base.geometry_proxy.metadata['object_geoms']})
        npwrite(out,'proxy_sample_buffers.npz',{'sample_geom_ids':sample_geom,'sample_body_ids':base.geometry_proxy.sample_body_ids.detach().numpy().copy(),
            'sample_group_ids':base.geometry_proxy.sample_group_ids.detach().numpy().copy(),'sample_points_body':base.geometry_proxy.sample_points_body.detach().numpy().copy()})
        write(out,'factory_receipt.json',objective.metadata)
        phases=np.asarray([s['phase'] for s in samples],np.int64);parts=[];penalties=[];bonuses=[];mins=[];pergeom=[]
        inputs_before=common.array_digest({'q':q,'v':v,'palm_pos':pp,'palm_R':R,'converted':converted})
        phase='fifteen_streamed_P_ONLY_forwards'
        for batch,start in enumerate(range(0,1808,128)):
            stop=min(start+128,1808)
            with torch.no_grad():score,terms,capture=call('objective_forward',common.evaluate_objective,objective,torch.from_numpy(converted[start:stop].copy()),phases[start:stop])
            sdf=capture['sampled_signed_distance_m'];require(sdf.shape==(stop-start,1344),'Actual SDF shape differs')
            P=terms['geometry_intent_penetration_prior'].numpy().copy();BP=score.numpy().copy();per=np.stack([sdf[:,sample_geom==g].min(-1) for g in unique],axis=-1)
            chunk={'row_indices':np.arange(start,stop),'sampled_sdf_m':sdf,'P':P,'BP':BP,'minimum_sdf_m':capture['minimum_signed_distance_m'],'geom_ids':np.asarray(unique),'per_geom_min_sdf_m':per}
            npwrite(out,'chunk_%02d.npz'%batch,chunk);parts.append(sdf);penalties.append(P);bonuses.append(BP);mins.append(capture['minimum_signed_distance_m']);pergeom.append(per);completed=stop
        sdf=np.concatenate(parts);P=np.concatenate(penalties);BP=np.concatenate(bonuses);minimum=np.concatenate(mins);per=np.concatenate(pergeom)
        require(counts['objective_forward']['returned']==15 and completed==1808,'Exact15batches/all1808 required')
        results=[]
        for i,s in enumerate(samples):
            info=pair_coverage(s['contacts'],hand,objects,set(unique),{g:float(per[i,j]) for j,g in enumerate(unique)})
            observed=bool(P[i]>0);info.update(row_id=i,path=s['path'],mode=s['mode'],k=s['phase'],branch_id=s['branch_id'],substep=s['substep'],
               canonical_pre_tick=s['canonical_pre_tick'],source_frame=s['source_frame'],P=float(P[i]),BP=float(BP[i]),minimum_sampled_sdf_m=float(minimum[i]),
               proxy_sample_depth_m=max(0.,-float(minimum[i])),proxy_P_positive=observed,proxy_min_sdf_below_minus3mm=bool(minimum[i]<-.003),
               proxy_min_sdf_exactly_minus_FP32_allowance=bool(minimum[i]==np.float32(-.003)),
               native_depth_minus_nominal_3mm_m=info['native_generated_HO_depth_m']-.003,
               primary_miss=bool(info['native_HO_depth_gt3mm'] and not observed),proxy_alert_without_observed_native_HO3mm=bool(observed and not info['native_HO_depth_gt3mm']))
            results.append(info)
        def summary(selected):
            return {'rows':len(selected),'native_HO_depth_gt3mm':sum(r['native_HO_depth_gt3mm'] for r in selected),'proxy_P_positive':sum(r['proxy_P_positive'] for r in selected),
                    'both_positive':sum(r['native_HO_depth_gt3mm'] and r['proxy_P_positive'] for r in selected),'native_positive_proxy_zero':sum(r['primary_miss'] for r in selected),
                    'proxy_alert_without_nativeHO3mm':sum(r['proxy_alert_without_observed_native_HO3mm'] for r in selected),
                    'both_nonpositive':sum(not r['native_HO_depth_gt3mm'] and not r['proxy_P_positive'] for r in selected),
                    'primary_deepest_tie_geom_gap_rows':sum(r['primary_miss'] and bool(r['primary_deepest_tie_coverage']['geom_set_gap_ids']) for r in selected),
                    'primary_deepest_tie_covered_undersensitivity_rows':sum(r['primary_miss'] and bool(r['primary_deepest_tie_coverage']['covered_proxy_undersensitivity_ids']) for r in selected),
                    'overlapping_categories_not_independent_N':True}
        stats={'all1808':summary(results)}
        for path in ('A0','A2'):
            for mode in ('donor','baseline','direction'):stats[path+'/'+mode]=summary([r for r in results if r['path']==path and r['mode']==mode])
        npwrite(out,'all1808_proxy_scores.npz',{'sampled_sdf_m':sdf,'P':P,'BP':BP,'minimum_sdf_m':minimum,'geom_ids':np.asarray(unique),'per_geom_min_sdf_m':per,
            'native_HO_depth_m':np.asarray([r['native_generated_HO_depth_m'] for r in results]),'proxy_P_positive':P>0,'phases':phases})
        write(out,'rows_all1808.json',results);write(out,'statistics.json',stats)
        fields=['row_id','path','mode','k','branch_id','substep','canonical_pre_tick','source_frame','native_generated_HO_count','native_generated_HO_depth_m','native_HO_depth_gt3mm','P','BP','minimum_sampled_sdf_m','proxy_P_positive','primary_miss']
        with (out/'rows_all1808.csv').open('x',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(results)
        invariants={'objective_parameters_buffers_unchanged':common._tensor_digest(objective.state_dict())==before_buffers,'RNG_unchanged':rng()==rng_before,
           'converted_inputs_unchanged':common.array_digest({'q':q,'v':v,'palm_pos':pp,'palm_R':R,'converted':converted})==inputs_before,
           'CUDA_uninitialized':not torch.cuda.is_initialized(),'Warp_not_imported':'warp' not in sys.modules,'no_parameter_gradients':all(t.grad is None for t in objective.parameters())}
        require(all(invariants.values()),'Read-only CPU invariance failed')
        for value in inputs.values():checked(value)
        for value in c['source_pins'].values():checked(value)
        require(pin(config_path)==p['config_pin'],'Actual CPU plan changed')
        report={'schema':'full151105_same_solver_phase_penalty_CPU_analysis:1','status':'COMPLETED_SAME_SOLVER_PHASE_PROXY_OBSERVABILITY_NOT_PHYSICS_CERTIFICATE',
          'config':p['config_pin'],'protocol':c['protocol'],'source_pins':c['source_pins'],'input_pins':inputs,'counts':counts,'converted_rows':2260,'scored_rows':1808,
          'actual_factory_passive_MJB_read':'1source-accounted inside factory, not separately instrumented; explicitmapping load1 counted above',
          'converter_gates':gates,'invariants':invariants,'statistics':stats,'primary_deepest_pairs':'EXACT all-tied deepest set; no firstargmax choice',
          'threshold_arithmetic':{'native_nominal_gate_m':.003,'proxy_FP32_scalar_allowance_m':float(np.float32(.003)),'unchanged_primary_values':True,'exact_FP32_boundary_flag_saved':True},
          'secondary_all_violating_pairs':'separately named overlapping flags, not replacing primarydeepest rule',
          'scope':'CPU FP32 originalP_ONLY on solverPREstate. Native label generatedHOcandidate depth>3mm, not positiveforce, selfcontact or global geometrytruth. Nsource1/Nseed1/Nego0, repeatedbranches dependent. No wholehidden/physicalgradient/forceclosure/taskquality upgrade.',
          'WM_Actor_weight_loads':0,'new_AD':0,'Env_physics_nativeFK_collision_GPU':0,'physical_gradient_qualified':False,'wholehiddenstate_qualified':False,'elapsed_seconds':time.perf_counter()-started}
        write(out,'report.json',report);write(out,'output_pins.json',{f.name:pin(f) for f in out.iterdir() if f.is_file()});return report
    except BaseException as e:
        write(out,'failure.json',{'schema':'full151105_same_solver_phase_penalty_CPU_failure:1','status':'FAILED_PARTIAL_PRESERVED_NO_RETRY','error':repr(e),'phase':phase,'counts':counts,'completed_scored_rows':completed,'remaining_rows':'NOT_RUN','invariants_available':invariants,'config':p['config_pin'],'new_physics_GPU':0});raise
