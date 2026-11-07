"""SOURCE DRAFT: released two-component, all152/all64 CPU confirmation.

Metadata defaults never decode the sealed test. Explicit Root --run alone loads
three trusted component files and constructs two NEW eval native WM instances;
no original119f load, Actor, optimizer, AD, MJB, state conversion or physics.
X152 repeats directly captured snapshot_x8: branch PRE full58 was NOT captured.
"""
from __future__ import annotations
import copy
import csv
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from . import response_calibration105_v1 as c

ROOT=c.ROOT
SCHEMA='full151105_WM_component_confirmation_CPU_protocol:1'
ROLE='fixed_common_pre_component_confirmation_all_fresh152'
CRITERION='ALL64_RESPONSE_STRICTLY_LOWER_AND_ALL152_POINTWISE_NOT_HIGHER'
OUTPUT_PREFIX='full151105_WM_component_confirmation_CPU_'
PHASES=(24,74,124,148)
ARMS=c.ARMS
OWN={'consumer':ROOT/'src/full151_gradient/component_confirmation105_v1.py',
     'CLI':ROOT/'scripts/confirm_full151_WM_component_cpu_v1.py'}
INPUTS={'calibration_recipe','training_protocol','calibration_result','initial_component','initial_validation',
        'pointwise_component','response_component','final_checkpoint_saved_review','sealed_manifest','release',
        'test_full_config','test_full_report','test_full_arrays','test_branch_index','test_snapshot_records','test_full_engineering_review'}
RAW58={'q':(0,20,'rad'),'qd':(20,40,'rad/s'),'p':(40,43,'m'),
       'R_raw9':(43,52,'dimensionless'),'v':(52,55,'m/s'),'omega':(55,58,'rad/s')}
ARCH={'dynamics_dim':278,'contact_dim':15,'hidden':[512,512,512],'state_dim':58,
      'delta_dim':52,'physical_xi_dim':46,'amp':False,'precision':'FP32_CPU_SHADOW'}
require=c.require;read=c.read;pin=c.pin;checked=c.checked;same=c.same


def no_backend():
    require(not any(k.split('.')[0] in ('torch','warp','mujoco','mujoco_warp','mjlab','dex_supermimic') for k in sys.modules),'Fresh metadata-only process')


def validate_protocol(p,require_authorized=False):
    require(type(p) is dict and p.get('schema')==SCHEMA and p.get('role')==ROLE,'New component confirmation protocol')
    for key in ('execution_authorized','CPU_run_authorized','GPU_run_authorized'):
        require(type(p.get(key)) is bool,'Explicit bool: '+key)
    require(p['GPU_run_authorized'] is False and type(p.get('cpu_threads')) is int and p['cpu_threads']==4,'CPU4/noGPU')
    require(p.get('criterion')==CRITERION and p.get('arms')==list(ARMS) and type(p.get('checkpoint_iteration')) is int and p['checkpoint_iteration']==1000,'Fixed final arms/criterion')
    require(p.get('test_phases')==list(PHASES) and p.get('architecture')==ARCH and p.get('output_prefix')==OUTPUT_PREFIX,'Fixed fresh phases/native architecture')
    require(all(type(p['architecture'][key]) is type(value) for key,value in ARCH.items()),'Architecture JSON types, including amp bool')
    require(type(p.get('inputs')) is dict and set(p['inputs'])==INPUTS,'Exact16 semantic aliases')
    require(type(p.get('source_pins')) is dict and p['source_pins'],'Current source closure')
    if require_authorized:require(p['execution_authorized'] and p['CPU_run_authorized'],'Root explicit evaluation authorization required')
    return p


def declaration():
    no_backend()
    return {'schema':'full151105_WM_component_confirmation_declaration:1','status':'SOURCE_DRAFT_NOT_EXECUTED',
            'protocol_schema':SCHEMA,'role':ROLE,'input_aliases':sorted(INPUTS),'criterion':CRITERION,
            'architecture':ARCH,'arms':list(ARMS),'test_phases':list(PHASES),'cpu_threads':4,
            'source_pins':{k:pin(v) for k,v in OWN.items()},'actual_test_values_read_model_FW':0}


def metadata_preflight(protocol_path,output=None):
    """JSON/hash joins only. No branch_index/snapshot values or NPZ decoding."""
    no_backend();p=validate_protocol(read(protocol_path));i=p['inputs']
    for v in i.values():checked(v)
    final=read(checked(i['final_checkpoint_saved_review']));c.allchecks(final)
    require(final['schema']=='full151105_WM_component_saved_checkpoint_CPU_review:1' and final['status']=='PASS_SERIALIZED_COMPONENT_HEALTH_NOT_ACCURACY'
            and final['engineering_qualified'] is True and final['iteration']==1000 and final['final_iteration_scope'] is True,'Final1000 independent health')
    require(same(final['protocol'],i['training_protocol']) and same(final['initial_component'],i['initial_component']),'Actual component initial/protocol aliases')
    for arm,alias in zip(ARMS,('pointwise_component','response_component')):
        require(same(final['checkpoints'][arm],i[alias]),'Final component checkpoint identity: '+arm)
    recipe=read(checked(i['calibration_recipe']));training=read(checked(i['training_protocol']))
    require(recipe['recipe']==training['recipe']==c.RECIPE,'Original numerical recipe unchanged')
    result=read(checked(i['calibration_result']))
    require(result['status']=='COMPLETED_COMPONENT_CALIBRATION_NO_TEST' and result['completed_iterations_per_arm']==1000
            and result['fresh_test_evaluated'] is False and same(result['protocol'],i['training_protocol']),'Complete still-blind calibration result')
    train_run=Path(final['run_dir']).resolve(strict=True)
    for alias,name in (('initial_component','initial_component_state.pt'),('initial_validation','initial_validation.json'),('calibration_result','result.json')):
        require(checked(i[alias]).samefile(train_run/name),'Actual component source route')
    seal=read(checked(i['sealed_manifest']))
    require(seal['schema']=='full151105_fresh_calibration_response_sealed_manifest:1' and seal['status']=='SEALED_ENGINEERING540_PASSED_SCIENTIFIC_RESPONSE_NOT_EVALUATED'
            and seal['scientific_release_authorized'] is False and seal['branch_pre58_qualified'] is False and seal['whole_hidden_state_qualified'] is False,'Retained original seal scope')
    require(same(seal['producer_report'],i['test_full_report']) and same(seal['config'],i['test_full_config']) and same(seal['engineering_review'],i['test_full_engineering_review'])
            and same(seal['frozen_before_collection_recipe'],i['calibration_recipe']),'Seal/source/recipe aliases')
    for v in seal['all_output_files'].values():checked(v)
    test_run=Path(seal['run_dir']).resolve(strict=True)
    for alias,name in (('test_full_arrays','physical_arrays.npz'),('test_branch_index','branch_index.json'),('test_snapshot_records','snapshot_records.json')):
        require(same(i[alias],seal['all_output_files'][name]) and checked(i[alias]).samefile(test_run/name),'Exact sealed test artifact: '+alias)
    engineering=read(checked(i['test_full_engineering_review']));c.allchecks(engineering)
    require(engineering['schema']=='full151105_fresh_response_engineering_saved_independent:1' and engineering['engineering_qualified'] is True
            and engineering['completed_branches']==152 and engineering['baseline_count']==24 and engineering['episode_step_returned']==1808
            and engineering['snapshot_phases']==list(PHASES) and engineering['branch_PRE_full58_qualified'] is False
            and engineering['WM_accuracy_or_response_benefit_evaluated'] is False,'Fresh engineering only, no inferred branchPRE58')
    require(same(engineering['config'],i['test_full_config']) and same(engineering['producer_report'],i['test_full_report']),'Engineering actual source')
    release=read(checked(i['release']))
    require(release['schema']=='full151105_WM_common_pre_confirmation_release:1' and release['status']=='RELEASED_FOR_FROZEN_TWO_ARM_EVAL_ALL152_64_NOT_TUNING'
            and release['scientific_release_authorized'] is True and release['evaluation_only'] is True
            and release['all77_sealed_files_current'] is True and release['models_fixed_before_release'] is True,'Explicit Root release')
    for key,alias in (('sealed_manifest','sealed_manifest'),('final_checkpoint_saved_review','final_checkpoint_saved_review'),
                      ('calibration_recipe','calibration_recipe'),('training_protocol','training_protocol'),
                      ('test_full_report','test_full_report'),('test_full_engineering_review','test_full_engineering_review')):
        require(same(release[key],i[alias]),'Release alias: '+key)
    for arm,alias in zip(ARMS,('pointwise_component','response_component')):require(same(release['checkpoint_pins'][arm],i[alias]),'Release fixes both models')
    sources=c.canonical_sources(list(p['source_pins'].values())+list(training['source_pins'].values())+[pin(v) for v in OWN.values()])
    for f in OWN.values():require(any(same(pin(f),v) for v in p['source_pins'].values()),'Current evaluation code not frozen in protocol')
    out=None if output is None else Path(output).resolve()
    if out is not None:require(out.parent==ROOT/'outputs' and out.name.startswith(OUTPUT_PREFIX) and not out.exists(),'Fresh scoped confirmation output')
    no_backend()
    return {'status':'PASS_RELEASED_COMPONENT_CONFIRMATION_HASH_JSON_ONLY','protocol':p,'protocol_pin':pin(protocol_path),
            'sources':sources,'output':out,'sealed_file_count':len(seal['all_output_files']),
            'test_values_decoded':0,'model_loads_forwards':0}


def prepare_test(a,rows,snaps,np):
    """Pure mechanical mapping. snapshot_x8 is input design, NOT observed PRE152."""
    for key,shape in (('snapshot_x',(8,58)),('snapshot_u',(8,20)),('branch_x_next',(152,58)),('branch_u',(152,20)),
                      ('direction__centers',(8,20)),('direction__minus',(8,4,2,20)),('direction__plus',(8,4,2,20))):
        require(a[key].shape==shape and a[key].dtype==np.float32 and np.isfinite(a[key]).all(),'Finite native FP32 '+key)
    require(len(rows)==152 and len(snaps)==8,'All original branches/snapshots')
    require(all(a[k].dtype==np.bool_ and a[k].shape==shape and a[k].all() for k,shape in
                (('completed',(152,)),('snapshot_completed',(8,)),('restore_qualified',(152,)),('restore_inventory_bitexact',(152,)))),'Full nonvacuous empirical engineering masks')
    for key in ('direction__d_eff','direction__center_offset'):
        require(a[key].dtype==np.float64 and a[key].shape==(8,4,2,20) and np.isfinite(a[key]).all(),'Original finite F64 '+key)
    require(a['direction__clipped'].dtype==np.bool_ and a['direction__clipped'].shape==(8,4,2),'Original clipping mask ABI')
    require(np.array_equal(a['snapshot_u'].view(np.uint32),a['direction__centers'].view(np.uint32)),'Original centers')
    mi=[];pl=[];du=[];identities=[];pairs=[]
    u=a['branch_u'].reshape(8,19,20)
    require(np.array_equal(u[:,:3].view(np.uint32),np.broadcast_to(a['snapshot_u'][:,None],(8,3,20)).view(np.uint32)),'Three baseline controls byte equal')
    for s in range(8):
        path=('A0','A2')[s//4];k=PHASES[s%4]
        require(snaps[s]['snapshot_id']==s and snaps[s]['path']==path and snaps[s]['k']==k and snaps[s]['baseline3_qualified'] is True,'Exact fresh snapshot identity')
        for off in range(19):
            bid=s*19+off;row=rows[bid];base=off<3
            ei=None if base else (off-3)//8;j=None if base else ((off-3)%8)//2;sign=0 if base else (-1,1)[(off-3)%2]
            expected={'row_id':bid,'snapshot_id':s,'path':path,'k':k,'type':'baseline' if base else 'direction',
                      'baseline_repeat':off if base else None,'j':j,'epsilon_index':ei,'sign':sign,
                      'epsilon_rad':None if base else (.002,.001)[ei],'joint_index':None if base else (0,4,8,12)[j]}
            require(all(type(row.get(key)) is type(v) and row[key]==v for key,v in expected.items()),'All152 original EPS-major branch order')
            require(row['status']=='COMPLETED' and row['restorequalified'] is True,'Completed original branch')
            source=np.asarray(row['actualu'],np.float64);cast=source.astype(np.float32)
            require(source.shape==(20,) and np.array_equal(cast.astype(np.float64),source)
                    and np.array_equal(cast.view(np.uint32),a['branch_u'][bid].view(np.uint32)),'Exact original JSON/FP32 actualu')
            if not base:
                q=(0,4,8,12)[j];mask=np.arange(20)!=q
                require(np.array_equal(u[s,off,mask].view(np.uint32),a['snapshot_u'][s,mask].view(np.uint32)),'Non-j19 controls unchanged')
                endpoint=a['direction__plus' if sign==1 else 'direction__minus'][s,j,ei]
                require(np.array_equal(endpoint.view(np.uint32),u[s,off].view(np.uint32)),'Direction endpoint bytes')
            identities.append(expected)
        for j,q in enumerate((0,4,8,12)):
            for ei,eps in enumerate((.002,.001)):
                minus=s*19+3+8*ei+2*j;plus=minus+1
                delta=a['branch_u'][plus].astype(np.float64)-a['branch_u'][minus].astype(np.float64)
                effective=delta/(2*eps);offset=(a['branch_u'][plus].astype(np.float64)+a['branch_u'][minus].astype(np.float64))*.5-a['snapshot_u'][s].astype(np.float64)
                require(np.array_equal(effective,a['direction__d_eff'][s,j,ei]) and np.array_equal(offset,a['direction__center_offset'][s,j,ei]),'Original dimensionless direction and center offset')
                unit=np.eye(20,dtype=np.float32)[q];center=a['snapshot_u'][s]
                clipped=not(np.array_equal(a['branch_u'][plus],center+np.float32(eps)*unit) and np.array_equal(a['branch_u'][minus],center-np.float32(eps)*unit))
                require(bool(a['direction__clipped'][s,j,ei])==clipped,'Original numeric-equality clipping flag')
                mi.append(minus);pl.append(plus);du.append(delta[q])
                pairs.append({'pair_id':len(pairs),'snapshot_id':s,'path':path,'k':k,'j':j,'joint_index':q,'epsilon_index':ei,
                              'requested_epsilon_rad':eps,'minus_branch_id':minus,'plus_branch_id':plus,'actual_du_rad':float(delta[q]),'clipped':clipped,
                              'center_offset_rad':offset.tolist(),'legacy_d_eff_dimensionless':effective.tolist()})
    return {'x152_FP32':np.repeat(a['snapshot_x'],19,axis=0),'target152_FP32':a['branch_x_next'].copy(),'actual_u152_FP32':a['branch_u'].copy(),
            'minus_indices':np.asarray(mi,np.int64),'plus_indices':np.asarray(pl,np.int64),'actual_du_rad_F64':np.asarray(du,np.float64)},identities,pairs


def _bits(a,b,torch):
    return a.dtype==b.dtype and a.shape==b.shape and c._tensor_digest({'x':a},torch)==c._tensor_digest({'x':b},torch)


def make_eval_component(payload,initial,arm,torch):
    """New artifact validation; never relabel as105 to call the old consumer."""
    require(payload.get('schema')==c.ARTIFACT_SCHEMA and payload.get('arm')==arm and type(payload.get('iteration')) is int and payload['iteration']==1000,'Actual final1000 component namespace')
    require('format_version' not in payload and c._finite_tree(payload['WM_state'],torch),'Not a native105 resume checkpoint')
    require(same(payload['parent_checkpoint105'],initial['parent_checkpoint105']),'Same initial A2 parent')
    require(_bits(payload['physical_xi46'],initial['physical_xi46'],torch) and _bits(payload['frozen_s52'],initial['frozen_s52'],torch),'Exact shared xi/s52')
    norm=lambda state:{k:v for k,v in state.items() if k.startswith(('norm.','delta_norm.')) or k=='error_floor'}
    initial_norm_sha=c._tensor_digest(norm(initial['WM']),torch)
    require(c._tensor_digest(norm(payload['WM_state']),torch)==initial_norm_sha
            and payload['frozen_norm_sha256']==initial_norm_sha,'Frozen initial normalizers')
    require(payload['initial_WM_sha256']==c._tensor_digest(initial['WM'],torch) and payload['recipe']==c.RECIPE,'Fixed initial model and exact recipe')
    require(payload['counts']=={'WM_forwards':1000,'backward_calls':1000,'optimizer_steps':1000},'Actual final component update counters')
    require(payload['normalizer_host_configuration']==initial['normalizer_host_configuration'],'Frozen normalizer host config')
    from dex_supermimic.rl.dex_supermimic.networks import WorldModel
    with torch.random.fork_rng(devices=[]):
        wm=WorldModel(dynamics_dim=278,contact_dim=15,hidden=(512,512,512),amp=False)
        expected=wm.state_dict();weights=payload['WM_state']
        require(set(weights)==set(expected),'Exact native WM key set')
        require(all(isinstance(weights[k],torch.Tensor) and weights[k].device.type=='cpu' and weights[k].dtype==v.dtype and weights[k].shape==v.shape
                    and bool(torch.isfinite(weights[k]).all()) for k,v in expected.items()),'Exact WM architecture/dtype/finite state')
        wm.load_state_dict(weights,strict=True);wm.eval().requires_grad_(False)
    require(c._norm_config(wm)==payload['normalizer_host_configuration'],'Native ctor loaded norm host config')
    require(c._tensor_digest(wm.state_dict(),torch)==c._tensor_digest(weights,torch),'Strict component weights exact')
    require(_bits(torch.maximum(wm.delta_norm.std(),wm.error_floor),initial['frozen_s52'],torch),'Native frozen scale52 formula exact')
    return wm


def state_statistics(pred,target,np):
    error=pred.astype(np.float64)-target.astype(np.float64);summary={}
    for name,(lo,hi,unit) in RAW58.items():
        v=error[:,lo:hi];item={'unit':unit,'component_RMSE':float(np.sqrt(np.mean(v*v))),
            'MAE':float(np.mean(np.abs(v))),'maximum_absolute':float(np.max(np.abs(v))),'rows':152}
        if name in ('p','v','omega'):item['vector_RMS']=float(np.sqrt(np.mean(np.sum(v*v,axis=-1))))
        if name=='R_raw9':item['Frobenius_RMS']=float(np.sqrt(np.mean(np.sum(v*v,axis=-1))))
        summary[name]=item
    return error,summary


def confirmation_decision(metrics):
    a,b=(metrics[k] for k in ARMS)
    require(all(type(m[k]) is float and __import__('math').isfinite(m[k]) for m in (a,b) for k in ('pointwise_loss','common_pre_pair_loss')),'Finite full metrics')
    response=b['common_pre_pair_loss']<a['common_pre_pair_loss'];point=b['pointwise_loss']<=a['pointwise_loss']
    return {'criterion':CRITERION,'response_MSE_strictly_lower':response,'pointwise_MSE_not_higher':point,
            'local_component_confirmation':response and point,'all152_rows_all64_pairs_used':True,
            'policy_benefit_or_physical_gradient_qualified':False}


def save_raw_rng(out,label,rng,np):
    """Inspectable raw RNG values, not only a pickle-persistence hash claim."""
    np.savez_compressed(out/(label+'_RNG_arrays.npz'),numpy_MT=rng['numpy'][1],torch_CPU=rng['torch_CPU'].numpy())
    c._write_json(out/(label+'_RNG_python_numpy.json'),{'python':rng['python'],
        'numpy_header':[rng['numpy'][0],*rng['numpy'][2:]],'CUDA_initialized':rng['CUDA_initialized']})


def run_confirmation(protocol_path,output):
    started=time.perf_counter();m=metadata_preflight(protocol_path,output);p=validate_protocol(m['protocol'],True);i=p['inputs'];out=m['output'];out.mkdir()
    c._write_json(out/'protocol.json',p);c._write_json(out/'source_inventory.json',m['sources'])
    with zipfile.ZipFile(out/'execution_sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for k,v in m['sources'].items():z.write(checked(v),k)
    counts={'trusted_component_torch_loads':0,'native_WM_constructors':0,'WM_forwards':0,'predicted_rows':0,'AD_calls':0,'original119f_loads':0,'MJB_Env_physics_GPU':0}
    phase='released_saved_inputs'
    try:
        os.environ['CUDA_VISIBLE_DEVICES']='-1'
        import numpy as np
        import torch
        torch.set_num_threads(4)
        from dex_supermimic.rl.dex_supermimic.state import difference
        rng=c._rng(torch,np);rng_sha=c._rng_digest(rng)
        save_raw_rng(out,'before',rng,np)
        with np.load(checked(i['test_full_arrays']),allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
        data,rows,pairs=prepare_test(a,read(checked(i['test_branch_index'])),read(checked(i['test_snapshot_records'])),np)
        np.savez_compressed(out/'matched_test_inputs152.npz',**data)
        c._write_json(out/'all152_identities.json',rows);c._write_json(out/'all64_pairs.json',pairs)
        require(np.isfinite(data['actual_du_rad_F64']).all() and (data['actual_du_rad_F64']>0).all(),'Undefined/reversed pair denominator blocks complete confirmation; no subset')
        phase='trusted_component_files'
        initial=torch.load(checked(i['initial_component']),map_location='cpu',weights_only=False);counts['trusted_component_torch_loads']+=1
        require(initial['schema']==c.ARTIFACT_SCHEMA,'Trusted isolated initial schema')
        initial_receipt=read(checked(i['initial_validation']))
        require(c._tensor_digest(initial['WM'],torch)==initial_receipt['initial_WM_sha256'],'Original validated parent state')
        require(initial['parent_checkpoint105']['sha256']==c.FROZEN_CHECKPOINT_SHA
                and same(initial['parent_checkpoint105'],initial_receipt['strict_loader']['checkpoint']),'Original strict105 parent identity')
        norm={k:v for k,v in initial['WM'].items() if k.startswith(('norm.','delta_norm.')) or k=='error_floor'}
        require(c._tensor_digest(norm,torch)==initial_receipt['frozen_norm_sha256']
                and c._tensor_digest({'xi':initial['physical_xi46']},torch)==initial_receipt['xi46_sha256']
                and initial['normalizer_host_configuration']==initial_receipt['normalizer_host_configuration'],'Initial norm/xi receipt exact')
        payloads={}
        for arm,alias in zip(ARMS,('pointwise_component','response_component')):
            payloads[arm]=torch.load(checked(i[alias]),map_location='cpu',weights_only=False);counts['trusted_component_torch_loads']+=1
            require(same(payloads[arm]['protocol'],i['training_protocol']),'Component payload actual run protocol')
        scale=initial['frozen_s52'];xi=initial['physical_xi46']
        require(scale.dtype==xi.dtype==torch.float32 and scale.shape==(52,) and xi.shape==(46,)
                and bool(torch.isfinite(scale).all()) and bool((scale>0).all()) and bool(torch.isfinite(xi).all()),'Fixed initial xi46/s52')
        x,target,u=(torch.from_numpy(data[k]) for k in ('x152_FP32','target152_FP32','actual_u152_FP32'))
        minus,plus=(torch.from_numpy(data[k]) for k in ('minus_indices','plus_indices'));du=torch.from_numpy(data['actual_du_rad_F64'])
        with torch.inference_mode():
            target_common=difference(x,target)
            target_chart=c.chart_guard(x,target,target_common,.01,torch)
        memory_inputs={'x':x,'target':target,'u':u,'xi':xi,'scale':scale,'minus':minus,'plus':plus,'du':du,'target_common':target_common}
        memory_before=c._tensor_digest(memory_inputs,torch)
        metrics={};state_stats={};weight_hashes={};forward_checks={}
        for arm in ARMS:
            phase='component_'+arm
            model=make_eval_component(payloads[arm],initial,arm,torch);counts['native_WM_constructors']+=1
            before=c._tensor_digest(model.state_dict(),torch);norm_before=c._tensor_digest(c._norm_state(model),torch);cfg_before=c._norm_config(model)
            diagnostics={}
            with torch.inference_mode():
                dynamics=torch.cat((x,xi[None].expand(152,-1),x.new_zeros((152,174))),-1)
                prediction,self15=model(x,u,dynamics);counts['WM_forwards']+=1;counts['predicted_rows']+=len(prediction)
                require(prediction.shape==(152,58) and self15.shape==(152,15) and bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(self15).all()),'Finite fullbatch native component output')
                point,pair,charts=c.common_pre_losses(prediction,target,x,scale,target_common,minus,plus,du,difference,torch,diagnostics=diagnostics)
            pred=prediction.numpy().copy();error,stats=state_statistics(pred,data['target152_FP32'],np)
            np.savez_compressed(out/(arm+'_evaluation_arrays.npz'),pred152_FP32=pred,self15_logits_FP32=self15.numpy(),
                target152_FP32=data['target152_FP32'],point_delta52_FP32=diagnostics['point_delta52_FP32'].numpy(),
                pred_common52_FP32=diagnostics['pred_common52_FP32'].numpy(),target_common52_FP32=target_common.numpy(),
                scale52_FP32=scale.numpy(),actual_du_rad_F64=du.numpy(),minus_indices=minus.numpy(),plus_indices=plus.numpy(),
                state_error58_F64=error)
            # Full branch/feature coverage, no mixed-unit58 aggregate.
            with (out/(arm+'_all152_state_error58.csv')).open('x',newline='',encoding='utf-8') as f:
                writer=csv.writer(f);writer.writerow(['branch_id','snapshot_id','path','k','feature','unit','predicted','target','error_F64'])
                for row in rows:
                    for _,(lo,hi,unit) in RAW58.items():
                        for feature in range(lo,hi):
                            bid=row['row_id'];writer.writerow([bid,row['snapshot_id'],row['path'],row['k'],feature,unit,float(pred[bid,feature]),float(data['target152_FP32'][bid,feature]),float(error[bid,feature])])
            metrics[arm]={'pointwise_loss':float(point),'common_pre_pair_loss':float(pair),'point_rows':152,'pairs':64,'chart':charts}
            state_stats[arm]=stats;weight_hashes[arm]={'before':before,'after':c._tensor_digest(model.state_dict(),torch)}
            forward_checks[arm]=weight_hashes[arm]['before']==weight_hashes[arm]['after'] and norm_before==c._tensor_digest(c._norm_state(model),torch) and cfg_before==c._norm_config(model)
            require(forward_checks[arm],'Eval mutated component weights/normalizers')
            del model
        rng_after=c._rng(torch,np);save_raw_rng(out,'after',rng_after,np)
        require(c._rng_digest(rng_after)==rng_sha,'RNG changed; no restoration to conceal it')
        memory_after=c._tensor_digest(memory_inputs,torch)
        require(memory_after==memory_before,'Input/xi/scale tensor bytes changed')
        for v in i.values():checked(v)
        for v in m['sources'].values():checked(v)
        require(same(pin(protocol_path),m['protocol_pin']),'Confirmation protocol changed')
        decision=confirmation_decision(metrics)
        report={'schema':'full151105_WM_component_confirmation_CPU_analysis:1','status':'COMPLETED_ALL152_ALL64_COMPONENT_CONFIRMATION_NOT_POLICY_BENEFIT',
            'protocol':m['protocol_pin'],'input_pins':i,'counts':counts,'native52_frozen_criterion_metrics':metrics,
            'raw58_state_error_original_units':state_stats,'decision':decision,'target_chart':target_chart,
            'invariants':{'component_weights_normalizers_unchanged':forward_checks,'RNG_unchanged_without_repair':True,'all_inputs_sources_unchanged':True,'input_x_u_target_xi_s52_bytes_unchanged':True},
            'raw_RNG_saved_before_after':True,'RNG_digests':{'before':rng_sha,'after':c._rng_digest(rng_after)},
            'input_memory_digests':{'before':memory_before,'after':memory_after},
            'weight_digests':weight_hashes,'branch_PRE_full58_directly_observed':False,
            'input_design':'Direct native snapshot_x8 repeated19; conditioned on engineering540/primitive restoration, not an independently observed PRE58 per branch.',
            'whole_hidden_state_qualified':False,'physical_gradient_qualified':False,'physical_reference_qualified':False,
            'independent_ego_N':0,'source_lineage_N':1,'scope':'Single-lineage/seed known paths; fresh held-out perturbation responses at preregistered phases, not unseen-state/trajectory/ego generalization. No native policy or original GPUAMP gradient replay. self15 logits are not HO force labels.',
            'elapsed_seconds':time.perf_counter()-started}
        c._write_json(out/'report.json',report);c._write_json(out/'output_pins.json',{f.name:pin(f) for f in out.iterdir() if f.is_file()})
        return report
    except BaseException as error:
        try:c._write_json(out/'failure.json',{'schema':'full151105_WM_component_confirmation_failure:1','status':'BLOCKED_NO_COMPONENT_CONFIRMATION_NO_ROW_REPAIR','phase':phase,'error':repr(error),'counts':counts,'protocol':m['protocol_pin'],'criterion_evaluated':False})
        except BaseException as save_error:
            try:error.add_note('Failure evidence save also failed: '+repr(save_error))
            except BaseException:pass
        raise
