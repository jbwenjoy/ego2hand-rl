"""Primary raw-initial comparison: healthy completed vs live-prefix runs.

V2 canonicalizes absolute path spelling before comparing aliases; SHA and
resolved core source path equality remain strict. V1 source/proof are retained.

Each side must already have a content-bound v3 health audit. This comparison
never requires the right run to be terminal or compares final trained weights.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
AUDITOR_SHA='7a6a4e21d24f5acad0eccc6908bef03fd7e1c4acaec7196edf296cc75772850f'
CORE_REL=('src/rough_prefix/native_v2.py','src/rough_prefix/training.py','src/rough_prefix/runtime_v2.py','src/rough_prefix/controller.py','src/rough_prefix/io.py','src/rough_prefix/metadata.py','src/rough_prefix/capacity.py','src/rough_prefix/bundle.py','src/rough_prefix/bundle_v2.py','vendor/runtime_io.py')
PURE=Path('D:/workspace/supermimic-c2-training-study/src/target_inset.py')
COMMON=('schema','role','seed','input_bundle','protocol','training','controller','evaluation_gates','automatic_retry','compile_fallback','hard_process_seconds','soft_worker_seconds')
CONTROLLER=('schema','kind','format_version','baseline_u15_FP32_sha256','baseline_shape','margin_rad','target_lower_rad','target_upper_rad','native_reference_signature','pure_projection_sha256','controller_source_sha256','active_intervals','terminal_observation')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def canon(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def checked(p):
    if set(p)!={'path','sha256'} or not Path(p['path']).is_absolute() or pin(p['path'])['sha256']!=p['sha256']:raise ValueError('Missing/changed content-bound artifact')
    return Path(p['path'])
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def same_array(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def same_tensors(a,b):return a.dtype==b.dtype and a.shape==b.shape and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()) and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def common_config(a,b):return canon({k:a[k] for k in COMMON})==canon({k:b[k] for k in COMMON})
def same_controls(a,b):return canon({k:a[k] for k in CONTROLLER})==canon({k:b[k] for k in CONTROLLER})
def module_checks(a,b):
    checks={'four_module_names':set(a)==set(b)=={'actor','wm','fail_head','critic'}};counts={}
    if not checks['four_module_names']:return checks,counts
    for name in ('actor','wm','fail_head','critic'):
        x,y=a[name],b[name];checks[name+'_named_parameter_bytes_exact']=x['parameter_names']==y['parameter_names'] and x['shapes']==y['shapes'] and x['dtypes']==y['dtypes'] and len(x['parameters'])==len(y['parameters']) and all(same_tensors(v,w) for v,w in zip(x['parameters'],y['parameters']));counts[name]={'tensor_count':len(x['parameters']),'parameter_count':sum(v.numel() for v in x['parameters'])}
    return checks,counts
def core_sources(config,runtime):
    pins=config['source_pins'];by_path={}
    for p in pins.values():
        key=str(Path(p['path']).resolve())
        normalized={'path':key,'sha256':p['sha256']}
        if key in by_path and by_path[key]!=normalized:raise ValueError('Conflicting SHA pins for same resolved source path')
        by_path[key]=normalized
    wanted=[ROOT/rel for rel in CORE_REL]+[PURE]
    entry=[p for p in by_path.values() if p['sha256']==runtime['source_sha256'] and Path(p['path']).name.startswith('train_rough_prefix_pen')]
    if len(entry)!=1:raise ValueError('Runtime actual entry source missing/ambiguous')
    wanted.append(Path(entry[0]['path']));selected={}
    for path in wanted:
        p=by_path[str(path.resolve())];checked(p);selected[str(path.resolve())]=p['sha256']
    dex=[p for k,p in pins.items() if k.startswith('dex__')];mjlab=[p for k,p in pins.items() if k.startswith('mjlab__')]
    if not dex or not mjlab:raise ValueError('Declared native Dex/mjlab code closure missing')
    for p in dex+mjlab:checked(p);selected[str(Path(p['path']).resolve())]=p['sha256']
    return selected,{'Dex_declared_pins':len(dex),'mjlab_declared_pins':len(mjlab),'common_explicit_study_and_C2_and_actual_entry':len(wanted)}
def health_identity(config_path,config,run,health):
    return health.get('schema')=='rough104_checkpoint_CPU_health:1' and health.get('status')=='PASS_SAVED_ENGINEERING_HEALTH' and bool(health.get('checks')) and all(type(x) is bool and x for x in health['checks'].values()) and health.get('config')==pin(config_path) and health.get('method')==config['method'] and type(health.get('seed')) is int and health['seed']==config['seed'] and 1<=health['iteration']<=config['training']['max_iterations'] and Path(health['run_dir']).resolve()==run.resolve() and health['auditor']['sha256']==AUDITOR_SHA
def participant(config_path,run,health_path):
    config=read(config_path);health=read(health_path)
    if not (config.get('role')=='primary_matched' and config.get('method') in ('A1L','P_ONLY','A2') and health_identity(config_path,config,run,health) and Path(config['output']).resolve()==run.resolve() and pin(run/'config.json')['sha256']==pin(config_path)['sha256']):raise ValueError('Primary health/config/run identity is not qualified')
    checked(health['checkpoint'])
    if Path(health['checkpoint']['path']).resolve().parent!=run.resolve()/'train':raise ValueError('Health checkpoint from another run')
    raw_path=run/'initial_parameter_tensors.pt'
    if pin(raw_path)!=health['details']['initial_parameters']['raw_initial_tensor_artifact']:raise ValueError('Raw initial parameters changed since independent health')
    raw=torch.load(raw_path,map_location='cpu',weights_only=False)
    if raw.get('schema')!='rough_prefix_initial_parameters:1' or raw.get('capture_point')!='after_native_DexRunner_init_before_first_collect_or_optimizer':raise ValueError('Wrong initial capture artifact')
    runtime=read(run/'runtime.json');core,coverage=core_sources(config,runtime);bundle=read(checked(config['input_bundle']));baseline=arrays(checked(bundle['source_pins']['control_arrays']))['control__fp32_old_export__tracking_u_exec_15_FP32_rad']
    return {'config':config,'config_pin':pin(config_path),'health':health,'health_pin':pin(health_path),'initial':raw,'initial_pin':pin(raw_path),'controller':read(run/'controller_descriptor.json'),'core':core,'core_coverage':coverage,'physical':{fn:arrays(run/fn) for fn in ('actual_context_readbacks.npz','expected_context_fields.npz','fixed_physics_context.npz')},'baseline':baseline,'bundle':bundle,'method':config['method'],'run':run}
def compare(a,b):
    checks={};details={};checks['common_primary_budget_input_protocol_controller_exact']=common_config(a['config'],b['config']);checks['same_seed_distinct_arms']=a['config']['seed']==b['config']['seed'] and a['method']!=b['method'];checks['actual_core_execution_source_paths_and_SHA_equal']=a['core']==b['core'];checks['native_reference_baseline_C2_limits_exact']=same_controls(a['controller'],b['controller']);checks['baseline_control_array_bitexact']=same_array(a['baseline'],b['baseline'])
    m,c=module_checks(a['initial']['modules'],b['initial']['modules']);checks.update(m);details['raw_initial_modules']=c
    checks['raw_capture_point_and_source_pins_exact']=a['initial']['capture_point']==b['initial']['capture_point'] and a['initial']['source_pins']==b['initial']['source_pins']
    for fn,aa in a['physical'].items():
        bb=b['physical'][fn];checks['same_'+fn]=set(aa)==set(bb) and all(same_array(aa[k],bb[k]) for k in aa)
    rolepins=('reference_arrays','init_arrays','control_arrays','contact_reference','native_data','export_source__model','export_source__actual_context')
    checks['same_reference_init_native_asset_pins']=all(a['bundle']['source_pins'][k]==b['bundle']['source_pins'][k] for k in rolepins)
    ka,kb=a['config']['source_pins'],b['config']['source_pins'];details['auth_provenance_source_differences_expected_not_core']={'different_common_keys':[k for k in sorted(set(ka)&set(kb)) if ka[k]!=kb[k]],'left_only_keys':sorted(set(ka)-set(kb)),'right_only_keys':sorted(set(kb)-set(ka))}
    details['core_source_coverage']={a['method']:a['core_coverage'],b['method']:b['core_coverage']};details['core_source_path_SHA_map']=a['core'];details['prior_declarations']={}
    for row in (a,b):
        cp=row['config']['contact_binding'];declaration={'contact_binding':cp,'executed_prior':row['method']!='A1L'}
        if cp is not None:
            spec=read(checked(cp));binding=read(checked(spec['binding_manifest']));declaration['prior_math_pins']={k:p for k,p in binding['source_pins'].items() if k.endswith('_math')};declaration['binding_module_sha256']=spec['expected_hashes']['binding_module_sha256'];declaration['role']='Separate frozen declared-geometry shaping, not part of common baseline controller/native WMloss';checks[row['method']+'_prior_pins_current']=all(checked(p) for p in declaration['prior_math_pins'].values())
        details['prior_declarations'][row['method']]=declaration
    return checks,details
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for side in ('left','right'):
        p.add_argument('--'+side+'-config',type=Path,required=True);p.add_argument('--'+side+'-run',type=Path,required=True);p.add_argument('--'+side+'-health',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);x=p.parse_args();x.output.mkdir(parents=True,exist_ok=False)
    try:
        a=participant(x.left_config.resolve(),x.left_run.resolve(),x.left_health.resolve());b=participant(x.right_config.resolve(),x.right_run.resolve(),x.right_health.resolve());checks,details=compare(a,b);checks['CPU_no_GPU_or_Env']=not torch.cuda.is_initialized() and not any(k.split('.')[0] in ('warp','mujoco','mujoco_warp','mjlab','dex_supermimic') for k in sys.modules)
        report={'schema':'primary_raw_initial_prefix_or_final_pair:1','status':'PASS_INITIAL_AND_COMMON_INPUTS_ONLY' if all(checks.values()) else 'FAIL_INITIAL_OR_COMMON_INPUTS','checks':checks,'passed':sum(checks.values()),'total':len(checks),'participants':[{'method':r['method'],'config':r['config_pin'],'health':r['health_pin'],'health_iteration':r['health']['iteration'],'health_final_iteration_scope':r['health']['final_iteration_scope'],'raw_initial':r['initial_pin'],'health_checkpoint':r['health']['checkpoint']} for r in (a,b)],'details':details,'source':pin(__file__),'GPU_used':False,'scope':'Constructor raw parameter bytes plus actual8/reference/analytic requested reset/control/source identity only. Right run may remain live; no right finalcapacity/result/weights requirement and no trainedweights comparison, equalcompute, task benefit or new independentmotion claim. Simulator reset states were source-enforced but not independently saved as allworld raw initialization by training.'}
    except Exception as exc:report={'schema':'primary_raw_initial_prefix_or_final_pair:1','status':'INVALID_OR_MISSING_EVIDENCE','error':repr(exc),'source':pin(__file__),'GPU_used':False}
    out=x.output/'report.json';out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'status':report['status'],'passed':report.get('passed'),'total':report.get('total'),'failed':[k for k,v in report.get('checks',{}).items() if not v],'error':report.get('error'),'report':pin(out)},indent=2));return int(report['status']!='PASS_INITIAL_AND_COMMON_INPUTS_ONLY')
if __name__=='__main__':raise SystemExit(main())
