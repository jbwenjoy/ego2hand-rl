"""Compare completed smoke initial raw CPU parameters and common inputs only."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import torch

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):p=Path(p).resolve();return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def canon(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def same_tensor(a,b):return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def same_np(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',nargs='+',type=Path,required=True);p.add_argument('--run-dir',nargs='+',type=Path,required=True);p.add_argument('--health-report',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if len(a.config)!=len(a.run_dir) or len(a.config)!=len(a.health_report) or len(a.config)<2:raise ValueError('Two or more complete config/run/health triplets required')
    a.output.mkdir(parents=True,exist_ok=False);rows=[];checks={};differences={}
    for cp,rd,hp in zip(a.config,a.run_dir,a.health_report):
        c=read(cp);h=read(hp);raw=torch.load(rd/'initial_parameter_tensors.pt',map_location='cpu',weights_only=False);runtime=read(rd/'runtime.json');identity=read(rd.with_name(rd.name+'_watchdog')/'engineering_identity.json');controller=read(rd/'controller_descriptor.json')
        row={'config':c,'initial':raw,'runtime':runtime,'controller':controller,'run_dir':rd,'method':c['method'],'health':h,'config_pin':pin(cp),'health_pin':pin(hp),'initial_pin':pin(rd/'initial_parameter_tensors.pt'),'identity_pin':pin(rd.with_name(rd.name+'_watchdog')/'engineering_identity.json'),'effective':read(rd/'effective_training_config.json')};rows.append(row)
        checks[c['method']+'_completed_health_identity']=runtime['status']=='completed_rough_training' and identity['engineering_identity_qualified'] is True and h['status']=='PASS_SAVED_ENGINEERING_HEALTH' and h['config']==pin(cp) and h['checkpoint']['sha256']==runtime['checkpoint']['sha256']
    checks['unique_methods_shared_engineering_seed0']=len({r['method'] for r in rows})==len(rows) and all(r['config']['role']=='engineering_smoke' and r['config']['seed']==0 for r in rows)
    base=rows[0]
    for row in rows[1:]:
        label=row['method']+'_vs_'+base['method'];ca,cb=base['config'],row['config'];ignored={'method','contact_binding','output','source_pins'}
        checks[label+'_common_config_exact']=canon({k:v for k,v in ca.items() if k not in ignored})==canon({k:v for k,v in cb.items() if k not in ignored})
        checks[label+'_effective_native_config_exact']=canon(base['effective'])==canon(row['effective'])
        common=set(ca['source_pins'])&set(cb['source_pins']);mismatch=[k for k in sorted(common) if ca['source_pins'][k]!=cb['source_pins'][k]]
        checks[label+'_common_execution_source_pins_exact']=not mismatch
        differences[label]={'changed_common_source_pins':mismatch,'extra_source_pins':sorted(set(cb['source_pins'])-set(ca['source_pins'])),'base_extra_source_pins':sorted(set(ca['source_pins'])-set(cb['source_pins'])),'intended_different_fields':['method','contact_binding','output']}
        x,y=base['initial'],row['initial'];checks[label+'_initial_capture_metadata_exact']=x['schema']==y['schema']=='rough_prefix_initial_parameters:1' and x['capture_point']==y['capture_point']=='after_native_DexRunner_init_before_first_collect_or_optimizer' and x['source_pins']==y['source_pins'] and set(x['modules'])==set(y['modules'])
        module_counts={}
        for name,v in x['modules'].items():
            w=y['modules'][name];equal=v['parameter_names']==w['parameter_names'] and v['dtypes']==w['dtypes'] and v['shapes']==w['shapes'] and len(v['parameters'])==len(w['parameters']) and all(same_tensor(i,j) for i,j in zip(v['parameters'],w['parameters']))
            checks[label+'_all_initial_parameter_bytes_'+name]=equal;module_counts[name]={'tensors':len(v['parameters']),'parameters':sum(i.numel() for i in v['parameters']),'bit_exact':equal}
        differences[label]['raw_initial_modules']=module_counts
        for fn in ('actual_context_readbacks.npz','expected_context_fields.npz','fixed_physics_context.npz'):
            with np.load(base['run_dir']/fn,allow_pickle=False) as aa,np.load(row['run_dir']/fn,allow_pickle=False) as bb:equal=set(aa)==set(bb) and all(same_np(aa[k],bb[k]) for k in aa)
            checks[label+'_'+fn+'_actual_fields_bitexact']=equal
        ac,bc=base['controller'],row['controller'];fields=('baseline_u15_FP32_sha256','baseline_shape','margin_rad','target_lower_rad','target_upper_rad','native_reference_signature','pure_projection_sha256','controller_source_sha256','active_intervals','terminal_observation')
        checks[label+'_native_reference_baseline_C2_exact']=canon({k:ac[k] for k in fields})==canon({k:bc[k] for k in fields})
        checks[label+'_same_input_source_identity']=ac['root_descriptor']['input_bundle']==bc['root_descriptor']['input_bundle'] and ac['root_descriptor']['source_identity']==bc['root_descriptor']['source_identity'] and ac['root_descriptor']['source_pins']==bc['root_descriptor']['source_pins']
    checks['CPU_no_Env_noGPU']=not torch.cuda.is_initialized() and not any(k.split('.')[0] in ('warp','mujoco','mujoco_warp','mjlab','dex_supermimic') for k in sys.modules)
    report={'schema':'rough_prefix_smoke_raw_initial_fairness:1','status':'PASS_SAME_INITIAL_AND_COMMON_INPUTS' if all(checks.values()) else 'FAIL','checks':checks,'passed':sum(checks.values()),'total':len(checks),'participants':[{k:r[k] for k in ('method','config_pin','health_pin','initial_pin','identity_pin')} for r in rows],'comparison':differences,'source':pin(__file__),'GPU_used':False,'environment_created':False,'scope':'Engineering seed0 only. Raw constructor-return parameter bytes and saved actual8/common reference/control inputs match; final trained weights and task outcome are not equated. Does not add a training seed, establish physical benefit, or claim equal compute.'}
    out=a.output/'report.json';out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'failed':[k for k,v in checks.items() if not v],'report':pin(out)},indent=2));return int(not all(checks.values()))
if __name__=='__main__':raise SystemExit(main())
