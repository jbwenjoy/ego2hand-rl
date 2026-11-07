"""One authorized two-dt zero-callback migration comparison; no NN/learning.

All commands/assets and expected step counts come from the existing qualified
zero report. New native data is written to a fresh run directory and compared
with every saved reference NPZ array. No old experiment source is imported.
"""
from pathlib import Path
import argparse,json,os,sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def need(v,msg):
    if not v:raise ValueError(msg)

def run(manifest,zero_report,output):
    from supermimic_research.artifacts import ArtifactRegistry,file_pin
    from supermimic_research.runtime import native_evaluation as E
    registry=ArtifactRegistry(manifest)
    reference=Path(zero_report) if zero_report is not None else registry.roots['ego']/'phone_tripod_policy_evaluation_v1/zero_qualified_run2/report.json'
    old=json.loads(reference.read_text());need(old['status']=='PASS_CALLBACK_MATCHES_FROZEN_NATIVE_LOOP' and old['passed']==old['total']==12 and all(v is True for v in old['checks'].values()),'Existing actual two-dt zero qualification')
    protocol=json.loads(registry.resolve_pin(old['protocol']).read_text());need(protocol['dt_s']==E.DTS and len(old['cases'])==2,'Exactly existing two commands/dts')
    bundle_path=registry.resolve_pin(protocol['inputs']['bundle']);bundle=json.loads(bundle_path.read_text())
    model_path=registry.resolve_pin(protocol['inputs']['model']);binding=json.loads(registry.resolve_pin(protocol['inputs']['binding']).read_text())
    catalog_path=registry.resolve_pin(protocol['inputs']['catalog']);catalog=json.loads(catalog_path.read_text());metadata_path=registry.resolve_pin(catalog['provider_metadata']);metadata=json.loads(metadata_path.read_text())
    expected_steps=sum(c['finite_steps_saved'] for c in old['cases'])
    need(not output.exists() and output.parent==ROOT/'runs','Fresh runs/migration_native_loop only')
    output.mkdir(parents=True)
    intent=dict(schema='phone_native_migration_validation_intent:1',reference_report=file_pin(reference),reference_protocol=old['protocol'],artifact_manifest=file_pin(manifest),
        commands=[dict(dt_s=c['dt_s'],command=c['command'],trace=c['trace'],expected_steps=c['finite_steps_saved']) for c in old['cases']],expected_steps=expected_steps,
        source=file_pin(__file__),native_evaluation_source=file_pin(E.__file__),NN_constructors=0,NN_forward=0,optimizer_updates=0,automatic_retry=False)
    E.write(output/'run_intent.json',intent)
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch,mujoco
    torch.set_num_threads(4);torch.set_default_dtype(torch.float32)
    need(not torch.cuda.is_initialized(),'CPU only')
    from supermimic_research.runtime.directional import DirectionalGapProvider
    from supermimic_research.runtime.phone_task import PhoneTask
    from supermimic_research.runtime.controller import TaskController
    from supermimic_research.learning.imagination import CanonicalMemoryTask
    model=mujoco.MjModel.from_binary_path(str(model_path))
    phone,pg,table,pads=E.assert_model(model,mujoco,np,binding);digest=E.model_digest_except_dt(model,np)
    geometry=DirectionalGapProvider(catalog_path,metadata_path=metadata_path,geometry_arrays_path=registry.resolve_pin(metadata['geometry_arrays']),direction_arrays_path=registry.resolve_pin(catalog['direction_arrays']))
    task=CanonicalMemoryTask(PhoneTask(bundle_path,mode='A1_tracking',geometry=geometry,reference_arrays_path=registry.resolve_pin(bundle['reference_arrays'])))
    counts=dict(MJB_reads=1,MjData=0,initial_forward=0,integration_steps=0,contact_force_reads=0,controller_calls=0,controller_receipt_reads=0,NN_calls=0,observer_forward=0)
    cases=[];comparisons=[];phase='native_two_dt';started=time.perf_counter()
    try:
        for i,oldcase in enumerate(old['cases']):
            h=oldcase['dt_s'];variant=dict(oldcase['command']);phase='dt_'+str(h)
            need(np.array(variant['qpos27']).tobytes()==np.array(bundle['initial']['qpos27_F64']).tobytes() and np.array(variant['ctrl20']).tobytes()==np.array(bundle['initial']['ctrl20_F64']).tobytes(),'Same frozen task q0/u0')
            folder=output/('dt_'+str(h));folder.mkdir();callback=TaskController(task,geometry,dt=h,phone=phone,actor=None,device='cpu')
            summary,endpoint=E.evaluate_case_controlled(model,variant,h,folder,phone,pg,table,pads,mujoco,np,counts,callback)
            E.write(folder/'controller_records.json',callback.records)
            with np.load(registry.resolve_pin(oldcase['trace']),allow_pickle=False) as ref,np.load(folder/'trace.npz',allow_pickle=False) as actual:
                need(set(ref.files)==set(actual.files),'Same full native trace ABI')
                per_array={key:bool(ref[key].dtype==actual[key].dtype and ref[key].shape==actual[key].shape and ref[key].tobytes()==actual[key].tobytes()) for key in ref.files}
            checks=dict(all_trace_arrays_raw_bytes_equal=all(per_array.values()),original_step_abort_exact=summary['finite_steps_saved']==oldcase['finite_steps_saved'] and summary['integration_steps']==oldcase['integration_steps'] and summary['termination']==oldcase['termination'],
                original_metrics_gates_exact=summary['metrics']==oldcase['metrics'] and summary['gates']==oldcase['gates'] and summary['old_two_pad_metrics']==oldcase['old_two_pad_metrics'] and summary['old_two_pad_gates']==oldcase['old_two_pad_gates'],
                source_failed_holding_remains_failed=summary['nominal_holding_dynamics_pass']==oldcase['nominal_holding_dynamics_pass'] and summary['old_two_pad_pass']==oldcase['old_two_pad_pass'],
                zero_Actor_calls=callback.NN_FW==0)
            comparisons.append(dict(dt_s=h,checks=checks,per_array=per_array,reference_trace=oldcase['trace'],new_trace=summary['trace'],expected_steps=oldcase['finite_steps_saved'],actual_steps=summary['finite_steps_saved'],termination=summary['termination'],controller_calls=len(callback.records)))
            cases.append(summary)
            print(json.dumps(dict(dt_s=h,steps=summary['finite_steps_saved'],termination=summary['termination'],raw_trace_equal=all(per_array.values()))),flush=True)
        checks={f'dt{i}_{key}':value for i,row in enumerate(comparisons) for key,value in row['checks'].items()}
        checks['model_unchanged_except_dt']=digest==E.model_digest_except_dt(model,np)
        checks['expected_steps_two_reset_no_observer_NN_GPU']=counts['integration_steps']==expected_steps and counts['initial_forward']==counts['MjData']==2 and counts['observer_forward']==counts['NN_calls']==0 and not torch.cuda.is_initialized()
        result=dict(schema='phone_native_runtime_migration_validation:1',status='PASS_TWO_DT_ZERO_CALLBACK_NATIVE_TRACE_BYTE_PARITY' if all(checks.values()) else 'FAIL_PRESERVED_NO_RETRY',
            checks=checks,passed=sum(checks.values()),total=len(checks),source=file_pin(__file__),intent=file_pin(output/'run_intent.json'),artifact_manifest=file_pin(manifest),reference_report=file_pin(reference),
            native_evaluation_source=file_pin(E.__file__),controller_source=file_pin(sys.modules[TaskController.__module__].__file__),memory_wrapper_source=file_pin(sys.modules[CanonicalMemoryTask.__module__].__file__),
            cases=cases,comparisons=comparisons,counts=counts,NN_constructors=0,NN_forward=0,optimizer_updates=0,GPU_calls=0,render_or_observer_forward=0,elapsed_s=time.perf_counter()-started,
            scope='Exactly two prescribed zero-action migration replays of existing qualified cases. Original source failures, aborts, null tails and holding gates remain. This is numerical/native code-port equivalence, not a new policy/holding success, parameter calibration or ego generalization.')
        E.write(output/'report.json',result)
        return result
    except BaseException as e:
        E.write(output/'failure.json',dict(phase=phase,error=repr(e),traceback=traceback.format_exc(),counts=counts,completed_cases=cases,automatic_retry=False));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,default=ROOT/'configs/artifacts.toml');parser.add_argument('--zero-report',type=Path);parser.add_argument('--output',type=Path,default=ROOT/'runs/migration_native_loop');parser.add_argument('--report',type=Path,default=ROOT/'provenance/native_validation.json');a=parser.parse_args()
    need(not a.report.exists(),'Fresh provenance outcome, retain previous attempts')
    try:result=run(a.manifest,a.zero_report,a.output.resolve())
    except BaseException as e:
        a.report.write_text(json.dumps(dict(schema='phone_native_runtime_migration_validation:1',status='FAILED_PRESERVED_NO_RETRY',error=repr(e),traceback=traceback.format_exc()),indent=2),encoding='utf8');raise
    a.report.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(dict(report=str(a.report),passed=result['passed'],total=result['total'],counts=result['counts'])))
    raise SystemExit(0 if result['passed']==result['total'] else 1)
