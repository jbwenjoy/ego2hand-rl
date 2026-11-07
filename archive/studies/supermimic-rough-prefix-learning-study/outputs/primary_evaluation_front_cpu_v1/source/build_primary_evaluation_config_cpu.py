"""CPU-only builder for explicit primary2000 Actor evaluation, always Gfalse."""
from __future__ import annotations
import argparse,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rough_prefix.io import read,pin,checked,canonical
from rough_prefix.metadata import write


def build_config(training_config,health_report,actor_load_report,protocol,version,evidence=()):
    """Return the existing tested evaluation schema; never load Torch/weights."""
    if not re.fullmatch(r'[A-Za-z0-9_]+',version):raise ValueError('New simple version label required')
    training_config=Path(training_config).resolve(strict=True);health_report=Path(health_report).resolve(strict=True);actor_load_report=Path(actor_load_report).resolve(strict=True);protocol=Path(protocol).resolve(strict=True)
    train=read(training_config);health=read(health_report);load=read(actor_load_report);p=read(protocol)
    from rough_prefix.probe import GATES
    def equal_fields(value,expected):return canonical({key:value.get(key) for key in expected})==canonical(expected)
    if not equal_fields(train,{'schema':'rough_prefix_native_run:1','role':'primary_matched','seed':0}) or train['method'] not in ('A1L','P_ONLY','A2') or type(train['training'].get('max_iterations')) is not int or train['training']['max_iterations']!=2000:raise ValueError('Only declared primary_matched seed0 final2000 training is supported')
    if train['controller']!={'kind':'ROUGH_PREFIX_BASELINE_EMA_C2','format_version':104,'margin_rad':.020,'resume':'from_scratch_only'} or train['training_authorized'] is not True:raise ValueError('Wrong from-scratch format104 controller contract')
    checkpoint=pin(Path(train['output'])/'train/model_final.pt');train_pin=pin(training_config)
    if health.get('schema')!='rough104_checkpoint_CPU_health:1' or health.get('status')!='PASS_SAVED_ENGINEERING_HEALTH' or not equal_fields(health,{'method':train['method'],'seed':0,'iteration':2000,'final_iteration_scope':True}) or health.get('config')!=train_pin or health.get('checkpoint')!=checkpoint or type(health.get('checks')) is not dict or not health['checks'] or any(value is not True for value in health['checks'].values()):raise ValueError('Actual final2000 health is not bound to the selected training config/checkpoint')
    if load.get('schema')!='rough104_actual_actor_CPU_load:1' or load.get('status')!='PASS_ACTUAL_EVALUATION_ACTOR_CPU_LOAD' or load.get('actual_training_checkpoint_loaded') is not True or type(load.get('checks')) is not list or not load['checks'] or any(row.get('passed') is not True for row in load['checks']) or type(load.get('passed')) is not int or type(load.get('check_count')) is not int or load['passed']!=load['check_count'] or load['check_count']!=len(load['checks']) or load.get('checkpoint')!=checkpoint:raise ValueError('Actual CPU Actor consumption report is not qualified for selected final bytes')
    receipt=load['receipt']
    if not equal_fields(receipt,{'schema':'rough_prefix_104_actor_evaluation_load:1','format_version':104,'method':train['method'],'seed':0,'iteration':2000,'evaluation_only':True,'training_resume_supported':False}) or receipt.get('checkpoint')!=checkpoint or receipt.get('bundle')!=train['input_bundle']:raise ValueError('Actor receipt is not the same primary final2000/bundle')
    if p.get('schema')!='rough_prefix_policy_evaluation_protocol:1' or canonical(p.get('checkpoint_iterations'))!=canonical([2000]) or canonical(p.get('allowed_seeds'))!=canonical([0]) or train['method'] not in p['arms'] or p['input_bundle']!=train['input_bundle']:raise ValueError('Separate final2000 protocol selection/data mismatch')
    if canonical(p['evaluation_gates'])!=canonical(GATES) or not equal_fields(p,{'worlds':1,'reference_states':16,'control_intervals':15,'solver_substeps':60,'postinterval_forward_calls':15,'automatic_retry':False,'compile_fallback':False}):raise ValueError('Original evaluation threshold/cadence/no-retry contract differs')
    if type(p['hard_process_seconds']) is not int or type(p['soft_worker_seconds']) is not int or not 0<p['soft_worker_seconds']<p['hard_process_seconds']<=120:raise ValueError('Finite tested evaluation process cap differs')
    pins=dict(train['source_pins'])
    if pins.get('protocol')!=train['protocol']:raise ValueError('Original training protocol alias differs')
    checked(train['protocol']);training_protocol=read(train['protocol']['path'])
    if training_protocol.get('protocol_revision')!=4 or canonical(training_protocol.get('primary_matched_training'))!=canonical(train['training']):raise ValueError('Training protocol/budget differs from actual primary source')
    pins['training_protocol']=train['protocol'];pins['protocol']=pin(protocol)
    paths=[Path(__file__),ROOT/'scripts/evaluate_rough_prefix_pen_v2.py',ROOT/'scripts/run_rough_evaluation_v2.py']
    paths += [ROOT/'src/rough_prefix'/name for name in ('evaluation_v2.py','policy_probe.py','checkpoint_v2.py')]
    for path in paths:pins['primary_evaluation__'+path.relative_to(ROOT).as_posix()]=pin(path)
    for key,path in {'training_config':training_config,'checkpoint_health':health_report,'actor_cpu_load':actor_load_report}.items():pins['primary_evaluation_evidence__'+key]=pin(path)
    for key,value in {'health_auditor':health['auditor'],'actual_load_producer':load['source'],'actual_loader':receipt['loader']}.items():pins['primary_evaluation_producer__'+key]=value
    for index,path in enumerate(evidence):pins['primary_evaluation_independent__'+str(index)]=pin(path)
    for value in pins.values():checked(value)
    return {'schema':'rough_prefix_policy_evaluation_run:1','role':'native_policy_eval','method':train['method'],'seed':0,'input_bundle':train['input_bundle'],'output':str(ROOT/'outputs'/('eval_primary_'+train['method'].lower()+'_seed0_'+version)),'checkpoint':checkpoint,'checkpoint_iteration':2000,'training_config':train_pin,'health_report':pin(health_report),'actor_load_report':pin(actor_load_report),'source_pins':pins,'protocol':pin(protocol),'GPU_run_authorized':False,'worlds':1,'hard_process_seconds':p['hard_process_seconds'],'soft_worker_seconds':p['soft_worker_seconds'],'automatic_retry':False,'compile_fallback':False,'evaluation_gates':p['evaluation_gates']}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--training-config',type=Path,required=True);ap.add_argument('--health',type=Path,required=True);ap.add_argument('--actor-load',type=Path,required=True);ap.add_argument('--protocol',type=Path,default=ROOT/'protocols/rough-evaluation-v3.json');ap.add_argument('--version',required=True);ap.add_argument('--evidence',type=Path,action='append',default=[]);a=ap.parse_args()
    config=build_config(a.training_config,a.health,a.actor_load,a.protocol,a.version,a.evidence)
    destination=ROOT/'configs'/(config['method'].lower()+'-primary-policy-evaluation-'+a.version+'.json')
    if destination.exists():raise FileExistsError('Never overwrite frozen primary evaluation config')
    write(destination,config);print(canonical(pin(destination)))


if __name__=='__main__':main()
