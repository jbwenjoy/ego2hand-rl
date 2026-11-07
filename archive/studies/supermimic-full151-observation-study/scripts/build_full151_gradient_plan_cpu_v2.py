"""Freeze one bounded physical qualification, no framework import or execution."""
from pathlib import Path
import argparse,ast,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_native.io import read,pin,checked
from full151_native.metadata import write
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['engineering_restore_smoke','serial_full'],required=True);ap.add_argument('--tag',required=True);ap.add_argument('--foreign-review',type=Path);ap.add_argument('--restore-smoke-health',type=Path);ap.add_argument('--authorize-gpu',action='store_true');a=ap.parse_args()
    if not a.tag.replace('_','').isalnum():raise ValueError('Simple immutable study tag required')
    if any(k in sys.modules for k in ('torch','warp','mujoco_warp')):raise AssertionError('Metadata-only builder')
    cp=ROOT/f'configs/full151105-gradient-{a.tag}-v1.json';pp=ROOT/f'protocols/full151105-gradient-{a.tag}-v1.json'
    if cp.exists() or pp.exists():raise FileExistsError('Freeze once, never overwrite/retry')
    base=read(ROOT/'configs/full151-v2-policy-eval-primary2000_A2_seed0-v2.json')
    source=dict(base['source_pins']);source.pop('protocol',None)
    proposal=pin(ROOT/'protocols/full151_same_warp_control_gradient_proposal_v2.json')
    ready={'transaction_CPU_READY':pin(ROOT/'outputs/full151105_transaction_preparation_CPU_v1/READY.json'),
      'transaction_CPU_report':pin(ROOT/'outputs/full151105_transaction_preparation_CPU_v1/report.json'),
      'schema2_WM_CPU_report':pin(ROOT/'outputs/full151_schema2_pair_WM_loader_CPU_v1/report.json'),
      'serial_CPU_READY':pin(ROOT/'outputs/full151105_serial_probe_preparation_CPU_v2/READY.json'),
      'serial_CPU_report':pin(ROOT/'outputs/full151105_serial_probe_preparation_CPU_v2/report.json')}
    source['gradient_proposal']=proposal
    for key,value in read(checked(proposal))['source_pins'].items():source['proposal_execution_source_'+key]=value
    for key,value in ready.items():source['gradient_'+key]=value
    for key,value in read(checked(ready['transaction_CPU_READY']))['source_pins'].items():source['transaction_CPU_source_'+key]=value
    for key,value in read(checked(ready['serial_CPU_READY']))['source_pins'].items():source['serial_CPU_source_'+key]=value
    for key,value in read(checked(ready['schema2_WM_CPU_report']))['source_pins'].items():source['WM_CPU_source_'+key]=value
    for key,path in {'gradient_entry':ROOT/'scripts/execute_full151_gradient_v1.py','gradient_supervisor':ROOT/'scripts/run_full151_gradient_v1.py','gradient_builder':Path(__file__),'physical_serial_probe':ROOT/'src/full151_gradient/serial_probe105_v1.py','transaction105':ROOT/'src/full151_gradient/transaction105.py'}.items():
        ast.parse(path.read_text(encoding='utf-8'));source[key]=pin(path)
    package=ROOT/'src/full151_gradient/__init__.py'
    if package.exists():source['serial_package']=pin(package)
    donors={}
    names=('controller_descriptor.json','control_events.json','raw_states.npz','evaluation.json','config.json','fixed_physics_context.npz','native_reward_transitions.json')
    for method,folder in {'A0':'full151_native_A0_v1','A2':'full151_v2_policy_eval_primary2000_A2_seed0_v1'}.items():
        directory=ROOT/'outputs'/folder;values={name:pin(directory/name) for name in names};donors[method]={'directory':str(directory),'input_pins':values}
        for name,value in values.items():source['donor_'+method+'_'+name]=value
    review=None if a.foreign_review is None else pin(a.foreign_review)
    smoke=None if a.restore_smoke_health is None else pin(a.restore_smoke_health)
    if review is not None:source['foreign_inventory_review']=review
    if smoke is not None:source['restore_smoke_health']=smoke
    for value in source.values():checked(value)
    from execute_full151_gradient_v1 import MODES,SCHEMA,preflight
    protocol={'schema':'full151105_physical_probe_execution_protocol:1','input_bundle':base['input_bundle'],'proposal':proposal,'mode_counts':MODES,
      'scientific_scope':'Fixed two-donor8snapshot152branch proposal retained; smoke is only necessary native restore qualification. Inventory discovery/baseline failure never counts as physical gradient evidence.',
      'no_target_or_quality_selection':True,'training_authorized':False,'actual_reference_physical_qualified':False,'independent_ego_N':0,
      'exact_scalar':'matched_post20_gpu_score: controlled native-formula FP32GPU 2+8 score from captured postintegration state58 at20ms and preaction goal_k. Exact same state-time/function for WMnext20 comparison. OriginalRewardManager GPU rate terms use derived15ms pose and are secondary, never equality-gated to20ms. CPU/F64 rescores of20ms are same-phase arithmetic diagnostics only; k+1 diagnostic. Terminal150 has no policy bootstrap.',
      'foreign_rule':'Initial None permits discovery only if any opaque objects. Future explicit source-pinned type/path review and mutable sidecars; never blanket acceptance of all foreign internals.',
      'retry':False,'hard_process_seconds':240,'soft_worker_seconds':180}
    write(pp,protocol);source['gradient_protocol']=pin(pp)
    cfg={'schema':SCHEMA,'role':'full151105_serial_physical_probe','evaluation_seed':0,'probe_mode':a.mode,'input_bundle':base['input_bundle'],'proposal':proposal,'protocol':pin(pp),'source_pins':source,
      'donor_inputs':donors,'foreign_inventory_review':review,'restore_smoke_health':smoke,'readiness':ready,'output':str(ROOT/f'outputs/full151105_gradient_{a.tag}_v1'),
      'GPU_run_authorized':bool(a.authorize_gpu),'training_authorized':False,'soft_worker_seconds':180,'hard_process_seconds':240,'automatic_retry':False,'compile_fallback':False}
    write(cp,cfg);_,_,proof=preflight(cp)
    print(json.dumps({'config':pin(cp),'protocol':pin(pp),'source_pins':len(source),'CPU_preflight':proof,'GPU_started':False}))
if __name__=='__main__':main()
