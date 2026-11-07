"""Bind actual final-result/health/terminal JSON into a fresh CPUFalse protocol.

No checkpoint deserialization, Torch, native, NPZ-value read or execution.
Frozen result schema can omit protocol: only its matching terminal supplies
that pin, with the fallback recorded explicitly and checked at checkpoint load.
"""
from pathlib import Path
import argparse,copy,importlib.util,json
H=Path(__file__).resolve().parent

def bind(*,base_protocol,final_result,health,terminal,output_protocol,output_folder=None):
    spec=importlib.util.spec_from_file_location('online_binding_source',H/'collect_online_v2.py')
    O=importlib.util.module_from_spec(spec);spec.loader.exec_module(O)
    base=O.read(base_protocol);O.need(base['CPU_run_authorized'] is False,'Bind CPUFalse source only')
    O.metadata_preflight(base_protocol)
    r=O.read(final_result);h=O.read(health);t=O.read(terminal)
    O.need(r['schema']=='phone_tripod_matched_supermimic_training_result:1' and r['status']=='COMPLETED_FIXED_OFFLINE_MATCHED1000_NOT_NATIVE_POLICY_QUALIFICATION','Completed final own training, not first100')
    O.need(r['paired_round']==1000 and r['shared_WM_updates']==1100,'AFTER1000/WM1100')
    O.need(h['schema']=='phone_tripod_matched_learning_checkpoint_health:1' and h['paired_round']==1000 and h['shared_WM_updates']==1100,'Final saved health phase')
    O.need(h['checkpoint']==r['checkpoint'] and h['checks'] and all(v is True for v in h['checks'].values()),'Final raw health checkpoint')
    O.checked(r['checkpoint'])
    O.need(t['returncode']==0,'Real final terminal exit0')
    pp=r.get('protocol',t['protocol']);O.need(pp==t['protocol'],'Result/terminal actual executed final protocol')
    O.checked(pp);tp=O.read(pp['path']);seed=O.read(O.checked(base['inputs']['paired_training_protocol']))
    O.need(tp['schema']==seed['schema'] and tp['recipe']==seed['recipe'] and tp['inputs']==seed['inputs'] and tp['sources']==seed['sources'],'Same experiment inputs, numerical recipe and source bytes; resume protocol may differ')
    O.need(t['source']==tp['sources']['trainer']==base['sources']['paired_trainer'],'Actual terminal frozen trainer')
    for key,count in [('WM_optimizer_steps',1100),('Actor_optimizer_steps',2000),('Critic_optimizer_steps',2000)]:
        O.need(r['counts'][key]==count,'Final actual update count '+key)
    p=copy.deepcopy(base);p['inputs'].update(paired_training_protocol=pp,paired_checkpoint=r['checkpoint'],paired_health=O.pin(health))
    p['CPU_run_authorized']=False;p['training_authorized']=False;p['GPU_authorized']=False
    p['sources']['final_binder']=O.pin(__file__)
    if output_folder is not None:p['output']=str(Path(output_folder).resolve())
    p['final_binding']=dict(base_protocol=O.pin(base_protocol),final_result=O.pin(final_result),final_health=O.pin(health),final_terminal=O.pin(terminal),
        actual_final_training_protocol=pp,protocol_binding_source='result.protocol (same as terminal.protocol)' if 'protocol' in r else 'terminal.protocol; frozen final-result schema omits protocol',
        checkpoint_payload_protocol_rechecked_before_Actor_construction=True,checkpoint_deserializations=0,new_native_NN_calls=0)
    dest=Path(output_protocol).resolve();O.need(dest.parent==H and not dest.exists(),'Fresh local CPUFalse final binding only')
    dest.write_text(json.dumps(p,indent=2,allow_nan=False),encoding='utf8')
    O.metadata_preflight(dest)
    return dict(protocol=O.pin(dest),CPU_run_authorized=False,binding=p['final_binding'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--base-protocol',type=Path,default=H/'protocol_CPUFalse_v2.json')
    ap.add_argument('--final-result',type=Path,required=True);ap.add_argument('--health',type=Path,required=True);ap.add_argument('--terminal',type=Path,required=True)
    ap.add_argument('--output-protocol',type=Path,default=H/'protocol_CPUFalse_bound_final_v1.json');ap.add_argument('--output-folder',type=Path)
    a=ap.parse_args();print(json.dumps(bind(base_protocol=a.base_protocol,final_result=a.final_result,health=a.health,terminal=a.terminal,output_protocol=a.output_protocol,output_folder=a.output_folder)))
