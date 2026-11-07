"""Versioned multi-round collector contract builder. No NN/native execution."""
from pathlib import Path
import argparse,copy,importlib.util,json
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('online_collector_v3_source',H/'collect_online_v3.py');O=importlib.util.module_from_spec(s);s.loader.exec_module(O)
ROOT_KEYS=('root_training_protocol','root_result','root_health','root_terminal')

def make_round_protocol(base_protocol,*,collection_cycle,output_name,root_artifacts=None,actor_input=None,CPU_run_authorized=False):
    """Return one round dictionary; caller writes a fresh path before execution.

    base_protocol is frozen v2 source metadata, not an arbitrary Actor payload.
    root_artifacts holds exact four pins from the actual completed offline
    experiment. Later actor_input identifies the previous complete online100.
    """
    base=O.read(base_protocol)
    O.need(base['schema']=='phone_tripod_matched_online_collection_CPU_protocol:1' and base['CPU_run_authorized'] is False,'Frozen v2 CPUFalse base')
    O.need(type(collection_cycle) is int and 1<=collection_cycle<=3,'Finite declared cycles')
    out=H/output_name;O.need(out.parent.resolve()==H and not out.exists(),'One fresh round folder, no nested/overwritten pool')
    p=copy.deepcopy(base);p['schema']=O.SCHEMA;p['rules']=O.RULES;p['collection_cycle']=collection_cycle
    p['CPU_run_authorized']=bool(CPU_run_authorized);p['GPU_authorized']=False;p['training_authorized']=False;p['output']=str(out)
    p['inputs']['offline_seed_protocol']=p['inputs'].pop('paired_training_protocol')
    p['inputs'].pop('paired_checkpoint');p['inputs'].pop('paired_health')
    roots={key:None for key in ROOT_KEYS}
    if root_artifacts is not None:
        O.need(set(root_artifacts)==set(ROOT_KEYS),'Exactly four root artifacts');roots.update(root_artifacts)
    p['inputs'].update(roots)
    if actor_input is None:
        root_result=O.read(O.checked(roots['root_result'])) if roots['root_result'] is not None else None
        actor_input=dict(kind='offline1000' if collection_cycle==1 else 'online_stage',stage_id=collection_cycle-1,
            checkpoint=root_result['checkpoint'] if collection_cycle==1 and root_result is not None else None,
            health=roots['root_health'] if collection_cycle==1 else None,
            training_protocol=roots['root_training_protocol'] if collection_cycle==1 else None)
    p['actor_input']=copy.deepcopy(actor_input)
    p['sources']['collector']=O.pin(H/'collect_online_v3.py');p['sources']['actor_lineage']=O.pin(H/'actor_lineage_v3.py');p['sources']['preparation']=O.pin(__file__)
    p['multi_round_scope']='collection_cycle1 uses exact offline1000; cycles2/3 use previous complete online100 checkpoints. Firsthealthy10 only supports resume, never another collection. This collector performs zero learning.'
    p['root_binding_source']=O.pin(base_protocol)
    return p

def write_round_protocol(p,path):
    path=Path(path).resolve();O.need(path.parent==H and not path.exists(),'Fresh local protocol path')
    path.write_text(json.dumps(p,indent=2,allow_nan=False),encoding='utf8');O.metadata_preflight(path)
    return O.pin(path)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--base-protocol',type=Path,default=H/'protocol_CPUFalse_v2.json')
    ap.add_argument('--cycle',type=int,default=1);ap.add_argument('--output-name',default='stage_cycle01_run1');ap.add_argument('--protocol-name',default='protocol_CPUFalse_v3.json')
    for key in ROOT_KEYS:ap.add_argument('--'+key.replace('_','-'),type=Path)
    ap.add_argument('--online-checkpoint',type=Path);ap.add_argument('--online-health',type=Path);ap.add_argument('--online-training-protocol',type=Path)
    a=ap.parse_args();roots={key:O.pin(getattr(a,key)) if getattr(a,key) is not None else None for key in ROOT_KEYS}
    actor=None
    if a.cycle>1:actor=dict(kind='online_stage',stage_id=a.cycle-1,checkpoint=O.pin(a.online_checkpoint) if a.online_checkpoint else None,health=O.pin(a.online_health) if a.online_health else None,training_protocol=O.pin(a.online_training_protocol) if a.online_training_protocol else None)
    p=make_round_protocol(a.base_protocol,collection_cycle=a.cycle,output_name=a.output_name,root_artifacts=roots,actor_input=actor)
    print(json.dumps(dict(protocol=write_round_protocol(p,H/a.protocol_name),CPU_run_authorized=False,new_native_NN_calls=0)))
