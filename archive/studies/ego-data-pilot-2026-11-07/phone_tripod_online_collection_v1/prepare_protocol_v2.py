"""Create CPUFalse online-stage source/plan contract; no Torch/native/NPZ values."""
from pathlib import Path
import argparse,importlib.util,json
H=Path(__file__).resolve().parent;P=H.parent
s=importlib.util.spec_from_file_location('online_source',H/'collect_online_v2.py');O=importlib.util.module_from_spec(s);s.loader.exec_module(O)

def build(*,training_protocol,checkpoint=None,health=None,output_name='acquired_run1',protocol_name='protocol_CPUFalse_v2.json'):
    tp=O.read(training_protocol)
    cp=O.read(P/'phone_tripod_task_acquisition_v1/protocol_CPUFalse_v1.json')
    dp=O.read(P/'phone_tripod_replay_dataset_v1/protocol_CPUFalse_v1.json')
    planpath=H/'profile_plan_v1.json'
    plan=O.profile_plan()
    if planpath.exists():O.need(O.read(planpath)==plan,'Existing fixed profile plan unchanged')
    else:planpath.write_text(json.dumps(plan,indent=2,allow_nan=False),encoding='utf8')
    sources={k:v for k,v in tp['sources'].items() if k not in ['trainer']}
    sources.update(paired_trainer=tp['sources']['trainer'],native_reader=cp['sources']['native_reader'],
        raw_collector=O.pin(P/'phone_tripod_task_acquisition_v1/collect_tripod_task_v1.py'),
        label_adapter=O.pin(P/'phone_tripod_replay_dataset_v1/dataset_adapter_v1.py'),
        dataset_exporter=O.pin(P/'phone_tripod_replay_dataset_v1/export_dataset_v1.py'),
        collector=O.pin(H/'collect_online_v2.py'),preparation=O.pin(__file__))
    inputs=dict(paired_training_protocol=O.pin(training_protocol),paired_checkpoint=O.pin(checkpoint) if checkpoint else None,paired_health=O.pin(health) if health else None,
        task_bundle=tp['inputs']['task_bundle'],model_mjb=cp['inputs']['model_mjb'],geometry_catalog=tp['inputs']['geometry_catalog'],
        bridge_qualification=cp['inputs']['bridge_qualification'],offline_dataset=tp['inputs']['dataset'],offline_dataset_review=tp['inputs']['dataset_review'],
        physical_parameters=dp['inputs']['physical_parameters'],geometry_metadata=dp['inputs']['geometry_metadata'],profile_plan=O.pin(planpath))
    p=dict(schema=O.SCHEMA,role='matched_closed_loop_fixed_final_A1_A2_collection_into_common_nextstage_replay',
        CPU_run_authorized=False,GPU_authorized=False,training_authorized=False,
        output=str(H/output_name),rules=O.RULES,inputs=inputs,sources=sources,
        offline_budget='Frozen paired1000 offline recipe has online_H0. This is a separately pinned new stage, no update or reclassification of those counts.',
        matching='Each arm receives same4profile x2dt x20H maxima from same probe030 initial. Early physical/task failures stop that episode without replacement; actual row counts may differ.',
        contact_scope='canonical22 actual signed nativePRE impulses are future supervision, not Actor observation. A2 proxy remains engineered, not measured contact or force.',
        safety_scope='Frozen Reader tick safety and bridge boundaryPOSTH task terminal preserved; acquisition20H truncation may bootstrap. This is not original full2s holding evaluation.',
        replay_scope='Both arms enter common ONLINE_TRAIN pool with explicit arm/profile/stage/alias identities; no hidden split/random row split/deduplication or oldCHECK/fresh reads.',
        followup_learning='load_common_pool exposes arrays/masks/per-mode real rewards. A new nextstage matched updater recipe must explicitly authorize updates; collector never fits.')
    path=H/protocol_name;O.need(not path.exists(),'Fresh protocol only')
    path.write_text(json.dumps(p,indent=2,allow_nan=False),encoding='utf8')
    O.metadata_preflight(path)
    return path

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--training-protocol',type=Path,default=P/'phone_tripod_supermimic_v1/protocol_CPUFalse_v1.json')
    ap.add_argument('--checkpoint',type=Path);ap.add_argument('--health',type=Path);ap.add_argument('--output-name',default='acquired_run1');ap.add_argument('--protocol-name',default='protocol_CPUFalse_v2.json');a=ap.parse_args()
    path=build(training_protocol=a.training_protocol,checkpoint=a.checkpoint,health=a.health,output_name=a.output_name,protocol_name=a.protocol_name)
    print(json.dumps(dict(protocol=O.pin(path),status='CPUFALSE_PENDING_FINAL_ALLOWED_NO_NN_NATIVE_VALUES')))
