"""Fixed eighteen-case plan and closed evidence/checkpoint ownership."""
import json
from pathlib import Path
import tomllib
from ..artifacts import file_pin,verify_file_pin


def read(path): return json.loads(Path(path).read_text())


def schedule():
    algorithms=['INSTALLED_ORIGINAL_OR','STABLE_SAME_REGIME_FRICTION_OR']; rows=[]
    for repeat in range(3):
        for algorithm in algorithms if repeat!=1 else list(reversed(algorithms)):
            for label in ['INITIAL','RESIDUAL','SUPPORT']:
                rows.append(dict(repetition=repeat,algorithm=algorithm,policy=label,name=f'R{repeat}_{algorithm}_{label}'))
    return rows


def settings(cp):
    cp=Path(cp).resolve(); cfg=tomllib.loads(cp.read_text())
    keys=['schema','mode','worlds','policies','algorithms','repetitions','reference_controls','ticks_per_control','native_cap','optimizer_budget','normalizer_fit_budget','reserve_GiB']
    expected=[1,'FROZEN_ACTOR_COST_LONG_W1',1,['INITIAL','RESIDUAL','SUPPORT'],['INSTALLED_ORIGINAL_OR','STABLE_SAME_REGIME_FRICTION_OR'],3,368,8,52992,0,0,32]
    if [cfg[k] for k in keys]!=expected: raise ValueError('Exact18-case W1 frozen-Actor budget required')
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['study_config','local_evidence','backend_qualification','protocol','admission','registration','reader_registration','output','reader_output']}
    study=tomllib.loads(paths['study_config'].read_text())
    for k in ['training_report','plant_context_report','gates','objective_config','backend_manifest']: paths[k]=(paths['study_config'].parent/study[k]).resolve()
    training=read(paths['training_report']); plant=read(paths['plant_context_report']); q=read(paths['backend_qualification']); ledger=read(paths['local_evidence'])
    if training['schema']!='fresh_backend_long_control_training:1' or training['passed']!=training['total'] or training['actual_native_integrations']!=65536000 or training['optimizer_steps']!=128000:
        raise ValueError('Closed equal-budget final training required')
    if plant['passed']!=plant['total'] or plant['role']!='CHECK_TRAINED_ACTOR_PHYSICAL_RESPONSE_NEVER_FIT': raise ValueError('Closed independent W1 plant witnesses required')
    if q['passed']!=q['total'] or q['total']!=22 or not all(q['checks'].values()): raise ValueError('V4 arithmetic qualification required')
    local=read(verify_file_pin(ledger['review']))
    if local['passed']!=local['total'] or not all(local['checks'].values()): raise ValueError('Closed local native comparison evidence required')
    actors={'INITIAL':training['arms'][0]['initial'],**{a['arm']:a['checkpoint'] for a in training['arms']}}
    contexts={label:next(s['source'] for s in plant['sources'] if s['actor']==label and s['phase']==10 and s['available']) for label in ['RESIDUAL','SUPPORT']}
    inputs={k:file_pin(paths[k]) for k in ['study_config','local_evidence','backend_qualification','protocol','training_report','plant_context_report','gates','objective_config','backend_manifest']}
    for pin in list(inputs.values())+list(actors.values())+list(contexts.values()): verify_file_pin(pin)
    return cfg,paths,inputs,actors,contexts,q
