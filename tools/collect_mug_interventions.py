"""Acquire new TRAIN/CHECK counterfactuals at independent seeded native prefixes."""
from pathlib import Path
import argparse,json,sys,time,tomllib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.compound_transition import CompoundTransitionReader
from supermimic_research.mechanics.transition_store import save_transitions,exploration_action


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')


def exact_transition(a,b):
    return all(np.array_equal(a[k],b[k]) for k in ['source55','next55','source_native_state','next_native_state','warmstart26_before','warmstart26_after','mean_native_wrench_world_COM','mean_native_wrench_source_BODY_COM','mean_native_group_Fn_N']) and all(np.array_equal(a['trace'][k],b['trace'][k]) for k in a['trace'])


def main(args):
    config_path=args.config.resolve();config=tomllib.loads(config_path.read_text());collection_path=(config_path.parent/config['collection']).resolve();collection=json.loads(collection_path.read_text());out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or set(config['TRAIN_seeds'])&set(config['CHECK_seeds']):raise ValueError('Fresh unified output and independent seeds')
    if config['joints']!=list(range(20)) or config['prefix_k']!=[2,7,12] or config['amplitudes']!=[.04,.2]:raise ValueError('Declared all-joint intervention design')
    binding=json.loads(verify_file_pin(collection['binding']).read_text());reference=json.loads(verify_file_pin(collection['reference']).read_text());ref=np.load(verify_file_pin(reference['output']))
    for pin in collection['sources']:verify_file_pin(pin)
    sources=[file_pin(ROOT/p) for p in ['tools/collect_mug_interventions.py','src/supermimic_research/mechanics/transition_store.py',
        'src/supermimic_research/mechanics/transition_records.py','src/supermimic_research/mechanics/compound_transition.py',
        'src/supermimic_research/mechanics/residual_control.py','src/supermimic_research/mechanics/compound_contacts.py',
        'src/supermimic_research/mechanics/compound_holding.py','src/supermimic_research/runtime/native_evaluation.py']]
    out.mkdir(parents=True);write(out/'intent.json',dict(schema='mug_new_native_interventions:1',config=file_pin(config_path),collection=file_pin(collection_path),sources=sources,
        TRAIN_expected=2880,CHECK_expected=1920,role_boundary='Only fresh TRAIN branches fit. Prefix/center replay evidence not fit; CHECK and all old diagnostics never fit.',
        intervention='Actual profile center action plus/minus .04/.2 in one of20 joints. Actual control difference is the derivative denominator, including asymmetric EMA room.',
        max_native_steps=config['max_native_steps'],neural_updates=0,physics_changes=False))
    groups=[];prefixes=[];barriers=[];steps=0;start=time.perf_counter()
    for role in ['TRAIN','CHECK']:
        for seed in config[role+'_seeds']:
            for dt in config[role+'_dt']:
                reader=CompoundTransitionReader(binding,ref['initial_qpos27'],ref['feedforward_PD_target20'][0],dt_s=dt);snapshots={};prefix=[]
                while reader.k<=max(config['prefix_k']):
                    if reader.k in config['prefix_k']:snapshots[reader.k]=reader.snapshot()
                    action=exploration_action(seed,reader.k,reader.observe(),ref['grasp_q20_reference'][0]);t=reader.step(action);prefix.append(t);steps+=t['executed_native_steps']
                    if t['terminated'] or not t['complete_H']:raise ValueError('Prefix terminated; preserve evidence, no implicit seed replacement')
                name=f'{role}_{seed}_{dt}';prefixes.append(dict(role='PREFIX_ONLY_NEVER_FIT',seed=seed,dt=dt,**save_transitions(out/(name+'_prefix'),prefix,dt,[dict(seed=seed,dt=dt)]*len(prefix))))
                for k,state in snapshots.items():
                    reader.restore(state);center=exploration_action(seed,k,reader.observe(),ref['grasp_q20_reference'][0]);first=reader.step(center);reader.restore(state);second=reader.step(center);steps+=first['executed_native_steps']+second['executed_native_steps']
                    equal=exact_transition(first,second);barriers.append(dict(role=role,seed=seed,dt=dt,k=k,passed=equal))
                    if not equal or not first['complete_H'] or first['terminated']:raise ValueError('Full state/controller restore barrier failed')
                    save_transitions(out/(name+f'_k{k}_barrier'),[first,second],dt,[dict(seed=seed,dt=dt)]*2)
                    records=[];metadata=[]
                    for joint in config['joints']:
                        for amplitude in config['amplitudes']:
                            for sign in [-1,1]:
                                reader.restore(state);action=center.copy();action[joint]+=sign*amplitude;t=reader.step(action);steps+=t['executed_native_steps']
                                if not t['complete_H'] or t['terminated']:raise ValueError('Partial/terminated intervention; no silent fitting or resampling')
                                records.append(t);metadata.append(dict(seed=seed,dt=dt,joint=joint,amplitude=amplitude,sign=sign,center_action20=center))
                    item=dict(role=role,seed=seed,dt=dt,k=k,**save_transitions(out/(name+f'_k{k}'),records,dt,metadata));groups.append(item)
                    if steps>config['max_native_steps']:raise ValueError('Native integration budget exceeded')
                    write(out/'progress.json',dict(groups=len(groups),steps=steps,last={x:item[x] for x in ['role','seed','dt','k']},wall_s=time.perf_counter()-start))
                reader.verify_physics();print(json.dumps(dict(role=role,seed=seed,dt=dt,groups=len(groups),steps=steps,wall_s=time.perf_counter()-start)),flush=True)
    for pin in sources:verify_file_pin(pin)
    result=dict(schema='mug_new_native_interventions:1',sources=sources,intent=file_pin(out/'intent.json'),config=file_pin(config_path),collection=file_pin(collection_path),
        binding=collection['binding'],reference=collection['reference'],groups=groups,prefixes=prefixes,barriers=barriers,TRAIN_rows=sum(g['rows'] for g in groups if g['role']=='TRAIN'),
        CHECK_rows=sum(g['rows'] for g in groups if g['role']=='CHECK'),total_native_steps=steps,wall_s=time.perf_counter()-start,neural_updates=0,
        scope='Same nominal mug/reset/anchored task, independent prefix seeds and read-only CHECK seeds/dt. Complete20ms all-joint interventions, not successful demonstrations, unseen trajectories, independent RGB or source contact truth.')
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),TRAIN=result['TRAIN_rows'],CHECK=result['CHECK_rows'],steps=steps)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
