"""Versioned TRAIN augmentation; exactly one declared interval per new episode."""
import argparse
import json
import sys
import tomllib
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.controlled_pen_records import pack_case
from supermimic_research.learning.pen_response import fit_pen_scales,INPUT_SCHEMA,OUTPUT_SCHEMA,HEAD_SCHEMA
from supermimic_research.learning.pen_branch_profiles import branch_profiles
from supermimic_research.runtime.controlled_evaluation import write


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text())
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['collection','EFC_review','old_prepared','old_review']}
    r,e,old,old_review=[json.loads(paths[k].read_text()) for k in paths]
    if (a.output.exists() or not a.output.resolve().is_relative_to(ROOT/'runs')
            or verify_file_pin(e['run'])!=paths['collection']
            or verify_file_pin(old_review['prepared'])!=paths['old_prepared']
            or not all(old['checks'].values()) or old_review['passed']!=old_review['total']):
        raise ValueError('Fresh unified preparation with exact reviewed sources')
    cc=tomllib.loads(verify_file_pin(r['config']).read_text());_,profiles=branch_profiles(cc)
    expected={(p['name'],dt) for p in profiles for dt in cc['dt']}
    if {(c['name'],c['dt']) for c in r['cases']}!=expected or len(r['cases'])!=256:raise ValueError('Fixed256 source branches')
    for pin in r['sources']+r['task_sources']+old['sources']:verify_file_pin(pin)
    p=json.loads(verify_file_pin(r['proposal']).read_text());idx=np.load(verify_file_pin(p['artifacts']['selected']))
    old_train=dict(np.load(verify_file_pin(old['outputs']['TRAIN'])));verify_file_pin(old['outputs']['CHECK'])
    roles={'TRAIN':[],'CHECK':[]};rows=[];physics=old['physics']
    for case_id,c in enumerate(r['cases']):
        if c['physics']!=physics:raise ValueError('Unchanged physical prior')
        z=np.load(verify_file_pin(c['trace']));m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])))
        d=mujoco.MjData(m);mujoco.mj_setState(m,d,z['initial_native_state'],mujoco.mjtState.mjSTATE_INTEGRATION)
        receipt=next(k for k in e['cases'] if k['name']==c['name'] and k['dt']==c['dt'])
        if not receipt['checks']['valid_source_indices'] or not receipt['checks']['local_labels_exact']:raise ValueError('Auditable EFC labels')
        packet=pack_case(m,c,z,np.load(verify_file_pin(receipt['wrenches'])),idx['qids'],idx['dofs'],idx['aids'],d.qacc_warmstart,physics)
        k=c['anchor'];reached=k<len(packet['x'])
        if reached:
            selected={key:value[k:k+1] for key,value in packet.items()}
            selected['case_id']=np.array([1000+case_id]);roles[c['role']].append(selected)
        rows.append(dict(case_id=1000+case_id,collection_case_id=case_id,name=c['name'],role=c['role'],
            dt=c['dt'],anchor=k,direction=c['direction'],direction_seed=c['direction_seed'],epsilon=c['epsilon'],sign=c['sign'],
            reached=reached,valid=bool(reached and packet['valid'][k]),qualified=c['summary']['qualified']))
    new={role:{key:np.concatenate([v[key] for v in parts]) for key in parts[0]} for role,parts in roles.items()}
    train={key:np.concatenate([old_train[key],new['TRAIN'][key]]) for key in old_train}
    if set(train)!=set(new['TRAIN']):raise ValueError('Same explicit indexed chart')
    all_train_x={row.tobytes() for row in train['x']}
    checks=dict(all_anchors_reached=all(row['reached'] for row in rows),
        minimum_new_TRAIN=int(new['TRAIN']['valid'].sum())>=cfg['minimum_valid_new_TRAIN'],
        minimum_new_CHECK=int(new['CHECK']['valid'].sum())>=cfg['minimum_valid_new_CHECK'],
        no_identical_full_input_TRAIN_CHECK=not any(row.tobytes() in all_train_x for row in new['CHECK']['x']),
        fresh_case_roles_disjoint=not bool(set(new['TRAIN']['case_id'])&set(new['CHECK']['case_id'])),
        one_declared_interval_per_episode=len(new['TRAIN']['x'])+len(new['CHECK']['x'])==sum(row['reached'] for row in rows),
        old_TRAIN_prefix_exact=all(np.array_equal(train[k][:len(old_train[k])],old_train[k]) for k in old_train))
    valid=train['valid'];scales,heads=fit_pen_scales(train['x'][valid],train['y'][valid],train['heads'][valid])
    pairs=[]
    for role in ['TRAIN','CHECK']:
        for plus in [row for row in rows if row['role']==role and row['sign']==1 and row['reached']]:
            minus=next(row for row in rows if row['role']==role and row['sign']==-1 and row['anchor']==plus['anchor'] and row['direction']==plus['direction'] and row['dt']==plus['dt'])
            if not minus['reached']:continue
            z=new[role];ip=np.flatnonzero(z['case_id']==plus['case_id'])[0];im=np.flatnonzero(z['case_id']==minus['case_id'])[0]
            delta=z['x'][ip,:81]-z['x'][im,:81]
            pairs.append(dict(role=role,dt=plus['dt'],anchor=plus['anchor'],direction=plus['direction'],
                exact_source_warm=bool(np.array_equal(z['x'][ip,:81],z['x'][im,:81])),
                state_max_abs=float(abs(delta[:55]).max()),warm_max_abs=float(abs(delta[55:]).max())))
    a.output.mkdir();outputs={}
    for role,z in [('TRAIN',train),('CHECK',new['CHECK']),('NEW_TRAIN',new['TRAIN'])]:
        path=a.output/(role+'.npz');np.savez_compressed(path,**z);outputs[role]=file_pin(path)
    outputs['SEQUENCE_CHECK']=old['outputs']['CHECK']
    for name,values in [('scales',scales),('head_scales',heads)]:np.savez_compressed(a.output/(name+'.npz'),**values)
    sources=[file_pin(__file__)]+[file_pin(ROOT/p) for p in ['src/supermimic_research/mechanics/controlled_pen_records.py',
        'src/supermimic_research/mechanics/transition_records.py','src/supermimic_research/mechanics/response_decode.py',
        'src/supermimic_research/learning/pen_response.py','src/supermimic_research/learning/pen_branch_profiles.py']]
    result=dict(schema='pen_branch_response_preparation:1',config=file_pin(cp),inputs={k:file_pin(p) for k,p in paths.items()},
        sources=sources,outputs=outputs,scales=file_pin(a.output/'scales.npz'),head_scales=file_pin(a.output/'head_scales.npz'),
        input_schema=INPUT_SCHEMA,output_schema=OUTPUT_SCHEMA,head_schema=HEAD_SCHEMA,physics=physics,
        checks=checks,passed=sum(checks.values()),total=len(checks),rows=rows,pair_source_diagnostics=pairs,
        counts={r:dict(rows=len(z['x']),valid=int(z['valid'].sum())) for r,z in [('TRAIN',train),('CHECK',new['CHECK']),('NEW_TRAIN',new['TRAIN'])]},
        neural_updates=0,native_integration_steps=0,
        scope='Old157 TRAIN augmented by fresh declared branch TRAIN only; old CHECK remains separate sequence evaluation. No previous diagnostic reuse as TRAIN. New CHECK holds out action directions, not source state/clip/ego. Exact-source pairs alone permit unconfounded central response comparison; all actual source differences remain in inputs.')
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),counts=result['counts'],checks=checks,pairs_exact=sum(p['exact_source_warm'] for p in pairs),pairs_total=len(pairs))))
    if not all(checks.values()):raise ValueError('Retain incomplete preparation; no training')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
