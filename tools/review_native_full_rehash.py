"""Independently decode all scoped-verifier packets and failure snapshots."""
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import mujoco
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.explicit_collision_backend import load_contract
from supermimic_research.runtime.backend_pen_probe import require_backend_probe
from supermimic_research.learning.world_model import raw_tree_equal
from supermimic_research.mechanics.probe_row_review import review_recorded_row


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def main():
    rp=ROOT/'runs/native_full_rehash_integration_v1/report.json';r=read(rp);out=ROOT/'runs/native_full_rehash_review_v1'
    if out.exists() or r['schema']!='native_full_rehash_integration:1' or r['passed']!=r['total'] or not all(r['checks'].values()):
        raise ValueError('Fresh completed integration review required')
    if r['default_changed'] or r['source_epoch_used']:raise ValueError('Only bounded full-rehash integration admitted')
    verify_source_snapshot(read(verify_file_pin(r['source_receipt'])),live=True)
    registration=read(verify_file_pin(r['registration']));verify_source_snapshot(read(verify_file_pin(registration['source_receipt'])),live=True)
    contract=load_contract(verify_file_pin(r['inputs']['backend_manifest']));states={};models={};physical={};rows=[];groups={}
    cfg=dict(role='TRAIN_CONTROLLED_PHYSICAL_RESPONSE',object_penetration_m=.002,self_penetration_m=.0002,joint_excursion_rad=.001)
    with patch.object(mujoco,'mj_step',side_effect=AssertionError('Saved-packet review')),patch.object(mujoco,'mj_forward',side_effect=AssertionError('Saved-packet review')):
        for c in r['cases']:
            sp=verify_file_pin(c['source']);mp=verify_file_pin(c['model'])
            if str(sp) not in states:
                states[str(sp)]=torch.load(sp,map_location='cpu',weights_only=False);require_backend_probe(states[str(sp)],contract)
            if str(mp) not in models:
                models[str(mp)]=mujoco.MjModel.from_binary_path(str(mp));physical[str(mp)]=dict(np.load(verify_file_pin(c['physical_context'])))
            z=dict(np.load(verify_file_pin(c['trace'])));window=dict(np.load(verify_file_pin(c['window'])))
            packet,audit=review_recorded_row(c,z,window,states[str(sp)],models[str(mp)],physical[str(mp)],cfg)
            rows.append(dict(name=c['name'],source_name=c['source_name'],round=c['round'],mode=c['mode'],**audit))
            key=(c['source_name'],c['round'],c['original_name'])
            if c['mode'] in groups.setdefault(key,{}):raise ValueError('Duplicate matched integration row')
            groups[key][c['mode']]=(z,packet)
    comparisons=[]
    for key,variants in groups.items():
        if set(variants)!={'ORIGINAL_A','SINGLE_RESOLVE','ORIGINAL_B'}:raise ValueError('All original brackets required')
        for bracket in ['ORIGINAL_A','ORIGINAL_B']:
            a,ap=variants[bracket];b,bp=variants['SINGLE_RESOLVE']
            inputs={k:same(a[k],b[k]) for k in ['actions20','initial_warmstart','initial_native_state','pre_action_phase']}
            inputs.update(initial_qpos=same(a['pre_qpos'][0],b['pre_qpos'][0]),initial_qvel=same(a['pre_qvel'][0],b['pre_qvel'][0]),held_PD=same(a['post_ctrl'],b['post_ctrl']),packet_x=same(ap['x'],bp['x']))
            differences={k:float(np.max(abs(a[k]-b[k]))) for k in ['post_qpos','post_qvel','post_qacc_warmstart','post_qfrc_constraint']}
            differences['packet_y']=float(np.max(abs(ap['y']-bp['y'])))
            comparisons.append(dict(source_name=key[0],round=key[1],original_name=key[2],bracket=bracket,identical_inputs=inputs,observed_max_absolute_differences=differences))
    negative=[]
    for n in r['negative_tests']:
        before=torch.load(verify_file_pin(n['before']),map_location='cpu',weights_only=False);after=torch.load(verify_file_pin(n['after']),map_location='cpu',weights_only=False)
        checks={k:raw_tree_equal(before[k],after[k]) for k in ['native','python','rng']}
        checks.update(denied=n['denied'] and bool(n['error']),zero_writes=n['Write_apply_calls']==0,zero_native=n['native_integrations']==0)
        negative.append(dict(source_name=n['source_name'],mode=n['mode'],defect=n['defect'],checks=checks))
    timing=[]
    for source in sorted({c['source_name'] for c in r['cases']}):
        for repeat in range(3):
            entries={p['mode']:p for p in r['phases'] if p['source_name']==source and p['round']==repeat}
            if set(entries)!={'ORIGINAL_A','SINGLE_RESOLVE','ORIGINAL_B'}:raise ValueError('All bracket phase timings required')
            record=dict(source_name=source,round=repeat)
            for field in ['phase_wall_s','branch_wall_sum_s']:
                times={k:p[field] for k,p in entries.items()};candidate=times['SINGLE_RESOLVE'];mean=(times['ORIGINAL_A']+times['ORIGINAL_B'])/2
                record[field]=dict(times=times,original_bracket_mean_s=mean,candidate_reduction_vs_bracket_mean=1-candidate/mean,
                    candidate_reduction_vs_originalA=1-candidate/times['ORIGINAL_A'],candidate_reduction_vs_originalB=1-candidate/times['ORIGINAL_B'])
            stage_s={m:{k:sum(c['stages'][k] for c in r['cases'] if c['source_name']==source and c['round']==repeat and c['mode']==m) for k in r['cases'][0]['stages']} for m in entries}
            record['stages_s']=stage_s;timing.append(record)
    summary={}
    for source in sorted({c['source_name'] for c in r['cases']}):
        entries=[t for t in timing if t['source_name']==source];summary[source]={}
        for field in ['phase_wall_s','branch_wall_sum_s']:
            summary[source][field]=dict(median_times_s={m:float(np.median([t[field]['times'][m] for t in entries])) for m in ['ORIGINAL_A','SINGLE_RESOLVE','ORIGINAL_B']},
                median_candidate_reduction_vs_bracket_mean=float(np.median([t[field]['candidate_reduction_vs_bracket_mean'] for t in entries])),
                all_three_candidate_faster_than_both=all(t[field]['times']['SINGLE_RESOLVE']<min(t[field]['times'][m] for m in ['ORIGINAL_A','ORIGINAL_B']) for t in entries))
    checks=dict(all180_rows=len(rows)==180,all60_triples=len(groups)==60,all120_bracket_comparisons=len(comparisons)==120,
        exact_original_inputs=all(all(c['identical_inputs'].values()) for c in comparisons),original_physics_all180=all(x['valid'] for x in rows),
        all_original_row_flags=all(all(x['checks'].values()) for x in rows),all8_negative_snapshots=len(negative)==8 and all(all(n['checks'].values()) for n in negative),
        exact1440_native=r['actual_native_integrations']==r['attempted_native_integrations']==1440,
        zero_optimizer=r['optimizer_steps']==r['attempted_optimizer_steps']==0,
        exact_restore_counts=r['restore_completed']==202 and r['restore_attempts']==210,
        all18_clean_scopes=len(r['phases'])==18 and all(p['scope']['validated'] and p['scope']['bindings_restored'] and p['scope']['default_unchanged'] and not p['scope']['source_epoch_used'] and p['scope']['implementation_boundary_hash_calls']==4 for p in r['phases']),
        full_rehash_count_parity=all(len({p['scope']['full_rehash_calls'] for p in r['phases'] if p['source_name']==s})==1 for s in summary),
        all6_round_timing_pairs=len(timing)==6,
        no_default_or_epoch=not r['default_changed'] and not r['source_epoch_used'])
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__)],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt)
    result=dict(schema='native_full_rehash_packet_review:1',integration=file_pin(rp),source_receipt=file_pin(out/'source_receipt.json'),
        checks=checks,passed=sum(checks.values()),total=len(checks),rows=rows,valid_rows=sum(x['valid'] for x in rows),total_rows=len(rows),
        row_flags=[sum(sum(x['checks'].values()) for x in rows),sum(len(x['checks']) for x in rows)],matched_inputs=comparisons,negative_snapshot_reviews=negative,
        timing_rounds=timing,timing_summary=summary,actual_native_integrations=0,optimizer_steps=0,
        scope='All180 new packets independently decoded under original EFC/force/torque/5e-7Ns impulse/source/geometry gates. All bracket timings retained;phase includes warmup,boundary hashes,case JSON. Same wrapper overhead in both modes. No unscoped default/W32 training/Actor/ego claim;old epoch anomaly unresolved.')
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],valid=[result['valid_rows'],result['total_rows']],row_flags=result['row_flags'],timing=summary)),flush=True)
    if not all(checks.values()):raise ValueError('Integration review discrepancy retained')


if __name__=='__main__':main()
