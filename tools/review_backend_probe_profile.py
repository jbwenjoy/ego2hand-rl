"""Decode every new profiling packet under original physics gates,without stepping."""
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
from supermimic_research.mechanics.probe_row_review import review_recorded_row


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    rp=ROOT/'runs/pen_backend_probe_profile_v1/report.json';run=read(rp);out=ROOT/'runs/pen_backend_probe_profile_review_v1'
    if out.exists() or run.get('schema')!='backend_STRICT_probe_pipeline_profile:1' or run['source_epoch_used'] or run['passed']!=run['total']:
        raise ValueError('Fresh completed current STRICT profile review required')
    verify_source_snapshot(read(verify_file_pin(run['source_receipt'])),live=True)
    contract=load_contract(verify_file_pin(run['inputs']['backend_manifest']));states={};models={};physical={};rows=[];pairs={}
    cfg=dict(role='TRAIN_CONTROLLED_PHYSICAL_RESPONSE',object_penetration_m=.002,self_penetration_m=.0002,joint_excursion_rad=.001)
    with patch.object(mujoco,'mj_step',side_effect=AssertionError('Saved-packet review')),patch.object(mujoco,'mj_forward',side_effect=AssertionError('Saved-packet review')):
        for c in run['cases']:
            sp=verify_file_pin(c['source']);mp=verify_file_pin(c['model'])
            if str(sp) not in states:
                states[str(sp)]=torch.load(sp,map_location='cpu',weights_only=False);require_backend_probe(states[str(sp)],contract)
            if str(mp) not in models:
                models[str(mp)]=mujoco.MjModel.from_binary_path(str(mp));physical[str(mp)]=dict(np.load(verify_file_pin(c['physical_context'])))
            z=dict(np.load(verify_file_pin(c['trace'])));window=dict(np.load(verify_file_pin(c['window'])))
            _,audit=review_recorded_row(c,z,window,states[str(sp)],models[str(mp)],physical[str(mp)],cfg)
            rows.append(dict(name=c['name'],mode=c['profile_mode'],source_name=c['source_name'],original_name=c['original_name'],**audit))
            pairs.setdefault(c['original_name'],{})[c['profile_mode']]=z
    comparisons=[]
    for name,variants in pairs.items():
        a,b=variants['UNINSTRUMENTED'],variants['CPROFILE']
        inputs={k:a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes() for k in ['actions20','initial_warmstart','initial_native_state','pre_action_phase']}
        inputs['initial_qpos']=a['pre_qpos'][0].tobytes()==b['pre_qpos'][0].tobytes();inputs['initial_qvel']=a['pre_qvel'][0].tobytes()==b['pre_qvel'][0].tobytes()
        inputs['held_PD']=a['post_ctrl'].tobytes()==b['post_ctrl'].tobytes()
        differences={k:float(np.max(abs(a[k]-b[k]))) for k in ['post_qpos','post_qvel','post_qacc_warmstart','post_qfrc_constraint']}
        comparisons.append(dict(name=name,identical_inputs=inputs,observed_max_absolute_differences=differences))
    checks=dict(all40_rows=len(rows)==40,original_inputs_bytes=all(all(c['identical_inputs'].values()) for c in comparisons),
        all20_matched_actions=len(comparisons)==20,exact_original_native_work=run['actual_native_integrations']==run['attempted_native_integrations']==320 and run['optimizer_steps']==0,
        all_row_audit_flags=all(all(x['checks'].values()) for x in rows),
        physical_all_valid=all(x['valid'] for x in rows))
    stage={};restore=np.asarray([x['wall_s'] for x in run['restore_measurements']])
    for mode in ['UNINSTRUMENTED','CPROFILE']:
        cases=[c for c in run['cases'] if c['profile_mode']==mode];whole=sum(c['branch_complete_wall_s'] for c in cases)
        sums={k:sum(c['stages'][k] for c in cases) for k in cases[0]['stages']}
        stage[mode]=dict(branches=len(cases),whole_branch_s=whole,stages_s=sums,
            restore_fraction=sums['strict_restore_and_observation_s']/whole,
            recorded8_tick_step_fraction=sums['step_with8_recorded_ticks_s']/whole,
            other_unattributed_s=whole-sum(sums.values()))
    all_flags=sum(sum(r['checks'].values()) for r in rows);total_flags=sum(len(r['checks']) for r in rows)
    out.mkdir();paths=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__)]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt)
    result=dict(schema='backend_STRICT_probe_profile_review:1',profile=file_pin(rp),source_receipt=file_pin(out/'source_receipt.json'),
        rows=rows,matched_inputs=comparisons,checks=checks,passed=sum(checks.values()),total=len(checks),row_flags=[all_flags,total_flags],
        valid_rows=sum(r['valid'] for r in rows),total_rows=len(rows),stages=stage,
        restore_timing=dict(calls=len(restore),median_s=float(np.median(restore)),sum_s=float(restore.sum())),
        actual_native_integrations=0,optimizer_steps=0,
        physical_status='ALL40_SAVED_PROFILE_ROWS_PASS_ORIGINAL_GATES' if all(x['valid'] for x in rows) else 'PHYSICAL_PROFILE_FAILURES_RETAINED',
        scope='Independent saved-packet EFC/impulse/source/warm/mount/geometry review. New work0;profile itself320 native. CPU stage/cProfile attribution is not active GPU time or a tested optimization. Whole branch excludes final per-case JSON write. Old epoch anomaly remains unresolved.')
    write(out/'report.json',result)
    print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],valid=[result['valid_rows'],result['total_rows']],row_flags=result['row_flags'])),flush=True)
    if not all(checks.values()):raise ValueError('Original physical/input gates failed;retain original diagnostic records')


if __name__=='__main__':main()
