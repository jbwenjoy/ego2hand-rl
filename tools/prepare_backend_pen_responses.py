"""Independently decode new-backend raw responses; no fit or native stepping."""
import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import mujoco
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.learning.backend_probe_review import require_response_probe
from supermimic_research.learning.backend_response_study import response_spec
from supermimic_research.learning.pen_interventions import prefix_noise_schedule
from supermimic_research.learning.pen_matched_recovery import write_metadata as write
from supermimic_research.runtime.explicit_collision_backend import load_contract
from supermimic_research.runtime.backend_pen_probe import require_backend_probe
from supermimic_research.mechanics.probe_row_review import review_recorded_row,aligned_rows,response_groups


def read(p):return json.loads(Path(p).read_text())


def main(a):
    rp=a.report.resolve();run=read(rp);cp=verify_file_pin(run['config']);cfg=tomllib.loads(cp.read_text());out=a.output.resolve()
    contract=load_contract((cp.parent/cfg['backend_manifest']).resolve());reg_path=(cp.parent/cfg['registration']).resolve();reg=read(reg_path)
    require_response_probe(run,cfg,contract,file_pin(reg_path),file_pin(cp))
    if reg['specs'][cfg['role']]!=file_pin(cp) or reg['backend_id']!=contract['backend_id']:raise ValueError('Raw response registration differs')
    for r in [reg,run]:verify_source_snapshot(read(verify_file_pin(r['source_receipt'])),live=True)
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh backend preparation directory required')
    out.mkdir();protocol=ROOT/'docs/pen-backend-response-preparation-protocol-v1.md'
    paths=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,protocol]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt)
    checks={};metadata=[];rows=[];models={};physical={};states={};phase='source_admission'
    try:
        directions,profiles,_=response_spec(cfg);checks['directions_exact']=np.array_equal(directions,np.load(verify_file_pin(run['directions']))['directions'])
        noise=dict(np.load(verify_file_pin(run['prefix_noise'])))
        for seed in cfg['noise_seeds']:checks[f'noise{seed}_exact']=np.array_equal(noise[f'seed{seed}'],prefix_noise_schedule(seed,max(cfg['source_phases'])-1,cfg['prefix_raw_sigma']))
        producers=[read(verify_file_pin(pin)) for pin in run['producers']]
        checks['all_four_distinct_episodes']=len(producers)==4 and len({p['episode'] for p in producers})==4
        checks['producer_cost']=sum(p['native_integrations'] for p in producers)==run['producer_native_integrations']
        for producer in producers:
            actions=producer['prefix_actions'];schedule=noise[f'seed{producer["noise_seed"]}'];key=producer['episode'];audit=producer['actor_audit']
            checks[key+'_uninterrupted_clock']=producer['no_post_start_resets'] and len(actions)*8==producer['native_integrations'] and [r['phase'] for r in actions]==list(range(1,len(actions)+1)) and producer['final_phase']==len(actions)+1
            checks[key+'_exogenous_noise']=all(np.array_equal(np.asarray(r['submitted_raw'],np.float32),np.asarray(r['mean_raw'],np.float32)+schedule[r['control_index']]) and r['clamped_components']==int((abs(np.asarray(r['submitted_raw']))>1).sum()) for r in actions)
            checks[key+'_fresh_same_backend_actor']=audit['full_W1_model_arrays']==404 and audit['full_W1_model_scalars']==195 and audit['full_W1_model_bytes_equal'] and audit['actor_normalizer_bytes_equal'] and audit['backend_binding_verified'] and audit['collision_backend_id']==contract['backend_id'] and not audit['old_native_backend_imported']
        for source in run['sources']:
            group=[c for c in run['cases'] if c['source_name']==source['name']];checks[source['name']+'_profiles_retained']=[c['profile'] for c in group]==profiles
            if not source['available']:continue
            state=torch.load(verify_file_pin(source['source']),map_location='cpu',weights_only=False);require_backend_probe(state,contract)
            if state['role']!=run['role'] or state['actor_checkpoint']!=source['actor_checkpoint']:raise ValueError('Source role/actor pin differs')
            verify_file_pin(state['actor_checkpoint']);verify_source_snapshot(state['native']['sources'],live=True)
            for pin in state['native']['engine_sources']:verify_file_pin(pin)
            states[source['name']]=state;key=source['model']['path']
            if key not in models:models[key]=mujoco.MjModel.from_binary_path(str(verify_file_pin(source['model'])));physical[key]=dict(np.load(verify_file_pin(source['physical_context'])))
        # Decoding must never advance dynamics, even accidentally via a helper.
        from unittest.mock import patch
        with patch.object(mujoco,'mj_step',side_effect=AssertionError('Read-only preparation')),patch.object(mujoco,'mj_forward',side_effect=AssertionError('Saved packets only')):
            for ci,case in enumerate(run['cases']):
                if not case['available']:continue
                phase=case['name'];z=dict(np.load(verify_file_pin(case['trace'])));window=dict(np.load(verify_file_pin(case['window'])))
                row,audit=review_recorded_row(case,z,window,states[case['source_name']],models[case['model']['path']],physical[case['model']['path']],cfg)
                metadata.append(dict(row_index=len(rows),case_index=ci,name=case['name'],source_name=case['source_name'],actor=case['actor'],episode=case['episode'],noise_seed=case['noise_seed'],phase=case['phase'],profile=case['profile'],**audit));rows.append(row)
        data=aligned_rows(rows,metadata);groups=response_groups(metadata,run['sources'],cfg)
        for source in run['sources']:
            ids=[i for i,r in enumerate(metadata) if r['source_name']==source['name']]
            checks[source['name']+'_identical_state_warm']=all(np.array_equal(data['x'][i,:81],data['x'][ids[0],:81]) for i in ids)
        checks.update(all_available_rows_retained=len(rows)==sum(c['available'] for c in run['cases']),all_declared_groups_retained=len(groups)==cfg['expected_sources']*40,
            exact_cost=run['actual_native_integrations']==run['producer_native_integrations']+8*len(rows),finite_aligned_rows=all(np.isfinite(data[k]).all() for k in ['x','y','heads']))
        np.savez_compressed(out/'response_rows.npz',**data);verify_source_snapshot(receipt,live=True)
        report=dict(schema='backend_pen_response_preparation:1',probe=file_pin(rp),config=file_pin(cp),registration=file_pin(reg_path),collision_backend=contract,protocol=file_pin(protocol),
            source_receipt=file_pin(out/'source_receipt.json'),rows=file_pin(out/'response_rows.npz'),metadata=metadata,response_groups=groups,directions=run['directions'],
            role=run['role'],checks=checks,passed=sum(checks.values()),total=len(checks),total_rows=len(rows),qualified_rows=int(data['valid'].sum()),
            rejected_rows=[r['row_index'] for r in metadata if not r['valid']],unavailable_case_indices=[i for i,c in enumerate(run['cases']) if not c['available']],
            qualified_response_groups=sum(g['qualified'] for g in groups),row_gate_passed=sum(sum(r['checks'].values()) for r in metadata),row_gate_total=sum(len(r['checks']) for r in metadata),
            data_status='EMPTY_NOT_ELIGIBLE_FOR_FIT' if not rows else 'REJECTED_ROWS_RETAINED_AND_MASKED' if not data['valid'].all() else 'ALL_AVAILABLE_ROWS_PASS',
            scales_fitted=False,CHECK_response_rows_used=0,validation_rows_used_for_TRAIN=0,actual_native_integrations=0,optimizer_steps=0,
            scope='New-backend raw physics review only;explicit source/row masks,all declared groups retained. No normalization/fitting/gradient/Actor/task/ego admission.')
        write(out/'report.json',report)
        if not all(checks.values()):raise ValueError('Preparation accounting/contract failed: '+repr(checks))
        print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[report['passed'],report['total']],rows=len(rows),qualified=report['qualified_rows'])),flush=True)
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),phase=phase,reviewed_rows=len(rows),actual_native_integrations=0,optimizer_steps=0));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
