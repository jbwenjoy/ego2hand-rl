"""Saved packet/transport review retaining every natural-lifecycle physics mask."""
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.shared_snapshot_transport import load_shared
from supermimic_research.learning.pen_live_decision import from_owned_snapshot
from supermimic_research.mechanics.probe_row_review import review_recorded_row
from supermimic_research.mechanics.pen_reward_chain import native_components
from supermimic_research.strict_json import write_json


def read(p):return json.loads(Path(p).read_text())


def main():
    started=time.perf_counter();rp=ROOT/'runs/pen_terminal_lifecycle_v1/report.json';r=read(rp);out=ROOT/'runs/pen_terminal_lifecycle_review_v1'
    if out.exists() or r['passed']!=r['total'] or not all(r['checks'].values()):raise ValueError('Fresh review of terminal lifecycle required')
    verify_source_snapshot(read(verify_file_pin(r['source_receipt'])),live=True);objective=tomllib.loads(verify_file_pin(r['inputs']['objective_config']).read_text());adapter=(Path(r['inputs']['objective_config']['path']).parent/objective['adapter_config']).resolve();reward=tomllib.loads(adapter.read_text())['reward']
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__)],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt);rows=[];models={};physical={}
    with patch.object(mujoco,'mj_step',side_effect=AssertionError('No native replay in saved review')),patch.object(mujoco,'mj_forward',side_effect=AssertionError('No forward replay in saved review')):
        for case in r['rows']:
            source=load_shared(case['source']);end=load_shared(case['end']);z=dict(np.load(verify_file_pin(case['trace'])));w=dict(np.load(verify_file_pin(case['window'])));key=case['model']['path']
            if key not in models:models[key]=mujoco.MjModel.from_binary_path(str(verify_file_pin(case['model'])));physical[key]=dict(np.load(verify_file_pin(case['physical_context'])))
            q=source['native']['data']['arrays'];qa=case['model_layout']['object_qposadr'];body=case['model_layout']['object_body'];R=Rotation.from_quat(q['qpos'][0,qa+np.array([4,5,6,3])]).as_matrix();view=dict(native=source,phase=case['phase'],source_qpos_R=R,source_cached_R=q['xmat'][0,body]);decision=from_owned_snapshot(source,case['phase'],case,z['actions20'][0])
            packet,audit=review_recorded_row(case,z,w,view,models[key],physical[key],dict(role=case['role'],object_penetration_m=.002,self_penetration_m=.0002,joint_excursion_rad=.001));values,formula,errors=native_components(case,z,w,decision['controller_context'],reward,objective);endpoint=end['native']['data']['arrays']
            extra=dict(input102_bytes=np.array_equal(packet['x'].astype(np.float32),decision['input102']),source_reward=abs(float(values[-1])-case['source_reward'])<=2e-6,end_qpos=np.array_equal(z['post_qpos'][-1],endpoint['qpos'][0]),end_qvel=np.array_equal(z['post_qvel'][-1],endpoint['qvel'][0]),end_warm=np.array_equal(z['post_qacc_warmstart'][-1],endpoint['qacc_warmstart'][0]),actual8_native=case['native_integrations']==len(z['post_qpos'])==8,
                stored_physics=audit==case['physical_audit'],stored_reward=formula==case['reward_checks'],stored_formula_errors=errors==case['reward_errors'],original_sampled_RAW=np.array_equal(z['actions20'][0],np.asarray(case['sampled_RAW20'],np.float32)),original_clipped_RAW=np.array_equal(decision['applied_RAW20'],np.asarray(case['applied_RAW20'],np.float32)),original_clip_count=case['RAW_clip_count']==decision['clipped_components'])
            rows.append(dict(name=case['name'],physics=audit,reward=formula,reward_errors=errors,extra=extra,contract_valid=all(formula.values()) and all(extra.values())))
    invalid={a['arm']:[c['name'] for c in rows if c['name'].startswith(a['arm']+'_P') and not c['physics']['valid']] for a in r['arms']}
    checks=dict(all_rows_present=len(rows)==len(r['rows']),all_input_endpoint_reward=all(c['contract_valid'] for c in rows),original_masks_unchanged=all(invalid[a['arm']]==a['physical_invalid_rows'] for a in r['arms']),
        natural_stop_declared=all(a['selected_controls']+len(a['missing_controls'])==368 and a['missing_controls']==list(range(a['selected_controls']+1,369)) and a['terminal_kind'] in ['FAILURE','REFERENCE_COMPLETION'] for a in r['arms']),
        all6_original_restore_readbacks=sum(len(a['restores']) for a in r['arms'])==6 and all(all(v['audit']['all_array_bytes_equal'] and v['audit']['all_scalar_values_equal'] and all(v['complete_state_equal'].values()) and v['external_counter_unchanged'] for v in a['restores']) for a in r['arms']),
        original_terminal_guards=all(all(v for k,v in a['terminal_guard'].items() if k!='message') and a['preterminal_pending_cleared'] and a['terminal_pending_restored'] and a['phase_after_terminal']==a['expected_phase_after_terminal'] for a in r['arms']),
        exact_work=r['actual_native_integrations']==r['attempted_native_integrations']==len(rows)*8<=5920 and r['optimizer_steps']==r['attempted_optimizer_steps']==0)
    verify_source_snapshot(receipt,live=True);result=dict(schema='pen_terminal_lifecycle_saved_review:1',run=file_pin(rp),source_receipt=file_pin(out/'source_receipt.json'),checks=checks,passed=sum(checks.values()),total=len(checks),rows=rows,physical_invalid_rows=invalid,physical_valid_rows=sum(c['physics']['valid'] for c in rows),
        original_physics_flags=sum(len(c['physics']['checks']) for c in rows),original_reward_flags=sum(len(c['reward']) for c in rows),actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,wall_s=time.perf_counter()-started,
        scope='Saved independent CPU decode/input/reward/endpoint/mask review only. Online terminal pending/restore receipts checked,not new runtime replay. Natural failure lifecycle never implies full task or unsafe rows certified;unobserved reference completion remains unproven.')
    write_json(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],rows=len(rows),valid_physics=result['physical_valid_rows'],physics_flags=result['original_physics_flags'],reward_flags=result['original_reward_flags'])),flush=True)
    if not all(checks.values()):raise ValueError('Saved terminal lifecycle discrepancy retained')


if __name__=='__main__':main()
