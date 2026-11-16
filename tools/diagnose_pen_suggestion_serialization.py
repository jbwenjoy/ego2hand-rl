"""Retain the first failed probe and reconstruct its bounded work explicitly."""
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
from supermimic_research.mechanics.probe_row_review import review_recorded_row
from supermimic_research.mechanics.pen_reward_chain import native_components
from supermimic_research.strict_json import write_json


def read(p):return json.loads(Path(p).read_text())


def main():
    failed=ROOT/'runs/pen_native_suggestions_v1';out=ROOT/'runs/pen_native_suggestions_failure_diagnosis_v1'
    if out.exists():raise ValueError('Fresh failure diagnosis required')
    bank_path=ROOT/'runs/pen_native_suggestions_bank_v1/report.json';bank=read(bank_path);source=next(s for s in bank['sources'] if s['name']=='RESIDUAL_N8841_P012')
    traces=list(failed.rglob('trace.npz'));windows=list(failed.rglob('window.npz'));log=ROOT/'runs/pen_native_suggestions_v1.log';text=log.read_text();old_source=ROOT/'tools/probe_pen_native_suggestions.py'
    if len(traces)!=1 or len(windows)!=1 or traces[0].parent.name!='BASE_R0' or (failed/'report.json').exists() or (failed/'failure.json').exists():raise ValueError('Exact first-record failure layout required')
    verify_source_snapshot(read(failed/'source_receipt.json'),live=True)
    z=dict(np.load(traces[0]));w=dict(np.load(windows[0]));case=dict(source['template'],trace=file_pin(traces[0]),window=file_pin(windows[0]),qualified=False)
    case['actual_PD20']=z['post_ctrl'][0,case['aids']].tolist();state=torch.load(verify_file_pin(source['original_source']['source']),map_location='cpu',weights_only=False)
    evaluation=read(verify_file_pin(bank['inputs']['evaluation']));reward_audit=read(verify_file_pin(evaluation['inputs']['native_reward']));contexts=torch.load(verify_file_pin(reward_audit['source_contexts']),map_location='cpu',weights_only=False)
    objective=__import__('tomllib').loads(verify_file_pin(bank['inputs']['objective_config']).read_text());reward=__import__('tomllib').loads(verify_file_pin(reward_audit['inputs']['adapter_config']).read_text())['reward']
    with patch.object(mujoco,'mj_step',side_effect=AssertionError('Saved failure only')),patch.object(mujoco,'mj_forward',side_effect=AssertionError('Saved failure only')):
        model=mujoco.MjModel.from_binary_path(str(verify_file_pin(case['model'])));physical=dict(np.load(verify_file_pin(case['physical_context'])))
        _,physics=review_recorded_row(case,z,w,state,model,physical,dict(role=case['role'],object_penetration_m=.002,self_penetration_m=.0002,joint_excursion_rad=.001))
        values,formula,errors=native_components(case,z,w,contexts[source['name']],reward,objective)
    score=np.float64(values[7]);flag=abs(float(score)-score)<=2e-6;reproduced=False
    try:json.dumps(dict(source_reward=flag))
    except TypeError:reproduced=True
    checks=dict(first8_complete=len(z['post_qpos'])==len(z['post_qvel'])==8 and int(w['counts'][0])==8,
        source_raw=np.array_equal(z['actions20'][0],np.asarray(source['base_raw'],np.float32)),
        original_case_write_failure='line 92, in main' in text and 'During handling of the above exception' in text and 'not JSON serializable' in text,
        numpy_boolean_failure_reproduced=reproduced,all_saved_physics_flags=all(physics['checks'].values()),all_reward_formula_flags=all(formula.values()))
    if not all(checks.values()):raise ValueError('First failure evidence differs: '+repr(checks))
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),old_source],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt)
    report=dict(schema='pen_suggestion_serialization_failure:1',original_source=file_pin(old_source),original_log=file_pin(log),original_bank=file_pin(bank_path),original_run_receipt=file_pin(failed/'source_receipt.json'),
        trace=file_pin(traces[0]),window=file_pin(windows[0]),checks=checks,passed=sum(checks.values()),total=len(checks),physics=physics,reward_formula_checks=formula,reward_formula_errors=errors,
        source_receipt=file_pin(out/'source_receipt.json'),reconstructed_actual_native_integrations=8,reconstructed_attempted_native_integrations=8,reconstructed_successful_restores=2,
        reconstruction_basis='Exactly one complete8-tick packet;frozen first-source warmup+branch restore;traceback at first case write before next control. Original live counters/failure JSON were not persisted;counts reconstructed from complete packet and frozen control flow,not a resumed run.',
        returned_source_reward_retained=False,original_online_qualification='UNAVAILABLE_DO_NOT_INFER',new_actual_native_integrations=0,optimizer_steps=0,
        scope='Serialization failure preserved.20 saved physics flags and original reward formula can be reviewed;original online qualified flag/returned reward lost. Retain8 prior integrations in total costs;new run must use fresh output and same bank.')
    write_json(out/'report.json',report);print(json.dumps(dict(diagnosis=file_pin(out/'report.json'),checks=[report['passed'],report['total']],prior_native_work=8)),flush=True)


if __name__=='__main__':main()
