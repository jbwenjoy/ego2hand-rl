"""Freeze contract/PPO evidence without rewriting earlier task or training runs."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from archive_pen_source_ccd_evidence import pins,write
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot,verify_source_pin


def main():
    out=ROOT/'provenance/pen_contract_results';analysis=ROOT/'runs/pen_contract_archive_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh immutable evidence directories required')
    prior_path=ROOT/'provenance/pen_cv_prior_results/ledger.json'
    prior=json.loads(prior_path.read_text());prior_alias=verify_file_pin(prior['source_resolutions'])
    resolutions=json.loads(prior_alias.read_text())
    previous_manifest=json.loads(verify_file_pin(prior['verified_manifest']).read_text())
    for item in previous_manifest['records']:
        resolved=verify_source_pin(item['observed'],resolutions)
        if file_pin(resolved)!=item['resolved']:raise ValueError('Prior resolution changed')
    aliases={json.dumps(e['original'],sort_keys=True):e for e in resolutions['records']}
    audit_path=ROOT/'runs/pen_task_contract_audit_v2/report.json'
    replay_path=ROOT/'runs/pen_full_pose_replay_v1/report.json'
    audit=json.loads(audit_path.read_text());replay=json.loads(replay_path.read_text())
    if audit['passed']!=85 or audit['total']!=85 or not all(audit['checks'].values()):raise ValueError('Task audit failed')
    if replay['passed']!=3422 or replay['total']!=3422 or not replay['all_checks_passed']:raise ValueError('Feature replay failed')
    if len(replay['rows'])!=285 or any(audit[k] or replay[k] for k in ['native_integrations','optimizer_updates']):
        raise ValueError('Offline contract scope changed')
    test=ROOT/'runs/pen_contract_final_tests261.log';text=test.read_text(encoding='utf-8').replace('\r\n','\n')
    if not re.search(r'Ran 261 tests in ',text) or '\nOK\n' not in text or '\nFAILED ' in text:raise ValueError('Final suite failed')
    docs=[ROOT/'docs'/name for name in ['research-plan-v0.94.md','research-plan-v0.95.md',
        'research-plan-v0.96.md','pen-task-contract-results-2026-11-15.md']]
    analysis.mkdir()
    source_paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study']
        for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),ROOT/'tools/archive_pen_source_ccd_evidence.py',
        ROOT/'tools/audit_pen_task_contracts.py',ROOT/'tools/replay_pen_full_pose.py',ROOT/'tests/test_shared_ppo.py',
        ROOT/'tests/test_pen_pose_observation.py',ROOT/'tests/test_pen_task_contracts.py']+docs
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    receipt=freeze_sources(source_paths,analysis/'source_snapshot',ROOT,revision)
    write(analysis/'source_receipt.json',receipt)
    receipts=[ROOT/'runs'/folder/'source_receipt.json' for folder in ['pen_task_contract_audit_v1',
        'pen_task_contract_audit_v2','pen_full_pose_replay_v1','pen_contract_archive_v1']]
    receipts.append(ROOT/'archive/pen_pre_shared_ppo_v1/source_receipt.json')
    for path in receipts:
        r=json.loads(path.read_text());verify_source_snapshot(r)
        for e in r['records']:
            aliases.setdefault(json.dumps(e['observed'],sort_keys=True),dict(original=e['observed'],archived=e['snapshot'],
                reason='Exact observed source receipt; original path/size/hash all match, no basename/hash-only fallback'))
    resolutions=dict(schema='explicit_source_relocations:1',records=list(aliases.values()))
    for e in resolutions['records']:verify_source_pin(e['original'],resolutions)
    failed=[ROOT/'runs/pen_task_contract_audit_v1/actual_paths.npz',ROOT/'runs/pen_task_contract_audit_console_v1.log']
    logs=[ROOT/'runs'/n for n in ['pen_task_contract_tests250.log','pen_full_pose_contract_tests254.log',
        'pen_shared_ppo_tests6.log','pen_shared_ppo_tests7.log','pen_task_contract_audit_console_v2.log',
        'pen_full_pose_replay_console_v1.log']]+[test]
    roots=dict(task_audit=file_pin(audit_path),feature_replay=file_pin(replay_path))
    todo=[*roots.values(),*[file_pin(p) for p in receipts+docs+failed+logs],file_pin(prior_path),
        file_pin(ROOT/'provenance/pen_horizon_results/ledger.json')]
    allowed={audit_path.parent,replay_path.parent,*[p.parent for p in receipts]}
    verified={}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        path=verify_source_pin(pin,resolutions)
        verified[key]=dict(observed=pin,resolved=file_pin(path))
        if path.suffix=='.json' and any(path.is_relative_to(folder) for folder in allowed):
            todo+=list(pins(json.loads(path.read_text())))
    verify_source_snapshot(receipt,live=True)
    summary=dict(schema='pen_contract_evidence_summary:1',roots=roots,
        checks=dict(task=[85,85],fullpose=[3422,3422],unit_tests=261,focused_PPO_tests=7),
        first_breaches=[{k:c[k] for k in ['profile','mode','dt','first_breaches','first_full_angle_detail']} for c in audit['cases']],
        frames=audit['frame_audits'],common_prefix=audit['common_prefix'],policy_contract=audit['policy_contract'],
        missed_first_pose=sum(c['missed_existing_first_pose'] for c in audit['candidate_tables']),
        observation_metadata=replay['metadata'],axial_feature_L2=replay['axial_roll_feature_L2_differences'],
        max_current_axis_error_m=max(r['current_qpos_axis_max_error_m'] for r in replay['rows']),
        max_cached_axis_error_m=max(r['cached_axis_max_error_m'] for r in replay['rows']),
        legacy_PPO_fix='Detach fixed GAE targets. Frozen-old multibatch error reproduced; model/optimizer bytes equal after only bootstrap detachment. Old cube guard and remainder handling retained.',
        PPO_KL_scope='Historical kl compares same current distributions; new approx_kl compares sampled rollout log probabilities. No KL/task benefit inferred.',
        failed_audit_v1=dict(status='FAILED_REPORT_SERIALIZATION_NP_INT64',artifacts=[file_pin(p) for p in failed],
            scope='Full source receipt retained; only Python int serialization changed for fresh v2.'),
        additional_actual_task_integrations=0,actual_task_training_updates=0,
        test_scope='CPU synthetic optimizer/GAE/normalizer/RNG tests only; no native actor training, environment resume or GPU speed claim.',
        scope='Contract implementation progress, full Goal ACTIVE/incomplete. Original task/force/gradient failures and never-fit roles preserved; no new control video/ego/task success.')
    out.mkdir();write(out/'summary.json',summary);write(out/'source_resolutions.json',resolutions)
    shutil.copyfile(test,out/test.name);shutil.copyfile(logs[3],out/logs[3].name)
    write(analysis/'verified_pins.json',dict(records=list(verified.values())))
    write(out/'ledger.json',dict(schema='pen_contract_evidence_ledger:1',analysis_revision=revision,roots=roots,
        summary=file_pin(out/'summary.json'),source_resolutions=file_pin(out/'source_resolutions.json'),
        previous_evidence=file_pin(prior_path),prior_verified_pins=len(previous_manifest['records']),
        verified_pins=len(verified),exact_source_aliases=len(aliases),verified_manifest=file_pin(analysis/'verified_pins.json'),
        source_receipts=[file_pin(p) for p in receipts],test_logs=[file_pin(out/test.name),file_pin(out/logs[3].name)],
        documents=[file_pin(p) for p in docs],additional_actual_task_integrations=0,actual_task_training_updates=0,
        scope='Detailed current report/source/file pins verified. Earlier native task reports are exact boundary inputs backed by the retained horizon ledger; CV prior manifest independently reverified. Snapshot is an archive-time source superset, not proof every module executed. CPU tests do not qualify native portable physics resume.'))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),verified=len(verified),aliases=len(aliases),prior_verified=len(previous_manifest['records']))))


if __name__=='__main__':main()
