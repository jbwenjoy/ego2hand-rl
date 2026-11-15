"""Compact exact native-interface evidence, including its failed first version."""
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from archive_pen_source_ccd_evidence import pins,write
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot,verify_source_pin


def main():
    out=ROOT/'provenance/pen_native_policy_results';analysis=ROOT/'runs/pen_native_policy_archive_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh immutable archive')
    prior_path=ROOT/'provenance/pen_contract_results/ledger.json';prior=json.loads(prior_path.read_text())
    prior_alias=verify_file_pin(prior['source_resolutions']);resolutions=json.loads(prior_alias.read_text())
    prev=json.loads(verify_file_pin(prior['verified_manifest']).read_text())
    for e in prev['records']:verify_source_pin(e['observed'],resolutions)
    reports={k:ROOT/'runs'/v/'report.json' for k,v in dict(failed='pen_full_pose_native_v1',
        fixed='pen_full_pose_native_v2',final='pen_full_pose_native_v3',review_fixed='pen_full_pose_native_review_v1',
        review_final='pen_full_pose_native_review_v2').items()}
    data={key:json.loads(path.read_text()) for key,path in reports.items()}
    for key,expected in [('failed',[662,670]),('fixed',[674,674]),('final',[674,674]),('review_fixed',[256,256]),('review_final',[256,256])]:
        r=data[key]
        if [r['passed'],r['total']]!=expected:raise ValueError('Declared native result changed')
    if sum(data[k]['actual_native_integrations'] for k in ['failed','fixed','final'])!=12672:raise ValueError('Native accounting changed')
    test=ROOT/'runs/pen_full_pose_native_tests266.log';t=test.read_text(encoding='utf-8').replace('\r\n','\n')
    if 'Ran 266 tests in ' not in t or '\nOK\n' not in t:raise ValueError('Final test suite missing')
    docs=[ROOT/'docs'/name for name in ['research-plan-v0.96.md','research-plan-v0.97.md','pen-native-policy-results-2026-11-15.md']]
    configs=[ROOT/'configs'/name for name in ['pen_full_pose_native_v1.toml','controlled_evaluation_v4.toml','controlled_pen_single_clip.json','artifacts.toml','pen_gpu_step_audit_v1.toml']]
    analysis.mkdir();revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})
    paths += [Path(__file__).resolve(),ROOT/'tools/archive_pen_source_ccd_evidence.py',ROOT/'tools/qualify_pen_native_policy.py',
        ROOT/'tools/review_pen_native_policy.py',ROOT/'tests/test_pen_native_policy.py']+docs+configs
    receipt=freeze_sources(paths,analysis/'source_snapshot',ROOT,revision);write(analysis/'source_receipt.json',receipt)
    receipts=[p.parent/'source_receipt.json' for p in reports.values()]+[analysis/'source_receipt.json']
    aliases={json.dumps(e['original'],sort_keys=True):e for e in resolutions['records']}
    for path in receipts:
        r=json.loads(path.read_text());verify_source_snapshot(r)
        for e in r['records']:aliases.setdefault(json.dumps(e['observed'],sort_keys=True),dict(
            original=e['observed'],archived=e['snapshot'],reason='Exact observed source path/size/hash snapshot; no basename/hash-only fallback'))
    resolutions=dict(schema='explicit_source_relocations:1',records=list(aliases.values()))
    for e in resolutions['records']:verify_source_pin(e['original'],resolutions)
    cfg=tomllib.loads(configs[0].read_text());proposal=(configs[0].parent/cfg['initialization_proposal']).resolve()
    selected=json.loads(proposal.read_text())['artifacts']['selected'];verify_file_pin(selected)
    logs=[ROOT/'runs'/('pen_full_pose_native_console_v'+str(v)+'.log') for v in [1,2,3]]
    logs += [ROOT/'runs'/('pen_full_pose_native_review_console_v'+str(v)+'.log') for v in [1,2]]
    todos=[file_pin(p) for p in list(reports.values())+receipts+docs+configs+logs+[test,prior_path,proposal]]+[selected]
    allowed={p.parent for p in reports.values()}|{p.parent for p in receipts};verified={}
    while todos:
        pin=todos.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        resolved=verify_source_pin(pin,resolutions);verified[key]=dict(observed=pin,resolved=file_pin(resolved))
        if resolved.suffix=='.json' and any(resolved.is_relative_to(folder) for folder in allowed):
            todos+=list(pins(json.loads(resolved.read_text())))
    verify_source_snapshot(receipt,live=True)
    summary=dict(schema='pen_native_policy_evidence_summary:1',roots={k:file_pin(p) for k,p in reports.items()},
        native_checks={key:[data[key]['passed'],data[key]['total']] for key in data},
        retained_first_failure=[k for k,v in data['failed']['checks'].items() if not v],
        transitions_per_version=528,integrations_per_version=4224,total_native_integrations=12672,
        CPU_kinematics_queries=1056,actual_task_optimizer_updates=0,final_unit_tests=266,
        latest_max_policy_error=max(r['max_policy_error'] for r in data['review_final']['rows']),
        latest_max_critic_error=max(r['max_critic_error'] for r in data['review_final']['rows']),
        latest_max_reward_error=max(r['max_reward_error'] for r in data['review_final']['rows']),
        native_cases={key:[{k:c[k] for k in ['worlds','transitions','integrations','observed_setup_integrations',
            'graph_capture_invocations','terminal_reset_worlds','explicit_reset_worlds','model_unchanged','metadata','numerical']} for c in data[key]['cases']]
            for key in ['failed','fixed','final']},
        physical_comparisons=data['review_final']['physical_comparisons'],
        initialization_configuration=dict(proposal=file_pin(proposal),selected=selected,
            scope='Pre-start physical initialization only; historical diagnostic decision/response labels never fitted or promoted.'),
        retained_nonrepeat='v2/v3 terminal reset worlds41/42; no physical repeat/portable resume qualification or speedup inferred.',
        scope='Native161/188 frame/control/postphysics reward/partial-reset progress. All full-task/contact-wrench/gradient failures persist. No Actor training, WM fitting, full-task/portable-resume/ego/GPU-speedup claim; full Goal active.')
    out.mkdir();write(out/'summary.json',summary);write(out/'source_resolutions.json',resolutions);shutil.copyfile(test,out/test.name)
    write(analysis/'verified_pins.json',dict(records=list(verified.values())))
    write(out/'ledger.json',dict(schema='pen_native_policy_evidence_ledger:1',analysis_revision=revision,
        roots={k:file_pin(p) for k,p in reports.items()},summary=file_pin(out/'summary.json'),source_resolutions=file_pin(out/'source_resolutions.json'),
        prior_evidence=file_pin(prior_path),prior_verified_pins=len(prev['records']),verified_pins=len(verified),exact_source_aliases=len(aliases),
        verified_manifest=file_pin(analysis/'verified_pins.json'),source_receipts=[file_pin(p) for p in receipts],
        documents=[file_pin(p) for p in docs],test_log=file_pin(out/test.name),total_native_integrations=12672,
        actual_task_optimizer_updates=0,CPU_kinematics_queries=1056,
        scope='Current report/array/model/external-source pins verified. Historical H5/task reports are exact boundary inputs backed by prior evidence. Archive-time config/source superset is not proof every file executed. No native physics resume claim.'))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),verified=len(verified),aliases=len(aliases),prior_verified=len(prev['records']))))


if __name__=='__main__':main()
