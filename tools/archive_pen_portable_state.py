"""Archive portable-copy passes and retained native-continuation failures together."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from archive_pen_source_ccd_evidence import pins,write
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_pin,verify_source_snapshot


def main():
    import torch
    out=ROOT/'provenance/pen_portable_state_results';analysis=ROOT/'runs/pen_portable_state_archive_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh immutable evidence directories')
    prior_path=ROOT/'provenance/pen_native_policy_results/ledger.json';prior=json.loads(prior_path.read_text())
    aliases_path=verify_file_pin(prior['source_resolutions']);previous=json.loads(aliases_path.read_text())
    pm=json.loads(verify_file_pin(prior['verified_manifest']).read_text())
    for e in pm['records']:verify_source_pin(e['observed'],previous)
    aliases={json.dumps(e['original'],sort_keys=True):e for e in previous['records']}
    folders=['pen_native_closure_inventory_v1']
    workers=[]
    for v in range(1,5):
        for mode in ['producer','restorer']:
            folder=f'pen_native_resume_W1_{mode}_v{v}';workers.append(folder);folders.append(folder)
    for size in [1,32]:
        if size==32:
            for mode in ['producer','restorer']:
                folder=f'pen_native_resume_W32_{mode}_v1';workers.append(folder);folders.append(folder)
        for mode in ['producer','restorer']:
            folder=f'pen_native_resume_contiguous_W{size}_{mode}_v1';workers.append(folder);folders.append(folder)
        folders += [f'pen_native_resume_W{size}_review_v1',f'pen_native_resume_contiguous_W{size}_review_v1',f'pen_native_resume_sameprocess_W{size}_v1']
    reports={f:ROOT/'runs'/f/('failure.json' if f in [f'pen_native_resume_W1_restorer_v{i}' for i in [1,2,3]] else 'report.json') for f in folders}
    records={key:json.loads(path.read_text()) for key,path in reports.items()}
    expected={'pen_native_resume_contiguous_W1_review_v1':[1149,1149],
        'pen_native_resume_contiguous_W32_review_v1':[754,1149],
        'pen_native_resume_sameprocess_W1_v1':[1148,1148],
        'pen_native_resume_sameprocess_W32_v1':[756,1148],
        'pen_native_resume_W1_review_v1':[968,1149],'pen_native_resume_W32_review_v1':[599,1149]}
    for key,value in expected.items():
        if [records[key]['passed'],records[key]['total']]!=value:raise ValueError('Retained result changed')
    total=sum(records[f]['observed_integrations'] for f in workers)+sum(records[f'pen_native_resume_sameprocess_W{n}_v1']['actual_native_integrations'] for n in [1,32])
    if total!=19392:raise ValueError('Native accounting changed')
    test=ROOT/'runs/pen_portable_state_tests278_v1.log';text=test.read_text(encoding='utf-8').replace('\r\n','\n')
    if 'Ran 278 tests in ' not in text or '\nOK\n' not in text:raise ValueError('Full unit suite missing')
    docs=[ROOT/'docs'/n for n in ['research-plan-v0.97.md','research-plan-v0.98.md','pen-portable-state-results-2026-11-16.md']]
    configs=[ROOT/'configs'/n for n in ['pen_native_resume_v1.toml','pen_native_resume_v2.toml','pen_full_pose_native_v1.toml','pen_gpu_step_audit_v1.toml']]
    analysis.mkdir();revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    source_paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})
    source_paths += [ROOT/'tools'/n for n in ['archive_pen_portable_state.py','archive_pen_source_ccd_evidence.py',
        'inspect_pen_native_closure.py','pen_native_resume_worker.py','review_pen_native_resume.py','control_pen_native_resume.py']]
    source_paths += [ROOT/'tests/test_portable_pen_state.py']+docs+configs
    receipt=freeze_sources(source_paths,analysis/'source_snapshot',ROOT,revision);write(analysis/'source_receipt.json',receipt)
    receipts=[ROOT/'runs'/f/'source_receipt.json' for f in folders]+[analysis/'source_receipt.json']
    for path in receipts:
        r=json.loads(path.read_text());verify_source_snapshot(r)
        for e in r['records']:aliases.setdefault(json.dumps(e['observed'],sort_keys=True),dict(original=e['observed'],archived=e['snapshot'],
            reason='Exact observed source path/size/hash receipt; no basename/hash-only fallback'))
    resolutions=dict(schema='explicit_source_relocations:1',records=list(aliases.values()))
    for e in resolutions['records']:verify_source_pin(e['original'],resolutions)
    allowed={p.parent for p in reports.values()}|{p.parent for p in receipts};verified={}
    # Include external engine source pins actually recorded INSIDE every checkpoint.
    checkpoint_pins={json.dumps(records[f]['checkpoint'],sort_keys=True):records[f]['checkpoint'] for f in workers if 'checkpoint' in records[f]}
    engines=[];checkpoint_audits=[]
    for pin in checkpoint_pins.values():
        state=torch.load(verify_file_pin(pin),map_location='cpu',weights_only=False)
        verify_source_snapshot(state['sources'])
        for p in state['engine_sources']:verify_file_pin(p)
        engines+=state['engine_sources']
        checkpoint_audits.append(dict(checkpoint=pin,schema=state['schema'],worlds=state['worlds'],
            native_model_arrays=len(state['native']['model']['arrays']),native_data_arrays=len(state['native']['data']['arrays']),
            python_tensors=len(state['python']['tensors']),python_arrays=len(state['python']['arrays']),python_scalars=len(state['python']['scalars']),
            external_source_pins=len(state['engine_sources'])))
        del state
    logs=[]
    for f in folders:
        if f=='pen_native_closure_inventory_v1':name='pen_native_closure_inventory_console_v1.log'
        else:
            stem,version=f.rsplit('_v',1);name=stem+'_console_v'+version+'.log'
        p=ROOT/'runs'/name
        if p.exists():logs.append(p)
    logs += [ROOT/'runs'/n for n in ['pen_native_resume_layout_diagnosis_v1.log','pen_portable_state_unit8_v1.log',
        'pen_portable_state_unit9_v1.log','pen_portable_state_unit12_v1.log']]
    todos=[file_pin(p) for p in list(reports.values())+receipts+docs+configs+logs+[test,prior_path]]+engines
    while todos:
        pin=todos.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        resolved=verify_source_pin(pin,resolutions);verified[key]=dict(observed=pin,resolved=file_pin(resolved))
        if resolved.suffix=='.json' and any(resolved.is_relative_to(folder) for folder in allowed):
            todos+=list(pins(json.loads(resolved.read_text())))
    verify_source_snapshot(receipt,live=True)
    primary={str(size):records[f'pen_native_resume_contiguous_W{size}_review_v1'] for size in [1,32]}
    controls={str(size):records[f'pen_native_resume_sameprocess_W{size}_v1'] for size in [1,32]}
    summary=dict(schema='pen_portable_state_evidence_summary:1',roots={k:file_pin(p) for k,p in reports.items()},
        primary={k:{key:r[key] for key in ['passed','total','first_failed_substep','maximum_state_errors','maximum_contact_errors']} for k,r in primary.items()},
        sameprocess={k:{key:r[key] for key in ['passed','total','first_failed_substep','maximum_state_errors','maximum_contact_errors','immediate']} for k,r in controls.items()},
        legacy_boundary_reset=dict(scope='First worker inadvertently injected k8 partial reset; W1 all-reset. Kept as DIFFERENT diagnostic, not contiguous resume.',
            checks={str(n):expected[f'pen_native_resume_W{n}_review_v1'] for n in [1,32]}),
        checkpoint_audits=checkpoint_audits,task_integrations=19392,actual_task_optimizer_updates=0,final_unit_tests=278,
        worker_failures={f:records[f] for f in workers if 'type' in records[f]},
        scope='Exact array copy/RNG passes; W1 short continuation passes, W32 native future remains unqualified even same-process. Do not promote complete physical resume, contacts/WM gradients, policy/ego/speedup or full Goal success.')
    out.mkdir();write(out/'summary.json',summary);write(out/'source_resolutions.json',resolutions);shutil.copyfile(test,out/test.name)
    write(analysis/'verified_pins.json',dict(records=list(verified.values())))
    write(out/'ledger.json',dict(schema='pen_portable_state_evidence_ledger:1',analysis_revision=revision,
        roots={k:file_pin(p) for k,p in reports.items()},summary=file_pin(out/'summary.json'),source_resolutions=file_pin(out/'source_resolutions.json'),
        prior_evidence=file_pin(prior_path),prior_verified_pins=len(pm['records']),verified_pins=len(verified),exact_source_aliases=len(aliases),
        verified_manifest=file_pin(analysis/'verified_pins.json'),source_receipts=[file_pin(p) for p in receipts],
        documents=[file_pin(p) for p in docs],test_log=file_pin(out/test.name),task_integrations=19392,actual_task_optimizer_updates=0,
        scope='All current worker/checkpoint/contact/report/source pins verified, including external engine sources inside checkpoints and failed workers. Prior native evidence reverified. Archive-time superset is not proof every module executed. Full Goal remains active/incomplete.'))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),verified=len(verified),aliases=len(aliases),prior_verified=len(pm['records']))))


if __name__=='__main__':main()
