"""Archive original failure and serialization-only repair with independent sums."""
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.learning.world_model import raw_tree_equal
from archive_pen_source_ccd_evidence import pins


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    out=ROOT/'runs/pen_reward_chain_independent_review_v1';archive=ROOT/'provenance/pen_reward_chain_results_v1'
    if out.exists() or archive.exists():raise ValueError('Fresh reward-chain archive required')
    v1=ROOT/'runs/pen_backend_reward_chain_v1';v2=ROOT/'runs/pen_backend_reward_chain_v2';r=read(v2/'report.json')
    prior_path=ROOT/'provenance/pen_backend_response_results_v1/ledger.json';prior=read(prior_path)
    old_code=(ROOT/'tools/audit_backend_reward_chain.py').read_text();new_code=(ROOT/'tools/audit_backend_reward_chain_v2.py').read_text()
    checks=dict(serialization_only_source_change=new_code==old_code.replace('share_above_10_percent=sum(v>.1 for v in geometry))','share_above_10_percent=int(sum(v>.1 for v in geometry)))'),
        complete_original_failure=read(v1/'failure.json')['reviewed_rows']==4860 and read(v1/'failure.json')['type']=='TypeError',
        repaired6=r['passed']==r['total']==6 and all(r['checks'].values()),
        no_native_optimizer_normalization=all(r[k]==0 for k in ['actual_native_integrations','optimizer_steps','normalizer_fits']))
    contexts1=torch.load(v1/'source_contexts.pt',map_location='cpu',weights_only=False)
    contexts2=torch.load(verify_file_pin(r['source_contexts']),map_location='cpu',weights_only=False)
    checks['all36_source_contexts_byte_equal']=len(contexts1)==len(contexts2)==36 and raw_tree_equal(contexts1,contexts2)
    omission={};arithmetic=[]
    for role,c in r['cohorts'].items():
        old=dict(np.load(v1/(role+'_native_components.npz')));new=dict(np.load(verify_file_pin(c['outputs'])))
        checks[role+'_all_numeric_outputs_unchanged']=old.keys()==new.keys() and all(raw_tree_equal(v,new[k]) for k,v in old.items())
        checks[role+'_all14_row_checks']=c['row_check_passed']==c['row_check_total']==14*c['rows'] and all(all(m['checks'].values()) for m in c['row_audits'])
        prep=read(verify_file_pin(r['inputs'][role+'_preparation']));values=new['values'];mask=new['physical_valid']
        checks[role+'_original_masks']=np.array_equal(mask,[m['valid'] for m in prep['metadata']]) and int(mask.sum())==prep['qualified_rows']
        a=values.astype(np.float32)
        checks[role+'_full_reward_component_sums']=np.array_equal(values[:,7],(a[:,4]+a[:,5]).astype(np.float64)) and np.array_equal(values[:,8],(a[:,4]+a[:,5]+np.float32(.2)*a[:,6]).astype(np.float64))
        groups=c['native_reward_responses']['groups']
        for g,observed in zip(prep['response_groups'],groups):
            p,m=g['rows']['1'],g['rows']['-1'];complete=len(p)==len(m)==2
            arithmetic.append(observed['qualified']==g['qualified'] and observed['complete']==complete and observed['source_name']==g['source_name'])
            if complete:
                fd=(values[p[0]]+values[p[1]]-values[m[0]]-values[m[1]])/(16*g['epsilon'])
                arithmetic.append(bool(np.allclose(fd,observed['finite_response_per_RAW'],atol=1e-12,rtol=1e-10)))
            else:arithmetic.append(observed['finite_response_per_RAW'] is None)
        checks[role+'_all_groups_preserved']=len(groups)==len(prep['response_groups'])
        selected=[g for g in groups if g['qualified']];fd=np.asarray([g['finite_response_per_RAW'] for g in selected])
        screens={(x['source_name'],x['direction']):x['channels'] for x in c['native_reward_responses']['scalar_amplitude_screens']}
        omission[role]={}
        for channel,j,observable in [('residual_objective',7,fd[:,4]),('support_objective',8,fd[:,4]+.2*fd[:,6])]:
            omission[role][channel]={}
            for subset in ['ALL_PHYSICAL_VALID','FROZEN_SCALAR_SCREEN']:
                keep=np.ones(len(fd),bool) if subset=='ALL_PHYSICAL_VALID' else np.array([screens[g['source_name'],g['direction']][channel]['physical_screen'] for g in selected])
                target=fd[keep,j];partial=observable[keep];den=float(np.linalg.norm(target))
                omission[role][channel][subset]=dict(pairs=int(keep.sum()),omission_relative_L2=float(np.linalg.norm(partial-target)/den) if den else None,
                    opposite_sign_pairs=int(np.sum(partial*target<0)),zero_observable_pairs=int(np.sum(partial==0)),zero_full_pairs=int(np.sum(target==0)))
    checks['all2880_group_arithmetic_flags']=len(arithmetic)==2880 and all(arithmetic)
    preflight_log=ROOT/'runs/pen_reward_chain_preflight_v1.log';preflight=json.loads(preflight_log.read_text().splitlines()[-1])
    checks['original_one_case_preflight']=len(preflight['checks'])==14 and all(preflight['checks'].values())
    tests=ROOT/'runs/pen_reward_chain_tests510_v1.log';checks['full510_tests']='Ran 510 tests' in tests.read_text() and '\nOK\n' in tests.read_text()
    receipts=[r['source_receipt']]
    for version in [1,2]:
        reg=ROOT/f'runs/pen_backend_reward_chain_registration_v{version}/registration.json'
        receipts.append(read(reg)['source_receipt'])
    receipts.append(file_pin(v1/'source_receipt.json'))
    for pin in receipts:verify_source_snapshot(read(verify_file_pin(pin)),live=True)
    if not all(checks.values()):raise ValueError('Reward archive discrepancy: '+repr([k for k,v in checks.items() if not v]))
    docs=[ROOT/'docs'/n for n in ['pen-reward-chain-results-2026-11-16.md','research-plan-v1.35.md']]
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+docs+[Path(__file__),ROOT/'tests/test_reward_chain_serialization.py'],
        out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt)
    summary=dict(checks=[sum(checks.values()),len(checks)],reviewed_rows=4860,row_checks=[68040,68040],
        real_row_formula_reconstructions_including_failed_attempt_and_preflight=9721,
        source_contexts=36,omission=omission,
        geometry_share={k:v['native_reward_responses']['geometry_absolute_share_of_tracking_geometry_support'] for k,v in r['cohorts'].items()},
        stable_reward_direction_counts={k:v['native_reward_responses']['physical_screen_direction_counts'] for k,v in r['cohorts'].items()},
        actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,WM_or_Actor_updates=0,
        physical_status='NATIVE_COMPONENTS_RECONSTRUCTED_FULL_WM_REWARD_NOT_ADMITTED',
        scope='Actual RAW quarter-gain and native components checked;path-dependent geometry missing from current WM90. Oracle partial-objective omission can be large on all valid validation pairs but small on stable-screen subset;neither subset qualifies full reward/Actor/ego/task gradients.')
    write(out/'independent_review.json',dict(checks=checks,passed=sum(checks.values()),total=len(checks),summary=summary,source_receipt=file_pin(out/'source_receipt.json')))
    # Reverify the immediately preceding complete evidence before adding the new
    # read-only branch. No historical roots are silently rebound or recomputed.
    parent=read(verify_file_pin(prior['verified_manifest']));records=parent['records'];verified={}
    for pin in records:verify_file_pin(pin);verified[json.dumps(pin,sort_keys=True)]=pin
    todo=[file_pin(p) for p in [v2/'report.json',v1/'failure.json',out/'independent_review.json',out/'source_receipt.json',preflight_log,tests,prior_path]+docs]+receipts
    todo += [file_pin(p) for p in [v1/'TRAIN_native_components.npz',v1/'VALIDATE_native_components.npz',v1/'source_contexts.pt']]
    todo += [file_pin(ROOT/f'runs/pen_backend_reward_chain_registration_v{version}/registration.json') for version in [1,2]]
    todo += [file_pin(ROOT/f'runs/pen_backend_reward_chain_v{version}.log') for version in [1,2]]
    allowed={v1,v2,out,ROOT/'runs/pen_backend_reward_chain_registration_v1',ROOT/'runs/pen_backend_reward_chain_registration_v2'}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        path=verify_file_pin(pin);verified[key]=pin
        if path.suffix=='.json' and any(path.is_relative_to(root) for root in allowed):todo.extend(pins(read(path)))
    write(out/'verified_pins.json',dict(records=list(verified.values())));archive.mkdir();write(archive/'summary.json',summary)
    write(archive/'ledger.json',dict(schema='native_pen_reward_chain_results_ledger:1',prior_evidence=file_pin(prior_path),
        summary=file_pin(archive/'summary.json'),review=file_pin(out/'independent_review.json'),native_audit=file_pin(v2/'report.json'),
        original_failure=file_pin(v1/'failure.json'),source_receipts=receipts+[file_pin(out/'source_receipt.json')],documents=[file_pin(p) for p in docs],
        verified_manifest=file_pin(out/'verified_pins.json'),verified_pins=len(verified),added_native_integrations=0,added_optimizer_steps=0,normalizer_fits=0))
    print(json.dumps(dict(ledger=file_pin(archive/'ledger.json'),checks=summary['checks'],pins=len(verified),row_checks=summary['row_checks'])),flush=True)


if __name__=='__main__':main()
