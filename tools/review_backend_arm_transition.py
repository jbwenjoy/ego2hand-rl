"""CPU-only complete saved-state audit of R2000 to fresh-common S1 transition."""
import json
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.learning.backend_pen_ppo import require_saved_backend
from supermimic_research.learning.pen_support_ppo import fork_initial_state
from supermimic_research.learning.pen_matched_recovery import saved_bytes_equal
from supermimic_research.runtime.explicit_collision_backend import load_contract


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    out=ROOT/'runs/pen_backend_second_arm_review_v1';run=ROOT/'runs/pen_backend_long_training_v1'
    if out.exists():raise ValueError('Fresh transition review required')
    # Check both declared milestones before creating a review directory. This
    # helper does not wait/restart/evaluate/select or drive native physics.
    health_paths={arm:run/arm/f'health_{iteration:04d}.json' for arm,iteration in [('RESIDUAL',2000),('SUPPORT',1)]}
    health={arm:read(p) for arm,p in health_paths.items()}
    reg_path=ROOT/'runs/pen_backend_long_registration_v1/registration.json';reg=read(reg_path)
    prep_path=ROOT/'runs/pen_backend_long_preparation_v1/report.json';prep=read(prep_path)
    contract=load_contract(ROOT/'configs/collision_backend_face_normal_v1.json')
    verify_source_snapshot(read(verify_file_pin(reg['source_receipt'])),live=True)
    inputs={'registration':file_pin(reg_path),'preparation':file_pin(prep_path)};states={};initials={};initialization={}
    common=torch.load(verify_file_pin(prep['common']),map_location='cpu',weights_only=False);require_saved_backend(common,contract)
    verify_source_snapshot(common['native']['sources'],live=True);inputs['common']=prep['common']
    for arm in ['RESIDUAL','SUPPORT']:
        inputs[arm+'_milestone_health']=file_pin(health_paths[arm]);inputs[arm+'_initial']=file_pin(run/arm/'initial_state.pt');inputs[arm+'_checkpoint']=health[arm]['checkpoint']
        inputs[arm+'_initialization']=file_pin(run/arm/'initialization.json');initialization[arm]=read(verify_file_pin(inputs[arm+'_initialization']))
        for target,pin in [(initials,inputs[arm+'_initial']),(states,inputs[arm+'_checkpoint'])]:
            state=torch.load(verify_file_pin(pin),map_location='cpu',weights_only=False);require_saved_backend(state,contract);verify_source_snapshot(state['native']['sources'],live=True)
            for engine_pin in state['native']['engine_sources']:verify_file_pin(engine_pin)
            target[arm]=state
    expected,patches=fork_initial_state(initials['RESIDUAL'],initials['SUPPORT']['metadata'])
    full_equal=saved_bytes_equal(expected,initials['SUPPORT'])
    root_differences=[key for key in expected if not saved_bytes_equal(expected[key],initials['SUPPORT'].get(key))]
    checks=dict(same_backend=all(s['metadata']['collision_backend']==contract for s in list(initials.values())+list(states.values())),
        same_registration=reg['backend_id']==contract['backend_id'] and prep['registration']==file_pin(reg_path),
        common_preparation=prep['passed']==prep['total']==5 and prep['actual_native_integrations']==2048 and prep['optimizer_steps']==0,
        whole_initial_fork_bytes=full_equal,
        only_three_declared_changes=[p['path'] for p in patches]==['metadata.arm','metadata.support_weight','native.python.scalars.support_weight'],
        recorded_fork_changes=initialization['RESIDUAL']['patches']==[] and initialization['SUPPORT']['patches']==patches,
        same_common_source=all(v['common']==prep['common'] for v in initialization.values()),
        initial_empty_Adam=all(not s['optimizer']['state'] for s in initials.values()),
        initial_zero_work=all((s['iteration'],s['optimizer_steps'],s['training_transitions'],s['calibration_transitions'])==(0,0,0,256) for s in initials.values()),
        final_R_budget=(states['RESIDUAL']['iteration'],states['RESIDUAL']['optimizer_steps'],states['RESIDUAL']['training_transitions'],states['RESIDUAL']['calibration_transitions'])==(2000,64000,4096000,256),
        first_S_budget=(states['SUPPORT']['iteration'],states['SUPPORT']['optimizer_steps'],states['SUPPORT']['training_transitions'],states['SUPPORT']['calibration_transitions'])==(1,32,2048,256),
        R_actual_calls=(health['RESIDUAL']['actual_native_integrations'],health['RESIDUAL']['optimizer_steps'])==(32768000,64000),
        S_actual_calls=(health['SUPPORT']['actual_native_integrations'],health['SUPPORT']['optimizer_steps'])==(16384,32),
        S_first_serialization=all(health['SUPPORT']['full_first_readback'].values()))
    first=initials['RESIDUAL']
    checks['common_all_root_state_bytes']=common.keys()==first.keys() and all(saved_bytes_equal(v,first[k]) for k,v in common.items() if k!='native')
    checks['common_all_native_state_bytes']=common['native'].keys()==first['native'].keys() and all(saved_bytes_equal(v,first['native'][k]) for k,v in common['native'].items() if k!='sources')
    checks['common_source_inputs_exact']=[r['observed'] for r in common['native']['sources']['records']]==[r['observed'] for r in first['native']['sources']['records']]
    for arm,state in states.items():
        checks[arm+'_nonempty_Adam']=bool(state['optimizer']['state'])
        checks[arm+'_all_Adam_steps']=all(int(v['step'])==state['optimizer_steps'] for v in state['optimizer']['state'].values())
        checks[arm+'_finite_modules']=all(bool(torch.isfinite(v).all()) for module in state['modules'].values() for v in module.values())
        checks[arm+'_finite_Adam']=all(bool(torch.isfinite(v).all()) for node in state['optimizer']['state'].values() for v in node.values() if isinstance(v,torch.Tensor))
        collector=read(verify_file_pin(health[arm]['collector']));checks[arm+'_milestone_health25']=collector['passed']==collector['total']==25 and all(collector['checks'].values())
        native=initialization[arm]['load']['native'];checks[arm+'_initial_strict_load']=native['all_array_bytes_equal'] and native['all_scalar_values_equal'] and native['pointers_preserved']
    out.mkdir();receipt=freeze_sources([Path(__file__)],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt)
    report=dict(schema='new_backend_second_arm_saved_review:1',observed_at_UTC=datetime.now(timezone.utc).isoformat(),inputs=inputs,checks=checks,passed=sum(checks.values()),total=len(checks),
        full_initial_root_differences=root_differences,declared_patches=patches,source_receipt=file_pin(out/'source_receipt.json'),
        common_comparison_provenance_scope='All common/R-initial state bytes compared. Native sources have separate preparation/training snapshot destinations;both snapshots and exact observed full-path/bytes/SHA lists are verified. No state/source metadata is rewritten or restored by this CPU review.',
        training_prefix_native=32784384,training_prefix_Adam=64032,shared_preparation_native=2048,cohort_prefix_native=32786432,
        added_actual_native_integrations=0,added_optimizer_steps=0,
        scope='Complete saved-state/fresh-common equality and R2000/S1 milestone counters only;not future liveness,deterministic W32 paths,complete matched trial,strict task/WM/ego/performance qualification.')
    write(out/'report.json',report)
    if not all(checks.values()):raise ValueError('Retain transition mismatch: '+repr([k for k,v in checks.items() if not v])+' roots='+repr(root_differences))
    print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[report['passed'],report['total']],initial_fork_bytes=full_equal)),flush=True)


if __name__=='__main__':main()
