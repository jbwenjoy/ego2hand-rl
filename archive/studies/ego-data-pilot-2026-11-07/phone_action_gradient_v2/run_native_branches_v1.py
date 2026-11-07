"""Frozen full20-action finite branches; baseline restore checks precede changes."""
from pathlib import Path
import argparse,json,time
import numpy as np
from native_action_probe import NativeSavedActionProbe,baseline_checks
from native_action_probe import reader

H=Path(__file__).resolve().parent;P=H.parent
def pin(p):return reader.pin(p)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def checked(r):assert pin(r['path'])==r;return Path(r['path'])
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.bool_):return bool(x)
    return x
def write(p,j):
    with Path(p).open('x',encoding='utf8') as f:json.dump(clean(j),f,indent=2,allow_nan=False)
def save(p,**arrays):
    with Path(p).open('xb') as f:np.savez_compressed(f,**arrays)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--run',action='store_true');a=ap.parse_args()
    cfg=read(a.protocol);assert cfg['schema']=='phone_full20_native_action_branch_protocol:1'
    for r in [*cfg['inputs'].values(),*cfg['source_pins'].values()]:checked(r)
    assert cfg['source_pins']['runner']==pin(__file__)
    selection=read(checked(cfg['inputs']['selection']));proof=read(checked(cfg['inputs']['restore_contract']))
    assert proof['status']==cfg['restore_contract_status'] and all(v is True for v in proof['checks'].values())
    assert proof['counts']['integration_steps']==384
    with np.load(checked(cfg['inputs']['operands']),allow_pickle=False) as z:operands={k:z[k].copy() for k in z.files}
    acquired=read(checked(cfg['inputs']['acquisition_report']))
    assert len(selection['states'])==12 and selection['available_sides']==476
    out=Path(cfg['output']);assert not out.exists()
    if not a.run:print(json.dumps(dict(status='METADATA_ONLY_NO_NATIVE',states=12,branches_max=488,steps_max=46848)));return
    assert cfg['CPU_run_authorized'] is True and cfg['training_or_parameter_update_authorized'] is False
    out.mkdir();write(out/'protocol.json',cfg);started=time.perf_counter();probes={};records=[];phase='models'
    try:
        for h in [.0003125,.00015625]:
            expected=acquired['model_audits'][str(h)]['model_digest']
            probes[h]=NativeSavedActionProbe(checked(cfg['inputs']['model_mjb']),dt_s=h,expected_effective_model_digest=expected)
        for state in selection['states']:
            sid=state['state_id'];probe=probes[state['native_dt_s']];phase=f'state{sid}_baseline'
            with np.load(checked(state['source_trace']),allow_pickle=False) as z:original={k:z[k].copy() for k in z.files}
            baseline=probe.run(state,operands,state['baseline_ctrl20_F64'])
            checks=baseline_checks(baseline,state,operands,trace=original)
            write(out/f'state{sid:02d}_baseline_checks.json',dict(state_id=sid,checks=checks,passed=all(checks.values())))
            if not all(checks.values()):raise ValueError('Baseline restore failed; no new perturbations for this context')
            proposals=[dict(kind='baseline',control=state['baseline_ctrl20_F64'],value=baseline)]
            proposals.extend(dict(kind='finite_action_side',control=p['actual_ctrl20_F64'],plan=p) for p in state['perturbations'] if p['available'])
            for prop in proposals:
                phase=f'state{sid}_{prop["kind"]}_{len(records)}'
                value=prop.get('value')
                if value is None:value=probe.run(state,operands,prop['control'])
                idx=len(records);path=out/f'branch_{idx:03d}.npz'
                arrays=dict(value['trace'])|dict(source_state58=value['source_state58'],actual_control20=value['actual_control20'],
                    observed_prefix_target58=value['target_state58'],physical_world_delta52=value['physical_world_delta52'],
                    observable59=value['observable59'],source_integration_state=value['source_receipt']['integration_state'],
                    source_after_action_integration_state=value['source_after_action_receipt']['integration_state'],
                    target_integration_state=value['target_receipt']['integration_state'])
                save(path,**arrays)
                compact=lambda r:{k:v for k,v in r.items() if k!='integration_state'}
                record=dict(branch_id=idx,state_id=sid,family_id=state['family_id'],episode_id=state['episode_id'],
                    source_sample_id=state['source_sample_id'],split=state['split'],dt_s=state['native_dt_s'],
                    interval_index=state['interval_index'],source_branch=state['source_branch'],kind=prop['kind'],
                    perturbation=prop.get('plan'),output=pin(path),complete_H20ms=value['complete_H20ms'],
                    finite=value['finite'],usable_observable59=value['training_eligible'],termination=value['termination'],
                    returned_substeps=value['returned_substeps'],source_receipt=compact(value['source_receipt']),
                    source_after_action_receipt=compact(value['source_after_action_receipt']),target_receipt=compact(value['target_receipt']),
                    summary=value['summary'],warning_numbers=value['warning_numbers'],
                    hypothetical_branch_not_held_robot_or_training_sample=True)
                records.append(record)
            print(json.dumps(dict(state=sid,returned_branches=len(proposals),baseline_all_bits=True)),flush=True)
        audits={str(h):p.audit() for h,p in probes.items()}
        counts={k:sum(p.reader.counts[k] for p in probes.values()) for k in next(iter(probes.values())).reader.counts}
        assert len(records)==488 and counts['integration_steps']<=46848 and counts['initial_forward']==counts['observer_forward']==0
        write(out/'branches.json',records)
        result=dict(schema='phone_full20_native_action_branch_result:1',status='COMPLETE_BASELINE_MATCHED_FULL20_NATIVE_FINITE_ACTIONS',
            protocol=pin(a.protocol),counts=counts,audits=audits,states=12,baseline_branches=12,new_action_branches=476,
            unavailable_sides=4,records=pin(out/'branches.json'),usable_observables=sum(v['usable_observable59'] for v in records),
            partial_branches=sum(not v['complete_H20ms'] for v in records),elapsed_s=time.perf_counter()-started,
            new_learning_training_or_policy_or_physical_analytic_Jacobian_qualified=False,
            scope='Local hypothetical simulated one-sided finite-action branches of validation families; no state reset in an actual holding rollout and no parameter updates.')
        write(out/'report.json',result);print(json.dumps(dict(status=result['status'],counts=counts,partial=result['partial_branches'])),flush=True)
    except BaseException as err:
        write(out/'failure.json',dict(phase=phase,error=repr(err),returned_records=records,
            counters={str(h):p.reader.counts for h,p in probes.items()},automatic_retry=False));raise

if __name__=='__main__':main()
