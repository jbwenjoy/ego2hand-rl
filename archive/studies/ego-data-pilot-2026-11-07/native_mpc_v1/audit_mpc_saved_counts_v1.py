"""Saved-only counts/provenance review. No model/backend or physics imports."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'feasibility_fine_v1'
OUT=ROOT/'saved_count_review_v1'


def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def checked(rec):
    p=Path(rec['path']);assert pin(p)==rec,rec['path'];return p


def bits(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def main():
    assert not OUT.exists();OUT.mkdir()
    checks={};r=read(RUN/'report.json');p=read(checked(r['protocol']))
    assert read(RUN/'protocol.json')==p
    for rec in p['inputs'].values():checked(rec)
    fixture=read(checked(p['inputs']['branch_fixture_report']));fp=read(checked(fixture['protocol']))
    for rec in [*fp['inputs'].values(),*fp['source_pins'].values()]:checked(rec)
    binding=read(checked(p['inputs']['model_binding']));checked(binding['model']);checked(binding['source'])
    terminal=read(ROOT/'pilot_process_terminal_v1.json')
    for name in ('source','protocol','stdout','stderr'):checked(terminal[name])
    checks['all_direct_protocol_fixture_candidate_model_source_hashes_current']=True
    checks['terminal_exit0_source_protocol_stdout_stderr_join']=(terminal['exit_code']==0 and terminal['pid']==36344
        and terminal['source']==p['inputs']['controller_source'] and terminal['protocol']==r['protocol']
        and terminal['stderr']['bytes']==0 and terminal['automatic_retry'] is False)
    checks['fixture_branch_seed_model_and_initial_digest_join']=(fp['source_pins']['evaluator']==p['inputs']['branch_source']
        and fp['inputs']['initial_state']==p['inputs']['candidate_state']
        and fp['inputs']['model_binding']==p['inputs']['model_binding']
        and binding['model']==p['inputs']['model_mjb']
        and fixture['passed']==fixture['total']==42 and all(v is True for v in fixture['checks'].values())
        and r['initial_boundary_digest']==fixture['source_state_before']==fixture['source_state_after'])
    controller_path=checked(p['inputs']['controller_source']);tree=ast.parse(controller_path.read_text(encoding='utf8'))
    before=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='write'
        and n.args and 'selection_before_actual_' in ast.unparse(n.args[0])]
    loops=[n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=='local' and ast.unparse(n.iter)=='range(128)']
    checks['selection_file_closed_before_actual_loop_by_frozen_source']=(len(before)==1 and len(loops)==1 and before[0]<loops[0].lineno)
    checked(r['trace'])
    with np.load(r['trace']['path'],allow_pickle=False) as z:t={k:z[k].copy() for k in z.files}
    with np.load(p['inputs']['candidate_state']['path'],allow_pickle=False) as z:seed={k:z[k].copy() for k in ('qpos27','qvel26','ctrl20')}
    checks['trace_initial_arrays_exact_registered_candidate']=all(bits(t['initial_'+k],seed[k]) for k in ('qpos27','qvel26','ctrl20'))
    intervals=[read(RUN/f'interval_{i:03d}.json') for i in range(7)]
    selections=[read(RUN/f'selection_before_actual_{i:03d}.json') for i in range(7)]
    checks['exact_seven_before_and_after_records']=(len(list(RUN.glob('interval_*.json')))==7 and len(list(RUN.glob('selection_before_actual_*.json')))==7)
    branch_ids=[];comparison=[];saved_pins={};all_bind=True;all_mtime=True;all_source=True;all_controls=True
    for i,(item,selection) in enumerate(zip(intervals,selections)):
        assert item['interval']==selection['interval']==i and item['start_tick']==selection['start_tick']==i*128
        all_bind &= all(item[k]==v for k,v in selection.items())
        all_mtime &= (RUN/f'selection_before_actual_{i:03d}.json').stat().st_mtime_ns < (RUN/f'interval_{i:03d}.json').stat().st_mtime_ns
        ev=item['evaluations'];assert len(ev)==23
        for e in ev:
            branch_ids.append(e['branch_index'])
            assert e['status']=='COMPLETE_H2_PLANNING_ONLY' and e['actual_planning_steps']==256 and e['complete_clone_equal'] is True
            all_source &= e['source_state_before']==e['source_state_after']==item['source_digest_before_actual']
        if i>0:all_source &= item['source_digest_before_actual']['integration_sha256']==intervals[i-1]['actual_boundary_integration_sha256']
        chosen=next(e for e in ev if e['branch_index']==item['selected_branch'])
        assert chosen['original_horizon_safety_limits_pass'] is True
        u=np.asarray(item['selected_control20'],dtype=np.float64)
        all_controls &= bits(u,np.asarray(chosen['controls2x20'][0],dtype=np.float64)) and bits(u,np.asarray(chosen['controls2x20'][1],dtype=np.float64))
        stop=min((i+1)*128,len(t['post_tick']))
        all_controls &= bits(t['chosen_ctrl20'][i*128:stop],np.tile(u,(stop-i*128,1)))
        all_controls &= bits(u[8:],seed['ctrl20'][8:])
        snap=item['selected_first_snapshot'];state=np.asarray(snap['integration_state'],dtype=np.float64)
        sha=hashlib.sha256(state.tobytes()).hexdigest()
        assert state.shape==(300,) and sha==snap['integration_state_sha256'] and snap['integration_state_signature']==16383
        assert snap['post_tick']==128 and snap['pad_force_phase']=='PRE of final solver tick, not POST geometry force'
        detail={'interval':i,'start_tick':i*128,'planned_branches':23,'planned_steps':23*256,
            'actual_steps':stop-i*128,'selected_branch':item['selected_branch'],'snapshot_integration_sha256':sha}
        if i<6:
            idx=(i+1)*128-1
            assert item['actual_vs_selected_first_state_raw_bit_equal'] is True and item['actual_boundary_integration_sha256']==sha
            obs=snap['state']
            direct=(bits(np.asarray(obs['q20']),t['post_qpos'][idx,:20])
                and bits(np.asarray(obs['qd20']),t['post_qvel'][idx,:20])
                and bits(np.asarray(obs['phone_root_world_m']),t['post_qpos'][idx,20:23])
                and bits(np.asarray(obs['phone_quaternion_wxyz']),t['post_qpos'][idx,23:27])
                and bits(np.asarray(obs['v_root_world_m_s']),t['post_qvel'][idx,20:23]))
            assert direct
            detail.update(complete=True,producer_raw300_equality_claim=True,saved_actual_SHA_matches_predicted_bytes=True,independent_direct_POST_fields_bit_equal=direct)
        else:
            assert item['actual_vs_selected_first_state_raw_bit_equal']=='NOT_FULL_INTERVAL' and 'actual_boundary_integration_sha256' not in item
            detail.update(complete=False,producer_raw300_equality_claim='NOT_FULL_INTERVAL',full_comparison_not_claimed=True)
        comparison.append(detail)
        for name in (f'interval_{i:03d}.json',f'selection_before_actual_{i:03d}.json'):saved_pins[name]=pin(RUN/name)
    checks['all_seven_selection_contents_and_file_order_preserved']=bool(all_bind and all_mtime)
    checks['all161_branches_unique_contiguous_and_unchanged_source']=bool(branch_ids==list(range(1,162)) and all_source)
    checks['all_executed_controls_match_selected_prediction_and_other12_fixed']=bool(all_controls)
    pc=r['planning_counts']
    checks['planning_counts161_times256']=all(pc[k]==41216 for k in ('planning_steps_attempted','planning_steps_returned','planning_steps_finite')) and all(pc[k]==161 for k in ('branches_attempted','branches_returned','branch_MjData','mj_copyData')) and pc['extra_forward_calls']==0 and pc['actual_chosen_steps']==0
    checks['actual822_equals_six128_plus54']=(len(t['post_tick'])==r['actual_integration_steps']==r['actual_finite_rows']==6*128+54
        and np.array_equal(t['pre_tick'],np.arange(822)) and np.array_equal(t['post_tick'],np.arange(1,823))
        and t['contact_rows24'].shape==(r['actual_contact_force_reads'],24) and r['actual_contact_force_reads']==419)
    oldsrc=PILOT/'hand2_holding_v1/run_cpu_holding_v1.py'
    ot=ast.parse(oldsrc.read_text(encoding='utf8'));gate=next(n for n in ot.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='GATES' for x in n.targets))
    original=ast.literal_eval(gate.value);checks['all_original_GATES_unchanged']=p['gates']==original
    tail=t['post_tick']>=9600;pretail=t['pre_tick']>=9600
    metrics={'final_COM_drift_m':float(np.linalg.norm(t['post_COM'][-1]-t['initial_COM_world3'])),
        'final_angle_rad':float(t['post_angle'][-1]),'tail_linear_velocity_RMS_m_s':None,
        'tail_angular_velocity_RMS_rad_s':None,'tail_dual_pad_force_coverage':None,
        'max_native_penetration_m':float(t['pre_penetration_m'].max()),'max_self_penetration_m':float(t['pre_self_penetration_m'].max()),
        'max_phone_nonpad_normal_force_N':float(t['pre_phone_nonpad_normal_N'].max()),'max_table_normal_force_N':float(t['pre_table_normal_N'].max()),
        'max_joint_limit_excursion_rad':r['metrics']['max_joint_limit_excursion_rad'],'warning_count':r['metrics']['warning_count']}
    assert not tail.any() and not pretail.any()
    gates={'duration_complete':False}
    for k,v in metrics.items():gates[k]=v is not None and (v>=original[k] if k=='tail_dual_pad_force_coverage' else v<=original[k])
    checks['saved_gate_logic_and_all_overall_qualifications_false']=(metrics==r['metrics'] and gates==r['gates']
        and not all(gates.values()) and r['original_2s_single_dt_holding_pass'] is False
        and r['two_dt_qualified'] is False and r['training_or_policy_benefit_qualified'] is False
        and r['mass_friction_calibrated'] is False and r['termination']=='ABORT_LOSS_OF_HOLD')
    assert all(checks.values()),{k:v for k,v in checks.items() if not v}
    result={'schema':'native_MPC_saved_counts_provenance_review:1','status':'PASS_SAVED_COUNTS_AND_PROVENANCE_NOT_HOLDING',
        'producer_report':pin(RUN/'report.json'),'protocol':r['protocol'],'source_reviewer':pin(__file__),
        'checks':checks,'passed':sum(checks.values()),'total':len(checks),
        'counts':{'planning_branches':161,'branches_each_interval':[23]*7,'planning_steps':41216,
            'actual_steps':822,'complete_actual_intervals':6,'partial_actual_interval_steps':54,
            'actual_contact_rows_and_reads':419,'initial_forward_calls':1},
        'complete_and_partial_comparisons':comparison,'original_failed_gates':[k for k,v in gates.items() if not v],
        'source_selection_order_lines':{'write_before_actual':before[0],'actual128step_loop':loops[0].lineno},
        'scope_limits':[
            'Six complete300-element actual integration vectors are not separately saved. Independently rehashed predicted vectors match saved actual SHAs; full actual raw-equality is producer claim backed by pinned executed source. Direct saved q/qdot/root/quaternion fields were independently byte-compared.',
            'Selection file closure before stepping is established by pinned source control flow plus matching saved pre/post records and earlier filesystem mtimes, not independent OS syscall tracing.',
            'Planning contactForce13376 and native warning/joint-limit metric are producer/source evidence; full planning contact traces/model joint bounds/warning arrays are not independently reconstructed here.',
            'Model MJB bytes hashed only; no MJB deserialization, backend import, physics, optimization or rollout performed.'],
        'pins':{**saved_pins,'trace':pin(RUN/'trace.npz'),'terminal':pin(ROOT/'pilot_process_terminal_v1.json'),'old_gate_source':pin(oldsrc)},
        'reviewer_saved_NPZ_reads':2,'reviewer_model_backend_physics_calls':0,
        'original_2s_holding_qualified':False,'two_dt_or_policy_qualified':False}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'report':pin(OUT/'report.json'),'passed':result['passed'],'total':result['total'],'counts':result['counts']}))


if __name__=='__main__':
    PILOT=ROOT.parent
    main()
