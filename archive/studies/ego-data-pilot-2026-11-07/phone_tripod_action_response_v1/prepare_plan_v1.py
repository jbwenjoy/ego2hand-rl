"""Freeze mechanical8MON contexts and320 side plans; saved JSON/NumPy only."""
from pathlib import Path
import argparse,importlib.util,json
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent
s=importlib.util.spec_from_file_location('tripod_response_common',H/'common_v1.py');C=importlib.util.module_from_spec(s);s.loader.exec_module(C)

def prepare():
    report_pin=C.pin(P/'phone_tripod_task_acquisition_v1/acquired_v1/report.json');r=C.read(report_pin['path'])
    C.need(r['schema']=='phone_tripod_task_acquisition_result:1' and r['status']=='COMPLETE_NEW_TRIPOD_TASK_CONTROLLED_COLLECTION' and r['partition_qualified'] is True,'Frozen source acquisition')
    ids=C.read(C.checked(r['sample_identities']));eps=C.read(C.checked(r['episode_records']));task=C.read(C.checked(r['task_bundle']))
    byid={v['sample_id']:v for v in ids};lookup={(v['family_id'],float(v['dt_s']),v['step_k']):v for v in ids};b=np.array(task['actuator_ctrlrange20x2'],dtype=np.float64)
    D=C.mod('tripod_response_label_math',P/'phone_tripod_replay_dataset_v1/dataset_adapter_v1.py');maths=D.load_math()
    physical_pin=C.pin(P/'phone_object_wrench_labels_v1/completed_v1/physical_parameters.json');physical=C.read(physical_pin['path'])
    states=[];buffers={};baseline_labels=[]
    for fid in [12,15]:
        for h in [.0003125,.00015625]:
            for k in [0,3]:
                ident=lookup[(fid,h,k)];sid=len(states);ep=next(e for e in eps if e['episode_id']==ident['episode_id'])
                C.need(ident['split']=='TRIPOD_MONITOR' and ident['complete_H20ms'] and ident['valid'] and ident['physical_training_eligible'],'Selected fixedMON context complete/current known')
                C.need(all(not byid[i]['done'] for i in ep['interval_row_ids'][:k]),'Context before first boundary failure')
                a=C.npz(ident['trace']);initial=C.npz(ep['initial']['arrays']);initial_task=C.npz(ep['initial_task'])
                if k==0:q=initial['qpos27'];v=initial['qvel26'];previous_u=initial['ctrl20'];previous_receipt=initial['integration_state'];previous_pin=ep['initial']['arrays']
                else:
                    prev_ident=byid[ep['interval_row_ids'][k-1]];prev=C.npz(prev_ident['trace'])
                    q=prev['post_qpos27'][-1];v=prev['post_qvel26'][-1];previous_u=prev['actual_control20'];previous_receipt=prev['target_integration_state'];previous_pin=prev_ident['trace']
                C.need(C.bits(a['source_integration_state'],previous_receipt),'No guessed300 slice: source is actual reset or previous target')
                C.need(C.bits(initial['state58'],initial_task['state58_F64']) and not np.any(initial['qvel26']),'Original episode safety origin/zero-velocity initial')
                C.need(C.bits(a['source_state58'],maths['codec'].actual_state(q,v,np.array(physical['body_ipos_body_m']))),'Source worldCOM state from actual q/v')
                u=a['actual_control20'];before=a['source_integration_state'];after=a['source_after_action_integration_state']
                offsets=[j for j in range(281) if C.bits(before[j:j+20],previous_u) and C.bits(after[j:j+20],u) and C.bits(before[:j],after[:j]) and C.bits(before[j+20:],after[j+20:])]
                C.need(len(offsets)==1,'Measured unique original control-only300 join; no guessed layout')
                for name,value in dict(source_integration_F64=before,source_qpos27_F64=q,source_qvel26_F64=v,source_state58_F64=a['source_state58'],
                    initial_qpos27_F64=initial['qpos27'],initial_state58_F64=initial_task['state58_F64'],source_previous_u20_F64=previous_u,
                    original_target_integration_F64=a['target_integration_state'],original_target_qpos27_F64=a['post_qpos27'][-1],original_target_qvel26_F64=a['post_qvel26'][-1],
                    baseline_ctrl20_F64=u,control_bounds20x2_F64=b).items():buffers.setdefault(name,[]).append(value)
                labels=C.label_interval(a,a['source_state58'],a['post_state58'][-1],h=h,n=ident['requested_substeps'],complete=True,finite=True,warning_numbers=ident['warning_numbers'],D=D,maths=maths,physical=physical)
                baseline_labels.append(labels)
                sides=[]
                for axis in range(20):
                    for si,sign in enumerate([-1,1]):
                        z=C.planned_side(u,axis,sign,b);z.update(branch_id=8+sid*40+axis*2+si,baseline_branch_id=sid)
                        scale=np.array(task['action']['symmetric_residual_scale20_rad']);needed=(np.array(z['actual_ctrl20_F64'])-np.array(task['initial']['ctrl20_F64'])-.5*a['memory_before83_F64'][41:61])/(.5*scale)
                        z.update(Actor_EMA_reachable_descriptor=bool(np.all(abs(needed)<=1)),required_normalized_action20_F64=needed.tolist(),intervention_kind='absolute_control_hypothetical_not_policy_action')
                        sides.append(z)
                states.append(dict(context_id=sid,state_id=sid,family_id=fid,episode_id=ident['episode_id'],source_sample_id=ident['sample_id'],interval_index=k,step_k=k,
                    split='TRIPOD_MONITOR',data_role=C.ROLE,native_dt_s=h,n_substeps=ident['requested_substeps'],source_trace=ident['trace'],source_identity=ident,
                    episode_initial=ep['initial'],initial_task=ep['initial_task'],qpos_source=previous_pin,source_before_action_receipt=ident['source_receipt'],original_source_after_action_receipt=ident['source_after_action_receipt'],
                    original_target_receipt=ident['target_receipt'],baseline_ctrl20_F64=u.tolist(),baseline_ctrl20_raw_sha256=C.sha(u),source_control_segment_offset=offsets[0],
                    expected_effective_model_digest=r['model_audits'][str(h)]['model_digest'],baseline_branch_id=sid,perturbations=sides))
    operands={k:np.stack(v) for k,v in buffers.items()}
    for name in baseline_labels[0]:operands['baseline_'+name]=np.stack([v[name] for v in baseline_labels])
    C.save(H/'operands_v1.npz',operands)
    available=sum(z['available'] for s in states for z in s['perturbations'])
    plan=dict(schema='phone_tripod_heldout_action_response_plan:1',rules=C.RULES,data_role=C.ROLE,acquisition_report=report_pin,
        source_identities=r['sample_identities'],source_episodes=r['episode_records'],task_bundle=r['task_bundle'],actuator_bounds_source=dict(task_bundle=r['task_bundle'],key='actuator_ctrlrange20x2'),
        states=states,planned_branches=328,available_sides=available,unavailable_sides=320-available,maximum_executed_branches=8+available,
        planned_maximum_native_steps=31488,available_maximum_native_steps=sum((1+sum(z['available'] for z in s['perturbations']))*s['n_substeps'] for s in states),
        selection='Mechanical family12/15 x dt order x k0/3, before new physics/NN response; no reselection/rescue',source_scope='Original tripod MON only; permanently excluded from any fit or monitor checkpoint selection; known initial lineage, not unseen ego')
    C.write(H/'selection_v1.json',plan)
    labelprotocol=C.read(P/'phone_tripod_replay_dataset_v1/protocol_CPUFalse_v1.json')
    inputs=dict(selection=C.pin(H/'selection_v1.json'),operands=C.pin(H/'operands_v1.npz'),acquisition_report=report_pin,source_identities=r['sample_identities'],source_episodes=r['episode_records'],task_bundle=r['task_bundle'],model_mjb=task['model'],physical_parameters=physical_pin,
        source_integrity_review=C.pin(P/'phone_tripod_task_acquisition_v1/saved_review_v1/report.json'),source_label_review=C.pin(P/'phone_tripod_replay_dataset_v1/saved_review_v1/report.json'))
    sources=dict(runner=C.pin(H/'run_native_v1.py'),common=C.pin(H/'common_v1.py'),preparation=C.pin(__file__),probe=C.pin(P/'phone_action_gradient_v2/native_action_probe.py'),native_reader=task['sources']['native_reader'],label_adapter=C.pin(P/'phone_tripod_replay_dataset_v1/dataset_adapter_v1.py'),
        frozen_decode=labelprotocol['sources']['frozen_decode_window'],scalar_codec=labelprotocol['sources']['scalar_codec'],structured_encoder=labelprotocol['sources']['pure_structured_encoder_source'])
    protocol=dict(schema='phone_tripod_heldout_action_response_CPU_protocol:1',CPU_run_authorized=False,NN_training_GPU_authorized=False,rules=C.RULES,inputs=inputs,sources=sources,
        output=str(H/'native_completed_v1'),permanent_data_role=C.ROLE,baseline_barrier='All8 baselines complete/trace/300/labels bits pass before any finite-action side',automatic_retry=False,
        scope='Finite actions are hypothetical absolute-control native interventions. True actuator-range legality need not imply current Actor residual+EMA reachability. Reachability is descriptive only, not a clipping/filter gate. No mj_forward restoration, safety recentering, fullMjData/Markov/trueJacobian/holding certificate.')
    C.write(H/'protocol_CPUFalse_v1.json',protocol)
    return dict(protocol=C.pin(H/'protocol_CPUFalse_v1.json'),selection=C.pin(H/'selection_v1.json'),available_sides=available,maximum_native_steps=31488)

if __name__=='__main__':print(json.dumps(prepare()))
