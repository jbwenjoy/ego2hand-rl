"""Independent saved-query,selection,proposal and complete-state chain review."""
import json
from pathlib import Path
import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation
from ..artifacts import verify_file_pin
from ..runtime.shared_snapshot_transport import load_shared
from ..learning.pen_live_decision import from_owned_snapshot
from ..learning.pen_native_screening import proposal,random_direction
from ..learning.pen_path_response import PathResponseWM
from ..learning.pen_path_decomposition import gradients
from ..learning.backend_reward_gradient import batch_context
from ..learning.world_model import raw_tree_equal
from .probe_row_review import review_recorded_row
from .pen_reward_chain import native_components


def read(pin):return json.loads(verify_file_pin(pin).read_text())


def recompute_error(cpu,gpu,bound):
    finite=bool(np.isfinite(cpu).all() and np.isfinite(gpu).all())
    if not finite:return dict(finite=False,max_abs=None,L2=None,relative_L2=None,scaled_L2=None,within_predeclared_bound=False)
    c,g=np.asarray(cpu,np.float64),np.asarray(gpu,np.float64);d=float(np.linalg.norm(c-g));n=float(np.linalg.norm(g));scaled=d/(1+n)
    return dict(finite=True,max_abs=float(abs(c-g).max()),L2=d,relative_L2=d/n if n else None,scaled_L2=scaled,within_predeclared_bound=scaled<=bound)


class ScreenedCaseReviewer:
    """CPU-only;all native/contact packets reopened,no native step or fit."""
    def __init__(self,checkpoint,reward,objective):
        self.reward=reward;self.objective=objective;self.models={}
        for name,saved in checkpoint['arms'].items():
            ctx=saved['context'];m=PathResponseWM(ctx['scales'],ctx['physics'],ctx['path_units'],width=512);m.load_state_dict(saved['model']);m.eval();m.requires_grad_(False);self.models[name]=m

    def packet(self,case,source,model,physical):
        z=dict(np.load(verify_file_pin(case['trace'])));w=dict(np.load(verify_file_pin(case['window'])));end=load_shared(case['end']);q=source['native']['data']['arrays'];qa=case['model_layout']['object_qposadr'];body=case['model_layout']['object_body']
        R=Rotation.from_quat(q['qpos'][0,qa+np.array([4,5,6,3])]).as_matrix();view=dict(native=source,phase=case['phase'],source_qpos_R=R,source_cached_R=q['xmat'][0,body]);decision=from_owned_snapshot(source,case['phase'],case,z['actions20'][0])
        packet,audit=review_recorded_row(case,z,w,view,model,physical,dict(role=case['role'],object_penetration_m=.002,self_penetration_m=.0002,joint_excursion_rad=.001));values,formula,errors=native_components(case,z,w,decision['controller_context'],self.reward,self.objective);e=end['native']['data']['arrays']
        checks=dict(original_audit=audit==case['physical_audit'],original_reward_flags=formula==case['reward_checks'],original_reward_errors=errors==case['reward_errors'],all_reward_flags=all(formula.values()),source_reward=abs(float(values[-1])-case['source_reward'])<=2e-6,
            input102_bytes=np.array_equal(packet['x'].astype(np.float32),decision['input102']),end_qpos=np.array_equal(z['post_qpos'][-1],e['qpos'][0]),end_qvel=np.array_equal(z['post_qvel'][-1],e['qvel'][0]),end_warm=np.array_equal(z['post_qacc_warmstart'][-1],e['qacc_warmstart'][0]),
            original_sampled_RAW=np.array_equal(z['actions20'][0],np.asarray(case['sampled_RAW20'],np.float32)),original_applied_RAW=np.array_equal(decision['applied_RAW20'],np.asarray(case['applied_RAW20'],np.float32)),original_clip_count=decision['clipped_components']==case['RAW_clip_count'],eight_ticks=case['native_integrations']==len(z['post_qpos'])==8)
        return dict(name=case['name'],checks=checks,physical=audit,reward=formula,reward_errors=errors,terminated=case['terminated'],truncated=case['truncated'],reward_value=case['source_reward']),end,decision,z['actions20'][0]

    def case(self,result,folder):
        nodes=[];packets=[];previous=None;expected_restores=[];query_count=0;phase_checks=[];source_chains=[];proposal_rows=[];cpu_bounds=dict(values_scaled_L2=1e-5,RAW_scaled_L2=1e-4)
        first=read(result['nodes'][0]['BASE']);model=mujoco.MjModel.from_binary_path(str(verify_file_pin(first['model'])));physical=dict(np.load(verify_file_pin(first['physical_context'])))
        for index,node in enumerate(result['nodes']):
            phase=node['phase'];phase_checks.append(phase==index+1);prefix=result['name']+'_P'+str(phase).zfill(3);base=read(node['BASE']);source=load_shared(base['source'])
            if previous is not None:source_chains.append(dict(phase=phase,**{k:raw_tree_equal(previous[k],source[k]) for k in ['native','python','rng']}))
            br,bend,decision,raw=self.packet(base,source,model,physical);packets.append(br);query_count+=1;alt=None;ar=None;aend=None;prop=None
            if result['controller']!='UNMODIFIED_ACTOR':
                nf=folder/('P'+str(phase).zfill(3));saved_prop=json.loads((nf/'proposal.json').read_text());p=dict(np.load(nf/'proposal.npz'));direction=np.zeros(20);finite=True;comparisons=None
                if result['controller']=='WM':
                    ga=dict(np.load(nf/'gradient.npz'));mi=0 if result['score_mode']=='OBSERVABLE' else 2;goal=0 if result['teacher']=='RESIDUAL' else 1;direction=ga['RAW'][0,mi,goal];finite=bool(np.isfinite(ga['values']).all())
                    cg=gradients(self.models[result['model']],torch.tensor(decision['input102'][None]),torch.tensor(raw[None]),batch_context({'live':decision['controller_context']},['live'],'cpu'),self.reward,self.objective)
                    # Predeclared broad cross-device math checks,not physics
                    # tolerances,native-gradient truth or byte device parity.
                    comparisons={k:recompute_error(cg[k].numpy(),ga[k],cpu_bounds['values_scaled_L2' if k=='values' else 'RAW_scaled_L2']) for k in ['values','RAW']}
                    gradient_PD=np.array_equal(ga['computed_PD'][0],decision['actual_PD20'])
                elif result['controller']=='NATIVE_RANDOM':direction=random_direction(9901,result['repetition'],result['teacher_index'],phase)
                prop=proposal(raw,direction,prediction_finite=finite);prop_checks=dict(direction=np.array_equal(direction,p['direction'],equal_nan=True),submitted_RAW=np.array_equal(prop['raw'],p['raw']),delta=np.array_equal(prop['actual_raw_delta'],p['actual_raw_delta']),
                    fallback=all(prop[k]==saved_prop[k] for k in ['duplicate_BASE','reason','direction_norm']))
                if result['controller']=='WM':prop_checks.update(gradient_actual_PD=gradient_PD,CPU_recompute=all(v['within_predeclared_bound'] for v in comparisons.values()) if finite and np.isfinite(direction).all() else prop['duplicate_BASE'])
                proposal_rows.append(dict(phase=phase,checks=prop_checks,fallback=prop['reason'],CPU_device_comparisons=comparisons))
            if node['ALT'] is not None:
                alt=read(node['ALT']);ar,aend,_,submitted=self.packet(alt,source,model,physical);packets.append(ar);query_count+=1;expected_restores.append(prefix+'_source')
                same_source=alt['source']==base['source'];proposal_matches=prop is not None and np.array_equal(submitted,prop['raw'])
            else:same_source=proposal_matches=True
            if not br['physical']['valid']:
                selected='BASE';accepted=False;stop=True;reason='BASE_PHYSICAL_INVALID';ALT_missing=node['ALT'] is None
            elif result['controller']=='UNMODIFIED_ACTOR':
                selected='BASE';accepted=False;stop=base['terminated'] or base['truncated'];reason='UNMODIFIED_ACTOR';ALT_missing=node['ALT'] is None
            else:
                if ar is None:raise ValueError('Valid screened BASE has missing ALT')
                accepted=bool(ar['physical']['valid'] and not alt['terminated'] and alt['source_reward']>base['source_reward']+.0001);selected='ALT' if accepted else 'BASE';chosen=alt if accepted else base;stop=chosen['terminated'] or chosen['truncated'];reason='IMPROVED_VALID_NONFAILURE' if accepted else 'BASE_RETAINED';ALT_missing=False
                if selected=='BASE':expected_restores.append(prefix+'_end_BASE')
            chosen=alt if selected=='ALT' else base;previous=aend if selected=='ALT' else bend
            checks=dict(same_full_source=same_source,submitted_proposal=proposal_matches,selection=node['selection']['selected']==selected and node['selection']['accepted']==accepted and node['selection']['stop']==stop and node['selection']['reason']==reason,
                selected_reward=node['selected_reward']==chosen['source_reward'],selected_failure=node['selected_terminated']==chosen['terminated'],selected_completion=node['selected_truncated']==chosen['truncated'],only_last_stops=stop==(index==len(result['nodes'])-1),ALT_declaration=(node['ALT_missing_reason'] is not None)==ALT_missing)
            nodes.append(dict(phase=phase,checks=checks,selected=selected,accepted=accepted,stop=stop,reason=reason,selected_physical_valid=br['physical']['valid'] if selected=='BASE' else ar['physical']['valid']))
        actual_restores=result['restores'];restore_checks=dict(expected_labels=[v['label'] for v in actual_restores]==expected_restores,all_complete_byte_readbacks=all(v['audit']['all_array_bytes_equal'] and v['audit']['all_scalar_values_equal'] and all(v['complete_state_equal'].values()) and v['external_counter_unchanged'] for v in actual_restores))
        final=nodes[-1];derived_stop='BASE_PHYSICAL_INVALID' if final['reason']=='BASE_PHYSICAL_INVALID' else ('REFERENCE_COMPLETION' if result['nodes'][-1]['selected_truncated'] else 'TASK_FAILURE');count=len(nodes);accepted=sum(n['accepted'] for n in nodes);success=derived_stop=='REFERENCE_COMPLETION' and count==368
        checks=dict(all_packet_contracts=all(all(p['checks'].values()) for p in packets),all_proposals=all(all(p['checks'].values()) for p in proposal_rows),all_node_decisions=all(all(n['checks'].values()) for n in nodes),all_complete_source_chains=all(all(v[k] for k in ['native','python','rng']) for v in source_chains),all_exact_phases=all(phase_checks),all_restores=all(restore_checks.values()),
            missing_declared=result['missing_controls']==list(range(count+1,369)),actual_controls=count==result['selected_controls'],actual_queries=query_count==result['native_queries'],actual_duration=.02*count==result['selected_duration_s'],accepted_exact=accepted==result['accepted_proposals'],outcome_exact=result['stop_reason']==derived_stop and result['full_task_success']==success,
            original_scope=result['scope']['validated'] and result['scope']['bindings_restored'] and not result['scope']['source_epoch_used'],actor_readback=result['actor_unchanged'])
        valid_prefix=next((i for i,n in enumerate(nodes) if not n['selected_physical_valid']),len(nodes));invalid=[p['name'] for p in packets if not p['physical']['valid']]
        return dict(checks=checks,passed=sum(checks.values()),total=len(checks),nodes=nodes,packets=packets,proposals=proposal_rows,source_chains=source_chains,restore_checks=restore_checks,CPU_recompute_bounds=cpu_bounds,
            summary=dict(name=result['name'],teacher=result['teacher'],controller=result['controller'],model=result['model'],score_mode=result['score_mode'],repetition=result['repetition'],selected_controls=count,valid_selected_prefix_controls=valid_prefix,executed_selected_duration_s=.02*count,valid_selected_prefix_s=.02*valid_prefix,native_queries=query_count,accepted_proposals=accepted,stop_reason=derived_stop,full_task_success=success,
                physically_valid_queries=sum(p['physical']['valid'] for p in packets),invalid_queries=invalid,all_declared_controls=368,original_physics_flags=sum(len(p['physical']['checks']) for p in packets),original_reward_flags=sum(len(p['reward']) for p in packets)))
