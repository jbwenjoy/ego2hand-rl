"""Continuous frozen-Actor W1 packets with two full snapshots and no restores."""
import time
from pathlib import Path
from unittest.mock import patch
import mujoco
import numpy as np
import torch

from ..artifacts import file_pin,verify_file_pin
from ..learning.world_model import raw_tree_equal
from ..mechanics.compound_holding import object_state
from ..mechanics.controlled_pen_records import native_state55
from ..mechanics.probe_row_review import review_recorded_row
from ..mechanics.pen_reward_chain import controller,native_components,ACTION,MOTION,ROBOT,CACHE,A,I
from ..strict_json import write_json
from . import owned_pen_state as native
from .pen_owned_probe_recorder import OwnedProbeRecorder,save_physics
from .shared_snapshot_transport import SharedImmutableTransport
from .pen_continuous_math_view_v1 import record_dynamic,context,physics_view,policy,terminal_reason
from .controlled_evaluation import summarize_trace


def near_error(cpu,gpu):
    a,b=np.asarray(cpu,float),np.asarray(gpu,float)
    return float(np.linalg.norm(a-b)/(1+np.linalg.norm(b))) if a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() else None


def reference(runtime):
    return dict(time_s=np.arange(369)*.02,COM_world_m=runtime.support_table.pose[:,:3].copy(),quaternion_wxyz=runtime.support_table.pose[:,3:].copy())


def static_math_fields(state):
    keys=[ACTION+(A(k),) for k in ['_target_ids','_position_ctrl_ids','_position_ctrl_limited']]
    keys += [ROBOT+(A(k),) for k in ['joint_pos_limits','encoder_bias','default_joint_pos','default_joint_vel']]
    keys += [MOTION+(A('group'),A(k)) for k in ['joint_pos','link_pos']]+[MOTION+(A('_trajectory_starts'),),CACHE+(A('world_pose'),),(A('support_table'),A('targets')),(A('support_table'),A('valid'))]
    cfg=(A('env'),A('cfg'),A('actions'),I('joint_pos'))
    scalars=[ACTION+(A('_scale'),),ACTION+(A('_mask_enabled'),),cfg+(A('alpha'),),cfg+(A('residual_clip'),I(0)),cfg+(A('residual_clip'),I(1)),(A('support_weight'),)]
    return dict(tensors={k:state['python']['tensors'][k] for k in keys},scalars={k:state['python']['scalars'][k] for k in scalars})


def augmented_trace(z,model,template):
    out={k:v for k,v in z.items() if k in ['pre_qpos','pre_qvel','post_qpos','post_qvel','post_ctrl','post_qacc','post_qacc_warmstart','pre_object_COM','post_qfrc_constraint','post_qfrc_applied','post_xfrc_applied','post_GPU_time','overflow','post_mocap_pos','post_mocap_quat','pre_penetration_m','pre_self_penetration_m','post_joint_limit_excursion_rad']}
    layout=template['model_layout']; qa,da=layout['object_qposadr'],layout['object_dofadr']; body=layout['object_body']
    values=[object_state(model,q,v,qa,da,np.asarray(layout['COM_BODY'])) for q,v in zip(z['post_qpos'],z['post_qvel'])]
    for k,index in [('post_COM',0),('post_quaternion_wxyz',1),('post_COM_velocity',2),('post_omega_world',3)]: out[k]=np.asarray([v[index] for v in values])
    hand=np.zeros(8); nonhand=np.zeros(8)
    for row in z['raw_contact_tick_plus23']:
        tick,g,h=int(row[0]),int(row[1]),int(row[2]); on=model.geom_bodyid[[g,h]]==body
        if on.sum()==1:
            other=h if on[0] else g; target=hand if (model.geom(other).name or '').startswith('robot/') else nonhand
            target[tick]+=max(0.,float(row[18]))
    out['pre_hand_normal_N']=hand; out['pre_object_nonhand_normal_N']=nonhand
    return out


def run_case(runtime,actor,decl,folder,receipt,counter,activate,profile,gates,reward_cfg):
    started=time.perf_counter(); folder=Path(folder); folder.mkdir()
    physics=save_physics(runtime,folder); m=runtime.env.sim.mj_model; f=runtime.features; body=f.body; qa=f.qa; da=int(m.jnt_dofadr[m.body_jntadr[body]])
    if m.neq or np.any(m.body_gravcomp) or int(m.body('robot/right_palm_link').jntnum)!=0 or not np.all(m.actuator_trntype==int(mujoco.mjtTrn.mjTRN_JOINT)):
        raise ValueError('Original fixed-wrist joint-only/no-aid structure required')
    template=dict(dt=.0025,qids=f.qids.cpu().tolist(),dofs=f.dofs.cpu().tolist(),aids=runtime.action._position_ctrl_ids.cpu().tolist(),physics=physics,
        model=file_pin(folder/'model.mjb'),physical_context=file_pin(folder/'physical_context.npz'),role='FROZEN_COST_LONG_EVALUATION_NEVER_FIT',
        model_layout=dict(object_body=body,object_qposadr=qa,object_dofadr=da,COM_BODY=m.body_ipos[body].tolist()))
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(template['model']))); physical=dict(np.load(verify_file_pin(template['physical_context'])))
    store=SharedImmutableTransport(folder/'shared_immutable.pt.gz'); initial=native.capture(runtime,receipt); ip=store.save(folder/'initial.pt.gz',initial)
    frozen={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}; recorder=OwnedProbeRecorder(runtime); activate(recorder); rows=[]; traces=[]; n0=counter.native; stop=None
    try:
        for index in range(368):
            stage=f'P{index+1:03}'; qf=folder/stage; qf.mkdir(); dynamic=record_dynamic(runtime); obs=runtime.observation()['policy']; raw=actor.distribution(obs,validate_args=False).mean
            if dynamic['phase']!=index+1 or not bool(torch.isfinite(obs).all()) or not bool(torch.isfinite(raw).all()): raise ValueError('Exact finite fresh-LIVE phase/Actor input required')
            cpu_obs=policy(initial,dynamic,template).numpy(); obs_cpu=obs.detach().cpu().numpy().copy(); obs_error=near_error(cpu_obs,obs_cpu)
            if obs_error is None or obs_error>1e-5: raise ValueError('CPU reconstructed policy observation differs')
            raw_cpu=raw.detach().cpu().numpy()[0].copy(); ctx=context(initial,dynamic,template); predicted,ema,applied=controller(torch.from_numpy(raw_cpu[None]),ctx)
            torch.save(dict(dynamic=dynamic,policy_observation=obs_cpu,CPU_policy_observation=cpu_obs,sampled_mean_RAW=raw_cpu),qf/'control_source.pt')
            recorder.begin(raw_cpu); before=counter.native
            try:
                _,reward,term,trunc,info=runtime.step(raw); z=recorder.finish()
            except BaseException:
                torch.save(dict(trace=recorder.trace,rows=recorder.rows,efc=recorder.efc,addresses=recorder.addresses),qf/'partial_NOT_RESUMABLE.pt'); raise
            w={k:v.detach().cpu().numpy().copy()[0] for k,v in runtime.contact_views.items()}
            np.savez_compressed(qf/'trace.npz',**z); np.savez_compressed(qf/'window.npz',**w)
            actual=z['post_ctrl'][0,template['aids']]; quality=dict(eight_ticks=counter.native-before==8,actual_PD=bool(abs(actual-predicted[0].numpy()).max()<=2e-7),no_aid=not bool(np.any(z['post_qfrc_applied']) or np.any(z['post_xfrc_applied'])))
            case=dict(template,name=decl['name']+'_'+stage,phase=dynamic['phase'],actual_PD20=actual.tolist(),qualified=all(quality.values()),quality=quality,
                sampled_RAW20=raw_cpu.tolist(),applied_RAW20=applied[0].numpy().tolist(),RAW_clip_count=int(np.count_nonzero(abs(raw_cpu)>1)),source_reward=float(reward[0]),
                terminated=bool(term[0]),truncated=bool(trunc[0]),termination_reasons={k:bool(v[0]) for k,v in info['termination_reasons'].items()},native_integrations=counter.native-before)
            write_json(qf/'execution.json',dict(case,trace=file_pin(qf/'trace.npz'),window=file_pin(qf/'window.npz'),control_source=file_pin(qf/'control_source.pt')))
            with patch.object(mujoco,'mj_step',side_effect=AssertionError('Audit cannot step')),patch.object(mujoco,'mj_forward',side_effect=AssertionError('Audit cannot forward')):
                packet,audit=review_recorded_row(case,z,w,physics_view(initial,dynamic,case),model,physical,dict(role=template['role'],object_penetration_m=.002,self_penetration_m=.0002,joint_excursion_rad=.001))
                component,flags,errors=native_components(case,z,w,ctx,reward_cfg,runtime.objective_cfg)
            expected_x=np.r_[native_state55(dynamic['arrays']['qpos'][0],dynamic['arrays']['qvel'][0],case['qids'],case['dofs'],qa,da),dynamic['arrays']['qacc_warmstart'][0],predicted[0].numpy(),.0025].astype(np.float32)
            checks=dict(native8=case['native_integrations']==8,observation=obs_error<=1e-5,source_state=np.array_equal(z['pre_qpos'][0],dynamic['arrays']['qpos'][0]) and np.array_equal(z['pre_qvel'][0],dynamic['arrays']['qvel'][0]),
                canonical_input=np.array_equal(packet['x'].astype(np.float32),expected_x),reward=abs(float(component[-1])-case['source_reward'])<=2e-6 and all(flags.values()),
                EMA_readback=np.array_equal(runtime.action._processed_actions.detach().cpu().numpy()[0],ema[0].numpy()),RAW_readback=np.array_equal(runtime.action._raw_actions.detach().cpu().numpy()[0],applied[0].numpy()),profile=profile.verify())
            stop=terminal_reason(audit['valid'],case['terminated'],case['truncated'],case['phase'])
            case.update(control_source=file_pin(qf/'control_source.pt'),trace=file_pin(qf/'trace.npz'),window=file_pin(qf/'window.npz'),physical_audit=audit,reward_checks=flags,reward_errors=errors,observation_scaled_L2=obs_error,
                checks={k:bool(v) for k,v in checks.items()},passed=sum(bool(v) for v in checks.values()),total=len(checks),stop_reason=stop)
            write_json(qf/'report.json',case); rows.append(file_pin(qf/'report.json')); traces.append(augmented_trace(z,model,template))
            write_json(folder/'work.json',dict(stage=stage,completed_controls=len(rows),**counter.metadata()))
            if not all(checks.values()): raise ValueError('Continuous source/controller/reward discrepancy')
            if stop is not None: break
        if stop is None: raise ValueError('Reference boundary was not reached within declared controls')
        end=native.capture(runtime,receipt); ep=store.save(folder/'end.pt.gz',end)
        torch.save({k:v.detach().cpu().clone() for k,v in actor.state_dict().items()},folder/'actor_after.pt')
        trace={k:np.concatenate([t[k] for t in traces]) for k in traces[0]}; ref=reference(runtime)
        np.savez_compressed(folder/'trace.npz',**trace,**ref,initial_qpos=initial['native']['data']['arrays']['qpos'][0],initial_mocap_pos=initial['native']['data']['arrays']['mocap_pos'],initial_mocap_quat=initial['native']['data']['arrays']['mocap_quat'])
        complete=stop=='REFERENCE_COMPLETION' and len(rows)==368
        summary=summarize_trace(trace,ref,.0025,complete,gates)
        valid_prefix=len(rows)-int(stop=='PHYSICAL_AUDIT_STOP')
        checks=dict(all_control_checks=all(read_json(p)['passed']==read_json(p)['total'] for p in rows),exact_native=counter.native-n0==8*len(rows),
            model_bytes=raw_tree_equal(initial['native']['model'],end['native']['model']),static_math_fields=raw_tree_equal(static_math_fields(initial),static_math_fields(end)),actor_norm_unchanged=all(torch.equal(v.cpu(),frozen[k]) for k,v in actor.state_dict().items()),
            final_state=np.array_equal(trace['post_qpos'][-1],end['native']['data']['arrays']['qpos'][0]) and np.array_equal(trace['post_qvel'][-1],end['native']['data']['arrays']['qvel'][0]),
            fixed_mount=np.array_equal(trace['post_mocap_pos'],np.tile(initial['native']['data']['arrays']['mocap_pos'],(len(trace['post_qpos']),1,1))) and np.array_equal(trace['post_mocap_quat'],np.tile(initial['native']['data']['arrays']['mocap_quat'],(len(trace['post_qpos']),1,1))),profile=profile.verify())
        result=dict(schema='pen_frozen_cost_continuous_case:1',**decl,**template,initial=ip,end=ep,controls=rows,selected_controls=len(rows),native_integrations=counter.native-n0,
            selected_duration_s=.02*len(rows),valid_prefix_controls=valid_prefix,valid_prefix_s=.02*valid_prefix,stop_reason=stop,missing_controls=list(range(len(rows)+1,369)),
            summary=summary,full_task_success=bool(complete and summary['qualified']),trace=file_pin(folder/'trace.npz'),actor_after=file_pin(folder/'actor_after.pt'),iteration=0 if decl['policy']=='INITIAL' else 2000,
            checks={k:bool(v) for k,v in checks.items()},passed=sum(bool(v) for v in checks.values()),total=len(checks),optimizer_steps=0,normalizer_fits=0,wall_s=time.perf_counter()-started,
            scope='Fresh frozen actor/normalizer only,new cost backend identity;continuous raw packets and full endpoints. Missing future censored;no adaptation/ego/W32/speed admission.')
        write_json(folder/'report.json',result)
        if not all(checks.values()): raise ValueError('Continuous final source/model/frozen actor discrepancy')
        return result
    finally: activate(None)


def read_json(pin):
    import json
    return json.loads(verify_file_pin(pin).read_text())
