"""Shared actual native objective comparisons with immutable source receipts."""
import json
import time
import subprocess
import tomllib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[3]
from supermimic_research.artifacts import ArtifactRegistry,file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot,frozen_import_sources
from supermimic_research.runtime.controlled_evaluation import evaluate_case,write
from supermimic_research.runtime.pen_gpu_initialization import initialize_and_query_pen
from supermimic_research.runtime.gpu_model_snapshot import capture_model
from supermimic_research.runtime.native_pen_planner import NativePenPlanner
from supermimic_research.runtime.pen_native_planning_math import candidate_offsets,candidate_targets
from supermimic_research.mechanics.mean_reference_wrench import mean_reference_wrench
from supermimic_research.mechanics.feedback_support import feedback_support_target


def main(a):
    import torch
    import warp as wp
    from mjlab.utils.lab_api.math import combine_frame_transforms
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());bp=(cp.parent/cfg['base_config']).resolve();base=tomllib.loads(bp.read_text())
    pp=(cp.parent/cfg['proposal']).resolve();proposal=json.loads(pp.read_text());s=np.load(verify_file_pin(proposal['artifacts']['selected']))
    profile=cfg.get('experiment','FEEDBACK')
    profiles={'FEEDBACK':['STATE_ONLY','STATE_PLUS_SUPPORT','STATE_PLUS_FEEDBACK_SUPPORT'],
              'VELOCITY':['STATE_ONLY','STATE_PLUS_SUPPORT','STATE_PLUS_VELOCITY','STATE_PLUS_VELOCITY_SUPPORT']}
    if profile not in profiles:raise ValueError('Unknown predeclared objective profile')
    if (a.output.exists() or not a.output.resolve().is_relative_to(ROOT/'runs') or cfg['candidates']!=57 or cfg['horizon_s']!=.02
            or cfg['controllers']!=profiles[profile] or cfg['dt']!=[.005,.0025]
            or (profile=='FEEDBACK' and (cfg['feedback_position_gain_s2']!=100. or cfg['feedback_velocity_gain_s']!=20.))
            or (profile=='VELOCITY' and (cfg['velocity_scale_m_s']!=.25 or cfg['velocity_weight']!=1.))
            or cfg['residual_reference']!='ZERO_PHYSICAL_EMA' or not cfg['numerical_device_override']):raise ValueError('Fixed fresh native search experiment')
    torch.set_num_threads(4);wp.set_device('cuda:0');a.output.mkdir();registry=ArtifactRegistry(bp.parent/base['artifact_manifest'])
    offsets=candidate_offsets(seed=cfg['candidate_seed']);np.savez_compressed(a.output/'candidates.npz',raw_offsets=offsets)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    paths=sorted({p for package in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/package).rglob('*.py')})
    paths.extend(ROOT/p for p in ['tools/run_pen_native.py','tools/run_pen_native_feedback.py'])
    source_receipt=freeze_sources(paths,a.output/'source_snapshot',ROOT,revision)
    verify_source_snapshot(source_receipt,live=True)
    write(a.output/'source_receipt.json',source_receipt)
    sources=[r['snapshot'] for r in source_receipt['records']]
    intent=dict(schema='pen_native_objective_search:2',experiment=profile,source_receipt=file_pin(a.output/'source_receipt.json'),code_revision=revision,config=file_pin(cp),base_config=file_pin(bp),proposal=file_pin(pp),sources=sources,
        candidates=file_pin(a.output/'candidates.npz'),neural_updates=0,Actor_updates=0,
        scope='Native20ms57-candidate finite-action objective comparison; profile and fixed gains/scales declared by config. Objectives only, no applied corrective force. Development on same clip, not held-out. Equal candidates/decision, not equal total compute under unequal survival. Full original7.36s clock, first native terminal, no object force/reset aid. Not learned-policy A1/A2, real-time, surrogate speedup or ego qualification.')
    write(a.output/'intent.json',intent);cases=[];keepers=[]
    for dt in cfg['dt']:
        for mode in cfg['controllers']:
            folder=a.output/f'{mode}_{dt}';planning={};decisions=[];model_before={};model_scalars={};reference={}
            scorecfg={**cfg,**{k:base[k] for k in ['max_penetration_m','max_self_penetration_m','max_joint_limit_excursion_rad','max_object_nonhand_normal_force_N']}}
            offset_tensor=torch.tensor(offsets,device='cuda');center=torch.tensor((2*s['delta20']).astype(np.float32),device='cuda')[None]
            if np.max(abs(offsets+2*s['delta20']))>1:raise ValueError('Illegal raw candidate table')
            def initialize(env):
                receipt=initialize_and_query_pen(env,folder,s['source_qpos'],s['qpos'],s['delta20'],s['target_BODY'],support_all_robot_contacts=True)
                command=env.command_manager.get_term('motion');obj=command._frames['object_pose'];wrist=command._frames['wrist_pose_w']
                pos,quat=combine_frame_transforms(wrist[:,:3],wrist[:,3:],obj[:,:3],obj[:,3:])
                COM,Q=pos.detach().cpu().numpy(),quat.detach().cpu().numpy()
                b=env.sim.mj_model.body('object/cube').id;dm=env.sim.wp_model
                iq=dm.body_iquat.numpy()[0,b];R=Rotation.from_quat(iq[[1,2,3,0]]).as_matrix();I=R@np.diag(dm.body_inertia.numpy()[0,b])@R.T
                targets=mean_reference_wrench(COM,Q,float(dm.body_mass.numpy()[0,b]),I,dm.opt.gravity.numpy()[0])
                np.savez_compressed(folder/'reference.npz',COM_world_m=COM,quaternion_wxyz=Q,reference_hand=command._frames['joint_pos'].detach().cpu().numpy(),
                    **{k:v for k,v in targets.items() if isinstance(v,np.ndarray)})
                reference.update(mass=float(dm.body_mass.numpy()[0,b]),velocity=torch.tensor(targets['velocity_world'],dtype=torch.float32,device='cuda'),COM=torch.tensor(COM,device='cuda'),shaft=torch.tensor(Rotation.from_quat(Q[:,[1,2,3,0]]).as_matrix()[:,:,0],dtype=torch.float32,device='cuda'),
                    wrench=torch.tensor(targets['wrench_world'],dtype=torch.float32,device='cuda'),valid=targets['valid_interval'])
                arrays,scalars,hashes=capture_model(dm);model_before.update(hashes);model_scalars.update(scalars)
                np.savez_compressed(folder/'fixed_model_arrays.npz',**arrays);write(folder/'fixed_model_scalars.json',scalars)
                return dict(**receipt,nominal_mean_wrench=file_pin(folder/'reference.npz'),wrench_schema=targets['schema'],reference_estimator=targets['estimator'])
            def provider(env,obs,k):
                started=time.perf_counter();action=env.action_manager.get_term('joint_pos');command=env.command_manager.get_term('motion')
                if action._mask_enabled or action.cfg.alpha!=.5 or action.cfg.scale!=.5:raise ValueError('Original unmasked .5/.5 controller')
                statekeys=['_raw_actions','_processed_actions','_targets','_targets_valid','_mask_active','_mask_remaining','_mask_dofs']
                controller={key:getattr(action,key).detach().cpu().numpy().copy() for key in statekeys};controller['command_phase']=command.time_steps.detach().cpu().numpy().copy()
                if 'planner' not in planning:
                    wp.synchronize();planner=NativePenPlanner(env.sim.mj_model,env.sim.wp_model,env.sim.wp_data,s['qids'],s['aids'],round(.02/dt))
                    planning['planner']=planner;keepers.append(planner)
                planner=planning['planner'];wp.synchronize();planner.load_source()
                raw=center+offset_tensor[k];lower,upper=action.target_position_limits;bias=env.scene['robot'].data.encoder_bias[:,action._target_ids]
                targets,ema,intended=candidate_targets(raw,action._processed_actions,command.joint_pos,bias,lower,upper)
                valid=bool(reference['valid'][k])
                source_p=wp.to_torch(env.sim.wp_data.qpos)[0,planner.qa:planner.qa+3]
                source_v=wp.to_torch(env.sim.wp_data.qvel)[0,planner.da:planner.da+3]
                nominal=reference['wrench'][k]
                # Tail-invalid intervals have no invented reference derivative.
                position_error=reference['COM'][k]-source_p
                velocity_error=reference['velocity'][k]-source_v if valid else torch.full_like(source_v,float('nan'))
                correction=torch.zeros_like(source_p);support_target=nominal
                if mode=='STATE_PLUS_FEEDBACK_SUPPORT' and valid:
                    feedback=feedback_support_target(nominal,reference['mass'],source_p,source_v,
                        reference['COM'][k],reference['velocity'][k],cfg['feedback_position_gain_s2'],cfg['feedback_velocity_gain_s'])
                    support_target=feedback['target_world'];correction=feedback['correction_force_world']
                endpoint_velocity_valid=k+1<len(reference['velocity'])
                endpoint_velocity=reference['velocity'][k+1] if endpoint_velocity_valid else torch.zeros_like(source_v)
                velocity_weight=cfg['velocity_weight'] if mode in ['STATE_PLUS_VELOCITY','STATE_PLUS_VELOCITY_SUPPORT'] else 0.
                support_weight=cfg['support_weight'] if mode in ['STATE_PLUS_SUPPORT','STATE_PLUS_FEEDBACK_SUPPORT','STATE_PLUS_VELOCITY_SUPPORT'] else 0.
                selected,arrays,scores,cuda_ms=planner.search(targets,ema,reference['COM'][k+1],reference['shaft'][k+1],command.joint_pos[0],
                    support_target,valid,support_weight,scorecfg,reference_velocity=endpoint_velocity,
                    velocity_valid=endpoint_velocity_valid,velocity_weight=velocity_weight)
                compute_wall=time.perf_counter()-started
                audit=planner.source.audit(env.sim.wp_data)
                controller_same=all(np.array_equal(controller[key],getattr(action,key).detach().cpu().numpy()) for key in statekeys)
                controller_same &= np.array_equal(controller['command_phase'],command.time_steps.detach().cpu().numpy())
                if not audit['all_array_bytes_equal'] or not audit['scalar_equal'] or not controller_same:raise ValueError('Live planning isolation failed')
                path=folder/f'decision_{k:04d}.npz'
                np.savez_compressed(path,endpoint_velocity_world=endpoint_velocity.detach().cpu().numpy(),
                    endpoint_velocity_valid=endpoint_velocity_valid,velocity_weight=velocity_weight,support_weight=support_weight,support_target_world=support_target.detach().cpu().numpy(),
                    support_feedback_force_world=correction.detach().cpu().numpy(),
                    support_source_position_error_world_m=position_error.detach().cpu().numpy(),
                    support_source_velocity_error_world_m_s=velocity_error.detach().cpu().numpy(),**arrays,**{'score_'+key:value for key,value in scores.items()},raw_candidates=raw.detach().cpu().numpy(),
                    EMA_candidates=ema.detach().cpu().numpy(),PD_targets=targets.detach().cpu().numpy(),intended_PD=intended.detach().cpu().numpy(),
                    **{'controller_'+key:value for key,value in controller.items()})
                source=folder/f'source_{k:04d}.npz';np.savez_compressed(source,**planner.source.host_values())
                decision=dict(k=k,phase=int(controller['command_phase'][0]),selected=selected,candidates=57,
                    any_feasible=bool(scores['feasible'].any()),selected_feasible=bool(scores['feasible'][selected]),
                    selected_finite=bool(scores['finite'][selected]),selected_capacity=bool(scores['capacity'][selected]),
                    selected_violation=float(scores['violation'][selected]),endpoint_velocity_valid=endpoint_velocity_valid,velocity_weight=velocity_weight,support_weight=support_weight,nominal_wrench_valid=bool(reference['valid'][k]),
                    live_data_controller_untouched=True,source=file_pin(source),prediction=file_pin(path),
                    candidate_cuda_ms=cuda_ms,compute_and_readback_wall_s=compute_wall,total_with_artifact_wall_s=time.perf_counter()-started)
                decisions.append(decision);write(folder/'planning_progress.json',dict(decisions=decisions))
                print(json.dumps(dict(mode=mode,dt=dt,k=k,selected=selected,feasible=decision['selected_feasible'],violation=decision['selected_violation'],cuda_ms=cuda_ms)),flush=True)
                if not decision['selected_finite'] or not decision['selected_capacity']:raise ValueError('No numerical/capacity-valid selected forecast')
                return raw[selected:selected+1]
            c=evaluate_case('hand2-pen-spin',mode,None,{**base,'physics_dt':dt,'numerical_options':cfg['numerical_options'],'numerical_device_override':True},
                bp.parent,registry,folder,initializer=initialize,action_provider=provider)
            planner=planning['planner'];_,after_scalars,after_hashes=capture_model(planner.model)
            c.update(name=mode,planning=decisions,planning_model_unchanged=after_hashes==model_before and after_scalars==model_scalars,
                fixed_model_arrays=file_pin(folder/'fixed_model_arrays.npz'),fixed_model_scalars=file_pin(folder/'fixed_model_scalars.json'),
                candidates_evaluated=57*len(decisions),candidate_physics_ticks=57*len(decisions)*round(.02/dt),
                supervision='NATIVE_CONTROLLER_PROBE_NONEXPERT_UNLESS_ALL_ORIGINAL_GATES_PASS')
            cases.append(c);write(folder/'report.json',c);write(a.output/'progress.json',dict(cases=cases))
    task_sources=frozen_import_sources(source_receipt,ROOT/'src','dex_supermimic')
    result=dict(**intent,cases=cases,task_sources=task_sources,source_import_enumeration_available=True)
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),cases=[dict(mode=c['name'],dt=c['dt'],duration=c['summary']['metrics']['actual_duration_s'],qualified=c['summary']['qualified'],candidates=c['candidates_evaluated']) for c in cases])))
