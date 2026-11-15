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
from supermimic_research.runtime.pen_native_planning_math import candidate_offsets,candidate_targets,candidate_sequence_targets
from supermimic_research.mechanics.mean_reference_wrench import mean_reference_wrench
from supermimic_research.mechanics.feedback_support import feedback_support_target
from supermimic_research.runtime.collision_numerics import validate_collision_numerics
from supermimic_research.runtime.response_parent_split import parent_roles


def main(a):
    import torch
    import warp as wp
    from mjlab.utils.lab_api.math import combine_frame_transforms
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());bp=(cp.parent/cfg['base_config']).resolve();base=tomllib.loads(bp.read_text())
    pp=(cp.parent/cfg['proposal']).resolve();proposal=json.loads(pp.read_text());s=np.load(verify_file_pin(proposal['artifacts']['selected']))
    profile=cfg.get('experiment','FEEDBACK')
    profiles={'FEEDBACK':['STATE_ONLY','STATE_PLUS_SUPPORT','STATE_PLUS_FEEDBACK_SUPPORT'],
              'STATE_SUPPORT':['STATE_ONLY','STATE_PLUS_SUPPORT'],
              'VELOCITY':['STATE_ONLY','STATE_PLUS_SUPPORT','STATE_PLUS_VELOCITY','STATE_PLUS_VELOCITY_SUPPORT']}
    if profile not in profiles:raise ValueError('Unknown predeclared objective profile')
    ccd=cfg.get('collision_numerics')
    if profile=='STATE_SUPPORT':
        ccd=validate_collision_numerics(ccd)
        if ccd['ccd_iterations']!=50 or ccd['ccd_tolerance'] not in [1e-6,1e-7]:raise ValueError('Predeclared two CCD profiles only')
    elif ccd is not None:raise ValueError('CCD comparison requires explicit STATE_SUPPORT profile')
    orientation=cfg.get('orientation_objective','SHAFT')
    if orientation not in ['SHAFT','FULL_SO3']:raise ValueError('Declared orientation objective')
    if 'orientation_objective' in cfg and (profile!='STATE_SUPPORT' or ccd!=dict(ccd_iterations=50,ccd_tolerance=1e-6) or cfg['shaft_scale_deg']!=5.):
        raise ValueError('Orientation matrix fixes original CCD and5deg scale')
    multiplier=cfg.get('candidate_offset_multiplier',1)
    horizon=cfg.get('prediction_controls',1)
    if isinstance(horizon,bool) or horizon not in [1,5] or horizon>1 and (profile!='STATE_SUPPORT' or orientation!='FULL_SO3'):
        raise ValueError('Predeclared H1/H5 full-SO3 state/support prediction')
    response_spec=cfg.get('response_parent_split');roles=parent_roles(response_spec) if response_spec is not None else {}
    if roles and (multiplier!=4 or orientation!='FULL_SO3' or profile!='STATE_SUPPORT' or horizon!=1):raise ValueError('Frozen fresh X4 response-parent collection')
    if 'candidate_offset_multiplier' in cfg and (profile!='STATE_SUPPORT' or orientation!='FULL_SO3'):
        raise ValueError('Amplitude matrix requires declared full-SO3 state/support profile')
    offsets=candidate_offsets(seed=cfg['candidate_seed'],multiplier=multiplier)
    raw_table=offsets+(2*s['delta20']).astype(np.float32)
    if not np.isfinite(raw_table).all() or abs(raw_table).max()>1:raise ValueError('Entire bounded raw candidate table required before execution')
    if (a.output.exists() or not a.output.resolve().is_relative_to(ROOT/'runs') or cfg['candidates']!=57 or cfg['horizon_s']!=.02*horizon
            or cfg['controllers']!=profiles[profile] or cfg['dt']!=[.005,.0025]
            or (profile=='FEEDBACK' and (cfg['feedback_position_gain_s2']!=100. or cfg['feedback_velocity_gain_s']!=20.))
            or (profile=='VELOCITY' and (cfg['velocity_scale_m_s']!=.25 or cfg['velocity_weight']!=1.))
            or cfg['residual_reference']!='ZERO_PHYSICAL_EMA' or not cfg['numerical_device_override']):raise ValueError('Fixed fresh native search experiment')
    torch.set_num_threads(4);wp.set_device('cuda:0');a.output.mkdir();registry=ArtifactRegistry(bp.parent/base['artifact_manifest'])
    np.savez_compressed(a.output/'candidates.npz',raw_offsets=offsets)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    paths=sorted({p for package in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/package).rglob('*.py')})
    paths.extend(ROOT/p for p in ['tools/run_pen_native.py','tools/run_pen_native_feedback.py'])
    source_receipt=freeze_sources(paths,a.output/'source_snapshot',ROOT,revision)
    verify_source_snapshot(source_receipt,live=True)
    write(a.output/'source_receipt.json',source_receipt)
    import mujoco_warp as mw
    package=Path(mw.__file__).resolve().parent
    installed=freeze_sources(sorted(p for p in package.rglob('*.py') if not p.name.endswith('_test.py') and not p.name.startswith('test_')),
        a.output/'installed_source_snapshot',package,'installed-package-source-not-git')
    write(a.output/'installed_source_receipt.json',installed)
    import mjlab.sim.sim as engine_module
    engine_path=Path(engine_module.__file__).resolve()
    engine_sources=freeze_sources([engine_path],a.output/'engine_source_snapshot',engine_path.parents[1],'installed-mjlab-simulation-source-not-git')
    write(a.output/'engine_source_receipt.json',engine_sources)
    sources=[r['snapshot'] for r in source_receipt['records']]
    intent=dict(schema='pen_native_objective_search:4' if horizon>1 else 'pen_native_objective_search:3',experiment=profile,source_receipt=file_pin(a.output/'source_receipt.json'),installed_source_receipt=file_pin(a.output/'installed_source_receipt.json'),collision_numerics=ccd,
        engine_source_receipt=file_pin(a.output/'engine_source_receipt.json'),
        collision_profile_fidelity='UNQUALIFIED_DEVELOPMENT' if ccd is not None else 'NOT_NEWLY_QUALIFIED',code_revision=revision,config=file_pin(cp),base_config=file_pin(bp),proposal=file_pin(pp),sources=sources,
        candidates=file_pin(a.output/'candidates.npz'),orientation_objective=orientation,neural_updates=0,Actor_updates=0,
        candidate_offset_multiplier=multiplier,entire_raw_table_max_abs=float(abs(raw_table).max()),entire_raw_table_valid=True,
        planning_horizon_controls=horizon,actual_applied_controls_per_decision=1,
        response_parent_split=response_spec,response_data_role='FRESH_WM_PARENTS' if roles else 'CONTROLLER_DIAGNOSTIC',
        scope=f'Native57-candidate H{horizon} receding-horizon objective comparison, apply only first20ms. Future per-phase raw/EMA/reference PD, mean per-control cost; full forecast geometry. Unequal forecast/survival compute explicitly counted. Full original7.36s clock, first native terminal, no object force/reset aid. Diagnostic same-clip development, not learning data, learned-policy A1/A2, real-time, surrogate speedup or ego qualification.')
    write(a.output/'intent.json',intent);cases=[];keepers=[];response_rows=[]
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
                reference.update(mass=float(dm.body_mass.numpy()[0,b]),velocity=torch.tensor(targets['velocity_world'],dtype=torch.float32,device='cuda'),COM=torch.tensor(COM,device='cuda'),quaternion=torch.tensor(Q,dtype=torch.float32,device='cuda'),shaft=torch.tensor(Rotation.from_quat(Q[:,[1,2,3,0]]).as_matrix()[:,:,0],dtype=torch.float32,device='cuda'),
                    hand=torch.tensor(command._frames['joint_pos'].detach().cpu().numpy(),device='cuda'),
                    wrench=torch.tensor(targets['wrench_world'],dtype=torch.float32,device='cuda'),valid=targets['valid_interval'])
                arrays,scalars,hashes=capture_model(dm);model_before.update(hashes);model_scalars.update(scalars)
                np.savez_compressed(folder/'fixed_model_arrays.npz',**arrays);write(folder/'fixed_model_scalars.json',scalars)
                return dict(**receipt,nominal_mean_wrench=file_pin(folder/'reference.npz'),wrench_schema=targets['schema'],reference_estimator=targets['estimator'])
            def provider(env,obs,k):
                started=time.perf_counter();action=env.action_manager.get_term('joint_pos');command=env.command_manager.get_term('motion')
                if action._mask_enabled or action.cfg.alpha!=.5 or action.cfg.scale!=.5:raise ValueError('Original unmasked .5/.5 controller')
                statekeys=['_raw_actions','_processed_actions','_targets','_targets_valid','_mask_active','_mask_remaining','_mask_dofs']
                controller={key:getattr(action,key).detach().cpu().numpy().copy() for key in statekeys};controller['command_phase']=command.time_steps.detach().cpu().numpy().copy()
                effective_horizon=min(horizon,368-k);planner_key=f'planner_H{effective_horizon}'
                if planner_key not in planning:
                    wp.synchronize();planner=NativePenPlanner(env.sim.mj_model,env.sim.wp_model,env.sim.wp_data,s['qids'],s['aids'],round(.02/dt)*effective_horizon,response_heads=bool(roles),horizon_controls=effective_horizon)
                    planning[planner_key]=planner;keepers.append(planner)
                    if not (folder/'observed_Data_layout.json').exists():write(folder/'observed_Data_layout.json',dict(schema='observed_live_dynamic_Data_layout:1',
                        scalars=planner.live_scalars,layout=planner.layout,
                        arrays={key:dict(shape=list(shape),dtype=str(dtype),strides=list(strides))
                            for key,(_,shape,dtype,strides) in planner.live_layout.items()},
                        scope='Observed in this fresh native episode; static layout/scalars rechecked at every source load. Never backfilled into historical runs.'))
                planner=planning[planner_key];planning['planner']=planner;wp.synchronize();planner.load_source()
                raw=center+offset_tensor[k];lower,upper=action.target_position_limits;bias=env.scene['robot'].data.encoder_bias[:,action._target_ids]
                targets,ema,intended=candidate_targets(raw,action._processed_actions,command.joint_pos,bias,lower,upper)
                sequence_reference=None;sequence_payload={}
                if effective_horizon>1:
                    raw_sequence=center+offset_tensor[k:k+effective_horizon].permute(1,0,2)
                    targets,ema,intended=candidate_sequence_targets(raw_sequence,action._processed_actions,
                        reference['hand'][k+1:k+1+effective_horizon],bias,lower,upper)
                    sequence_reference={key:reference[key][k+1:k+1+effective_horizon] for key in ['COM','shaft','hand','quaternion']}
                    sequence_reference.update(wrench=reference['wrench'][k:k+effective_horizon],valid=reference['valid'][k:k+effective_horizon])
                    sequence_payload=dict(raw_sequence=raw_sequence.detach().cpu().numpy(),PD_schedule=targets.detach().cpu().numpy(),
                        EMA_schedule=ema.detach().cpu().numpy(),intended_PD_schedule=intended.detach().cpu().numpy(),
                        sequence_reference_COM=sequence_reference['COM'].detach().cpu().numpy(),
                        sequence_reference_quaternion=sequence_reference['quaternion'].detach().cpu().numpy())
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
                    velocity_valid=endpoint_velocity_valid,velocity_weight=velocity_weight,reference_quaternion=reference['quaternion'][k+1],reference_sequence=sequence_reference)
                primary_search_timing=dict(planner.last_search_timing)
                compute_wall=time.perf_counter()-started
                audit=planner.source.audit(env.sim.wp_data)
                controller_same=all(np.array_equal(controller[key],getattr(action,key).detach().cpu().numpy()) for key in statekeys)
                controller_same &= np.array_equal(controller['command_phase'],command.time_steps.detach().cpu().numpy())
                if not audit['all_array_bytes_equal'] or not audit['scalar_equal'] or not controller_same:raise ValueError('Live planning isolation failed')
                path=folder/f'decision_{k:04d}.npz'
                first_ema=ema[:,0] if effective_horizon>1 else ema
                first_PD=targets[:,0] if effective_horizon>1 else targets
                first_intended=intended[:,0] if effective_horizon>1 else intended
                payload=dict(prediction_controls=effective_horizon,**sequence_payload,endpoint_velocity_world=endpoint_velocity.detach().cpu().numpy(),
                    PD_lower20=lower.detach().cpu().numpy(),PD_upper20=upper.detach().cpu().numpy(),encoder_bias20=bias.detach().cpu().numpy(),
                    orientation_reference_quaternion_wxyz=reference['quaternion'][k+1].detach().cpu().numpy(),
                    endpoint_velocity_valid=endpoint_velocity_valid,velocity_weight=velocity_weight,support_weight=support_weight,support_target_world=support_target.detach().cpu().numpy(),
                    support_feedback_force_world=correction.detach().cpu().numpy(),
                    support_source_position_error_world_m=position_error.detach().cpu().numpy(),
                    support_source_velocity_error_world_m_s=velocity_error.detach().cpu().numpy(),**arrays,**{'score_'+key:value for key,value in scores.items()},raw_candidates=raw.detach().cpu().numpy(),
                    EMA_candidates=first_ema.detach().cpu().numpy(),PD_targets=first_PD.detach().cpu().numpy(),intended_PD=first_intended.detach().cpu().numpy(),
                    **{'controller_'+key:value for key,value in controller.items()})
                np.savez_compressed(path,**payload)
                source=folder/f'source_{k:04d}.npz';np.savez_compressed(source,**planner.source.host_values())
                if k in roles:
                    parent=dict(role=roles[k],mode=mode,dt=dt,k=k,parent_key=f'{a.output.name}/{mode}/{dt}/{k}',source=file_pin(source),
                        primary_prediction=file_pin(path),native_physics_ticks=57*round(.02/dt))
                    response_rows.append(dict(**parent,repeat=0,trace=file_pin(path),selected=selected,candidate_cuda_ms=cuda_ms))
                    for repeat in [1,2]:
                        planner.load_source()
                        extra_index,extra_arrays,extra_scores,extra_cuda=planner.search(targets,ema,reference['COM'][k+1],reference['shaft'][k+1],command.joint_pos[0],
                            support_target,valid,support_weight,scorecfg,reference_velocity=endpoint_velocity,velocity_valid=endpoint_velocity_valid,
                            velocity_weight=velocity_weight,reference_quaternion=reference['quaternion'][k+1])
                        extra=folder/f'response_{k:04d}_R{repeat}.npz'
                        np.savez_compressed(extra,**{**payload,**extra_arrays,**{'score_'+key:value for key,value in extra_scores.items()}})
                        response_rows.append(dict(**parent,repeat=repeat,trace=file_pin(extra),selected=extra_index,candidate_cuda_ms=extra_cuda))
                    after_extra=planner.source.audit(env.sim.wp_data)
                    after_controller=all(np.array_equal(controller[key],getattr(action,key).detach().cpu().numpy()) for key in statekeys)
                    after_controller &= np.array_equal(controller['command_phase'],command.time_steps.detach().cpu().numpy())
                    if not after_extra['all_array_bytes_equal'] or not after_extra['scalar_equal'] or not after_controller:raise ValueError('Extra response repeats changed live source/controller')
                    write(a.output/'response_progress.json',dict(rows=response_rows))
                decision=dict(k=k,phase=int(controller['command_phase'][0]),selected=selected,candidates=57,
                    prediction_controls=effective_horizon,native_candidate_ticks=57*effective_horizon*round(.02/dt),
                    candidate_offset_multiplier=multiplier,
                    orientation_objective=orientation,
                    any_feasible=bool(scores['feasible'].any()),selected_feasible=bool(scores['feasible'][selected]),
                    selected_finite=bool(scores['finite'][selected]),selected_capacity=bool(scores['capacity'][selected]),
                    selected_violation=float(scores['violation'][selected]),endpoint_velocity_valid=endpoint_velocity_valid,velocity_weight=velocity_weight,support_weight=support_weight,nominal_wrench_valid=bool(reference['valid'][k]),
                    live_data_controller_untouched=True,source=file_pin(source),prediction=file_pin(path),
                    candidate_cuda_ms=cuda_ms,search_timing=primary_search_timing,compute_and_readback_wall_s=compute_wall,total_with_artifact_wall_s=time.perf_counter()-started)
                decisions.append(decision);write(folder/'planning_progress.json',dict(decisions=decisions))
                print(json.dumps(dict(mode=mode,dt=dt,k=k,selected=selected,feasible=decision['selected_feasible'],violation=decision['selected_violation'],cuda_ms=cuda_ms)),flush=True)
                if not decision['selected_finite'] or not decision['selected_capacity']:raise ValueError('No numerical/capacity-valid selected forecast')
                return raw[selected:selected+1]
            c=evaluate_case('hand2-pen-spin',mode,None,{**base,'physics_dt':dt,'numerical_options':cfg['numerical_options'],'numerical_device_override':True,
                **({'collision_numerics':ccd} if ccd is not None else {})},
                bp.parent,registry,folder,initializer=initialize,action_provider=provider)
            planner=planning['planner'];_,after_scalars,after_hashes=capture_model(planner.model)
            c.update(name=mode,planning=decisions,planning_model_unchanged=after_hashes==model_before and after_scalars==model_scalars,
                candidate_offset_multiplier=multiplier,
                planning_horizon_controls=horizon,
                orientation_objective=orientation,
                observed_Data_layout=file_pin(folder/'observed_Data_layout.json'),
                fixed_model_arrays=file_pin(folder/'fixed_model_arrays.npz'),fixed_model_scalars=file_pin(folder/'fixed_model_scalars.json'),
                candidates_evaluated=57*len(decisions),candidate_physics_ticks=sum(d['native_candidate_ticks'] for d in decisions),
                supervision='NATIVE_CONTROLLER_PROBE_NONEXPERT_UNLESS_ALL_ORIGINAL_GATES_PASS')
            cases.append(c);write(folder/'report.json',c);write(a.output/'progress.json',dict(cases=cases))
    task_sources=frozen_import_sources(source_receipt,ROOT/'src','dex_supermimic')
    verify_source_snapshot(source_receipt,live=True);verify_source_snapshot(installed,live=True);verify_source_snapshot(engine_sources,live=True)
    result=dict(**intent,cases=cases,task_sources=task_sources,source_import_enumeration_available=True,response_rows=response_rows,
        response_candidate_ticks_including_parent_primary=sum(e['native_physics_ticks'] for e in response_rows),
        extra_response_candidate_ticks=sum(e['native_physics_ticks'] for e in response_rows if e['repeat']>0))
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),cases=[dict(mode=c['name'],dt=c['dt'],duration=c['summary']['metrics']['actual_duration_s'],qualified=c['summary']['qualified'],candidates=c['candidates_evaluated']) for c in cases])))
