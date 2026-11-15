"""Read-only actual sources; independent full-Data counterfactual responses."""
import argparse
import gc
import json
import sys
import tomllib
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import ArtifactRegistry,file_pin,verify_file_pin
from supermimic_research.runtime.controlled_evaluation import evaluate_case,write
from supermimic_research.runtime.pen_gpu_initialization import initialize_and_query_pen
from supermimic_research.runtime.gpu_model_snapshot import capture_model
from supermimic_research.runtime.gpu_data_snapshot import ResidentDataSnapshot
from supermimic_research.runtime.gpu_data_fork import fork_dynamic_data
from supermimic_research.runtime.warp_response_recorder import record_response


def main(a):
    import mujoco
    import mujoco_warp as mw
    import torch
    import warp as wp
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());bp=(cp.parent/cfg['base_config']).resolve();base=tomllib.loads(bp.read_text())
    pp=(cp.parent/cfg['proposal']).resolve();proposal=json.loads(pp.read_text());selected=np.load(verify_file_pin(proposal['artifacts']['selected']))
    gp=(cp.parent/cfg['gates']).resolve()
    if (cfg['anchors']!=[0,2,4,6] or cfg['dt']!=[.005,.0025] or cfg['amplitudes']!=[.002,.005]
            or cfg['repeats']!=3 or cfg['direction_seed']!=7801 or cfg['horizon_s']!=.02
            or a.output.exists() or not a.output.resolve().is_relative_to(ROOT/'runs')):raise ValueError('Fixed120 diagnostic response protocol')
    torch.set_num_threads(4);wp.set_device('cuda:0');a.output.mkdir();registry=ArtifactRegistry(bp.parent/base['artifact_manifest'])
    direction=np.random.default_rng(cfg['direction_seed']).normal(size=20);direction/=np.linalg.norm(direction)
    np.savez_compressed(a.output/'direction.npz',direction20=direction)
    profiles=[dict(name='BASE',epsilon=0.,sign=0)]+[dict(name=f'E{eps}_S{sign}',epsilon=eps,sign=sign) for eps in cfg['amplitudes'] for sign in [1,-1]]
    sources=[file_pin(__file__)]+[file_pin(ROOT/p) for p in ['src/supermimic_research/runtime/gpu_data_fork.py',
        'src/supermimic_research/runtime/gpu_data_snapshot.py','src/supermimic_research/runtime/gpu_model_snapshot.py',
        'src/supermimic_research/runtime/warp_response_recorder.py','src/supermimic_research/runtime/controlled_evaluation.py',
        'src/supermimic_research/runtime/pen_gpu_initialization.py','src/supermimic_research/mechanics/warp_contact_labels.py']]
    intent=dict(schema='pen_same_source_responses:1',config=file_pin(cp),base_config=file_pin(bp),proposal=file_pin(pp),gates=file_pin(gp),
        direction=file_pin(a.output/'direction.npz'),profiles=profiles,sources=sources,neural_updates=0,data_role='DIAGNOSTIC_NEVER_FIT',
        contract='8 read-only ACTUAL source captures;5 profiles x3 repeats each, exact20ms held absolute PD. Independent full dynamic Data/graph, fixed read-only actual GPU model. Original actual trajectory is never restored after start. No response becomes TRAIN.')
    write(a.output/'intent.json',intent);actual=[];responses=[];captures=[];retained=[]
    for dt in cfg['dt']:
        folder=a.output/f'actual_{dt}';raw=torch.as_tensor((2*selected['delta20']).astype(np.float32),device='cuda')[None]
        def initialize(env):
            return initialize_and_query_pen(env,folder,selected['source_qpos'],selected['qpos'],selected['delta20'],selected['target_BODY'],support_all_robot_contacts=True)
        def provider(env,obs,k):
            if k not in cfg['anchors']:return raw
            m=env.sim.mj_model;device_model=env.sim.wp_model;live=env.sim.wp_data;action=env.action_manager.get_term('joint_pos');command=env.command_manager.get_term('motion')
            if m.na or m.nhistory or m.nplugin or m.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP) or action._mask_enabled:
                raise ValueError('No unmodeled activation/history/plugin/sleep/masking')
            statekeys=['_raw_actions','_processed_actions','_targets','_targets_valid','_mask_active','_mask_remaining','_mask_dofs']
            controller={key:getattr(action,key).detach().cpu().numpy().copy() for key in statekeys}
            controller.update(command_phase=command.time_steps.detach().cpu().numpy().copy(),command_joint_pos=command.joint_pos.detach().cpu().numpy().copy())
            bias=env.scene['robot'].data.encoder_bias[:,action._target_ids]
            residual=(raw*action._scale).clamp(*action.cfg.residual_clip)
            ema=action.cfg.alpha*residual+(1-action.cfg.alpha)*action._processed_actions
            baseline=action.limit_target(command.joint_pos+ema-bias)[0].detach().cpu().numpy().copy()
            lower,upper=[v[0].detach().cpu().numpy().copy() for v in action.target_position_limits]
            aids=action._position_ctrl_ids.detach().cpu().numpy();n=round(.02/dt)
            wp.synchronize();live_snapshot=ResidentDataSnapshot(live);wp.synchronize()
            values,scalars,model_before=capture_model(device_model)
            source_folder=folder/f'source_A{k}';source_folder.mkdir()
            np.savez_compressed(source_folder/'live_arrays.npz',**live_snapshot.host_values())
            write(source_folder/'live_scalars.json',live_snapshot.scalars);np.savez_compressed(source_folder/'controller.npz',**controller,encoder_bias=bias.detach().cpu().numpy(),baseline_PD20=baseline,lower20=lower,upper20=upper)
            clone,layout=fork_dynamic_data(live);wp.synchronize();clone_snapshot=ResidentDataSnapshot(clone);wp.synchronize()
            initial=clone_snapshot.audit(live)
            was_gc=gc.isenabled();gc.disable()
            try:
                with wp.ScopedDevice(env.sim.wp_device),wp.ScopedCapture() as capture:mw.step(device_model,clone)
            finally:
                if was_gc:gc.enable()
            graph=capture.graph;clone_snapshot.restore(clone);wp.synchronize();retained.append((clone,graph,clone_snapshot,device_model))
            for profile in profiles:
                target=np.clip(baseline+profile['sign']*profile['epsilon']*direction,lower,upper).astype(np.float32)
                for repeat in range(cfg['repeats']):
                    clone_snapshot.restore(clone);wp.synchronize();before=clone_snapshot.audit(clone)
                    name=f'A{k}_{profile["name"]}_R{repeat}';out=source_folder/name;out.mkdir()
                    packet=record_response(m,device_model,clone,graph,aids,target,n)
                    np.savez_compressed(out/'trace.npz',**packet)
                    body=m.body('object/cube').id;joint=m.body_jntadr[body]
                    entry=dict(task='hand2-pen-spin',controller='HELD_PD_FORK',name=name,dt=dt,anchor=k,repeat=repeat,profile=profile['name'],epsilon=profile['epsilon'],sign=profile['sign'],
                        role='DIAGNOSTIC_NEVER_FIT',contact_label_schema='GPU_constraint_indexed_no_adhesion:1',
                        source_capture=str(source_folder),source_array_bytes_exact=bool(before['all_array_bytes_equal'] and before['scalar_equal']),
                        intended_PD20=(baseline+profile['sign']*profile['epsilon']*direction).tolist(),executed_PD20=target.tolist(),
                        clamp_max_rad=float(abs(target-(baseline+profile['sign']*profile['epsilon']*direction)).max()),
                        trace=file_pin(out/'trace.npz'),host_topology=file_pin(folder/'host_topology.mjb'),GPU_physical_context=file_pin(folder/'GPU_physical_context.npz'),
                        model_layout=dict(object_body=body,object_qposadr=int(m.jnt_qposadr[joint]),object_dofadr=int(m.jnt_dofadr[joint]),COM_BODY=device_model.body_ipos.numpy()[0,body].tolist()))
                    responses.append(entry)
            wp.synchronize();after=live_snapshot.audit(live);_,after_scalars,model_after=capture_model(device_model)
            controller_same=all(np.array_equal(controller[key],getattr(action,key).detach().cpu().numpy()) for key in statekeys)
            controller_same &= np.array_equal(controller['command_phase'],command.time_steps.detach().cpu().numpy()) and np.array_equal(controller['command_joint_pos'],command.joint_pos.detach().cpu().numpy())
            checks=dict(initial_fork_full_bytes=initial['all_array_bytes_equal'] and initial['scalar_equal'],
                live_data_untouched=after['all_array_bytes_equal'] and after['scalar_equal'],live_model_untouched=model_before==model_after and scalars==after_scalars,
                controller_command_untouched=bool(controller_same),all_response_sources_exact=all(e['source_array_bytes_exact'] for e in responses if e['dt']==dt and e['anchor']==k))
            receipt=dict(anchor=k,dt=dt,layout=layout,checks=checks,arrays=file_pin(source_folder/'live_arrays.npz'),scalars=file_pin(source_folder/'live_scalars.json'),
                controller=file_pin(source_folder/'controller.npz'),model_fingerprints=model_before)
            captures.append(receipt);write(source_folder/'receipt.json',receipt)
            write(a.output/'progress.json',dict(captures=captures,responses=responses))
            print(json.dumps(dict(dt=dt,anchor=k,responses=len(responses),checks=checks)),flush=True)
            if not all(checks.values()):raise ValueError('Isolation/source contract failed; actual execution stops')
            return raw
        c=evaluate_case('hand2-pen-spin','SOURCE_CAPTURE',None,{**base,'physics_dt':dt,'numerical_options':cfg['numerical_options'],'numerical_device_override':True},
            bp.parent,registry,folder,initializer=initialize,action_provider=provider)
        c['name']='ACTUAL_BASELINE';actual.append(c)
    for pin in sources:verify_file_pin(pin)
    task_sources=[]
    for name,module in list(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if name.startswith('dex_supermimic') and path:
            if not Path(path).resolve().is_relative_to(ROOT/'src'):raise ValueError('Unexpected historical task import')
            task_sources.append(file_pin(path))
    write(a.output/'actual_report.json',dict(**intent,cases=actual,task_sources=task_sources))
    write(a.output/'responses_report.json',dict(**{**intent,'schema':'pen_same_source_response_cases:1'},cases=responses,task_sources=task_sources))
    result=dict(**intent,actual=file_pin(a.output/'actual_report.json'),responses=file_pin(a.output/'responses_report.json'),source_captures=captures,
        actual_cases=len(actual),response_cases=len(responses),diagnostic_physics_ticks=sum(round(.02/c['dt']) for c in responses),
        passed=sum(sum(c['checks'].values()) for c in captures),total=sum(len(c['checks']) for c in captures))
    write(a.output/'report.json',result);print(json.dumps(dict(report=file_pin(a.output/'report.json'),responses=len(responses),passed=result['passed'],total=result['total'])))


if __name__=='__main__':
    import torch
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    with torch.jit.optimized_execution(False):main(p.parse_args())
