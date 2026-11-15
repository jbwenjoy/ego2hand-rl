"""Native161/188 frame/control/reset checks; no optimizer or task success claim."""
import argparse
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_native_policy import NativePenPolicyAdapter
from supermimic_research.runtime.gpu_model_snapshot import capture_model
from supermimic_research.runtime.numerical_options import numerical_receipt


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def cpu(x):return x.detach().cpu().numpy().copy()


def state(adapter):
    d=adapter.env.sim.data;a=adapter.action
    return {**{k:cpu(getattr(d,k)) for k in ['qpos','qvel','ctrl','qacc_warmstart','time','mocap_pos','mocap_quat']},
        'EMA':cpu(a._processed_actions),'raw':cpu(a._raw_actions),'target':cpu(a._targets),
        'entity_position_targets':cpu(adapter.features.robot.data.joint_pos_target),
        'phase':cpu(adapter.motion.time_steps),'pending':cpu(adapter.pending)}


def feature_check(adapter,obs):
    f=adapter.features;d=adapter.env.sim.data
    pose=cpu(f.pose());mount=cpu(f.mount());ref=cpu(f.reference_pose_world())
    W=Rotation.from_quat(mount[:,[4,5,6,3]])
    def nine(p):
        R=(W.inv()*Rotation.from_quat(p[:,[4,5,6,3]])).as_matrix()
        return np.c_[.15*np.tanh(W.inv().apply(p[:,:3]-mount[:,:3])/.15),R[:,:,0],R[:,:,1]]
    p=cpu(obs['policy']);c=cpu(obs['critic']);qd=cpu(d.qvel);qpos=cpu(d.qpos)
    local=f.local_joints;data=f.robot.data
    q=qpos[:,cpu(f.qids)]+cpu(data.encoder_bias[:,local])-cpu(data.default_joint_pos[:,local])
    qv=(qd[:,cpu(f.dofs)]-cpu(data.default_joint_vel[:,local]))*.25
    reference_q=cpu(adapter.motion.joint_pos);links=cpu(adapter.motion.link_pos).reshape(adapter.num_envs,63)
    expected=np.c_[reference_q,nine(ref),links,q,qv,nine(pose),cpu(adapter.action._raw_actions)]
    m=adapter.env.sim.mj_model;da=m.jnt_dofadr[m.body_jntadr[f.body]]
    angular_world=Rotation.from_quat(pose[:,[4,5,6,3]]).apply(qd[:,da+3:da+6])
    ec=np.c_[expected,cpu(adapter.action._processed_actions),W.inv().apply(qd[:,da:da+3]),W.inv().apply(angular_world),
        cpu(adapter.motion.frame_ids)/(len(f.reference['joint_pos'])-1)]
    return dict(policy161=p.shape==(adapter.num_envs,161),critic188=c.shape==(adapter.num_envs,188),
        policy_matches_independent_numpy=bool(np.max(abs(p-expected))<=1e-6),
        critic_matches_independent_numpy=bool(np.max(abs(c-ec))<=2e-5),finite=bool(np.isfinite(p).all() and np.isfinite(c).all())),dict(
        policy_max_error=float(np.max(abs(p-expected))),critic_max_error=float(np.max(abs(c-ec))))


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or cfg['role']!='DIAGNOSTIC_NEVER_FIT':raise ValueError('Fresh diagnostic output')
    if cfg['worlds']!=[1,32] or cfg['controls_per_case']!=16 or cfg['partial_reset_before_control']!=8:raise ValueError('Declared qualification budget')
    out.mkdir()
    paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),cp]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    write(out/'source_receipt.json',receipt)
    from mjlab.sim.sim import Simulation
    import mjlab.envs.manager_based_rl_env as env_module
    import mjlab.sim.sim as sim_module
    external={k:file_pin(v.__file__) for k,v in [('environment',env_module),('simulation',sim_module)]}
    original_step=Simulation.step;ticks={};captures={};original_graph=Simulation.create_graph
    def counted_step(sim):
        ticks[id(sim)]=ticks.get(id(sim),0)+sim.num_envs
        return original_step(sim)
    def counted_graph(sim):
        captures[id(sim)]=captures.get(id(sim),0)+1
        return original_graph(sim)
    Simulation.step=counted_step;Simulation.create_graph=counted_graph
    rows=[];cases=[];all_arrays={};started=time.perf_counter()
    try:
        with torch.jit.optimized_execution(False):
            for worlds in cfg['worlds']:
                adapter=None;folder=out/f'worlds{worlds}';folder.mkdir()
                try:
                    adapter=NativePenPolicyAdapter(cp,worlds);env=adapter.env;sid=id(env.sim);setup=ticks.get(sid,0)
                    model,scalars,hashes=capture_model(env.sim.wp_model)
                    np.savez_compressed(folder/'model_arrays.npz',**model);write(folder/'model_scalars.json',scalars)
                    initial=state(adapter);ob=adapter.observation();initial_flags,initial_error=feature_check(adapter,ob)
                    resets=[];snapshots=[];pending_count=0
                    # Include an over-range sample witness, but always execute bounded raw.
                    for k in range(16):
                        if k==8:
                            ids=torch.arange(0,worlds,2,device=adapter.device);before=state(adapter)
                            ob=adapter.reset(ids);after=state(adapter);untouched=np.setdiff1d(np.arange(worlds),cpu(ids))
                            flags={key+'_other_worlds_unchanged':np.array_equal(before[key][untouched],after[key][untouched]) for key in before}
                            flags.update({key+'_reset_source':np.array_equal(after[key][cpu(ids)],initial[key][cpu(ids)]) for key in initial})
                            np.savez_compressed(folder/'partial_reset.npz',**{'before__'+key:value for key,value in before.items()},
                                **{'after__'+key:value for key,value in after.items()},reset_ids=cpu(ids))
                            resets.append(dict(k=k,ids=cpu(ids).tolist(),checks={key:bool(v) for key,v in flags.items()}))
                        previous=state(adapter)
                        w=torch.arange(worlds,device=adapter.device)[:,None];j=torch.arange(20,device=adapter.device)[None,:]
                        sample=cfg['scripted_amplitude']*torch.sin(.2*k+.13*w+.03*j)
                        if k==2:sample[0,0]=1.5
                        ob,reward,term,trunc,info=adapter.step(sample)
                        after=state(adapter);p=cpu(info['applied_raw']);ema=.5*previous['EMA']+.25*p
                        refq=cpu(info['reference_joints']);lo,hi=[cpu(v) for v in adapter.action.target_position_limits]
                        target=np.clip(refq+ema,lo,hi);fchecks,errors=feature_check(adapter,ob)
                        d=env.sim.data
                        flags=dict(**fchecks,phase_used=np.array_equal(cpu(info['reference_phase']),previous['phase']),
                            next_phase=np.array_equal(after['phase'],previous['phase']+1),
                            applied_raw=np.array_equal(p,np.clip(cpu(sample),-1,1)),
                            EMA=bool(np.max(abs(after['EMA']-ema))<=1e-7),
                            target_PD=bool(np.max(abs(after['target']-target))<=2e-7),
                            transmitted_ctrl=bool(np.max(abs(after['ctrl'][:,cpu(adapter.action._position_ctrl_ids)]-target))<=2e-7),
                            no_applied_force=not np.any(cpu(d.qfrc_applied)) and not np.any(cpu(d.xfrc_applied)),
                            no_overflow=not np.any(cpu(d.overflow)),
                            fixed_mocap=np.array_equal(initial['mocap_pos'],after['mocap_pos']) and np.array_equal(initial['mocap_quat'],after['mocap_quat']),
                            clock=bool(np.max(abs(after['time']-previous['time']-.02))<=2e-7),
                            finite_physics=all(np.isfinite(after[key]).all() for key in ['qpos','qvel','ctrl','qacc_warmstart','time']),
                            reward_finite=bool(torch.isfinite(reward).all()),
                            terminal_observation_before_reset=torch.equal(adapter.pending,term|trunc))
                        # Independent postphysics reward against the USED frame, not next ref.
                        pose=cpu(adapter.features.pose());rp=cpu(info['reference_pose_world'])
                        angle=(Rotation.from_quat(pose[:,[4,5,6,3]])*Rotation.from_quat(rp[:,[4,5,6,3]]).inv()).magnitude()
                        distance=np.linalg.norm(pose[:,:3]-rp[:,:3],axis=-1);rw=cfg['reward'];q=after['qpos'][:,cpu(adapter.features.qids)]
                        expected=(rw['COM_weight']*np.exp(-(distance/rw['COM_scale_m'])**2)+rw['angle_weight']*np.exp(-(angle/rw['angle_scale_rad'])**2)
                            +rw['hand_weight']*np.exp(-np.mean((q-refq)**2,axis=-1)/rw['hand_scale_rad']**2)-rw['action_weight']*np.mean(p*p,axis=-1))
                        flags['postphysics_reward_used_reference']=bool(np.max(abs(cpu(reward)-expected))<=1e-4)
                        rows.append(dict(worlds=worlds,k=k,checks={key:bool(v) for key,v in flags.items()},**errors,
                            reward_max_error=float(np.max(abs(cpu(reward)-expected))),terminated=int(term.sum()),truncated=int(trunc.sum())))
                        snapshots.append(dict(**after,policy=cpu(ob['policy']),critic=cpu(ob['critic']),reward=cpu(reward),sample=cpu(sample),applied=p,
                            used_reference_pose=rp,used_phase=cpu(info['reference_phase'])))
                        ids=(term|trunc).nonzero().flatten()
                        pending_count+=len(ids)
                        if len(ids):adapter.reset(ids)
                    for key in snapshots[0]:all_arrays[f'worlds{worlds}__{key}']=np.array([s[key] for s in snapshots])
                    final_model,final_scalars,final_hashes=capture_model(env.sim.wp_model)
                    case=dict(worlds=worlds,metadata=adapter.metadata(),initial_checks=initial_flags,initial_error=initial_error,resets=resets,
                        terminal_reset_worlds=pending_count,explicit_reset_worlds=adapter.explicit_reset_worlds,transitions=adapter.control_transitions,
                        integrations=adapter.physics_integrations,observed_total_integrations=ticks.get(sid,0),observed_setup_integrations=setup,
                        graph_capture_invocations=captures.get(sid,0),graph_scope='Captured step graph is not launched physics; forward/reset are nonintegrating.',
                        model_unchanged=hashes==final_hashes and scalars==final_scalars,
                        numerical=numerical_receipt(env.sim.mj_model,env.sim.wp_model),assets=adapter.assets,
                        model_arrays=file_pin(folder/'model_arrays.npz'),model_scalars=file_pin(folder/'model_scalars.json'))
                    cases.append(case);print(json.dumps(dict(worlds=worlds,transitions=case['transitions'],integrations=case['integrations'],terminal_resets=pending_count)),flush=True)
                finally:
                    if adapter:adapter.close()
    finally:
        Simulation.step=original_step;Simulation.create_graph=original_graph
    np.savez_compressed(out/'transitions.npz',**all_arrays)
    verify_source_snapshot(receipt,live=True)
    for pin in external.values():verify_file_pin(pin)
    checks={f'row{i}_{key}':v for i,row in enumerate(rows) for key,v in row['checks'].items()}
    for c in cases:
        prefix=f"worlds{c['worlds']}"
        checks.update({prefix+'_initial_'+key:v for key,v in c['initial_checks'].items()})
        checks[prefix+'_integration_count']=c['integrations']==16*c['worlds']*8 and c['observed_total_integrations']==c['integrations']+c['observed_setup_integrations']
        checks[prefix+'_model_unchanged']=c['model_unchanged']
        for reset in c['resets']:checks.update({prefix+'_reset_'+key:v for key,v in reset['checks'].items()})
    result=dict(schema='pen_native_full_pose_integration_qualification:1',config=file_pin(cp),source_receipt=file_pin(out/'source_receipt.json'),
        external_sources=external,cases=cases,rows=rows,checks=checks,passed=sum(checks.values()),total=len(checks),
        transitions=file_pin(out/'transitions.npz'),wall_s=time.perf_counter()-started,actual_task_optimizer_updates=0,
        actual_native_integrations=sum(c['observed_total_integrations'] for c in cases),
        scope='Diagnostic161/188 native frame/action/reward/partial-reset interface only. Explicit postphysics termination timing differs from legacy. No training, full-task success, portable physical resume or GPU speedup claim.')
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],integrations=result['actual_native_integrations'])),flush=True)
    if not all(checks.values()):raise ValueError('Retain failed contract checks; training not qualified')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
