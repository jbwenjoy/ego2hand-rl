"""Fresh guarded W1 response bank for two loaded fingers; owned holdout by policy."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from mjlab.sim.sim import Simulation
from benchmark_pen_scaled import ROOT, COLLISION, OBJECTIVE, factory
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources
from supermimic_research.learning.native_pen_ppo import model_pair
from supermimic_research.runtime.explicit_collision_backend import backend_scope
from supermimic_research.runtime.pen_stable_friction_profile_v2 import CostProfile, execution_identity
from supermimic_research.runtime import owned_pen_state as native
from supermimic_research.runtime.pen_owned_probe_recorder import OwnedProbeRecorder,save_physics
from supermimic_research.runtime.source_validation_epoch import SourceValidationEpoch
from supermimic_research.runtime.pen_frozen_cost_rollout_v2 import require_structure
from supermimic_research.mechanics.pen_contact_groups import normal_group
from supermimic_research.mechanics.controlled_pen_records import pack_case
from supermimic_research.mechanics.warp_packet_decode import decode_saved_case
from supermimic_research.strict_json import write_json

PHASES=[160,180,192,200,206,210,214,218,222,226]
ARMS={'TRAIN':'pen_scaled_train_v1','HOLDOUT':'pen_scaled_support_matched_v1'}


def probes(selected):
    out=[('sham',np.zeros(20))]
    for index in selected:
        delta=np.zeros(20);delta[index]=.04
        out.extend([(f'j{index}_minus',-delta),(f'j{index}_plus',delta)])
    mixed=np.zeros(20);mixed[selected]=.04*np.array([1,-1,-1,1,-1,1,1,-1])/np.sqrt(8)
    return out+[('mixed_minus',-mixed),('mixed_plus',mixed),('sham_repeat',np.zeros(20))]


def collect(runtime,actor,role,folder,receipt,activate):
    folder.mkdir()
    sim=runtime.env.sim;m=sim.mj_model;require_structure(m)
    physics=save_physics(runtime,folder)
    model=mujoco.MjModel.from_binary_path(str(folder/'model.mjb'))
    with np.load(folder/'physical_context.npz') as p:physical=dict(p)
    f=runtime.features;body=f.body;qa=f.qa;da=int(m.jnt_dofadr[m.body_jntadr[body]])
    qids=f.qids.cpu().tolist();dofs=f.dofs.cpu().tolist();aids=runtime.action._position_ctrl_ids.cpu().tolist()
    layout=dict(object_body=body,object_qposadr=qa,object_dofadr=da,COM_BODY=m.body_ipos[body].tolist())
    case=dict(dt=.0025,model_layout=layout)
    selected=[i for i,j in enumerate(aids) if normal_group(m.joint(int(m.actuator_trnid[j,0])).name) in (1,2)]
    if len(selected)!=8:raise ValueError('Wrong loaded finger action axes')
    sources=[]
    for phase in range(1,max(PHASES)+1):
        assert int(runtime.motion.time_steps[0])==phase
        raw=actor.distribution(runtime.observation()['policy']).mean
        if phase in PHASES:
            path=folder/f'source_{phase}.pt'
            torch.save(native.capture(runtime,receipt),path)
            sources.append((phase,file_pin(path),raw.detach().cpu().numpy().copy()))
        if phase==max(PHASES): break
        _,_,term,trunc,_=runtime.step(raw)
        if (term|trunc).any():raise ValueError(f'Prefix ended at {phase}')
    recorder=OwnedProbeRecorder(runtime);activate(recorder);rows=[]
    for phase,source,raw in sources:
        state=torch.load(verify_file_pin(source),map_location='cpu',weights_only=False)
        with SourceValidationEpoch(state['sources'],state['engine_sources']) as epoch:
            for code,delta in probes(selected):
                audit=native.restore(runtime,state,source_epoch=epoch)
                if not (audit['all_array_bytes_equal'] and audit['all_scalar_values_equal']):raise ValueError('Restore mismatch')
                cmd=(raw+delta[None]).astype(np.float32)
                recorder.begin(cmd[0]);_,_,term,trunc,_=runtime.step(torch.from_numpy(cmd).to(runtime.device));z=recorder.finish()
                metrics,decoded=decode_saved_case(case,z,model,physical)
                record=pack_case(model,case,z,decoded,qids,dofs,aids,z['initial_warmstart'],physics)
                ok=bool(record['valid'][0] and all(metrics['checks'].values()) and np.all(record['geometry_flags'][0]<=[.002,.0002,.001]))
                name=f'P{phase:03}_{code}'; target=folder/(name+'.npz')
                np.savez_compressed(target,**record,base_raw=raw,requested_raw=cmd,pd=z['post_ctrl'][0,aids],delta=delta)
                rows.append(dict(role=role,phase=phase,name=code,source=source,packet=file_pin(target),valid=ok,decoder=metrics,
                    terminal=bool(term[0] or trunc[0]),clipped=bool(np.any(abs(cmd)>1))))
        write_json(folder/'progress.json',dict(last_phase=phase,rows=len(rows),valid=sum(r['valid'] for r in rows)))
    activate(None)
    write_json(folder/'report.json',dict(role=role,selected=selected,rows=rows))


def main(out):
    out.mkdir(exist_ok=False)
    manifest=factory()
    inputs={role:file_pin(ROOT/'runs'/run/'actor_02781.pt') for role,run in ARMS.items()}
    write_json(out/'intent.json',dict(arms=inputs,phases=PHASES,raw_basis_step=.04,control_dt=.02,shams_per_phase=2,
        training='Residual fresh rollout, basis responses only',validation='Mixed action excluded from fitting, support-policy sources fully held out',
        scope='Single reference. Fresh no-aid state-action response data; earlier final-evaluation packets excluded.'))
    receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'tools/benchmark_pen_scaled.py',OBJECTIVE,COLLISION],out/'snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write_json(out/'source_receipt.json',receipt)
    original=Simulation.step;recorder=None;n=0
    def activate(r):
        nonlocal recorder
        recorder=r
    def step(sim):
        nonlocal n
        if recorder is not None and recorder.enabled:recorder.before_step()
        value=original(sim);n+=sim.num_envs
        if recorder is not None and recorder.enabled:recorder.after_step()
        return value
    Simulation.step=step
    try:
        with torch.no_grad(),torch.jit.optimized_execution(False),backend_scope(COLLISION) as collision:
            identity=execution_identity('STABLE_SAME_REGIME_FRICTION_OR',collision.contract['backend_id'],manifest)
            for role,pin in inputs.items():
                with CostProfile(identity,manifest) as profile:
                    runtime=profile.create_runtime(OBJECTIVE,1)
                    try:
                        runtime.set_fixed_frame_features(False);runtime.support_weight=.2 if role=='HOLDOUT' else 0.
                        state=torch.load(verify_file_pin(pin),map_location='cpu',weights_only=False)
                        actor,_,_=model_pair(runtime.device,SimpleNamespace(**state['config']));actor.load_state_dict(state['actor']);actor.eval()
                        collect(runtime,actor,role,out/role,receipt,activate)
                    finally:activate(None);runtime.close()
        expected=2*(225+20*len(PHASES))*8
        if n!=expected:raise ValueError(f'Unexpected {n} world steps vs {expected}')
        write_json(out/'report.json',dict(native=n,inputs=inputs,arms={k:file_pin(out/k/'report.json') for k in inputs}))
    except BaseException as e:
        write_json(out/'failure.json',dict(error=repr(e),native=n));raise
    finally:Simulation.step=original
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
