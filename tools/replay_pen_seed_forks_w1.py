"""Verify two candidate pools on restored complete W1 source states at the identified split."""
import argparse
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.stats import qmc
from audit_pen_seed_action_forks import select
from evaluate_pen_dev_clip import b,clip_start_constructor,file_pin,verify_file_pin,read,write_json
from supermimic_research.mechanics.pen_task_contracts import pose_components
from supermimic_research.runtime.pen_cpu_model_export import export_pen_cpu_model


def run(out):
    root=b.ROOT/'runs'
    fork=file_pin(root/'pen_seed_action_forks_v1/report.json')
    identified=read(fork)['results']
    source=file_pin(root/'pen_receding_seeds_v1/report.json')
    previous=read(source)
    intent=read(read(read(previous['intent'])['source'])['intent'])
    config=read(read(intent['baseline'])['intent'])
    cfg=read(config['configs'])['clips']['clip_003']
    with np.load(verify_file_pin(cfg['item']['source'])) as f:q=f['reference_qpos'].astype(np.float32)
    with np.load(verify_file_pin(cfg['item']['series'])) as f:ref=dict(f)
    rot=Rotation.from_quat(ref['quat_wxyz'][:,[1,2,3,0]])
    omega=(rot[1:]*rot[:-1].inv()).as_rotvec()/.02
    seeds=(112026,112027)
    dirs={s:(2*qmc.Sobol(20,scramble=True,seed=s).random_base2(int(np.ceil(np.log2(len(q)*8))))[:8*len(q)]-1).reshape(-1,8,20) for s in seeds}
    out.mkdir(exist_ok=False)
    files=sorted((b.ROOT/'src').rglob('*.py'))+[Path(__file__),Path(__file__).with_name('audit_pen_seed_action_forks.py')]
    receipt=b.freeze_sources(files,out/'snapshot',b.ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=b.ROOT,text=True).strip())
    write_json(out/'source_receipt.json',receipt)
    write_json(out/'intent.json',dict(sources=source,phase_evidence=fork,seeds=seeds,horizon=8,
        scope='Re-execute archived RAW prefix under native audit. Capture exact live runtime state and re-rank both pools on that state. Restore before each of the two treatments and a repeat of the source treatment; use the same archived subsequent RAW for seven more controls while updating each branch EMA. Stop on any native task or physical violation.'))
    manifest=b.factory()
    rows=[]
    original=b.Simulation.step
    active=None
    def step(sim):
        if active is not None and active.enabled:active.before_step()
        val=original(sim)
        if active is not None and active.enabled:active.after_step()
        return val
    with torch.no_grad(), torch.jit.optimized_execution(False),patch.object(b.Simulation,'step',step),b.backend_scope(b.COLLISION) as col:
        identity=b.execution_identity('STABLE_SAME_REGIME_FRICTION_OR',col.contract['backend_id'],manifest)
        for witness in identified:
            seed=witness['source_seed'];phase=witness['phase']
            saved=next(r['case'] for r in previous['rows'] if r['mode']=='pose_support' and r['clip']=='clip_003' and r['seed']==seed)
            with np.load(verify_file_pin(saved['inputs'])) as f:archive=dict(f)
            folder=out/str(seed)
            folder.mkdir()
            with b.CostProfile(identity,manifest) as profile:
                with clip_start_constructor():rt=profile.create_runtime(verify_file_pin(cfg['objective']),1)
                try:
                    rt.set_fixed_frame_features(False);rt.support_weight=0.
                    state=torch.load(verify_file_pin(config['actor']),map_location='cpu',weights_only=False)
                    actor,_,_=b.model_pair(rt.device,SimpleNamespace(**state['config']))
                    actor.load_state_dict(state['actor']);actor.eval()
                    export=export_pen_cpu_model(rt,folder/'cpu')
                    assert export['model']['sha256']==config['anchors']['model']['sha256']
                    m=mujoco.MjModel.from_binary_path(str(verify_file_pin(export['model'])))
                    with np.load(verify_file_pin(export['physical_context'])) as f:phys=dict(f)
                    body=rt.features.body;qa=rt.features.qa
                    layout=dict(object_body=body,object_qposadr=qa,object_dofadr=0,COM_BODY=m.body_ipos[body].tolist())
                    case=dict(dt=.0025,model_layout=layout,aids=rt.action._position_ctrl_ids.cpu().tolist(),
                              qids=rt.features.qids.cpu().tolist(),dofs=rt.features.dofs.cpu().tolist())
                    active=b.OwnedProbeRecorder(rt)
                    def apply(cmd,p):
                        assert int(rt.motion.time_steps[0])==p
                        active.begin(cmd)
                        _,_,term,trunc,_=rt.step(torch.as_tensor(cmd,device=rt.device)[None])
                        z=active.finish()
                        checks,decoded=b.decode_saved_case(case,z,m,phys)
                        pkt=b.pack_case(m,case,z,decoded,case['qids'],case['dofs'],case['aids'],z['initial_warmstart'],export['physics'])
                        ok=bool(pkt['valid'][0] and all(checks['checks'].values()) and np.all(pkt['geometry_flags'][0]<=[.002,.0002,.001]) and not term[0] and not trunc[0])
                        assert not z['post_xfrc_applied'].any() and not z['post_qfrc_applied'].any()
                        tip=z['post_qpos'][-1]
                        ang=pose_components(tip[3:7],ref['quat_wxyz'][p])
                        loads=np.zeros(6)
                        for r in z['raw_contact_tick_plus23']:
                            g,h=map(int,r[1:3]);on=m.geom_bodyid[[g,h]]==body
                            if on.sum()==1:
                                digit=b.normal_group(m.geom(h if on[0] else g).name)
                                if digit>=0:loads[digit]+=max(0.,r[18])/8
                        return dict(phase=p,passed=ok,COM_mm=1000*np.linalg.norm(tip[:3]-ref['COM'][p]),
                            shaft_deg=np.rad2deg(ang['shaft_angle_rad']),twist_deg=np.rad2deg(ang['reference_axis_twist_rad']),
                            peak_mm=1000*z['pre_penetration_m'].max(),normal_digit12345_palm_N=loads.tolist(),
                            qpos=tip.tolist(),pd=z['post_ctrl'][0].tolist())
                    for p in range(1,phase):
                        if not apply(archive['selected_raw'][p-1],p)['passed']:
                            raise ValueError('Prefix failed before measured split')
                    snapshot=b.native.capture(rt,receipt)
                    a=rt.action;d=rt.env.sim.wp_data
                    def host(x):return x.detach().cpu().numpy().copy()
                    lo,hi=a.target_position_limits
                    x={k:getattr(d,k).numpy()[0].copy() for k in ('qpos','qvel','mocap_pos','mocap_quat')}
                    x.update(warmstart=d.qacc_warmstart.numpy()[0].copy(),actor_raw=host(actor.distribution(rt.observation()['policy']).mean[0]),
                        previous_raw=host(a._raw_actions[0]),previous_ema=host(a._processed_actions[0]),
                        encoder_bias=host(a._entity.data.encoder_bias[0,a._target_ids]),lower=host(lo[0]),upper=host(hi[0]))
                    indexed={k:np.broadcast_to(v,(phase,*v.shape)) for k,v in x.items()}
                    proposals={s:select(m,indexed,phase-1,ref,q,dirs[s],omega) for s in seeds}
                    with b.SourceValidationEpoch(snapshot['sources'],snapshot['engine_sources']) as epoch:
                        for order,treatment in enumerate([seed,*[s for s in seeds if s!=seed],seed]):
                            audit=b.native.restore(rt,snapshot,source_epoch=epoch)
                            assert audit['all_array_bytes_equal'] and audit['all_scalar_values_equal']
                            idx,raw,_=proposals[treatment];path=[]
                            for off in range(8):
                                path.append(apply(raw if off==0 else archive['selected_raw'][phase+off-1],phase+off))
                                if not path[-1]['passed']:break
                            rows.append(dict(source=seed,treatment=treatment,order=order,candidate=idx,raw=raw.tolist(),
                                exact_restore=True,path=path))
                    write_json(out/'progress.json',rows)
                finally:active=None;rt.close()
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    run(p.parse_args().out.resolve())
