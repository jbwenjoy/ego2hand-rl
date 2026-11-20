"""Prospective closed-loop test of the five-candidate capacity selector on a frozen Actor."""
from pathlib import Path
from types import SimpleNamespace
import json,sys,tomllib
import numpy as np
import torch
import mujoco
import mujoco_warp as mw
from check_pen_paired_capacity_suggestions import setup,proposals,load_capacity_head,old,b
from check_pen_capacity_geometry_oracle import geometry,balanced_directions
from supermimic_research.runtime.pen_frozen_cost_rollout_v2 import reference,augmented_trace
from supermimic_research.runtime.controlled_evaluation import summarize_trace
from supermimic_research.mechanics.pen_capacity_trace import packet_capacity
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json


def cpu_scores(m,state,actions,body,dirs,target):
    scores={}
    for name,act in actions.items():
        d=mujoco.MjData(m);mujoco.mj_setState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION)
        d.ctrl[:]=np.asarray(act['pd'],np.float32);assert not d.xfrc_applied.any() and not d.qfrc_applied.any()
        view=mujoco.MjData(m);view.mocap_pos[:]=d.mocap_pos;view.mocap_quat[:]=d.mocap_quat
        caps=[]
        for _ in range(8):
            caps.append(geometry(m,view,d.qpos,body,dirs));mujoco.mj_step(m,d)
        scores[name]=float(np.abs(np.mean(caps,axis=0)-target).mean())
    return scores


def main(out,pose_fit=None,secant_fit=None):
    root=b.ROOT/'runs';out.mkdir(exist_ok=False)
    read=lambda p:json.loads(verify_file_pin(p).read_bytes())
    fit=file_pin(root/'pen_capacity_pair_fit_v1/report.json')
    nets={k:load_capacity_head(torch.load(verify_file_pin(read(fit)['variants'][k]['model']),map_location='cpu',weights_only=True),dtype=torch.float64,device='cpu').eval().requires_grad_(False) for k in ('absolute','response')}
    source=file_pin(root/'pen_paired_imagination_v1/A1_7413.pt')
    target=read(file_pin(root/'pen_capacity_targets_v1/report.json'))['refs']['clip_001']['series']
    with np.load(verify_file_pin(target)) as z:targets=z['capacity'];eligible=z['eligible']
    dirs=balanced_directions();order=['BASE','LEARNED','CPU','CPU','BASE','LEARNED','LEARNED','CPU','BASE']
    pose_nets={};pose_pins={}
    if pose_fit is not None:
        from supermimic_research.mechanics.pen_substep_pose import load_substep_pose
        pose_report=read(file_pin(pose_fit/'report.json'))
        for key in ('absolute','paired'):
            pose_pin=pose_report['models'][key]['checkpoint'];pose_pins[key]=pose_pin
            pose_nets[key.upper()]=load_substep_pose(torch.load(verify_file_pin(pose_pin),map_location='cpu',weights_only=False),dtype=torch.float64).eval()
        order=['BASE','ABSOLUTE','PAIRED','PAIRED','BASE','ABSOLUTE','ABSOLUTE','PAIRED','BASE']
    if secant_fit is not None:
        from supermimic_research.mechanics.pen_substep_secant import load_local_secant
        assert pose_fit is not None
        pin=read(file_pin(secant_fit/'report.json'))['checkpoint'];pose_pins={'local_secant':pin}
        pose_nets={'SECANT':load_local_secant(torch.load(verify_file_pin(pin),map_location='cpu',weights_only=True))}
        order=['BASE','SECANT','SECANT','BASE','BASE','SECANT']
    write_json(out/'intent.json',dict(pose_models=pose_pins,pose_modes='ABSOLUTE/PAIRED score the same five candidates using predicted substep poses plus CPU collision geometry. Separate exploratory native trial on a previously inspected behavior.',actor=source,fit=fit,target=target,schedule=order,phase_range=[206,222],
        arms='BASE original, LEARNED lowest response-head cost across all 5 candidates, CPU lowest predicted cost across same 5 candidates using CPU dynamics from observed initial state. All choices before physical execution. No learned parameters updated. Inside qualified geometry windows only.',endpoints='Mean actual 8-tick capacity L1 in intervention region, strict full duration and orientation, score only matched qualified intervals across arms.'))
    st=torch.load(verify_file_pin(source),map_location='cpu',weights_only=False);manifest=b.factory()
    original=b.Simulation.step;rec=None;ticks=0;rows=[]
    def step(sim):
        nonlocal ticks
        if rec is not None and rec.enabled:rec.before_step()
        val=original(sim);ticks+=sim.num_envs
        if rec is not None and rec.enabled:rec.after_step()
        return val
    b.Simulation.step=step
    try:
        with torch.no_grad(),torch.jit.optimized_execution(False),b.backend_scope(b.COLLISION) as col:
            ident=b.execution_identity('STABLE_SAME_REGIME_FRICTION_OR',col.contract['backend_id'],manifest)
            for trial,mode in enumerate(order):
                folder=out/f'{trial}_{mode}';folder.mkdir()
                with b.CostProfile(ident,manifest) as profile:
                    rt=profile.create_runtime(b.OBJECTIVE,1)
                    try:
                        rt.set_fixed_frame_features(False);rt.support_weight=0.
                        assert st['batch_backend']['collision_backend']==col.contract['backend_id'] and st['batch_backend']['factory']==manifest['generated']
                        actor,_,_=b.model_pair(rt.device,SimpleNamespace(**st['config']));actor.load_state_dict(st['actor']);actor.eval()
                        ph=b.save_physics(rt,folder);m=mujoco.MjModel.from_binary_path(str(folder/'model.mjb'));b.require_structure(m)
                        with np.load(folder/'physical_context.npz') as f:phys=dict(f)
                        c=setup(rt);rec=b.OwnedProbeRecorder(rt);records=[];traces=[];stop=None;query_ticks=0;query_geometry=0
                        for p in range(1,369):
                            assert int(rt.motion.time_steps[0])==p
                            raw=actor.distribution(rt.observation()['policy']).mean;key='BASE';scores={};options={}
                            targeted=bool(206<=p<=222 and eligible[p-1])
                            if targeted:
                                options,_=proposals(rt,nets,raw,c,old.tensor(targets[p-1]))
                                if mode=='LEARNED':scores={n:a['predicted']['response'] for n,a in options.items()}
                                elif mode=='CPU':
                                    sig=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(m,sig))
                                    mw.get_data_into(rec.host,rec.model,rt.env.sim.wp_data);mujoco.mj_getState(rec.model,rec.host,state,sig)
                                    scores=cpu_scores(m,state,options,c['model_layout']['object_body'],dirs,targets[p-1]);query_ticks+=8*len(scores)
                                elif mode in pose_nets:
                                    data=rt.env.sim.wp_data;layout=c['model_layout']
                                    state=old.native_state55(data.qpos.numpy()[0],data.qvel.numpy()[0],c['qids'],c['dofs'],layout['object_qposadr'],layout['object_dofadr'])
                                    pd=torch.as_tensor(np.array([a['pd'] for a in options.values()]),dtype=torch.float64)
                                    poses=pose_nets[mode].predict_qpos(old.tensor(state).expand(len(pd),-1),old.tensor(data.qacc_warmstart.numpy()[0]).expand(len(pd),-1),pd).numpy()
                                    assert poses.shape==(len(pd),8,27) and np.isfinite(poses).all()
                                    geo=mujoco.MjData(m);geo.mocap_pos[:]=data.mocap_pos.numpy()[0];geo.mocap_quat[:]=data.mocap_quat.numpy()[0]
                                    for label,trajectory in zip(options,poses):
                                        capacities=[geometry(m,geo,q,layout['object_body'],dirs) for q in trajectory]
                                        scores[label]=float(np.mean(np.abs(np.mean(capacities,0)-targets[p-1])))
                                    query_geometry+=poses.shape[0]*8
                                if scores:key=min(scores,key=scores.get)
                                raw=torch.tensor(options[key]['raw'],device=rt.device,dtype=raw.dtype)[None]
                            rec.begin(raw[0].cpu().numpy());_,_,term,trunc,info=rt.step(raw);z=rec.finish()
                            metrics,dec=b.decode_saved_case(c,z,m,phys);pkt=b.pack_case(m,c,z,dec,c['qids'],c['dofs'],c['aids'],z['initial_warmstart'],ph)
                            valid=bool(all(metrics['checks'].values()) and pkt['valid'][0] and np.all(pkt['geometry_flags'][0]<=[.002,.0002,.001]))
                            err=None
                            if targeted:
                                np.testing.assert_allclose(z['post_ctrl'][:,c['aids']],np.broadcast_to(options[key]['pd'],(8,20)),atol=2e-7,rtol=0)
                                cap=packet_capacity(z,m,c['model_layout'],phys['geom_friction'])
                                err=float(np.abs(cap['capacity'].mean(0)-targets[p-1]).mean())
                                path=folder/f'raw_{p}.npz';np.savez_compressed(path,**z)
                            records.append(dict(phase=p,selected=key,scores=scores,mismatch=err,valid=valid,metrics=metrics,geometry=pkt['geometry_flags'][0].tolist()))
                            traces.append(augmented_trace(z,m,c))
                            if not valid:stop='PHYSICAL_AUDIT_STOP'
                            elif term[0]:stop='TASK_FAILURE'
                            elif trunc[0]:stop='REFERENCE_COMPLETION'
                            if stop:break
                        assert stop is not None
                        trace={k:np.concatenate([a[k] for a in traces]) for k in traces[0]};ref=reference(rt)
                        summary=summarize_trace(trace,ref,.0025,stop=='REFERENCE_COMPLETION' and p==368,tomllib.loads((b.ROOT/'configs/controlled_evaluation_v4.toml').read_text()))
                        np.savez_compressed(folder/'trace.npz',**trace,**ref)
                        write_json(folder/'report.json',dict(records=records,stop=stop,summary=summary,trace=file_pin(folder/'trace.npz')))
                        valid_s=.02*(p-int(not valid));scored=[a['mismatch'] for a in records if a['valid'] and a['mismatch'] is not None]
                        rows.append(dict(mode=mode,case=file_pin(folder/'report.json'),valid_s=valid_s,stop=stop,success=bool(summary['qualified']),phase_count=len(scored),
                            local_capacity_error=float(np.mean(scored)) if scored else None,rollout_ticks=8*p,planning_ticks=query_ticks,predicted_pose_geometry_queries=query_geometry,changed=sum(a['selected']!='BASE' for a in records)))
                        print(mode,valid_s,rows[-1]['local_capacity_error'],flush=True);write_json(out/'progress.json',dict(rows=rows))
                    finally:rec=None;rt.close()
        assert ticks==sum(r['rollout_ticks'] for r in rows)
        write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),native=ticks,rows=rows))
    finally:b.Simulation.step=original
if __name__=='__main__':main(*map(lambda s:Path(s).resolve(),sys.argv[1:]))

