"""Run frozen policies on the first dynamically screened development start."""
from pathlib import Path
import json,re,sys,tomllib
from contextlib import nullcontext
from types import SimpleNamespace
import numpy as np
import mujoco
import torch
import collect_guarded_finger_response as b
from supermimic_research.runtime.pen_clip_start import clip_start_constructor
from supermimic_research.runtime.pen_frozen_cost_rollout_v2 import reference,augmented_trace
from supermimic_research.runtime.controlled_evaluation import summarize_trace
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json


def read(pin):return json.loads(verify_file_pin(pin).read_bytes())


def config(out,name,preload=None):
    root=b.ROOT;pre=json.loads((preload or root/'runs/pen_dev_static_preload_v1/report.json').read_bytes())
    proj=read(read(pre['intent'])['source']);screen=read(read(proj['intent'])['screen']);item=screen['results'][name]
    pose=next(r for r in proj['rows'] if r['clip']==name)['variants']['projection']['trace']
    ctrl=next(r for r in pre['rows'] if r['clip']==name)['holds']['preloaded']['trace']
    def absolute_cfg(path,changes):
        text=path.read_text()
        for k,v in changes.items():
            text,n=re.subn('(?m)^'+k+' = ".*"$',lambda m:k+' = '+json.dumps(Path(v).resolve().as_posix()),text)
            if n!=1:raise ValueError(k)
        dest=out/path.name;dest.write_text(text);return dest
    recipe=out/'recipe.json';write_json(recipe,dict(schema='wuji_pen_spin_recipe:1',clips=['01_single_axis/'+name]))
    orig=root/'configs/controlled_evaluation_v4.toml';orig_cfg=tomllib.loads(orig.read_text())
    base=absolute_cfg(orig,dict(pen_recipe=recipe,**{k:orig.parent/orig_cfg[k] for k in ('clips_root','artifact_manifest','cube_checkpoint','pen_checkpoint')}))
    ap=absolute_cfg(root/'configs/pen_full_pose_native_v1.toml',dict(base_config=base))
    txt=ap.read_text()
    for k,pin in [('clip_pin',item['source']),('fixed_reference_pin',item['series']),('projected_pin',pose),('control_pin',ctrl)]:
        txt+='\n['+k+']\n'+'\n'.join(n+' = '+json.dumps(v) for n,v in pin.items())+'\n'
    ap.write_text('frame_count = '+str(item['frames'])+'\n'+txt)
    obj=absolute_cfg(b.OBJECTIVE,dict(adapter_config=ap))
    return obj,item


def main(out):
    out.mkdir(exist_ok=False)
    obj,item=config(out,'clip_003')
    roots=b.ROOT/'runs'
    actors={k:file_pin(roots/folder/'actor_02781.pt') for k,folder in [('residual','pen_scaled_train_v1'),('support','pen_scaled_support_matched_v1')]}
    write_json(out/'intent.json',dict(reference=item['source'],actors=actors,repetitions=3,scope='Fixed-preload dev clip 003. Each fresh policy starts at phase 1; truncate on native termination or original strict physical audit failure.'))
    return evaluate_cases(out,obj,item,actors,{'residual':0.,'support':.2},3)


def evaluate_cases(out,obj,item,actors,environment_weights,repetitions,training_labels=None,initial_histories=None,*,custom_start=True,refit_export=False):
    """Shared clip-bound native loop. Actor names preserve the registered run order."""
    if repetitions < 1 or set(actors) != set(environment_weights):
        raise ValueError('Specify the native support weight for each Actor')
    if training_labels is not None and set(actors) != set(training_labels):
        raise ValueError('Missing imagined training arm metadata')
    rec=None;ticks=0;original=b.Simulation.step
    def step(sim):
        nonlocal ticks
        if rec is not None and rec.enabled:rec.before_step()
        value=original(sim);ticks+=sim.num_envs
        if rec is not None and rec.enabled:rec.after_step()
        return value
    b.Simulation.step=step;rows=[]
    try:
        with torch.no_grad(),torch.jit.optimized_execution(False),b.backend_scope(b.COLLISION) as col:
            manifest=b.factory();ident=b.execution_identity('STABLE_SAME_REGIME_FRICTION_OR',col.contract['backend_id'],manifest)
            for rep in range(repetitions):
                for arm in actors:
                    folder=out/f'{arm}_{rep}';folder.mkdir()
                    with b.CostProfile(ident,manifest) as profile:
                        with (clip_start_constructor() if custom_start else nullcontext()):rt=profile.create_runtime(obj,1)
                        try:
                            rt.set_fixed_frame_features(False);rt.support_weight=environment_weights[arm]
                            ref=reference(rt)
                            # Older reference helper hardcodes the original clip clock; use actual catalog frame times.
                            ref['time_s']=np.arange(item['frames'])*.02
                            with np.load(verify_file_pin(item['series'])) as fixed:
                                np.testing.assert_allclose(ref['COM_world_m'],fixed['COM'],atol=1e-6,rtol=0)
                                assert np.min(abs((ref['quaternion_wxyz']*fixed['quat_wxyz']).sum(1)))>1-1e-6
                            st=torch.load(verify_file_pin(actors[arm]),map_location='cpu',weights_only=False)
                            actor,_,_=b.model_pair(rt.device,SimpleNamespace(**st['config']));actor.load_state_dict(st['actor']);actor.eval()
                            if refit_export:
                                from supermimic_research.runtime.pen_cpu_model_export import export_pen_cpu_model
                                bundle=export_pen_cpu_model(rt,folder/'cpu_export')
                                ph=bundle['physics'];model_file=verify_file_pin(bundle['model']);physical_file=verify_file_pin(bundle['physical_context'])
                            else:
                                ph=b.save_physics(rt,folder);model_file=folder/'model.mjb';physical_file=folder/'physical_context.npz'
                            m=mujoco.MjModel.from_binary_path(str(model_file));b.require_structure(m)
                            with np.load(physical_file) as f:phys=dict(f)
                            body=rt.features.body;layout=dict(object_body=body,object_qposadr=rt.features.qa,object_dofadr=int(m.jnt_dofadr[m.body_jntadr[body]]),COM_BODY=m.body_ipos[body].tolist())
                            case=dict(dt=.0025,model_layout=layout,aids=rt.action._position_ctrl_ids.cpu().tolist(),qids=rt.features.qids.cpu().tolist(),dofs=rt.features.dofs.cpu().tolist())
                            rec=b.OwnedProbeRecorder(rt);parts=[];stop=None;checks=[]
                            for phase in range(1,item['frames']):
                                assert int(rt.motion.time_steps[0])==phase
                                obs=rt.observation()['policy']
                                if phase==1:
                                    initial_obs=obs[0].detach().cpu().numpy().copy()
                                    if initial_histories is not None:
                                        obs=obs.clone()
                                        obs[0,-20:]=torch.as_tensor(initial_histories[arm],device=obs.device,dtype=obs.dtype)
                                    first_obs=obs[0].detach().cpu().numpy().copy()
                                raw=actor.distribution(obs).mean;rec.begin(raw[0].cpu().numpy())
                                if phase==1:first_raw=raw[0].detach().cpu().numpy().copy()
                                _,_,term,trunc,info=rt.step(raw);z=rec.finish()
                                if phase==1:first_pd=z['post_ctrl'][0,case['aids']].copy()
                                metrics,decoded=b.decode_saved_case(case,z,m,phys);pkt=b.pack_case(m,case,z,decoded,case['qids'],case['dofs'],case['aids'],z['initial_warmstart'],ph)
                                ok=bool(all(metrics['checks'].values()) and pkt['valid'][0] and np.all(pkt['geometry_flags'][0]<=[.002,.0002,.001]))
                                checks.append(dict(ok=ok,geometry=pkt['geometry_flags'][0].tolist(),metrics=metrics))
                                parts.append(augmented_trace(z,m,case))
                                if not ok:stop='PHYSICAL_AUDIT_STOP'
                                elif term[0]:stop='TASK_FAILURE'
                                elif trunc[0]:stop='REFERENCE_COMPLETION'
                                if stop:break
                            if stop is None:raise ValueError('Missing reference end')
                            trace={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
                            summary=summarize_trace(trace,ref,.0025,stop=='REFERENCE_COMPLETION' and phase==item['frames']-1,tomllib.loads((out/'controlled_evaluation_v4.toml').read_text()))
                            np.savez_compressed(folder/'trace.npz',**trace,**ref)
                            row=dict(arm=arm,rep=rep,controls=phase,stop=stop,reasons={k:bool(v[0]) for k,v in info['termination_reasons'].items()},valid_s=.02*(phase-int(not ok)),summary=summary)
                            if training_labels is not None:
                                row.update(training=training_labels[arm],environment_support_weight=rt.support_weight,
                                           actor=actors[arm],strict_success=bool(summary['qualified']))
                            if initial_histories is not None:
                                initial_file=folder/'first_action.npz'
                                np.savez_compressed(initial_file,native_obs=initial_obs,actor_obs=first_obs,raw=first_raw,transmitted_pd=first_pd)
                                row.update(first_action=file_pin(initial_file))
                            write_json(folder/'report.json',dict(**row,checks=checks,trace=file_pin(folder/'trace.npz')))
                            rows.append(dict(**row,case=file_pin(folder/'report.json')))
                        finally:rec=None;rt.close()
        assert ticks==8*sum(r['controls'] for r in rows)
        write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=rows,native=ticks))
        print([(r['arm'],r['valid_s'],r['stop'],r['reasons']) for r in rows])
        return rows
    finally:b.Simulation.step=original
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
