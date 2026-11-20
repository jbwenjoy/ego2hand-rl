"""Run a frozen balanced Actor with two prospective contact filters in the native W1 task."""
import argparse
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from evaluate_pen_dev_clip import b,evaluate_cases,file_pin,read,verify_file_pin,write_json
from supermimic_research.mechanics import pen_contact_action_filter as af


ARMS = ('actor','paired','cpu_dynamics')
CLIPS = ('clip_002','clip_003')


def host(x):
    return x.detach().cpu().numpy().copy()


@contextmanager
def filter_policy(folder, model, anchors, reference, nets):
    pair, original_recorder = b.model_pair,b.OwnedProbeRecorder
    cases,live = [],{}

    def make_actor(*args,**kwargs):
        actor, critic, state = pair(*args,**kwargs)
        original = actor.distribution
        arm, rep = ARMS[len(cases)%len(ARMS)], len(cases)//len(ARMS)
        record = dict(arm=arm,rep=rep,decisions=[],inputs=[])
        cases.append(record)

        def distribution(obs):
            start = time.perf_counter()
            mean = original(obs).mean
            runtime = live['runtime']
            phase = int(runtime.motion.time_steps[0])
            if phase != len(record['inputs'])+1:
                raise ValueError('Unexpected phase or number of policy calls')
            a = runtime.action
            d = runtime.env.sim.wp_data
            previous_raw=host(a._raw_actions[0])
            np.testing.assert_array_equal(previous_raw,host(obs[0,-20:]))
            upper,lower=live['upper'],live['lower']
            inputs = dict(qpos=d.qpos.numpy()[0].copy(),qvel=d.qvel.numpy()[0].copy(),
                warmstart=d.qacc_warmstart.numpy()[0].copy(),mocap_pos=d.mocap_pos.numpy()[0].copy(),
                mocap_quat=d.mocap_quat.numpy()[0].copy(),actor_raw=host(mean[0]),previous_raw=previous_raw,
                previous_ema=host(a._processed_actions[0]),refq=host(runtime.motion.joint_pos[0]),
                encoder_bias=host(a._entity.data.encoder_bias[0,a._target_ids]),lower=lower,upper=upper)
            target_pose = host(runtime.features.reference_pose_world()[0])
            np.testing.assert_allclose(target_pose[:3],reference['COM'][phase],atol=1e-6,rtol=0)
            if abs(target_pose[3:] @ reference['quat_wxyz'][phase]) < 1-1e-6:
                raise ValueError('Wrong native reference phase')
            if arm=='actor':
                selected = inputs['actor_raw'].clip(-1,1)
                pd = np.clip(inputs['refq'] + (.5*inputs['previous_ema'] + .25*selected) - inputs['encoder_bias'],lower,upper)
                decision = dict(triggered=False,choice='actor',evaluations={},pd=pd.tolist(),planning_ticks=0,predicted_poses=0)
                elapsed = 0.
            else:
                before = time.perf_counter()
                selected,decision = live['filter'].select_action(**inputs)
                elapsed = time.perf_counter()-before
            selected=np.asarray(selected,dtype=np.float32)
            if selected.shape!=(20,) or not np.isfinite(selected).all() or np.any(abs(selected)>1):
                raise ValueError('Filter selected an inadmissible action')
            record['decisions'].append(dict(**decision,phase=phase,planning_wall_s=elapsed,
                decision_wall_s=time.perf_counter()-start))
            record['inputs'].append(dict(**inputs,obs=host(obs[0]),target_pose=target_pose,selected_raw=selected.copy()))
            return SimpleNamespace(mean=torch.as_tensor(selected,device=obs.device,dtype=obs.dtype)[None])

        actor.distribution=distribution
        return actor,critic,state

    def bind(runtime):
        record = cases[-1]
        path = folder/f"{record['arm']}_{record['rep']}"/'cpu_export/report.json'
        export=read(file_pin(path))
        for key,expected in [('model',anchors['model']),('physical_context',anchors['physics'])]:
            if (export[key]['sha256'],export[key]['bytes']) != (expected['sha256'],expected['bytes']):
                raise ValueError(f'Filter and live world differ: {key}')
        a = runtime.action
        np.testing.assert_array_equal(host(a._position_ctrl_ids), np.arange(model.nu))
        np.testing.assert_array_equal(host(runtime.features.qids), model.jnt_qposadr[model.actuator_trnid[:,0]])
        np.testing.assert_array_equal(host(runtime.features.dofs), model.jnt_dofadr[model.actuator_trnid[:,0]])
        if a._mask_enabled or float(a.cfg.alpha)!=.5 or float(a._scale)!=.5:
            raise ValueError('Unexpected action mask or EMA configuration')
        lower,upper=a.target_position_limits
        arm=record['arm']
        live.update(runtime=runtime,lower=host(lower[0]),upper=host(upper[0]))
        if arm!='actor':
            live['filter'] = af.PenContactActionFilter(model,arm,pose_checkpoint=nets if arm=='paired' else None)
        return original_recorder(runtime)

    with patch.object(b,'model_pair',make_actor),patch.object(b,'OwnedProbeRecorder',bind):
        yield cases


def summarize(folder,native,case,frames):
    if (native['arm'],native['rep']) != (case['arm'],case['rep']):
        raise ValueError('Unexpected world order')
    record=read(native['case'])
    decisions=case['decisions']
    observed=native['controls']
    if len(decisions)!=observed:
        raise ValueError('Missing decision')
    audit=folder/f"{case['arm']}_{case['rep']}"
    inputs={k:np.stack([r[k] for r in case['inputs']]) for k in case['inputs'][0]}
    applied=inputs['selected_raw']
    np.testing.assert_array_equal(inputs['previous_raw'][1:],applied[:-1])
    np.testing.assert_allclose(inputs['previous_ema'][1:],.5*inputs['previous_ema'][:-1]+.25*applied[:-1],rtol=0,atol=2e-7)
    with np.load(verify_file_pin(record['trace'])) as tr:
        actual_pd=tr['post_ctrl'].reshape(observed,8,20)
        np.testing.assert_array_equal(tr['pre_qpos'][::8],inputs['qpos'])
        np.testing.assert_array_equal(tr['pre_qvel'][::8],inputs['qvel'])
        warm_gap=float(abs(tr['post_qacc_warmstart'][7::8][:-1]-inputs['warmstart'][1:]).max(initial=0.))
        np.testing.assert_array_equal(tr['post_mocap_pos'][::8],inputs['mocap_pos'])
        np.testing.assert_array_equal(tr['post_mocap_quat'][::8],inputs['mocap_quat'])
        planned=np.asarray([v['pd'] for v in decisions])
        pd_error=abs(planned[:,None,:]-actual_pd).max(axis=(1,2))
        np.testing.assert_allclose(pd_error,0,rtol=0,atol=3e-7)
        for i,dec in enumerate(decisions):
            dec.update(actual_object_mm=1000*float(tr['pre_penetration_m'][8*i:8*i+8].max()),
                actual_self_mm=1000*float(tr['pre_self_penetration_m'][8*i:8*i+8].max()),
                max_pd_error_rad=float(pd_error[i]),native_checks_passed=record['checks'][i]['ok'])
        inputs['actual_pd_by_tick']=actual_pd
    data=audit/'filter_inputs.npz'
    np.savez_compressed(data,**inputs)
    details=audit/'filter_decisions.json'
    write_json(details,decisions)
    complete=native['stop']=='REFERENCE_COMPLETION'
    n=observed-int(not complete)
    return dict(arm=case['arm'],rep=case['rep'],controls=observed,passing=n,passing_s=.02*n,
        passed_halfway=2*n>=frames-1,complete=complete,strict=native['summary']['qualified'],stop=native['stop'],
        reasons=native['reasons'],native_case=native['case'],inputs=file_pin(data),decisions=file_pin(details),
        triggers=sum(d['triggered'] for d in decisions),choices=dict(Counter(d['choice'] for d in decisions)),
        changed=sum(bool(np.any(abs(r['selected_raw']-r['actor_raw'].clip(-1,1))>0)) for r in case['inputs']),
        cpu_planning_ticks=sum(d['planning_ticks'] for d in decisions),predicted_substep_poses=sum(d['predicted_poses'] for d in decisions),
        planning_wall_s=sum(d['planning_wall_s'] for d in decisions),decision_wall_s=sum(d['decision_wall_s'] for d in decisions),
        max_pd_error_rad=float(pd_error.max()),boundary_warmstart_max_gap=warm_gap,last_decision=decisions[-1])


def run(out):
    start=time.perf_counter()
    root=b.ROOT/'runs'
    previous=file_pin(root/'pen_prospective_prefix_v1/registration.json')
    protocol=read(previous)
    forecast=read(file_pin(root/'pen_transfer_contact_forecasts_v1/intent.json'))
    anchors=forecast['anchors']
    refit=read(file_pin(root/'pen_refit_catalog_v2/refit.json'))
    if refit['model']!=anchors['model']:
        raise ValueError('The calibrated model anchor differs from the refit export')
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(anchors['model'])))
    verify_file_pin(anchors['physics'])
    net=forecast['models']['paired']
    netpath=verify_file_pin(net)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'intent.json',dict(configs=previous,actor=protocol['actor'],anchors=anchors,paired=net,
        arms=ARMS,clips=CLIPS,repetitions=3,dt_s=.0025,control_s=.02,candidates=af.LABELS,gates_mm=af.GATES_MM.tolist(),
        code=[file_pin(__file__),file_pin(af.__file__),file_pin(b.ROOT/'tools/evaluate_pen_dev_clip.py')],
        scope='Run from original phase one through native termination under zero assistance. Forecast one control from live state. Preserve applied RAW/EMA and free-object dynamics. Direct baseline has no lookahead. Report numerical repeats and original strict complete criteria.'))
    results, budget={},dict(native_ticks=0,cpu_planning_ticks=0,predicted_substep_poses=0,planning_wall_s=0.)
    for clip in CLIPS:
        cfg=protocol['clips'][clip]
        folder=out/clip
        folder.mkdir()
        shutil.copyfile(verify_file_pin(cfg['evaluation']),folder/'controlled_evaluation_v4.toml')
        write_json(folder/'intent.json',dict(parent=file_pin(out/'intent.json'),clip=clip,reference=cfg['item']['series'],
            projected=cfg['projected'],control=cfg['control'],objective=cfg['objective']))
        clip_start=time.perf_counter()
        with np.load(verify_file_pin(cfg['item']['series'])) as ref, filter_policy(folder,model,anchors,ref,netpath) as cases:
            rows=evaluate_cases(folder,verify_file_pin(cfg['objective']),cfg['item'],dict.fromkeys(ARMS,protocol['actor']),
                dict.fromkeys(ARMS,0.),3,refit_export=True)
        summaries=[summarize(folder,r,c,cfg['item']['frames']) for r,c in zip(rows,cases,strict=True)]
        results[clip]=dict(frames=cfg['item']['frames'],wall_s=time.perf_counter()-clip_start,
            cases={k:[r for r in summaries if r['arm']==k] for k in ARMS},native_report=file_pin(folder/'report.json'))
        budget['native_ticks']+=8*sum(r['controls'] for r in summaries)
        for key in ('cpu_planning_ticks','predicted_substep_poses','planning_wall_s'):
            budget[key]+=sum(r[key] for r in summaries)
        write_json(out/'progress.json',dict(clips=results,budget=budget))
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),clips=results,budget=budget,wall_s=time.perf_counter()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    run(p.parse_args().out.resolve())
