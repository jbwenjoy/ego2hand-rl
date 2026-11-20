"""Factorial test of finger-4 versus other-digit control at the clip_003 handoff."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import tomllib
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from evaluate_pen_prefix_w1 import DEFAULT_ACTOR, b, config, evaluate_cases, file_pin, load_selection, read, summarize_case, verify_file_pin, write_json


ARMS = ('hold_finger4', 'actor_finger4')


def control_layout(runtime):
    model = runtime.env.sim.mj_model
    aids = runtime.action._position_ctrl_ids.cpu().numpy()
    jids = model.actuator_trnid[aids,0]
    joint_names = [model.joint(int(j)).name for j in jids]
    expected = {f'robot/right_finger{i}_joint{j}' for i in range(1,6) for j in range(1,5)}
    if set(joint_names) != expected or len(set(aids)) != 20 or len(aids) != 20:
        raise ValueError('Expected four joint transmissions on each of five fingers')
    if joint_names != ['robot/'+n for n in runtime.motion.joint_names]:
        raise ValueError('Raw-action and reference joint order differ')
    np.testing.assert_array_equal(model.jnt_qposadr[jids], runtime.features.qids.cpu().numpy())
    b.require_structure(model)
    finger4 = [i for i,n in enumerate(joint_names) if n.startswith('robot/right_finger4_')]
    return dict(raw_to_joint=joint_names, actuator_ids=aids.tolist(),
        actuator_names=[model.actuator(int(i)).name for i in aids], transmission_joint_ids=jids.tolist(),
        finger4_raw_indices=finger4, other_raw_indices=[i for i in range(20) if i not in finger4])


@contextmanager
def digit_policy(arm, final_prefix):
    """Wrap only the frozen Actor mean; leave the original W1 loop and runtime untouched."""
    pair, recorder = b.model_pair, b.OwnedProbeRecorder
    means, connected = [], {}

    def make_actor(*args, **kwargs):
        actor, critic, state = pair(*args, **kwargs)
        original = actor.distribution
        samples = []
        means.append(samples)

        def distribution(obs):
            native = original(obs)
            runtime = connected['runtime']
            phase = int(runtime.motion.time_steps[0])
            if phase != len(samples)+1:
                raise ValueError('Expected exactly one frozen-Actor observation per native control')
            mean = native.mean
            samples.append(mean[0].cpu().numpy().copy())
            if 9 <= phase <= 16:
                masked = torch.where(connected['actor_mask'], mean.clamp(-1,1), connected['held'])
                return SimpleNamespace(mean=masked)
            return native

        actor.distribution = distribution
        return actor, critic, state

    def bind(runtime):
        layout = control_layout(runtime)
        if 'layout' in connected and connected['layout'] != layout:
            raise ValueError('Live actuator mapping changed between repeats')
        indices = layout['other_raw_indices' if arm == 'hold_finger4' else 'finger4_raw_indices']
        use_actor = np.zeros(20, dtype=bool)
        use_actor[indices] = True
        connected.update(runtime=runtime,layout=layout,
            actor_mask=torch.as_tensor(use_actor,device=runtime.device),
            held=torch.as_tensor(final_prefix,device=runtime.device))
        return recorder(runtime)

    with patch.object(b,'model_pair',make_actor), patch.object(b,'OwnedProbeRecorder',bind):
        yield means, connected


def summarize_factor(case, original, folder, actor_indices):
    native = read(case['case'])
    with np.load(verify_file_pin(native['handoff_controls'])) as z:
        pd = z['post_ctrl']
        phases = len(pd)
        p9 = {}
        if phases >= 9:
            delta = abs(pd[8,0] - pd[7,-1])
            p9 = dict(max_pd_jump_rad=float(delta.max()),pd_jump_by_channel_rad=delta.tolist(),
                geometry=native['checks'][8]['geometry'],audit_pass=native['checks'][8]['ok'])
        if original is not None:
            mean, prefix = original
            actor = np.asarray(mean)
            if actor.shape != (phases,20):
                raise ValueError('Frozen mean sample count differs from native control count')
            expected = actor.copy()
            expected[:8] = prefix[:min(phases,8)]
            mask = np.zeros(20, dtype=bool)
            mask[actor_indices] = True
            for i in range(8,min(16,phases)):
                expected[i] = np.where(mask, np.clip(actor[i],-1,1), prefix[-1])
            np.testing.assert_array_equal(z['sent_raw'], expected)
            np.testing.assert_array_equal(z['applied_raw'], np.clip(expected,-1,1))
            np.testing.assert_array_equal(z['previous_raw'][1:],z['applied_raw'][:-1])
            target = folder / 'digit_controls.npz'
            np.savez_compressed(target,frozen_actor_raw=actor,controller_mean=z['actor_mean'],sent_raw=z['sent_raw'],
                applied_raw=z['applied_raw'],previous_raw=z['previous_raw'],actual_pd=pd,phase9_to_16_actor_mask=mask)
            case['digit_controls'] = file_pin(target)
    complete = case['reference_completion']
    passed = phases - int(not complete)
    with np.load(verify_file_pin(native['trace'])) as trace:
        failure = None if complete else dict(phase=phases,stop=case['stop'],reasons=case['reasons'],
            geometry=native['checks'][-1]['geometry'], max_obj_penetration_mm=float(1000*trace['pre_penetration_m'][-8:].max()),
            max_self_penetration_mm=float(1000*trace['pre_self_penetration_m'][-8:].max()),
            max_joint_limit_excursion_rad=float(trace['post_joint_limit_excursion_rad'][-8:].max()),
            endpoint_COM_error_mm=float(1000*np.linalg.norm(trace['post_COM'][-1]-trace['COM_world_m'][phases])))
    if not all(c['ok'] for c in native['checks'][:passed]):
        raise ValueError('Invalid consecutive passing count')
    case.update(phase9=p9,first_failure=failure,consecutive_task_and_physics_controls=passed,
        consecutive_task_and_physics_s=.02*passed)
    return case


def run(selection,out,actor_path,repetitions,comparison):
    if repetitions <= 0:
        raise ValueError('At least one repetition required')
    old_pin = file_pin(comparison)
    old = read(old_pin)
    old_intent = read(old['intent'])
    for code in old_intent['scripts']:
        verify_file_pin(code)
    actor = file_pin(actor_path)
    source = file_pin(selection)
    if actor != old_intent['actor'] or source != old_intent['selection'] or old_intent['repeats'] != repetitions:
        raise ValueError('Paired controls require identical frozen actor, selection and repeat count')
    runs = b.ROOT/'runs'
    preload = file_pin(runs/'pen_refit_dev_preload_v1/report.json')
    proj = read(read(read(preload)['intent'])['source'])
    pose = next(r['variants']['projection']['trace'] for r in proj['rows'] if r['clip']=='clip_003')
    with np.load(verify_file_pin(pose)) as projected:
        plan = load_selection(source,projected['initial_hand_q'])['action_only']
    q, raw, _ = plan
    if len(raw)!=8:
        raise ValueError('Expected the registered eight-control prefix')
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'intent.json',dict(selection=source,actor=actor,comparisons=old_pin,code=file_pin(__file__),
        repeats=repetitions,new_arms=list(ARMS),prefix='action_only',weights_window=[9,16],fully_actor_from=17,
        scope='Fixed observed development clip, 2 by 2 handoff test. Controls passing both task and W1 physical audit exclude every failure control.'))
    results = {}
    for arm in ('abrupt','hold'):
        results[arm]=[summarize_factor(dict(c),None,None,None) for c in old['arms'][arm]['cases']]
    layouts = {}
    for arm in ARMS:
        folder = out/arm
        folder.mkdir()
        objective,item = config(folder,'clip_003',Path(preload['path']),initial_hand_q=q)
        adapter = Path(tomllib.loads(objective.read_text())['adapter_config'])
        cfg = tomllib.loads(adapter.read_text())
        if cfg['control_pin'] != old_intent['preload'] or item['series'] != old_intent['reference']:
            raise ValueError('Matched native reference or preload changed')
        with np.load(verify_file_pin(cfg['projected_pin'])) as a, np.load(verify_file_pin(old_intent['projected'])) as orig:
            for key in orig.files:
                np.testing.assert_array_equal(a[key],orig[key])
        write_json(folder/'intent.json',dict(master=file_pin(out/'intent.json'),objective=file_pin(objective),
            adapter=file_pin(adapter),arm=arm,
            native_controller_mean='Composite digit-policy mean. The independent frozen Actor mean is saved in digit_controls.npz.'))
        with digit_policy(arm,raw[-1]) as (means,live):
            rows = evaluate_cases(folder,objective,item,{arm:actor},{arm:0.},repetitions,refit_export=True,
                raw_prefixes={arm:raw},handoff_weights={arm:[]})
        layout = live['layout']
        layouts[arm] = layout
        indices = layout['other_raw_indices' if arm == 'hold_finger4' else 'finger4_raw_indices']
        with np.load(verify_file_pin(cfg['control_pin'])) as ctrl, np.load(verify_file_pin(item['series'])) as ref:
            cases = [summarize_case(r,plan,ctrl['ctrl'],ref) for r in rows]
        results[arm] = [summarize_factor(c,(means[c['rep']],raw),folder/f"{arm}_{c['rep']}",indices) for c in cases]
    if len(layouts)!=2 or layouts[ARMS[0]]!=layouts[ARMS[1]]:
        raise ValueError('Factorial arm transmission order mismatch')
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),channel_layout=layouts[ARMS[0]],
        clip='clip_003',arms={k:dict(cases=v,full_count=sum(c['reference_completion'] for c in v),
        strict_count=sum(c['strict_success'] for c in v)) for k,v in results.items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--actor',type=Path,default=DEFAULT_ACTOR)
    p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--comparison',type=Path,default=b.ROOT/'runs/pen_prefix_handoff_w1_v1/report.json')
    args=p.parse_args()
    run(args.selection.resolve(),args.out.resolve(),args.actor.resolve(),args.repetitions,args.comparison.resolve())
