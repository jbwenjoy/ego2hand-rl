"""Fresh mean-policy executions and saved-pose video of the registered pilot."""
import argparse
import copy
import json
import math
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image
from dex_supermimic.rl.dex_supermimic.networks import GaussianPPOActor
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.learning.pen_support_ppo import SCHEMA, CENTERED_SCHEMA
from supermimic_research.mechanics.compound_holding import object_state
from supermimic_research.mechanics.warp_contact_labels import read_warp_contact_labels
from supermimic_research.runtime.controlled_evaluation import summarize_trace
from supermimic_research.runtime.pen_portable_state import reference_signature, host_model_digest
from supermimic_research.runtime.pen_pose_observation import require_full_pose_contract
from supermimic_research.runtime.pen_support_runtime import SupportPenRuntime
from tools.render_mug_progress_video import Scene, canvas, writer, FONT


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


@torch.no_grad()
def execute(pin, config, gates, folder):
    """Transfer only actor/norm weights; never transfer W32 native state to W1."""
    folder.mkdir()
    state = torch.load(verify_file_pin(pin), map_location='cpu', weights_only=False)
    if state['schema'] not in [SCHEMA,CENTERED_SCHEMA] or state['schema'] != state['metadata']['checkpoint_schema'] or state['metadata']['objective_config'] != file_pin(config):
        raise ValueError('Explicit registered training3/training4/config required')
    require_full_pose_contract(state['metadata'])
    verify_source_snapshot(state['native']['sources'], live=True)
    runtime = SupportPenRuntime(config, 1)
    runtime.support_weight = float(state['metadata']['support_weight'])
    sim = runtime.env.sim
    model = sim.mj_model
    old_native = state['native']
    if reference_signature(runtime) != old_native['reference_signature'] or host_model_digest(model) != old_native['host_model_digest']:
        raise ValueError('Fresh reference/topology differs')
    if list(runtime.motion.joint_names) != state['metadata']['joint_order']:
        raise ValueError('Joint order differs')
    cfg = state['config']
    actor = GaussianPPOActor(161, 20, init_std=cfg['ppo_init_std'], std_range=(cfg['ppo_std_min'], cfg['ppo_std_max'])).to(runtime.device)
    actor.load_state_dict(state['modules']['actor'], strict=True)
    actor.eval()
    render_model = copy.copy(model)
    physical_checks = {}
    for key in ['body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'geom_size', 'geom_pos', 'geom_quat', 'geom_friction', 'geom_solref', 'geom_solimp',
                'actuator_gainprm', 'actuator_biasprm', 'actuator_forcerange', 'actuator_ctrlrange', 'jnt_range', 'dof_damping', 'dof_armature']:
        value = getattr(sim.wp_model, key).numpy().copy()
        old = old_native['native']['model']['arrays'][key]
        physical_checks[key] = bool(np.array_equal(np.broadcast_to(value, old.shape), old))
        target = getattr(render_model, key)
        target[:] = value if target.shape == value.shape else value[0]
    if not all(physical_checks.values()) or model.neq or np.any(model.body_gravcomp):
        raise ValueError('Physical parameters/aids differ')
    mujoco.mj_setConst(render_model, mujoco.MjData(render_model))
    mujoco.mj_saveModel(render_model, str(folder / 'model.mjb'))
    f = runtime.features
    body, qa = f.body, f.qa
    da = int(model.jnt_dofadr[model.body_jntadr[body]])
    com = sim.wp_model.body_ipos.numpy()[0, body].copy()
    qids = f.qids.cpu().numpy()
    jointids = [model.joint('robot/' + n).id for n in runtime.motion.joint_names]
    ranges = sim.wp_model.jnt_range.numpy()[0, jointids]
    d = sim.wp_data
    initial_qpos = d.qpos.numpy()[0].copy()
    initial_mocap_pos = d.mocap_pos.numpy().copy()
    initial_mocap_quat = d.mocap_quat.numpy().copy()
    reference = dict(time_s=np.arange(369) * .02, COM_world_m=runtime.support_table.pose[:, :3], quaternion_wxyz=runtime.support_table.pose[:, 3:])
    trace = {k: [] for k in ['post_qpos', 'post_qvel', 'post_COM', 'post_quaternion_wxyz', 'post_COM_velocity', 'post_omega_world',
                              'pre_penetration_m', 'pre_self_penetration_m', 'post_joint_limit_excursion_rad', 'pre_hand_normal_N',
                              'pre_object_nonhand_normal_N', 'post_qfrc_applied', 'post_xfrc_applied', 'overflow']}
    # Instrument the native resource below the owned hook; keep its stream and
    # resident metric sampling intact. No reset or simulation is done for rendering.
    from mjlab.sim.sim import Simulation
    original = Simulation.step
    fixed_mount = True

    def record_step(s):
        nonlocal fixed_mount
        original(s)
        if s is not sim:
            raise ValueError('Unexpected simulation')
        q, v = d.qpos.numpy()[0].copy(), d.qvel.numpy()[0].copy()
        center, quat, velocity, omega = object_state(model, q, v, qa, da, com)
        labels = read_warp_contact_labels(sim.wp_model, d)
        efc.extend(labels['efc_force'])
        offsets.append(len(efc))
        for i in range(len(labels['geom'])):
            raw['tick'].append(len(trace['post_qpos']))
            for key in raw:
                if key != 'tick':
                    raw[key].append(labels[key][i])
        pen = selfpen = hand = nonhand = 0.
        for i, (g, h) in enumerate(labels['geom']):
            g, h = int(g), int(h)
            on = model.geom_bodyid[[g, h]] == body
            gn, hn = model.geom(g).name or '', model.geom(h).name or ''
            if on.sum() == 1:
                pen = max(pen, max(0., -float(labels['dist'][i])))
                force = max(0., float(labels['local_force6'][i, 0]))
                if (hn if on[0] else gn).startswith('robot/'):
                    hand += force
                else:
                    nonhand += force
            elif gn.startswith('robot/') and hn.startswith('robot/'):
                selfpen = max(selfpen, max(0., -float(labels['dist'][i])))
        excursion = float(np.maximum(np.maximum(ranges[:, 0] - q[qids], q[qids] - ranges[:, 1]), 0).max())
        row = dict(post_qpos=q, post_qvel=v, post_COM=center, post_quaternion_wxyz=quat, post_COM_velocity=velocity, post_omega_world=omega,
                   pre_penetration_m=pen, pre_self_penetration_m=selfpen, post_joint_limit_excursion_rad=excursion, pre_hand_normal_N=hand,
                   pre_object_nonhand_normal_N=nonhand, post_qfrc_applied=d.qfrc_applied.numpy()[0].copy(), post_xfrc_applied=d.xfrc_applied.numpy()[0].copy(),
                   overflow=int(d.overflow.numpy()[0]) or int(labels['overflow']))
        for key, value in row.items():
            trace[key].append(value)
        fixed_mount &= np.array_equal(d.mocap_pos.numpy(), initial_mocap_pos) and np.array_equal(d.mocap_quat.numpy(), initial_mocap_quat)

    # The runtime retained a bound method at construction: explicitly replace
    # that resource temporarily; the public owned hook remains unchanged.
    native_resource = runtime._native_step_resource
    runtime._native_step_resource = record_step.__get__(sim, Simulation)
    actions, reasons, phases, windows = [], {}, [], []
    raw = {k: [] for k in ['tick', 'geom', 'dim', 'dist', 'pos', 'frame', 'friction', 'efc_address', 'local_force6']}
    efc, offsets = [], [0]
    try:
        obs = runtime.observation()
        for _ in range(368):
            phases.append(int(runtime.motion.time_steps[0]))
            action = actor.distribution(obs['policy'], validate_args=False).mean
            actions.append(action.cpu().numpy()[0].copy())
            obs, _, term, trunc, info = runtime.step(action)
            windows.append({k: v.detach().cpu().numpy().copy()[0] for k, v in runtime.contact_views.items()})
            if bool(term[0]) or bool(trunc[0]):
                reasons = {k: bool(v[0]) for k, v in info['termination_reasons'].items()}
                break
        arrays = {k: np.array(v) for k, v in trace.items()}
        summary = summarize_trace(arrays, reference, .0025, len(actions) == 368 and not bool(term[0]), gates)
        np.savez_compressed(folder / 'trace.npz', **arrays, **reference, initial_qpos=initial_qpos, initial_mocap_pos=initial_mocap_pos,
                            initial_mocap_quat=initial_mocap_quat, sampled_mean_raw=np.array(actions), pre_action_phase=np.array(phases))
        raw_arrays = {k: np.array(v).reshape((-1,) + {'geom': (2,), 'pos': (3,), 'frame': (3, 3), 'friction': (5,), 'efc_address': (6,), 'local_force6': (6,)}.get(k, ())) for k, v in raw.items()}
        np.savez_compressed(folder / 'raw_contacts.npz', **raw_arrays, efc_force=np.array(efc), efc_offsets=np.array(offsets))
        np.savez_compressed(folder / 'control_windows.npz', **{k: np.stack([w[k] for w in windows]) for k in windows[0]})
        result = dict(checkpoint=pin, iteration=state['iteration'], trained_optimizer_steps=state['optimizer_steps'], checkpoint_schema=state['schema'], arm=state['metadata']['arm'], summary=summary, controls=len(actions),
                      native_integrations=len(trace['post_qpos']), optimizer_steps=0, terminated=bool(term[0]), truncated=bool(trunc[0]),
                      termination_reasons=reasons, fixed_mount=bool(fixed_mount), post_start_resets=0, physical_matches=physical_checks,
                      model=file_pin(folder / 'model.mjb'), trace=file_pin(folder / 'trace.npz'),
                      raw_contacts=file_pin(folder / 'raw_contacts.npz'), control_windows=file_pin(folder / 'control_windows.npz'),
                      dt=.0025, cone=int(sim.wp_model.opt.cone))
        if result['native_integrations'] != 8 * len(actions) or not fixed_mount:
            raise ValueError('Native clock/mount mismatch')
        write(folder / 'report.json', result)
        return result
    finally:
        runtime._native_step_resource = native_resource
        runtime.close()


def render(cases, out):
    z = [dict(np.load(verify_file_pin(c['trace']))) for c in cases]
    durations = [c['summary']['metrics']['actual_duration_s'] for c in cases]
    scenes = []
    labels = ['共同初始化', f'几何约束：{cases[1]["iteration"]}轮训练', f'几何＋支持目标：{cases[2]["iteration"]}轮训练']
    for c, v in zip(cases, z):
        m = mujoco.MjModel.from_binary_path(str(verify_file_pin(c['model'])))
        m.vis.global_.offwidth = 768
        m.vis.global_.offheight = 560
        m.vis.headlight.ambient[:] = .45
        m.vis.headlight.diffuse[:] = .75
        s = Scene(m, 512, 550, v['COM_world_m'][0])
        s.data.mocap_pos[:] = v['initial_mocap_pos'][0]
        s.data.mocap_quat[:] = v['initial_mocap_quat'][0]
        scenes.append(s)
    path = out / 'geometry_support_pilot_comparison.mp4'
    frames = 0
    selected = []
    times = np.linspace(0, max(durations), math.ceil(max(durations) / .05 * 30) + 1)
    try:
        with writer(path) as video:
            for i, t in enumerate(times):
                panels, names, indices = [], [], []
                for label, v, s, duration in zip(labels, z, scenes, durations):
                    j = int(np.searchsorted(np.arange(1, len(v['post_qpos']) + 1) * .0025, min(t, duration) + 1e-12, side='right') - 1)
                    panels.append(s.draw(v['initial_qpos'] if j < 0 else v['post_qpos'][j], 250))
                    indices.append(j)
                    names.append(label + ('：终止静帧' if t >= duration else '：实测'))
                selected.append(indices)
                frame = canvas(1536, '最新几何／支持目标先导训练：真实仿真执行',
                    '20倍慢放；均值动作；保存的2.5ms姿态，无插值；受控参考、完美位姿，非ego数据', panels, names,
                    '各列物理时间：' + ' / '.join(f'{min(t, duration):.3f}' for duration in durations) + ' 秒',
                    [f'完整任务7.36秒；两训练组各{cases[1]["trained_optimizer_steps"]}次优化更新。终止后保持末帧，没有继续仿真。',
                     '无物体外力辅助；单参考、单种子的小规模试验，尚未通过完整任务。'])
                if i == 0:
                    Image.fromarray(frame).save(out / 'start.png')
                    for _ in range(18):
                        video.append_data(frame); frames += 1
                if i == len(times) // 2:
                    Image.fromarray(frame).save(out / 'mid.png')
                video.append_data(frame); frames += 1
            Image.fromarray(frame).save(out / 'end.png')
            for _ in range(30):
                video.append_data(frame); frames += 1
    finally:
        for s in scenes:
            s.close()
    reader = imageio.get_reader(path)
    try:
        if reader.count_frames() != frames:
            raise ValueError('Encoded frame count differs')
        for tag, j in [('start', 0), ('mid', frames // 2), ('end', frames - 1)]:
            frame = reader.get_data(j)
            if frame.shape != (800, 1536, 3):
                raise ValueError('Decoded dimensions differ')
            Image.fromarray(frame).save(out / f'decoded_{tag}.png')
    finally:
        reader.close()
    return dict(video=file_pin(path), frames=frames, fps=30, speed=.05, durations_s=durations,
                selected_post_indices=selected, render_native_integrations=0, render_optimizer_steps=0, font=file_pin(FONT))


def main(a):
    out = a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT / 'runs'):
        raise ValueError('Fresh run output required')
    report = json.loads(a.training.read_text())
    config = verify_file_pin(report['config'])
    verify_source_snapshot(json.loads(verify_file_pin(report['source_receipt']).read_text()), live=True)
    out.mkdir()
    receipt = freeze_sources([Path(__file__).resolve(), ROOT / 'tools/render_mug_progress_video.py',
                              ROOT / 'configs/controlled_evaluation_v4.toml'], out / 'source_snapshot', ROOT,
                              subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    write(out / 'source_receipt.json', receipt)
    gates = tomllib.loads((ROOT / 'configs/controlled_evaluation_v4.toml').read_text())
    cases = []
    pins = [report['arms'][0]['initial']] + [arm['checkpoint'] for arm in report['arms']]
    with torch.jit.optimized_execution(False):
        for label, pin in zip(['INITIAL', 'RESIDUAL', 'SUPPORT'], pins):
            case = execute(pin, config, gates, out / label)
            case['name'] = label
            cases.append(case)
            print(json.dumps(dict(name=label, summary=case['summary'])), flush=True)
    video = render(cases, out)
    verify_source_snapshot(receipt, live=True)
    result = dict(schema='geometry_support_pilot_actual_preview:1', training=file_pin(a.training), source_receipt=file_pin(out / 'source_receipt.json'),
                  cases=cases, rendering=video, actual_native_integrations=sum(c['native_integrations'] for c in cases), optimizer_steps=0,
                  scope='Fresh W1 actor/norm-only mean execution; original timing and strict gates, no native W32 state transfer or post-start reset. Controlled perfect-pose reference, not ego or established support benefit.')
    write(out / 'report.json', result)
    print(json.dumps(dict(video=video['video'], report=file_pin(out / 'report.json'))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--training', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
