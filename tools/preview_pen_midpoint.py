"""A preregistered intermediate diagnostic;never select or replace final policy."""
import argparse
import json
import math
import subprocess
import sys
import time
import tomllib
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'tools')]
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.learning.pen_w1_evaluation import actor_for_fresh_live_W1
from preview_pen_support_pilot import execute, write
from render_mug_progress_video import Scene, canvas, writer, FONT


def render_diagnostic(case, out):
    with np.load(verify_file_pin(case['trace'])) as source:
        z = dict(source)
    duration = case['summary']['metrics']['actual_duration_s']
    scenes = []
    for _ in range(2):
        model = mujoco.MjModel.from_binary_path(str(verify_file_pin(case['model'])))
        model.vis.global_.offwidth = 768
        model.vis.global_.offheight = 560
        model.vis.headlight.ambient[:] = .45
        model.vis.headlight.diffuse[:] = .75
        scene = Scene(model, 640, 550, z['COM_world_m'][0])
        scene.data.mocap_pos[:] = z['initial_mocap_pos'][0]
        scene.data.mocap_quat[:] = z['initial_mocap_quat'][0]
        scenes.append(scene)
    path = out / 'residual_1000_diagnostic.mp4'
    frames = 0
    selected = []
    try:
        with writer(path) as video:
            times = np.linspace(0, duration, math.ceil(duration / .05 * 30) + 1)
            for i, t in enumerate(times):
                index = int(np.searchsorted(np.arange(1, len(z['post_qpos']) + 1) * .0025,
                                            t + 1e-12, side='right') - 1)
                selected.append(index)
                pose = z['initial_qpos'] if index < 0 else z['post_qpos'][index]
                terminal = '；终止静帧' if t >= duration else ''
                frame = canvas(1280, 'RESIDUAL 1000轮：中途实测诊断',
                    '同一轨迹的两个视角；20倍慢放；2.5ms记录姿态；受控参考、完美位姿，非Ego',
                    [s.draw(pose, angle) for s, angle in zip(scenes, [250, 100])],
                    ['同一次执行：视角1' + terminal, '同一次执行：视角2' + terminal],
                    f'物理时间 {t:.3f} / {duration:.3f} 秒；完整任务7.36秒',
                    ['单个中途检查点；无物体外力辅助；终止后保持末帧，没有继续仿真。',
                     '未与长训练SUPPORT组做同预算对照；最终2000轮评测仍待完成。'])
                if i == 0:
                    Image.fromarray(frame).save(out / 'start.png')
                    for _ in range(18):
                        video.append_data(frame)
                        frames += 1
                if i == len(times) // 2:
                    Image.fromarray(frame).save(out / 'mid.png')
                video.append_data(frame)
                frames += 1
            Image.fromarray(frame).save(out / 'end.png')
            for _ in range(30):
                video.append_data(frame)
                frames += 1
    finally:
        for scene in scenes:
            scene.close()
    reader = imageio.get_reader(path)
    try:
        if reader.count_frames() != frames:
            raise ValueError('Encoded frame count differs')
        for name, index in [('start', 0), ('mid', frames // 2), ('end', frames - 1)]:
            decoded = reader.get_data(index)
            if decoded.shape != (800, 1280, 3):
                raise ValueError('Decoded dimensions differ')
            Image.fromarray(decoded).save(out / f'decoded_{name}.png')
    finally:
        reader.close()
    return dict(video=file_pin(path), frames=frames, fps=30, speed=.05,
                durations_s=[duration, duration], selected_post_indices=selected,
                render_native_integrations=0, optimizer_steps=0, font=file_pin(FONT))


def initialize(runtime, state, pin, *, context):
    if (state['iteration'], state['optimizer_steps'], state['training_transitions'],
            state['metadata']['arm']) != (1000, 32000, 2048000, 'RESIDUAL'):
        raise ValueError('Fixed RESIDUAL1000 diagnostic source required')
    return actor_for_fresh_live_W1(runtime, state, pin, context)


def main(a):
    cp = a.config.resolve()
    cfg = tomllib.loads(cp.read_text())
    if (cfg['schema'], cfg['arm'], cfg['iteration']) != (1, 'RESIDUAL', 1000):
        raise ValueError('Fixed preregistered midpoint required')
    paths = {k: (cp.parent / cfg[k]).resolve() for k in [
        'checkpoint', 'health', 'objective_config', 'gates', 'plant_context_report', 'protocol']}
    out = a.output.resolve()
    if not out.is_relative_to(ROOT / 'runs'):
        raise ValueError('Explicit runs output required')
    inputs = {k: file_pin(paths[k]) for k in ['objective_config', 'gates', 'plant_context_report', 'protocol']}
    if a.register:
        if out.exists() or paths['checkpoint'].exists() or paths['health'].exists():
            raise ValueError('Register before the fixed1000 checkpoint exists')
        out.mkdir()
        sources = sorted((ROOT / 'src').rglob('*.py')) + [Path(__file__), cp, paths['protocol'],
            paths['gates'], ROOT / 'tools/preview_pen_support_pilot.py', ROOT / 'tools/render_mug_progress_video.py']
        receipt = freeze_sources(sources, out / 'source_snapshot', ROOT,
                                 subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
        write(out / 'source_receipt.json', receipt)
        write(out / 'registration.json', dict(config=file_pin(cp), inputs=inputs,
              source_receipt=file_pin(out / 'source_receipt.json'), iteration=1000,
              checkpoint_absent_at_registration=True, future_checkpoint=str(paths['checkpoint']), scope=cfg['scope']))
        print(json.dumps(dict(registration=file_pin(out / 'registration.json'))), flush=True)
        return
    if (out / 'report.json').exists() or (out / 'execution').exists() or (out / 'failure.json').exists():
        raise ValueError('One execution only;retain terminal artifacts')
    registration = json.loads((out / 'registration.json').read_text())
    if registration['config'] != file_pin(cp) or registration['inputs'] != inputs:
        raise ValueError('Diagnostic registration differs')
    receipt = json.loads(verify_file_pin(registration['source_receipt']).read_text())
    verify_source_snapshot(receipt, live=True)
    health = json.loads(paths['health'].read_text())
    checkpoint = file_pin(paths['checkpoint'])
    if health['checkpoint'] != checkpoint or health['collector']['passed'] != health['collector']['total']:
        raise ValueError('Saved fixed checkpoint/collector receipt differs')
    contexts = json.loads(paths['plant_context_report'].read_text())
    context = next(s['source'] for s in contexts['sources'] if s['actor'] == 'RESIDUAL' and s['phase'] == 10 and s['available'])
    from mjlab.sim.sim import Simulation
    original = Simulation.step
    observed = 0
    started = time.perf_counter()

    def step(sim):
        nonlocal observed
        original(sim)
        observed += sim.num_envs

    Simulation.step = step
    try:
        torch.set_num_threads(4)
        with torch.no_grad(), torch.jit.optimized_execution(False):
            case = execute(checkpoint, paths['objective_config'], tomllib.loads(paths['gates'].read_text()),
                           out / 'execution', actor_initializer=partial(initialize, context=context))
        if observed != case['native_integrations']:
            raise ValueError('Independent native integration count differs')
        execution_s = time.perf_counter() - started
        video = render_diagnostic(case, out)
        verify_source_snapshot(receipt, live=True)
        for pin in inputs.values():
            verify_file_pin(pin)
        result = dict(schema='pen_long_control_midpoint_diagnostic:1', config=file_pin(cp),
            registration=file_pin(out / 'registration.json'), source_receipt=file_pin(out / 'source_receipt.json'),
            health=file_pin(paths['health']), case=case, rendering=video,
            actual_native_integrations=observed, optimizer_steps=0, execution_record_wall_s=execution_s,
            render_wall_s=time.perf_counter() - started - execution_s, scope=cfg['scope'])
        write(out / 'report.json', result)
        print(json.dumps(dict(report=file_pin(out / 'report.json'), summary=case['summary'], video=video['video'])), flush=True)
    except Exception as error:
        write(out / 'failure.json', dict(type=type(error).__name__, message=str(error),
                                        actual_native_integrations=observed, optimizer_steps=0))
        raise
    finally:
        Simulation.step = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--register', action='store_true')
    main(parser.parse_args())
