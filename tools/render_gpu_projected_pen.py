"""Render saved GPU initialization comparisons without executing dynamics."""
from pathlib import Path
import argparse
import json
import math
import sys

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from supermimic_research.artifacts import file_pin, verify_file_pin
from tools.render_mug_progress_video import Scene, canvas, writer


def main(args):
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT / 'runs'):
        raise ValueError('Use a fresh output directory under unified runs')
    report = json.loads(args.report.read_text(encoding='utf8'))
    out.mkdir()
    videos = []
    for dt in [.005, .0025]:
        cases = [next(c for c in report['cases'] if c['name'] == name and c['dt'] == dt)
                 for name in ['ORIGINAL_ZERO_RESIDUAL', 'PROJECTED_CONSTANT_RESIDUAL']]
        traces = [dict(np.load(verify_file_pin(c['trace']))) for c in cases]
        scenes = []
        for c, z in zip(cases, traces):
            model = mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])))
            physical = np.load(verify_file_pin(c['GPU_physical_context']))
            for key in physical.files:
                actual = physical[key]
                target = getattr(model, key)
                if actual.shape != (1, *target.shape):
                    raise ValueError(f'Unexpected actual GPU physical table: {key}')
                target[:] = actual[0]
            model.vis.global_.offwidth = 768
            model.vis.global_.offheight = 560
            model.vis.headlight.ambient[:] = .45
            model.vis.headlight.diffuse[:] = .75
            scene = Scene(model, 640, 550, z['COM_world_m'][0])
            mujoco.mj_setState(model, scene.data, z['initial_native_state'],
                              mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_setConst(model, scene.data)
            scenes.append(scene)
        durations = [c['summary']['metrics']['actual_duration_s'] for c in cases]
        speed = .05
        path = out / f'GPU_pen_original_vs_projection_{dt:g}s.mp4'
        frames = 0
        indices = []
        try:
            with writer(path) as video:
                samples = np.linspace(0, max(durations), math.ceil(max(durations) / speed * 30) + 1)
                for i, t in enumerate(samples):
                    panels, labels, selected = [], [], []
                    for label, c, z, scene, duration in zip(
                        ['原始参考，零残差', '接触几何修正，固定残差'],
                        cases, traces, scenes, durations
                    ):
                        times = np.arange(1, len(z['post_qpos']) + 1) * c['dt']
                        index = int(np.searchsorted(times, min(t, duration) + 1e-12, side='right') - 1)
                        q = z['initial_qpos'] if index < 0 else z['post_qpos'][index]
                        selected.append(index)
                        panels.append(scene.draw(q, 250))
                        labels.append(label + ('：已终止，末帧' if t >= duration else '：GPU 实测状态'))
                    indices.append(selected)
                    frame = canvas(
                        1280, '最新 GPU 转笔诊断：两组均未稳定跟踪',
                        f'物理步长 {dt * 1000:g} ms；0.05× 慢放（20 倍）；保存状态回放，无插值',
                        panels, labels, f'物理时间 {t:.3f} / {max(durations):.3f} 秒',
                        ['计划 7.36 秒；两组均在 0.16 秒因物体位置误差超过 50 mm 终止。',
                         '上游受控转笔参考，非真实 ego 视频；固定残差诊断，未新增神经网络训练。']
                    )
                    if i == 0:
                        Image.fromarray(frame).save(out / f'{dt:g}_start.png')
                        for _ in range(18):
                            video.append_data(frame)
                            frames += 1
                    if i == len(samples) // 2:
                        Image.fromarray(frame).save(out / f'{dt:g}_mid.png')
                    video.append_data(frame)
                    frames += 1
                Image.fromarray(frame).save(out / f'{dt:g}_end.png')
                for _ in range(30):
                    video.append_data(frame)
                    frames += 1
        finally:
            for scene in scenes:
                scene.close()
        videos.append(dict(dt_s=dt, output=file_pin(path), frames=frames, fps=30,
                           speed=speed, physical_durations_s=durations, selected_post_indices=indices,
                           inputs=[{k: c[k] for k in ['trace', 'host_topology', 'GPU_physical_context']}
                                   for c in cases]))
    result = dict(schema='projected_pen_GPU_actual_replay:1', run=file_pin(args.report),
                  sources=[file_pin(__file__), file_pin(ROOT / 'tools/render_mug_progress_video.py')],
                  videos=videos, neural_updates=0, simulation_integration_steps=0,
                  scope='Saved GPU qpos, actual physical tables and fixed wrist mocap. Native forward refreshes graphics only. No interpolation, extrapolation, new controller execution, successful manipulation or ego-input claim.')
    (out / 'report.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(videos=[v['output']['path'] for v in videos])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    main(parser.parse_args())
