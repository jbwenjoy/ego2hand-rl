"""Original, geometry-only and support-aware actual GPU state replay."""
from pathlib import Path
import argparse
import json
import math
import sys
import numpy as np
import mujoco
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from supermimic_research.artifacts import file_pin, verify_file_pin
from tools.render_mug_progress_video import Scene, canvas, writer


def main(args):
    if args.output.exists():
        raise ValueError('Fresh three-way replay output')
    original = json.loads(args.baseline.read_text())
    current = json.loads(args.report.read_text())
    args.output.mkdir()
    videos = []
    for dt in [.005, .0025]:
        cases = [next(c for c in original['cases'] if c['name'] == name and c['dt'] == dt)
                 for name in ['ORIGINAL_ZERO_RESIDUAL', 'PROJECTED_CONSTANT_RESIDUAL']]
        cases.append(next(c for c in current['cases'] if c['dt'] == dt))
        zs = [dict(np.load(verify_file_pin(c['trace']))) for c in cases]
        scenes = []
        for c, z in zip(cases, zs):
            m = mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])))
            for k, v in dict(np.load(verify_file_pin(c['GPU_physical_context']))).items():
                getattr(m, k)[:] = v[0]
            m.vis.global_.offwidth = 768
            m.vis.global_.offheight = 560
            m.vis.headlight.ambient[:] = .45
            m.vis.headlight.diffuse[:] = .75
            scene = Scene(m, 512, 550, z['COM_world_m'][0])
            mujoco.mj_setState(m, scene.data, z['initial_native_state'], mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_setConst(m, scene.data)
            scenes.append(scene)
        durations = [c['summary']['metrics']['actual_duration_s'] for c in cases]
        frames = 0
        selected = []
        path = args.output / f'GPU_pen_three_way_{dt:g}s.mp4'
        try:
            with writer(path) as video:
                samples = np.linspace(0, max(durations), math.ceil(max(durations) / .05 * 30) + 1)
                for i, t in enumerate(samples):
                    panels, labels, ids = [], [], []
                    for label, c, z, scene, duration in zip(
                            ['原始参考，零残差', '仅接触几何修正', '动态扳手支持初态'], cases, zs, scenes, durations):
                        times = np.arange(1, len(z['post_qpos']) + 1) * dt
                        index = int(np.searchsorted(times, min(t, duration) + 1e-12, side='right') - 1)
                        ids.append(index)
                        q = z['initial_qpos'] if index < 0 else z['post_qpos'][index]
                        panels.append(scene.draw(q, 250))
                        labels.append(label + ('：终止静帧' if t >= duration else '：实测'))
                    selected.append(ids)
                    frame = canvas(1536, 'GPU 受控转笔：首帧支持已改善，连续跟踪仍失败',
                        f'物理步长 {dt * 1000:g} ms；20 倍慢放；保存实测状态，无插值、无新增神经训练',
                        panels, labels, '各列物理时间：' + ' / '.join(f'{min(t, d):.3f}' for d in durations) + ' 秒',
                        [f'计划 7.36 秒；实际终止：{durations[0]:.2f} / {durations[1]:.2f} / {durations[2]:.2f} 秒。结束后保持末帧展示。',
                         '三组均未通过位置/姿态/完整时长门槛；上游受控参考，非真实 ego 数据。'])
                    if i == 0:
                        Image.fromarray(frame).save(args.output / f'{dt:g}_start.png')
                        for _ in range(18):
                            video.append_data(frame)
                            frames += 1
                    if i == len(samples) // 2:
                        Image.fromarray(frame).save(args.output / f'{dt:g}_mid.png')
                    video.append_data(frame)
                    frames += 1
                Image.fromarray(frame).save(args.output / f'{dt:g}_end.png')
                for _ in range(30):
                    video.append_data(frame)
                    frames += 1
        finally:
            for scene in scenes:
                scene.close()
        videos.append(dict(dt_s=dt, video=file_pin(path), frames=frames, fps=30, speed=.05,
                           physical_durations_s=durations, selected_post_indices=selected,
                           inputs=[{k: c[k] for k in ['trace', 'host_topology', 'GPU_physical_context']} for c in cases]))
    result = dict(schema='GPU_support_pen_three_way_replay:1', sources=[file_pin(__file__), file_pin(ROOT / 'tools/render_mug_progress_video.py')],
                  baseline=file_pin(args.baseline), run=file_pin(args.report), videos=videos,
                  neural_updates=0, simulation_integration_steps=0,
                  scope='Actual saved GPU poses with actual physical tables and saved wrist mocap; graphics-only native forward. No inference, simulation integration, interpolation or successful full task.')
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(videos=[v['video']['path'] for v in videos])))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
