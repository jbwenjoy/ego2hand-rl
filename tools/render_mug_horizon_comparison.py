"""Compare recorded zero/60ms/120ms coarse-dt runs without new integration."""
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
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT / 'runs'):
        raise ValueError('Fresh unified video output required')
    baseline = json.loads(args.baseline.read_text(encoding='utf8'))
    extension = json.loads(args.extension.read_text(encoding='utf8'))
    # A progress manifest is sufficient only for its explicitly completed case.
    extended_cases = extension.get('cases', [extension.get('last')])
    cases = [next(c for c in baseline['cases'] if c['task'] == 'REFERENCE'
                  and c['controller'] == controller and c['dt'] == .0003125)
             for controller in ['ZERO', 'NATIVE_MPC']]
    cases.append(next(c for c in extended_cases if c and c['task'] == 'REFERENCE'
                      and c['controller'] == 'NATIVE_MPC' and c['dt'] == .0003125))
    if cases[-1]['horizon'] != 6:
        raise ValueError('Expected preregistered H6 case')
    traces = [dict(np.load(verify_file_pin(c['execution']['trace']))) for c in cases]
    durations = [c['summary']['metrics']['actual_duration_s'] for c in cases]
    if any(len(z['post_qpos']) * c['dt'] != duration
           for z, c, duration in zip(traces, cases, durations)):
        raise ValueError('Exact trace durations required')
    if any(not np.array_equal(z['pre_qpos'][0], traces[0]['pre_qpos'][0]) for z in traces):
        raise ValueError('Same initial pose required')
    binding_path = verify_file_pin(baseline['inputs']['binding'])
    binding = json.loads(binding_path.read_text(encoding='utf8'))
    model = mujoco.MjModel.from_binary_path(str(verify_file_pin(binding['model'])))
    model.vis.global_.offwidth = 768
    model.vis.global_.offheight = 560
    model.vis.headlight.ambient[:] = .45
    model.vis.headlight.diffuse[:] = .75
    data = mujoco.MjData(model)
    data.qpos[:] = traces[0]['pre_qpos'][0]
    mujoco.mj_forward(model, data)
    center = data.xipos[binding['mug_body_id']].copy()
    center[2] -= .015
    scene = Scene(model, 512, 550, center)
    out.mkdir()
    path = out / 'zero_60ms_120ms_actual.mp4'
    labels = ['零残差基线', '60 ms 前瞻规划', '120 ms 前瞻规划']
    frames = 0
    try:
        with writer(path) as video:
            samples = np.linspace(0, max(durations), math.ceil(max(durations) / .25 * 30) + 1)
            for i, t in enumerate(samples):
                panels, captions = [], []
                for label, c, z, duration in zip(labels, cases, traces, durations):
                    index = min(len(z['post_qpos']) - 1, max(0, round(t / c['dt']) - 1))
                    q = z['pre_qpos'][0] if i == 0 else z['post_qpos'][index]
                    panels.append(scene.draw(q, 250))
                    captions.append(label + ('：终止后静帧' if t >= duration else '：实际记录'))
                frame = canvas(1536, '最新实际仿真对照：仍未通过 2 秒参考跟踪',
                    '同一初态与步长；0.25× 慢放；固定展示三个粗步长案例；三列均为记录的物理执行',
                    panels, captions,
                    '各列物理时间：' + ' / '.join(f'{min(t,d):.3f}' for d in durations) + ' 秒',
                    ['各列在姿态超过 45° 时停止；随后明确显示末帧，不延长仿真、不插入复位。',
                     '规划使用完整仿真状态和原生前瞻；尚非学习 Actor。120 ms 前瞻增加计算成本。'])
                if i == 0:
                    Image.fromarray(frame).save(out / 'start.png')
                    for _ in range(18):
                        video.append_data(frame)
                        frames += 1
                if i == len(samples) // 2:
                    Image.fromarray(frame).save(out / 'mid.png')
                video.append_data(frame)
                frames += 1
            Image.fromarray(frame).save(out / 'end.png')
            for _ in range(30):
                video.append_data(frame)
                frames += 1
    finally:
        scene.close()
    result = dict(schema='mug_horizon_recorded_comparison:1',
        sources=[file_pin(__file__), file_pin(ROOT / 'tools/render_mug_progress_video.py')],
        manifests=[file_pin(args.baseline), file_pin(args.extension)],
        model=binding['model'], traces=[c['execution']['trace'] for c in cases],
        output=file_pin(path), frames=frames, fps=30, speed=.25,
        physical_durations_s=durations,
        qualified=[c['summary']['task_qualified'] for c in cases],
        termination=[c['summary']['termination'] for c in cases],
        neural_updates=0, native_integration_steps=0,
        scope='Fixed first coarse-dt cases, actual saved qpos only; terminal holds explicit. No learned-policy or success claim.')
    (out / 'report.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(output=str(path), frames=frames, durations=durations)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--extension', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
