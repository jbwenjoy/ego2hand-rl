"""Plot saved physical trajectories; optional offscreen replay does not integrate dynamics."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--output', type=Path, default=HERE / 'cpu_visualization_v1')
    args = parser.parse_args()
    result_path = HERE / 'cpu_v1/result.json'
    result = json.loads(result_path.read_text())
    dest = args.output
    dest.mkdir(exist_ok=True)
    colors = {0.0025: '#1b6ca8', 0.00125: '#d16e14'}
    tests = ('drop', 'table_settling', 'force_push_release')
    labels = ('Drop from 0.10 m', 'Settle on support', '2 N push then release')
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), constrained_layout=True)
    for col, (test, label) in enumerate(zip(tests, labels)):
        selected = [r for r in result['cases'] if r['test'] == test and r['mass_kg'] == 0.2 and r['sliding_friction'] == 0.6]
        for row in selected:
            a = np.load(row['trajectory']['path'], allow_pickle=False)
            t, com = a['tick_time_s'], a['com_world_m']
            color = colors[row['dt_s']]
            name = f"dt={row['dt_s']*1000:g} ms"
            axes[0, col].plot(t, com[:, 2]*1000, color=color, label=name)
            axes[1, col].plot(t, com[:, 0]*1000, color=color)
            axes[2, col].plot(t, a['penetration_m']*1000, color=color)
        axes[0, col].set_title(label)
        axes[2, col].axhline(2, color='#b52222', ls='--', lw=1, label='2 mm engineering limit')
        axes[2, col].set_xlabel('Simulation time (s)')
        for ax in axes[:, col]:
            ax.grid(alpha=.2)
            if test == 'force_push_release':
                ax.axvspan(1, 1.2, color='orange', alpha=.12)
    axes[0, 0].set_ylabel('COM height (mm)')
    axes[1, 0].set_ylabel('COM displacement X (mm)')
    axes[2, 0].set_ylabel('Generated-contact penetration (mm)')
    axes[0, 0].legend(fontsize=9)
    axes[2, 0].legend(fontsize=8)
    fig.suptitle('HOT3D phone AABB bench: actual saved CPU trajectories\nAssumed mass 0.2 kg, friction 0.6; no hand, retargeting, or parameter calibration', fontsize=14)
    figure_path = dest / 'nominal_physics_traces.png'
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    products = {'figure': str(figure_path), 'source_result': str(result_path), 'actual_CPU_dynamics_already_recorded': True,
                'new_physics_steps': 0, 'Hand2_or_learning': False, 'native_video_rendered': False}
    if args.render:
        import imageio.v2 as imageio
        import mujoco
        import run_phone_cpu_bench_v1 as bench
        model, geometry = bench.make_model(bench.ROOT / 'asset_stage1/phone_asset.xml', 0.2, 0.6, 0.00125)
        # Appearance-only QA changes; replay the same saved states without steps.
        scene = ET.fromstring(geometry['xml'])
        asset = scene.find('asset')
        ET.SubElement(asset, 'texture', name='qa_checker', type='2d', builtin='checker',
                      rgb1='.85 .87 .90', rgb2='.55 .60 .67', width='256', height='256')
        ET.SubElement(asset, 'material', name='qa_floor', texture='qa_checker',
                      texrepeat='5 5', texuniform='true')
        scene.find("./worldbody/geom[@name='bench_support']").set('material', 'qa_floor')
        scene.find("./worldbody/geom[@name='bench_support']").set('rgba', '1 1 1 1')
        ET.SubElement(scene.find('worldbody'), 'light', name='qa_key', pos='.5 -.5 1.5',
                      dir='-.2 .2 -1', directional='true', diffuse='.6 .6 .6', ambient='.1 .1 .1')
        model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding='unicode'))
        model.vis.headlight.ambient[:] = .15
        model.vis.headlight.diffuse[:] = .35
        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.azimuth, camera.elevation = 125, -30
        option = mujoco.MjvOption()
        option.geomgroup[3] = 0
        output = dest / 'nominal_CPU_state_replay.mp4'
        count = 0
        with imageio.get_writer(output, fps=20, codec='libx264', macro_block_size=16) as video:
            for test, label in zip(tests, labels):
                row = next(r for r in result['cases'] if r['test'] == test and r['mass_kg'] == 0.2 and r['sliding_friction'] == 0.6 and r['dt_s'] == 0.00125)
                a = np.load(row['trajectory']['path'], allow_pickle=False)
                data = mujoco.MjData(model)
                travel = abs(a['com_world_m'][-1, 0] - a['com_world_m'][0, 0])
                camera.lookat[:] = [(a['com_world_m'][-1, 0] + a['com_world_m'][0, 0])/2, 0, .035]
                camera.distance = .45 + .5*travel
                for frame, index in enumerate(np.arange(0, len(a['qpos']), 40)):
                    data.qpos[:] = a['qpos'][index]
                    data.qvel[:] = a['qvel'][index]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    image = renderer.render()
                    video.append_data(image)
                    count += 1
                    if frame == 22:
                        imageio.imwrite(dest / (test + '_native_frame.png'), image)
        renderer.close()
        products.update(native_video_rendered=True, video=str(output), renderer_frames=count,
                        replay_extra_nonintegrating_forward_calls=count,
                        rendering_appearance_only_changes='Added checker support material and brighter lighting for visual QA; saved physical states unchanged.',
                        replay_scope='Saved CPU states displayed with visual mesh only; three tests in order; untextured geometry.')
    with (dest / ('render_manifest.json' if args.render else 'plot_manifest.json')).open('x') as stream:
        json.dump(products, stream, indent=2)
    print(json.dumps(products))

if __name__ == '__main__':
    main()
