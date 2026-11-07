"""Render the exact assembled model's initial state; zero integration or grasp claims."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import scene_loader_v1 as loader

ROOT = Path(__file__).resolve().parent

def main():
    model = loader.load_model()
    data = mujoco.MjData(model)
    initial = loader.initial_state()
    data.qpos[:] = initial['qpos']
    data.qvel[:] = initial['qvel']
    data.ctrl[:] = initial['ctrl']
    mujoco.mj_forward(model, data)
    palm = model.body('right_palm_link').id
    phone = model.body('hot3d_phone24').id
    focus = (data.xpos[palm] + data.xpos[phone]) / 2
    model.vis.headlight.ambient[:] = .3
    model.vis.headlight.diffuse[:] = .65
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    renderer = mujoco.Renderer(model, width=640, height=480)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = focus
    camera.distance, camera.elevation = .5, -20
    images = []
    for azimuth in (135, -45):
        camera.azimuth = azimuth
        renderer.update_scene(data, camera=camera, scene_option=options)
        images.append(renderer.render().copy())
    renderer.close()
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, image, azimuth in zip(axes, images, (135, -45)):
        ax.imshow(image)
        ax.set_title(f'Native viewer, azimuth {azimuth} degrees')
        ax.axis('off')
    figure.suptitle('Fixed-wrist Hand2 + real-scale HOT3D phone: staging only\nPhone parked outside the hand; no grasp, dynamics, or retargeting result', fontsize=13)
    output = ROOT/'staging_preview_v1.png'
    figure.savefig(output, dpi=140)
    plt.close(figure)
    with (ROOT/'staging_preview_v1.json').open('x') as stream:
        json.dump({'figure': str(output), 'model_source': str(loader.OUTPUT/'hand2_phone_scene.mjb'),
                   'nonintegrating_forward_calls': 1, 'render_calls': 2, 'integration_steps': 0,
                   'visual_lighting_only_changes': True, 'Hand2_grasp_validated': False,
                   'initial_contact_candidates': int(data.ncon)}, stream, indent=2)
    print(output)

if __name__ == '__main__':
    main()
