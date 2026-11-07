"""Native scene stills for inspection; not a rollout or holding demonstration."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin


def main(args):
    import mujoco,matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    bind=json.loads((args.reference/'binding.json').read_text());run=json.loads((args.engagement/'report.json').read_text())
    before=ROOT/'runs/mug_equilibrium_v1/seed00/final_state.npz';after=verify_file_pin(run['results'][1]['candidate'])
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(bind['model'])))
    # Display buffer only, fresh in-memory model; no physics array or saved MJB edit.
    model.vis.global_.offwidth=1100;model.vis.global_.offheight=780
    renderer=mujoco.Renderer(model,height=780,width=1100);fig,axes=plt.subplots(2,2,figsize=(12,8.6))
    center=None
    for col,path in enumerate([before,after]):
        state=np.load(path);data=mujoco.MjData(model);data.qpos[:]=state['qpos27'];data.ctrl[:]=state['ctrl20'];mujoco.mj_forward(model,data)
        if center is None:center=data.xipos[bind['mug_body_id']].copy()
        for row,azimuth in enumerate([100,250]):
            camera=mujoco.MjvCamera();camera.lookat[:]=center;camera.distance=.24;camera.azimuth=azimuth;camera.elevation=-35
            option=mujoco.MjvOption();renderer.update_scene(data,camera=camera,scene_option=option);pixels=renderer.render()
            axes[row,col].imshow(pixels);axes[row,col].axis('off')
            axes[row,col].set_title(('Original local-force pose' if col==0 else 'Source-region pose, 50 um engagement')+f' | view {row+1}',fontsize=11)
    renderer.close();fig.suptitle('Fixed object and wrist; native Hand2 geometry inspection',fontsize=15)
    fig.text(.5,.025,'Static scene stills only. New pose matches candidate regions and has three contacts, but fails the frozen 6D support test.\nSource regions are uncertain annotation geometry. No dynamic hold, independent RGB reconstruction or policy improvement.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.95));out.mkdir(parents=True);fig.savefig(out/'mug_patch_poses.png',dpi=160);plt.close(fig)
    verify_file_pin(bind['model'])
    report=dict(schema='mug_patch_native_stills:1',source=file_pin(__file__),model=bind['model'],before=file_pin(before),after=file_pin(after),
        output=file_pin(out/'mug_patch_poses.png'),cameras=[dict(distance=.24,elevation=-35,azimuth=a,lookat=center.tolist()) for a in [100,250]],
        native_static_forwards=2,integration_steps=0,neural_updates=0,display_only_buffer_changed=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['reference','engagement','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
