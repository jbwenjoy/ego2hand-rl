"""Compare from-start open-loop and feedback replays with the same one-millimetre offset."""
import argparse
import io
import tarfile
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from scipy.spatial.transform import Rotation
from render_mug_progress_video import Scene,font
from test_mug_witness_mpc import h3


def render(out):
    root=Path(__file__).resolve().parents[1]/'runs'
    closed=h3.file_pin(root/'mug_witness_mpc_v1/report.json')
    result=h3.physics.read_json(closed);spec=h3.physics.read_json(result['intent'])
    baseline=h3.physics.read_json(spec['prior']);cfg=h3.physics.read_json(h3.physics.read_json(spec['fit'])['intent'])['inputs']
    clip=1888;condition='start_x_minus_mm'
    open_case=next(r for r in baseline['rows'] if r['clip']==clip and r['condition']==condition and r['arm']=='open')
    feedback=next(r for r in result['rows'] if r['clip']==clip and r['condition']==condition and r['mode']=='wrench')
    item,=h3.physics.read_json(h3.file_pin(root/'mug_unilateral_interval_v1/report.json'))['rows']
    ref=h3.physics.load_arrays(item['reference']);old=h3.physics.load_arrays(cfg['reference'])
    model=mujoco.MjModel.from_binary_path(str(h3.verify_file_pin(cfg['model'])))
    qs=old['initial_qpos27'].copy();qa=h3.physics.read_json(cfg['binding'])['mug_free_qpos']
    qs[qa:qa+3]+=Rotation.from_quat(qs[qa+np.array([4,5,6,3])]).apply([-.001,0,0])
    traces=[h3.physics.load_arrays(r['trace']) for r in [open_case,feedback]]
    poses=[np.vstack([qs,z['qpos']]) for z in traces];limits=[open_case['passing'],feedback['passed']]
    model.vis.global_.offwidth=640;model.vis.global_.offheight=480
    model.vis.headlight.ambient[:]=.50;model.vis.headlight.diffuse[:]=.78
    d=mujoco.MjData(model);d.qpos[:]=qs;mujoco.mj_forward(model,d)
    points=np.vstack([ref['body'][:,:3,3],d.site_xpos[[model.site(f'right_finger{k}_tip').id for k in range(1,6)]]])
    center=(points.min(0)+points.max(0))*.5
    scene=Scene(model,464,390,center);camera=mujoco.MjvCamera()
    camera.lookat[:]=center;camera.distance=.52;camera.azimuth=250;camera.elevation=-22
    out.mkdir(exist_ok=False);dest=out/'offset_feedback.mp4';mapping=[]
    try:
        with tarfile.open(h3.verify_file_pin(item['archive'])) as archive,imageio.get_writer(str(dest),fps=25,codec='libx264',quality=8,pixelformat='yuv420p',macro_block_size=1) as vid:
            for i,t in enumerate(ref['t']):
                j=int(np.argmin(abs(ref['source_time']-ref['source_time'][0]-round(1e9*t))))
                frame=int(ref['frames'][j]);mapping.append(frame)
                rgb=Image.open(io.BytesIO(archive.extractfile(f'{frame:06}.image_214-1.jpg').read())).convert('RGB').transpose(Image.Transpose.ROTATE_270)
                rgb=ImageOps.contain(rgb,(464,390))
                im=Image.new('RGB',(1392,520),'#16252d');draw=ImageDraw.Draw(im)
                for n,label in enumerate(('HOT3D 原片 · 单右手时段','冻结指令重放','两步反馈 · 位姿与扳手代价')):
                    draw.text((464*n+10,7),label,font=font(22),fill='white')
                im.paste(rgb,((464-rgb.width)//2,44+(390-rgb.height)//2))
                for a,states in enumerate(poses):
                    k=min(i,len(states)-1)
                    scene.data.qpos[:]=states[k];mujoco.mj_forward(model,scene.data)
                    scene.renderer.update_scene(scene.data,camera=camera)
                    im.paste(Image.fromarray(scene.renderer.render().copy()),(464*(a+1),44))
                    metrics=traces[a]['metrics'][k-1,:2] if k else [1.,0.]
                    status='起点' if not i else ('连续合格' if i<=limits[a] else '首个失败窗 · 定格')
                    draw.text((464*(a+1)+10,444),f'{status}　{k*.02:.2f} s',font=font(19),fill='#e9c886')
                    draw.text((464*(a+1)+10,477),f'质心 {metrics[0]:.1f} mm　姿态 {metrics[1]:.1f}°',font=font(17),fill='white')
                draw.text((10,445),f'001888　原片帧 {frame}　0.5 倍速',font=font(19),fill='white')
                draw.text((10,478),'两组杯子初始位置均沿杯体 x 轴偏移 −1 mm',font=font(16),fill='white')
                vid.append_data(np.asarray(im))
    finally:scene.close()
    rd=imageio.get_reader(dest)
    try:
        assert rd.count_frames()==len(ref['t']) and rd.get_meta_data()['fps']==25
        Image.fromarray(rd.get_data(len(mapping)//2)).save(out/'decoded_mid.png')
    finally:rd.close()
    h3.write_json(out/'report.json',dict(closed=closed,open=spec['prior'],condition=condition,limits=limits,video=h3.file_pin(dest),frames=mapping))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);render(p.parse_args().out.resolve())
