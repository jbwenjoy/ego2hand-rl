"""Render RGB, anchored IK and executed free-mug dynamics for the new annotated window."""
import argparse
import io
import json
from pathlib import Path
import tarfile
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from scipy.spatial.transform import Rotation
from render_mug_progress_video import Scene,font
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json


def render(run):
    source=file_pin(run/'report.json')
    j=json.loads(verify_file_pin(source).read_bytes())
    cfg=json.loads(verify_file_pin(j['intent']).read_bytes())
    if 'original_protocol' not in cfg:
        parent=json.loads(verify_file_pin(cfg['parent']).read_bytes())
        cfg=json.loads(verify_file_pin(parent['intent']).read_bytes())
    orig=json.loads(verify_file_pin(cfg['original_protocol']).read_bytes())
    inputs=orig['inputs'];m=mujoco.MjModel.from_binary_path(str(verify_file_pin(inputs['model'])))
    old=np.load(verify_file_pin(inputs['reference']))
    for row in j['rows']:
        if 'stats' not in row:continue
        trial=next(r for r in row['stats'] if r['arm']=='lookahead_8201')
        with np.load(verify_file_pin(trial['trace'])) as f:
            qactual=np.vstack([old['initial_qpos27'],f['qpos']])
            mm=np.r_[0,f['com_mm']];deg=np.r_[0,f['so3_deg']]
        with np.load(verify_file_pin(row['reference'])) as f:ref=dict(f)
        T=ref['body'];quat=Rotation.from_matrix(T[:,:3,:3]).as_quat()[:,[3,0,1,2]]
        qkin=np.column_stack([ref['q'],T[:,:3,3],quat]);assert len(qactual)==len(qkin)
        m.vis.global_.offwidth=640;m.vis.global_.offheight=480
        m.vis.headlight.ambient[:]=.50;m.vis.headlight.diffuse[:]=.78
        prep=mujoco.MjData(m);prep.qpos[:]=qkin[0];mujoco.mj_forward(m,prep)
        points=np.vstack([T[:,:3,3],prep.site_xpos[[m.site(f'right_finger{j}_tip').id for j in range(1,6)]]])
        center=.5*(points.max(0)+points.min(0))
        scene=Scene(m,464,390,center)
        camera=mujoco.MjvCamera();camera.lookat[:]=center;camera.distance=.58;camera.azimuth=250;camera.elevation=-22
        def draw_state(q):
            scene.data.qpos[:]=q;mujoco.mj_forward(m,scene.data)
            scene.renderer.update_scene(scene.data,camera=camera)
            return Image.fromarray(scene.renderer.render().copy())
        folder=run/str(row['clip']);target=folder/'source_and_robot_wide.mp4'
        if target.exists():raise FileExistsError(target)
        mapping=[]
        try:
            with tarfile.open(verify_file_pin(row['archive'])) as tar,imageio.get_writer(str(target),fps=25,codec='libx264',quality=8,pixelformat='yuv420p',macro_block_size=1) as writer:
                for i,t in enumerate(ref['t']):
                    stamp=ref['source_time'][0]+round(1e9*t)
                    ix=int(np.argmin(abs(ref['source_time']-stamp)));frame=int(ref['frames'][ix])
                    jpg=tar.extractfile(f'{frame:06d}.image_214-1.jpg').read()
                    rgb=ImageOps.contain(Image.open(io.BytesIO(jpg)).convert('RGB').transpose(Image.Transpose.ROTATE_270),(464,390))
                    im=Image.new('RGB',(1392,500),'#16252d');draw=ImageDraw.Draw(im)
                    for col,label in enumerate(('原片 · HOT3D 第一视角','固定腕参考 · 五指 IK','自由杯仿真 · 三步前瞻')):
                        draw.text((464*col+10,8),label,font=font(22),fill='white')
                    im.paste(rgb,((464-rgb.width)//2,44+(390-rgb.height)//2))
                    im.paste(draw_state(qkin[i]),(464,44));im.paste(draw_state(qactual[i]),(928,44))
                    flag='首次越界后' if trial['first_breach'] and t>=trial['first_breach']['time_s'] else '阈值内'
                    draw.text((12,452),f'片段 {row["clip"]:06d}　原片帧 {frame}　参考 {t:.2f} 秒　0.5 倍速',font=font(17),fill='white')
                    draw.text((780,452),f'{flag}　质心偏差 {mm[i]:.1f} mm　姿态偏差 {deg[i]:.1f}°',font=font(17),fill='#f0ca86')
                    writer.append_data(np.asarray(im));mapping.append(frame)
        finally:scene.close()
        reader=imageio.get_reader(target)
        try:
            assert reader.count_frames()==len(mapping) and reader.get_meta_data()['fps']==25
            Image.fromarray(reader.get_data(len(mapping)//2)).save(folder/'video_wide_midpoint.png')
        finally:reader.close()
        write_json(folder/'video_wide.json',dict(source=source,method=trial['arm'],video=file_pin(target),frames=mapping,
            RGB_display_rotation='90deg clockwise',camera_center=center.tolist(),camera_distance=.58,playback_speed=.5))
    old.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);render(p.parse_args().run.resolve())
