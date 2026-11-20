"""Render fixed repeat zero of the two costs on the same clip003 clock."""
import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from evaluate_pen_dev_clip import read,file_pin,verify_file_pin,write_json
from render_mug_progress_video import Scene,font
from supermimic_research.mechanics.pen_task_contracts import pose_components


def render(run):
    source=file_pin(run/'report.json')
    j=read(source)
    intent=read(j['intent'])
    config=read(read(read(intent['baseline'])['intent'])['configs'])['clips']['clip_003']
    with np.load(verify_file_pin(config['item']['source'])) as f:
        hand=f['reference_qpos']
    trajectories=[]
    for arm in ('pose','pose_support'):
        row=next(x for x in j['clips']['clip_003'] if x['rep']==0 and x['arm']==arm)
        case=read(row['native_case'])
        folder=Path(row['native_case']['path']).parent
        model_pin=read(file_pin(folder/'cpu_export/report.json'))['model']
        with np.load(verify_file_pin(case['trace'])) as f:
            ref=np.column_stack([f['COM_world_m'],f['quaternion_wxyz'],hand])
            qs=np.vstack([f['pre_qpos'][0],f['post_qpos'][7::8]])
            mount=(f['post_mocap_pos'][0],f['post_mocap_quat'][0])
        angles=pose_components(qs[:,3:7],ref[:len(qs),3:7])
        trajectories.append((qs,np.rad2deg(angles['shaft_angle_rad']),row))
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(model_pin)))
    model.vis.global_.offwidth=512
    model.vis.global_.offheight=400
    model.vis.headlight.ambient[:]=.45
    model.vis.headlight.diffuse[:]=.75
    scene=Scene(model,448,352,ref[0,:3])
    scene.data.mocap_pos[:],scene.data.mocap_quat[:]=mount
    length=max(len(q) for q,_,_ in trajectories)
    path=run/'clip003_replay.mp4'
    if path.exists():raise FileExistsError(path)
    labels=['原始运动学参考','位姿代价','位姿＋接触扳手代价']
    try:
        with imageio.get_writer(str(path),fps=50,codec='libx264',quality=8,macro_block_size=1,pixelformat='yuv420p') as video:
            for i in range(length):
                im=Image.new('RGB',(1344,444),'#16252d')
                draw=ImageDraw.Draw(im)
                for n,label in enumerate(labels):
                    pos=448*n
                    draw.text((pos+12,7),label,font=font(22),fill='white')
                    if n==0:
                        q=ref[i]
                        caption=f'参考时间 {i*.02:.2f} / {(len(ref)-1)*.02:.2f} 秒'
                    else:
                        qs,angles,row=trajectories[n-1]
                        idx=min(i,len(qs)-1)
                        q=qs[idx]
                        status='终止画面' if idx==len(qs)-1 else '原生仿真'
                        caption=f'{status}  {idx*.02:.2f} 秒  笔轴偏差 {angles[idx]:.1f}°'
                    im.paste(scene.draw(q,250),(pos,42))
                    draw.text((pos+10,402),caption,font=font(17),fill='#e1eced')
                video.append_data(np.asarray(im))
    finally:
        scene.close()
    reader=imageio.get_reader(str(path))
    try:
        assert reader.count_frames()==length and reader.get_meta_data()['fps']==50
        for idx in (0,length//2,length-1):
            assert reader.get_data(idx).shape==(444,1344,3)
    finally:reader.close()
    write_json(run/'video.json',dict(source=source,rep=0,dt=.02,frames=length,video=file_pin(path)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    render(p.parse_args().run.resolve())
