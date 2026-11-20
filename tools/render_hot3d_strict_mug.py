"""Show the selected 001888 source interval, target and independently replayed strict-search path."""
import argparse
import io
from pathlib import Path
import tarfile
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from scipy.spatial.transform import Rotation
from render_mug_progress_video import Scene,font
from test_mug_learned_planning import physics,file_pin,verify_file_pin,write_json


def render(run):
    root=Path(__file__).resolve().parents[1]/'runs'
    source=file_pin(run/'report.json');result=physics.read_json(source)
    clip=result['clip']
    refpin=file_pin(root/('mug_unilateral_interval_v1' if clip==1888 else 'mug_annotated_windows_original_v1')/'report.json')
    item,=physics.read_json(refpin)['rows']
    inputs=physics.read_json(file_pin(root/'mug_annotated_windows_original_v1/intent.json'))['inputs']
    ref=physics.load_arrays(item['reference']);data=physics.load_arrays(result['trace'])
    init=physics.load_arrays(inputs['reference'])['initial_qpos27'];states=np.vstack([init,data['qpos']])
    T=ref['body'];qref=np.column_stack([ref['q'],T[:,:3,3],Rotation.from_matrix(T[:,:3,:3]).as_quat()[:,[3,0,1,2]]])
    assert len(qref)==len(states)==len(ref['t'])
    m=mujoco.MjModel.from_binary_path(str(verify_file_pin(inputs['model'])))
    m.vis.global_.offwidth=640;m.vis.global_.offheight=480
    m.vis.headlight.ambient[:]=.50;m.vis.headlight.diffuse[:]=.78
    d=mujoco.MjData(m);d.qpos[:]=init;mujoco.mj_forward(m,d)
    points=np.vstack([T[:,:3,3],d.site_xpos[[m.site(f'right_finger{k}_tip').id for k in range(1,6)]]])
    center=.5*(points.max(0)+points.min(0));scene=Scene(m,464,390,center)
    camera=mujoco.MjvCamera();camera.lookat[:]=center;camera.distance=.52;camera.azimuth=250;camera.elevation=-22
    def draw_state(q):
        scene.data.qpos[:]=q;mujoco.mj_forward(m,scene.data);scene.renderer.update_scene(scene.data,camera=camera)
        return Image.fromarray(scene.renderer.render().copy())
    target=run/'source_reference_replay.mp4'
    if target.exists():raise FileExistsError(target)
    mapping=[]
    try:
        with tarfile.open(verify_file_pin(item['archive'])) as archive,imageio.get_writer(str(target),fps=25,codec='libx264',quality=8,pixelformat='yuv420p',macro_block_size=1) as video:
            for i,t in enumerate(ref['t']):
                j=int(np.argmin(abs(ref['source_time']-(ref['source_time'][0]+round(1e9*t)))))
                frame=int(ref['frames'][j]);mapping.append(frame)
                encoded=archive.extractfile(f'{frame:06d}.image_214-1.jpg').read()
                rgb=ImageOps.contain(Image.open(io.BytesIO(encoded)).convert('RGB').transpose(Image.Transpose.ROTATE_270),(464,390))
                im=Image.new('RGB',(1392,510),'#16252d');draw=ImageDraw.Draw(im)
                for col,label in enumerate(('HOT3D 第一视角','固定腕运动学参考','自由杯原生重放')):
                    draw.text((464*col+12,7),label,font=font(22),fill='white')
                im.paste(rgb,((464-rgb.width)//2,42+(390-rgb.height)//2));im.paste(draw_state(qref[i]),(464,42));im.paste(draw_state(states[i]),(928,42))
                status='起点' if i==0 else ('连续合格' if i<=result['strict_controls'] else '已越过连续合格窗')
                mm=0. if i==0 else data['com_mm'][i-1];deg=0. if i==0 else data['so3_deg'][i-1]
                draw.text((12,453),f'{clip:06d}　原片帧 {frame}　参考 {t:.2f} 秒　0.5 倍速',font=font(18),fill='white')
                draw.text((790,453),f'{status}　质心偏差 {mm:.1f} mm　姿态偏差 {deg:.1f}°',font=font(18),fill='#f0ca86')
                video.append_data(np.asarray(im))
    finally:scene.close()
    reader=imageio.get_reader(target)
    try:
        assert reader.count_frames()==len(states) and reader.get_meta_data()['fps']==25
        Image.fromarray(reader.get_data(len(states)//2)).save(run/'decoded_mid.png')
    finally:reader.close()
    write_json(run/'video.json',dict(source=source,reference=refpin,video=file_pin(target),original_frames=mapping,
                                     display_speed=.5,solver='nominal CPU MuJoCo',strict_controls=result['strict_controls']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);render(p.parse_args().run.resolve())
