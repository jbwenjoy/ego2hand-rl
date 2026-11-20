"""Time-aligned source RGB, kinematic reference and executed shooting trial."""
import io,json,sys,tarfile
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageOps
from render_mug_progress_video import Scene,writer,font
from evaluate_mug_live_wrench import ROOT,verify_file_pin,file_pin,write_json


def main(run,out):
    out.mkdir(exist_ok=False)
    read=lambda pin:json.loads(verify_file_pin(pin).read_bytes())
    results=json.loads((run/'report.json').read_bytes())
    row=next(r for r in results['rows'] if r['mode']=='support' and r['seed']==7411)
    ff=read(read(results['intent'])['parent']);trial=read(read(ff['intent'])['parent'])
    contract=read(read(trial['dynamic'])['intent']);reference=read(contract['reference'])
    with np.load(verify_file_pin(reference['output'])) as f:z=dict(f)
    with np.load(verify_file_pin(read(trial['dynamic'])['trajectory'])) as f:qref=f['q20']
    with np.load(verify_file_pin(row['trace'])) as f:actual=np.vstack([z['initial_qpos27'],f['qpos']])
    clip=verify_file_pin(json.loads((ROOT/'runs/mug_rgb_projection_v1/member_receipts.json').read_bytes())['clip'])
    m=mujoco.MjModel.from_binary_path(str(verify_file_pin(reference['model'])));m.vis.global_.offwidth=640;m.vis.global_.offheight=480
    scene=Scene(m,464,390,z['COM_world_m'][0]);cache={};path=out/'source_reference_shooting.mp4';frames=0;mapping=[]
    try:
        with tarfile.open(clip) as tar,writer(path) as video:
            for k in range(len(qref)):
                clock=z['timestamps_ns'][k]
                idx=int(np.argmin(abs(z['source_timestamps_ns']-clock)));frame=int(z['source_frame_ids'][idx])
                if frame not in cache:
                    raw=tar.extractfile(f'{frame:06d}.image_214-1.jpg').read()
                    cache[frame]=ImageOps.contain(Image.open(io.BytesIO(raw)).convert('RGB'),(464,390))
                q=np.r_[qref[k],z['T_world_from_BODY'][k,:3,3],z['quaternion_wxyz'][k]]
                active=min(k,len(actual)-1)
                im=Image.new('RGB',(1392,544),'#141b24');d=ImageDraw.Draw(im)
                for x,label in enumerate(('真实第一视角 · HOT3D','固定腕参考 · 运动学','原生仿真 · 接触扳手前瞻')):
                    d.text((x*464+12,10),label,font=font(23),fill='white')
                img=cache[frame];im.paste(img,((464-img.width)//2,52+(390-img.height)//2))
                im.paste(scene.draw(q,250),(464,52));im.paste(scene.draw(actual[active],250),(928,52))
                d.text((14,455),f'参考时刻 {k*.02:.2f} s  ·  原片帧 {frame}  ·  0.3× 慢放',font=font(21),fill='white')
                status='达到停止阈值，保留末帧' if k>=len(actual)-1 and not row['full'] else '执行中'
                d.text((740,455),f'仿真时刻 {active*.02:.2f} s  ·  {status}',font=font(21),fill='#f0ca86')
                d.text((14,500),'右手与杯子标注 · clip 001884 · 源图取最近时间戳，仿真只绘制实测姿态。',font=font(17),fill='#b3c0ce')
                for _ in range(2):video.append_data(np.asarray(im));frames+=1
                mapping.append((k,frame,active))
    finally:scene.close()
    reader=imageio.get_reader(path)
    try:
        assert reader.count_frames()==frames
        Image.fromarray(reader.get_data(30)).save(out/'decoded_030.png')
    finally:reader.close()
    write_json(out/'report.json',dict(video=file_pin(path),run=file_pin(run/'report.json'),frames=frames,mapping=mapping,scope='Fixed seed 7411 support arm; reference and actual share camera. RGB and robot are different viewpoints.'))
if __name__=='__main__':main(*map(Path,sys.argv[1:]))

