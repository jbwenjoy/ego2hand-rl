"""Actual saved initial/iteration2/iteration32 mean-policy trajectories, no physics."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image
from tools.render_mug_progress_video import Scene,canvas,writer,FONT
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main(a):
    rp=a.report.resolve();r=json.loads(rp.read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or [c['name'] for c in r['cases']]!=['INITIAL_MEAN','ITER2_MEAN','ITER32_MEAN']:
        raise ValueError('Fresh declared actual-policy matrix video')
    out.mkdir();receipt=freeze_sources([Path(__file__).resolve(),ROOT/'tools/render_mug_progress_video.py'],out/'source_snapshot',ROOT,
        subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    cases=r['cases'];z=[dict(np.load(verify_file_pin(c['trace']))) for c in cases];scenes=[]
    durations=[c['summary']['metrics']['actual_duration_s'] for c in cases]
    labels=['初始化（已校准观测）','2轮 / 64次优化更新','32轮 / 1024次优化更新']
    for c,v in zip(cases,z):
        model=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['model'])))
        model.vis.global_.offwidth=768;model.vis.global_.offheight=560;model.vis.headlight.ambient[:]=.45;model.vis.headlight.diffuse[:]=.75
        s=Scene(model,512,550,v['COM_world_m'][0]);s.data.mocap_pos[:]=v['initial_mocap_pos'][0];s.data.mocap_quat[:]=v['initial_mocap_quat'][0];scenes.append(s)
    path=out/'native_PPO_initial_iter2_iter32.mp4';frames=0;selected=[];sampletimes=np.linspace(0,max(durations),math.ceil(max(durations)/.05*30)+1)
    try:
        with writer(path) as video:
            for i,t in enumerate(sampletimes):
                panels=[];names=[];indices=[]
                for label,trace,scene,d in zip(labels,z,scenes,durations):
                    j=int(np.searchsorted(np.arange(1,len(trace['post_qpos'])+1)*.0025,min(t,d)+1e-12,side='right')-1)
                    panels.append(scene.draw(trace['initial_qpos'] if j<0 else trace['post_qpos'][j],250));indices.append(j)
                    names.append(label+('：终止静帧' if t>=d else '：实测'))
                selected.append(indices)
                frame=canvas(1536,'原生 PPO 受控转笔：初始化、2轮、32轮（实际轨迹）',
                    '20倍慢放；2.5ms物理 / 50Hz控制；均值动作；保存姿态，无插值；完美位姿，非ego',panels,names,
                    '各列物理时间：'+' / '.join(f'{min(t,d):.3f}' for d in durations)+' 秒',
                    ['原任务7.36秒；本批实际终止：'+' / '.join(f'{d:.2f}' for d in durations)+'秒。终止后未继续仿真。',
                     '无物体外力辅助；同一训练参考。姿态/穿透等严格门槛仍失败，不宣称完成任务或CHORD收益。'])
                if i==0:
                    Image.fromarray(frame).save(out/'start.png')
                    for _ in range(18):video.append_data(frame);frames+=1
                if i==len(sampletimes)//2:Image.fromarray(frame).save(out/'mid.png')
                video.append_data(frame);frames+=1
            Image.fromarray(frame).save(out/'end.png')
            for _ in range(30):video.append_data(frame);frames+=1
    finally:
        for scene in scenes:scene.close()
    reader=imageio.get_reader(path);decoded=[]
    try:
        if reader.count_frames()!=frames:raise ValueError('Encoded frame count mismatch')
        for tag,j in [('start',0),('mid',frames//2),('end',frames-1)]:
            value=reader.get_data(j)
            if value.shape!=(800,1536,3):raise ValueError('Decoded shape mismatch')
            target=out/f'decoded_{tag}.png';Image.fromarray(value).save(target);decoded.append(file_pin(target))
    finally:reader.close()
    verify_source_snapshot(receipt,live=True)
    result=dict(schema='native_PPO_actual_policy_video:1',evaluation=file_pin(rp),source_receipt=file_pin(out/'source_receipt.json'),
        font=file_pin(FONT),video=file_pin(path),frames=frames,fps=30,speed=.05,durations_s=durations,
        selected_post_indices=selected,decoded_frames=decoded,render_forward_calls=3*len(sampletimes),native_integrations=0,optimizer_updates=0,
        scope='Actual saved native poses, explicit times/terminal stills, no interpolation or dynamics. Controlled same-reference training diagnostic, not successful task/ego/CHORD claim.')
    write(out/'report.json',result);print(json.dumps(dict(video=result['video'],frames=frames,durations=durations)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
