"""Saved four-cell CCD actual control videos; no re-integration or Actor inference."""
import argparse
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
import imageio.v2 as imageio

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from tools.render_mug_progress_video import Scene,canvas,writer


def main(a):
    r,b=[json.loads(p.read_text()) for p in [a.report,a.baseline]]
    if a.output.exists():raise ValueError('Fresh native control replay')
    for pin in r['sources']+b['sources']:verify_file_pin(pin)
    a.output.mkdir();videos=[]
    for dt in [.005,.0025]:
        cases=[next(c for c in source['cases'] if c['dt']==dt and c['name']==mode) for source in [b,r] for mode in ['STATE_ONLY','STATE_PLUS_SUPPORT']]
        if any(c['summary']['qualified'] for c in cases):raise ValueError('Refresh failure-specific annotations before rendering a qualified result')
        traces=[dict(np.load(verify_file_pin(c['trace']))) for c in cases];scenes=[]
        for c,z in zip(cases,traces):
            m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])))
            for key,value in dict(np.load(verify_file_pin(c['GPU_physical_context']))).items():getattr(m,key)[:]=value[0]
            m.vis.global_.offwidth=768;m.vis.global_.offheight=560;m.vis.headlight.ambient[:]=.45;m.vis.headlight.diffuse[:]=.75
            scene=Scene(m,512,550,z['COM_world_m'][0]);mujoco.mj_setState(m,scene.data,z['initial_native_state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_setConst(m,scene.data);scenes.append(scene)
        durations=[c['summary']['metrics']['actual_duration_s'] for c in cases];path=a.output/f'GPU_pen_CCD_matrix_{dt:g}s.mp4';frames=0;selected=[]
        try:
            samples=np.linspace(0,max(durations),math.ceil(max(durations)/.05*30)+1)
            with writer(path) as video:
                for i,t in enumerate(samples):
                    panels=[];labels=[];indices=[]
                    for name,z,scene,duration in zip(['CCD 1e-6：状态','CCD 1e-6：状态＋支持','CCD 1e-7：状态','CCD 1e-7：状态＋支持'],traces,scenes,durations):
                        j=int(np.searchsorted(np.arange(1,len(z['post_qpos'])+1)*dt,min(t,duration)+1e-12,side='right')-1)
                        panels.append(scene.draw(z['initial_qpos'] if j<0 else z['post_qpos'][j],250));indices.append(j)
                        labels.append(name+('：终止静帧' if t>=duration else '：实测'))
                    selected.append(indices)
                    frame=canvas(2048,'最新 GPU 受控转笔：CCD 数值配置与接触支持对照，八组均未完成任务',
                        f'步长 {dt*1000:g} ms；每步57候选、预测20ms；20倍慢放，保存实测姿态，无插值',panels,labels,
                        '各列物理时间：'+' / '.join(f'{min(t,d):.3f}' for d in durations)+' 秒',
                        ['计划7.36秒；实际终止：'+' / '.join(f'{d:.2f}' for d in durations)+'秒。终止列停留末帧，未继续仿真。',
                         '无物体外力辅助；未训练Actor；受控参考，非真实ego。CCD候选未通过力预测资格，全部失败保留。'])
                    if i==0:
                        Image.fromarray(frame).save(a.output/f'{dt:g}_start.png')
                        for _ in range(18):video.append_data(frame);frames+=1
                    if i==len(samples)//2:Image.fromarray(frame).save(a.output/f'{dt:g}_mid.png')
                    video.append_data(frame);frames+=1
                Image.fromarray(frame).save(a.output/f'{dt:g}_end.png')
                for _ in range(30):video.append_data(frame);frames+=1
        finally:
            for scene in scenes:scene.close()
        reader=imageio.get_reader(path);decoded=[]
        try:
            for tag,index in [('start',0),('mid',frames//2),('end',frames-1)]:
                decoded_frame=reader.get_data(index)
                if decoded_frame.shape!=(800,2048,3):raise ValueError('Decoded video shape changed')
                output=a.output/f'{dt:g}_decoded_{tag}.png';Image.fromarray(decoded_frame).save(output);decoded.append(file_pin(output))
            if reader.count_frames()!=frames:raise ValueError('Encoded/decoded frame count mismatch')
        finally:reader.close()
        videos.append(dict(dt=dt,video=file_pin(path),frames=frames,fps=30,speed=.05,physical_durations_s=durations,selected_post_indices=selected,decoded_frames=decoded,
            inputs=[{key:c[key] for key in ['trace','host_topology','GPU_physical_context']} for c in cases]))
    result=dict(schema='pen_native_CCD_matrix_actual_video:1',run=file_pin(a.report),baseline=file_pin(a.baseline),sources=[file_pin(__file__),file_pin(ROOT/'tools/render_mug_progress_video.py')],videos=videos,
        neural_updates=0,native_integrations=0,scope='Actual saved native GPU trajectories, explicit per-column clocks and terminal stills. Graphics forward only, no new dynamics, interpolation or successful policy/ego claim.')
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(videos=[v['video'] for v in videos])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
