"""Fixed repeat0 saved-pose comparison videos after all18 independent reviews."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_frozen_cost_long_design_v1 import settings,read
from supermimic_research.strict_json import write_json
from render_mug_progress_video import Scene,writer,FONT


def font(n): return ImageFont.truetype(FONT,n)


def main(a):
    started=time.perf_counter(); cp=a.config.resolve(); cfg,paths,_,_,_,_=settings(cp); rp=file_pin(paths['output']/'report.json'); vp=file_pin(paths['reader_output']/'report.json'); run=read(verify_file_pin(rp)); reviewed=read(verify_file_pin(vp))
    if any(r['passed']!=r['total'] or not all(r['checks'].values()) for r in [run,reviewed]) or reviewed['producer']!=rp: raise ValueError('Complete independently reviewed18-case execution required')
    out=ROOT/'runs/pen_frozen_cost_long_video_v1'
    if out.exists(): raise ValueError('Fresh video output required')
    out.mkdir(); receipt=freeze_sources([Path(__file__),ROOT/'tools/render_mug_progress_video.py',cp],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()); write_json(out/'source_receipt.json',receipt)
    case_map={r['declaration']['name']:r['report'] for r in run['rows']}; reviews={r['name']:read(verify_file_pin(r['report'])) for r in reviewed['rows']}; outputs={}; checks={}; indices={}; display_forwards=0
    original_forward=mujoco.mj_forward
    def forward(m,d):
        nonlocal display_forwards
        display_forwards+=1; return original_forward(m,d)
    with patch.object(mujoco,'mj_step',side_effect=AssertionError('No integration while rendering')),patch.object(mujoco,'mj_forward',side_effect=forward):
        for algorithm in cfg['algorithms']:
            folder=out/algorithm; folder.mkdir(); scenes=[]; cases=[]; traces=[]; pins=[]
            for label in cfg['policies']:
                name=f'R0_{algorithm}_{label}'; p=case_map[name]; c=read(verify_file_pin(p)); review=reviews[name]
                if review['passed']!=review['total'] or not all(review['checks'].values()): raise ValueError('All fixed repeat0 case reviews required')
                z=dict(np.load(verify_file_pin(c['trace']))); m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['model']))); m.vis.global_.offwidth=768; m.vis.global_.offheight=560; m.vis.headlight.ambient[:]=.45; m.vis.headlight.diffuse[:]=.75
                scene=Scene(m,512,530,z['COM_world_m'][0]); scene.data.mocap_pos[:]=z['initial_mocap_pos'][0]; scene.data.mocap_quat[:]=z['initial_mocap_quat'][0]
                scenes.append(scene); cases.append(c); traces.append(z); pins.append(p)
            durations=[c['selected_duration_s'] for c in cases]; times=np.minimum(np.arange(math.ceil(max(durations)/.05*30)+1)*(.05/30),max(durations)); chosen=[]; frames=0; path=folder/'frozen_actor_comparison.mp4'
            try:
                with writer(path) as video:
                    for fi,t in enumerate(times):
                        im=Image.new('RGB',(1536,832),(17,22,32)); draw=ImageDraw.Draw(im)
                        draw.text((18,10),'冻结策略长时评估：'+('原版求解器' if algorithm=='INSTALLED_ORIGINAL_OR' else '有条件稳定摩擦代价'),font=font(28),fill='white')
                        draw.text((18,50),'固定第0次数值重复 · 20倍慢放 · 既有权重，本轮未训练 · 无物体外力辅助',font=font(21),fill=(180,200,220)); ids=[]
                        for i,(c,z,s,duration,label) in enumerate(zip(cases,traces,scenes,durations,cfg['policies'])):
                            j=int(np.searchsorted(np.arange(1,len(z['post_qpos'])+1)*.0025,min(float(t),duration)+1e-12,side='right')-1); ids.append(j)
                            pose=z['initial_qpos'] if j<0 else z['post_qpos'][j]; x=512*i
                            draw.text((x+12,88),{'INITIAL':'INITIAL · 共同初始化','RESIDUAL':'RESIDUAL · 冻结2000轮权重','SUPPORT':'SUPPORT · 冻结2000轮权重'}[label],font=font(21),fill=(100,215,230))
                            stopped=t>=duration
                            status='实测执行' if not stopped else {'PHYSICAL_AUDIT_STOP':'物理审计停止，末帧','TASK_FAILURE':'任务终止，末帧','REFERENCE_COMPLETION':'参考结束，末帧'}[c['stop_reason']]
                            draw.text((x+12,118),f'{min(float(t),duration):.3f}/{duration:.3f}s  '+status,font=font(18),fill=(220,120,120) if stopped and not c['full_task_success'] else (145,215,160))
                            im.paste(s.draw(pose,250),(x,154))
                        draw.text((18,703),'保存的2.5ms姿态，无插值；停止后保持末帧，缺失未来未继续仿真。',font=font(22),fill=(220,225,235))
                        draw.text((18,746),'完整参考7.36秒；物理审计停止与任务失败分别保留，严格任务结果以报告为准。',font=font(20),fill=(180,200,220))
                        draw.text((18,783),'受控单条参考、完美位姿；不是ego数据、真机或新策略训练的效果。',font=font(20),fill=(180,200,220))
                        chosen.append(ids); frame=np.asarray(im)
                        if fi==0:
                            im.save(folder/'start.png')
                            for _ in range(18): video.append_data(frame); frames+=1
                        if fi==len(times)//2: im.save(folder/'mid.png')
                        video.append_data(frame); frames+=1
                    im.save(folder/'end.png')
                    for _ in range(30): video.append_data(frame); frames+=1
            finally:
                for scene in scenes: scene.close()
            decoded=imageio.get_reader(path)
            try:
                checks[algorithm+'_frames']=decoded.count_frames()==frames
                for tag,index in [('start',0),('mid',frames//2),('end',frames-1)]:
                    frame=decoded.get_data(index); checks[algorithm+'_'+tag]=frame.shape==(832,1536,3); Image.fromarray(frame).save(folder/('decoded_'+tag+'.png'))
            finally: decoded.close()
            indices[algorithm]=np.asarray(chosen,np.int32); outputs[algorithm]=dict(video=file_pin(path),cases=pins,durations_s=durations,frames=frames,fps=30,speed=.05)
            checks[algorithm+'_repeat0']=all(c['repetition']==0 for c in cases)
    np.savez_compressed(out/'indices.npz',**indices); verify_source_snapshot(receipt,live=True)
    write_json(out/'report.json',dict(schema='pen_frozen_cost_long_saved_video:1',producer=rp,review=vp,source_receipt=file_pin(out/'source_receipt.json'),outputs=outputs,indices=file_pin(out/'indices.npz'),font=file_pin(FONT),
        checks=checks,passed=sum(checks.values()),total=len(checks),display_forward_calls=display_forwards,actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,wall_s=time.perf_counter()-started,scope='All fixed repeat0 policies per backend,actual saved native poses,explicit stop stills. No interpolation,replay integration,new Actor,ego or hardware result.'))
    print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[sum(checks.values()),len(checks)])),flush=True)
    if not all(checks.values()): raise ValueError('Video encoding/readback discrepancy')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); main(p.parse_args())
