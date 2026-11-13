"""Replay all four fixed controlled traces, preserving saved wrist mocap state."""
from pathlib import Path
import argparse
import json
import math
import sys
import numpy as np
import mujoco
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from tools.render_mug_progress_video import Scene,canvas,writer


def main(args):
    if args.output.exists():raise ValueError('Fresh video output')
    report=json.loads(args.report.read_text(encoding='utf8'));args.output.mkdir();outputs=[]
    for task in ['hand2-reorient','hand2-pen-spin']:
        cases=[next(c for c in report['cases'] if c['task']==task and c['controller']==mode) for mode in ['ZERO','SAVED_ACTOR']]
        zs=[dict(np.load(verify_file_pin(c['trace']))) for c in cases]
        models=[mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology']))) for c in cases]
        scenes=[]
        for m,z in zip(models,zs):
            m.vis.global_.offwidth=768;m.vis.global_.offheight=560
            m.vis.headlight.ambient[:]=.45;m.vis.headlight.diffuse[:]=.75
            scene=Scene(m,640,550,z['COM_world_m'][0])
            mujoco.mj_setState(m,scene.data,z['initial_native_state'],mujoco.mjtState.mjSTATE_INTEGRATION)
            scenes.append(scene)
        durations=[c['summary']['metrics']['actual_duration_s'] for c in cases]
        speed=.25 if task=='hand2-reorient' else .1;name='方块' if task=='hand2-reorient' else '单条转笔'
        path=args.output/f'{task}_zero_vs_saved_actor.mp4';frames=0
        try:
            with writer(path) as video:
                samples=np.linspace(0,max(durations),math.ceil(max(durations)/speed*30)+1)
                for i,t in enumerate(samples):
                    panels=[];labels=[]
                    for label,case,z,scene,duration in zip(['零动作基线','已有短训练 Actor'],cases,zs,scenes,durations):
                        index=min(len(z['post_qpos'])-1,max(0,round(t/case['dt'])-1))
                        q=z['initial_qpos'] if i==0 else z['post_qpos'][index]
                        panels.append(scene.draw(q,250))
                        labels.append(label+('：结束后静帧' if t>=duration else '：实际 GPU 记录'))
                    frame=canvas(1280,f'{name}：统一无辅助、无复位评估，均未通过',
                        f'同一初始完整状态；{speed:.2f}× 慢放；首次终止即停止；以下为保存状态回放',panels,labels,
                        '各列物理时间：'+' / '.join(f'{min(t,d):.3f}' for d in durations)+' 秒',
                        ['方块满 2 秒但跟踪与穿透不合格。' if task=='hand2-reorient' else '固定 clip_001：计划 7.36 秒，两组均在 0.16 秒位置误差超限。',
                         '原始 smoke 权重，未新增训练；不代表成功操作或真实 ego 重定向。'])
                    if i==0:
                        Image.fromarray(frame).save(args.output/f'{task}_start.png')
                        for _ in range(18):video.append_data(frame);frames+=1
                    if i==len(samples)//2:Image.fromarray(frame).save(args.output/f'{task}_mid.png')
                    video.append_data(frame);frames+=1
                Image.fromarray(frame).save(args.output/f'{task}_end.png')
                for _ in range(30):video.append_data(frame);frames+=1
        finally:
            for scene in scenes:scene.close()
        outputs.append(dict(task=task,output=file_pin(path),frames=frames,fps=30,speed=speed,physical_durations_s=durations,
            inputs=[dict(trace=c['trace'],host_topology=c['host_topology']) for c in cases]))
    result=dict(schema='controlled_actual_replay_videos:1',sources=[file_pin(__file__),file_pin(ROOT/'tools/render_mug_progress_video.py')],
        run=file_pin(args.report),videos=outputs,neural_updates=0,native_integration_steps=0,
        scope='All fixed coarse task runs; recorded qpos with saved fixed mocap, no physics integration or extrapolation. Host model graphics, actual GPU physical context remains separate.')
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(outputs=[str(v['output']['path']) for v in outputs])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
