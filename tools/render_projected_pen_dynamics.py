"""Actual native primary-dt original/projection replay; terminal holds explicit."""
from pathlib import Path
import argparse,json,math,sys
import numpy as np
import mujoco
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.controlled_snapshot import native_controlled_snapshot
from tools.render_mug_progress_video import Scene,canvas,writer


def main(args):
    if args.output.exists():raise ValueError('Fresh native replay output')
    r=json.loads(args.report.read_text());p=json.loads(verify_file_pin(r['projection']).read_text());cases=[next(c for c in r['cases'] if c['name']==name and c['dt']==.005) for name in ['ORIGINAL_ZERO_RESIDUAL','PROJECTED_CONSTANT_RESIDUAL']];zs=[dict(np.load(verify_file_pin(c['trace']))) for c in cases]
    args.output.mkdir();scenes=[]
    for z in zs:
        m,d,source,_=native_controlled_snapshot(p['case']);m.vis.global_.offwidth=768;m.vis.global_.offheight=560;m.vis.headlight.ambient[:]=.45;m.vis.headlight.diffuse[:]=.75
        scene=Scene(m,640,550,z['COM_world_m'][0]);mujoco.mj_setState(m,scene.data,source['initial_native_state'],mujoco.mjtState.mjSTATE_INTEGRATION);scenes.append(scene)
    durations=[c['summary']['metrics']['actual_duration_s'] for c in cases];path=args.output/'native_pen_original_vs_projection.mp4';frames=0
    try:
        with writer(path) as video:
            samples=np.linspace(0,max(durations),math.ceil(max(durations)/.1*30)+1)
            for i,t in enumerate(samples):
                panels=[];labels=[]
                for label,c,z,scene,duration in zip(['原始参考，零残差','关节几何修正，固定残差'],cases,zs,scenes,durations):
                    idx=min(len(z['post_qpos'])-1,max(0,round(t/c['dt'])-1));q=z['initial_qpos'] if i==0 else z['post_qpos'][idx];panels.append(scene.draw(q,250));labels.append(label+('：终止后静帧' if t>=duration else '：实际记录'))
                frame=canvas(1280,'转笔：初始几何通过，原生动力学仍未通过','固定主步长 5 ms；0.10× 慢放；CPU 原生实测，不是 GPU 或学习 Actor 的新结果',panels,labels,
                    '各列物理时间：'+' / '.join(f'{min(t,d):.3f}' for d in durations)+' 秒',
                    ['计划 7.36 秒：原始 0.16 秒、修正 0.22 秒位置误差超限；末帧静止展示。',
                     '另一 2.5 ms 步长修正组仅持续 0.18 秒；不能把主步长延长当作稳定跟踪。'])
                if i==0:
                    Image.fromarray(frame).save(args.output/'start.png')
                    for _ in range(18):video.append_data(frame);frames+=1
                if i==len(samples)//2:Image.fromarray(frame).save(args.output/'mid.png')
                video.append_data(frame);frames+=1
            Image.fromarray(frame).save(args.output/'end.png')
            for _ in range(30):video.append_data(frame);frames+=1
    finally:
        for scene in scenes:scene.close()
    result=dict(schema='projected_pen_native_actual_replay:1',source=file_pin(__file__),run=file_pin(args.report),output=file_pin(path),frames=frames,fps=30,speed=.1,physical_durations_s=durations,
        neural_updates=0,native_integration_steps=0,scope='Fixed original primary dt; both recorded actual states, saved wrist pose, explicit terminal holds. Fine-dt failure disclosed, no success or learned-control claim.')
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(output=str(path),frames=frames)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
