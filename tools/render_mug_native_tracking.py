"""Side-by-side actual recorded ZERO/MPC trajectories; no policy/model inference."""
from pathlib import Path
import argparse,json,math,sys
import numpy as np
import mujoco
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from tools.render_mug_progress_video import Scene,canvas,writer


def main(args):
    run,out=args.run.resolve(),args.output.resolve();report=json.loads((run/'report.json').read_text())
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified video output')
    binding=json.loads(verify_file_pin(report['inputs']['binding']).read_text());model=mujoco.MjModel.from_binary_path(str(verify_file_pin(binding['model'])));model.vis.global_.offwidth=768;model.vis.global_.offheight=560;model.vis.headlight.ambient[:]=.45;model.vis.headlight.diffuse[:]=.75
    cases=[next(c for c in report['cases'] if c['task']=='REFERENCE' and c['controller']==mode and c['dt']==.0003125) for mode in ['ZERO','NATIVE_MPC']];traces=[dict(np.load(verify_file_pin(c['execution']['trace']))) for c in cases];durations=[c['summary']['metrics']['actual_duration_s'] for c in cases]
    data=mujoco.MjData(model);data.qpos[:]=traces[0]['pre_qpos'][0];mujoco.mj_forward(model,data);center=data.xipos[binding['mug_body_id']].copy();center[2]-=.015;scene=Scene(model,640,550,center);out.mkdir();frames=0;path=out/'reference_zero_vs_native_mpc.mp4'
    try:
        with writer(path) as video:
            for i,t in enumerate(np.linspace(0,max(durations),math.ceil(max(durations)/.25*30)+1)):
                panels=[];labels=[]
                for j,(case,z,duration) in enumerate(zip(cases,traces,durations)):
                    index=min(len(z['post_qpos'])-1,max(0,round(t/case['dt'])-1));q=z['pre_qpos'][0] if i==0 else z['post_qpos'][index];panels.append(scene.draw(q,250));labels.append(('零残差执行' if j==0 else '原生滚动规划')+('：终止后静帧' if t>=duration else '：实际记录'))
                image=canvas(1280,'新原生闭环对照：参考跟踪，仍未通过 2 秒任务','同一初态、模型与步长；0.25× 慢放；两列都是已执行仿真，不是世界模型预测',panels,labels,
                    f'各列物理时间：{min(t,durations[0]):.3f} / {min(t,durations[1]):.3f} 秒',
                    [f'零残差 {durations[0]:.3f}s、原生规划 {durations[1]:.3f}s 触发姿态超限；终止后只展示静帧。','原生规划使用完整仿真状态与昂贵 CPU 前瞻；不是学习 Actor，也不代表实时运行。'])
                if i==0:
                    Image.fromarray(image).save(out/'start.png')
                    for _ in range(18):video.append_data(image);frames+=1
                video.append_data(image);frames+=1
            Image.fromarray(image).save(out/'end.png')
            for _ in range(30):video.append_data(image);frames+=1
    finally:scene.close()
    verify_file_pin(binding['model']);result=dict(schema='mug_native_feedback_comparison_video:1',sources=[file_pin(__file__),file_pin(ROOT/'tools/render_mug_progress_video.py')],run=file_pin(run/'report.json'),model=binding['model'],cases=[dict(task=c['task'],controller=c['controller'],dt=c['dt'],trace=c['execution']['trace']) for c in cases],
        output=file_pin(path),frames=frames,fps=30,physical_durations_s=durations,speed=.25,neural_updates=0,native_integration_steps=0,scope='Actual recorded native ZERO/MPC reference runs, fixed first coarse-dt cases, no favorable-case selection. Explicit terminal holds, no frame extrapolation or learned-policy claim.')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
