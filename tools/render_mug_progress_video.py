"""Render saved native failure and fixed latest WM predictions, with honest clocks."""
from pathlib import Path
import argparse,json,math,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
import mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.transition_records import state_rotation

FONT='C:/Windows/Fonts/msyh.ttc'
def font(size):return ImageFont.truetype(FONT,size)


def qpos_from_state(s,COM):
    R=state_rotation(s)
    if not np.isfinite(s).all() or np.max(abs(R.T@R-np.eye(3)))>1e-8:raise ValueError('Saved finite proper predicted state')
    q=np.empty(27);q[:20]=s[:20];q[20:23]=s[40:43]-R@COM;mujoco.mju_mat2Quat(q[23:27],R.ravel())
    return q


class Scene:
    def __init__(self,model,width,height,center):
        self.model=model;self.data=mujoco.MjData(model);self.renderer=mujoco.Renderer(model,width=width,height=height);self.width=width;self.height=height;self.center=center
    def draw(self,q,azimuth):
        self.data.qpos[:]=q;self.data.qvel[:]=0;mujoco.mj_forward(self.model,self.data)
        camera=mujoco.MjvCamera();camera.lookat[:]=self.center;camera.distance=.33;camera.azimuth=azimuth;camera.elevation=-30
        self.renderer.update_scene(self.data,camera=camera);return Image.fromarray(self.renderer.render().copy())
    def close(self):self.renderer.close()


def canvas(width,title,subtitle,panels,labels,clock,footer,*,status=None):
    height=800;im=Image.new('RGB',(width,height),(17,22,32));d=ImageDraw.Draw(im)
    d.text((24,13),title,font=font(29),fill=(245,247,250));d.text((24,55),subtitle,font=font(20),fill=(178,197,217))
    for i,(panel,label) in enumerate(zip(panels,labels)):
        x=i*panel.width;im.paste(panel,(x,120));d.text((x+15,92),label,font=font(21),fill=(99,211,225))
    d.text((24,687),clock,font=font(23),fill=(240,244,250))
    for i,line in enumerate(footer):d.text((24,727+29*i),line,font=font(19),fill=(186,199,217))
    if status:
        box=d.textbbox((0,0),status,font=font(22));w=box[2]-box[0]+32;d.rectangle((width-w-18,683,width-16,720),fill=(135,40,42));d.text((width-w-2,687),status,font=font(22),fill='white')
    return np.asarray(im)


def writer(path):return imageio.get_writer(str(path),fps=30,codec='libx264',quality=8,pixelformat='yuv420p',macro_block_size=1,ffmpeg_log_level='error',output_params=['-movflags','+faststart'])


def main(args):
    out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified visual output')
    binding_path=ROOT/'runs/right_mug_frames_v4/binding.json';binding=json.loads(binding_path.read_text());model_path=verify_file_pin(binding['model']);model=mujoco.MjModel.from_binary_path(str(model_path))
    # Rendering buffers/light only in this new in-memory copy; saved physics intact.
    model.vis.global_.offwidth=768;model.vis.global_.offheight=560;model.vis.headlight.ambient[:]=.45;model.vis.headlight.diffuse[:]=.75
    COM=model.body_ipos[binding['mug_body_id']].copy();original=json.loads((ROOT/'runs/mug_H20_collection_v1/report.json').read_text())
    native=next(e for e in original['episodes'] if e['role']=='CHECK' and e['dt_s']==.0003125);trace=dict(np.load(verify_file_pin(native['trace'])));duration=len(trace['post_qpos'])*native['dt_s'];q0=trace['pre_qpos'][0]
    d=mujoco.MjData(model);d.qpos[:]=q0;mujoco.mj_forward(model,d);center=d.xipos[binding['mug_body_id']].copy();center[2]-=.015
    out.mkdir();scene=Scene(model,640,550,center)
    images=[];frames=0;last_frame=None
    path=out/'native_recorded_control_slow.mp4'
    try:
        with writer(path) as video:
            samples=np.linspace(0,duration,math.ceil(duration/.25*30)+1)
            for i,t in enumerate(samples):
                index=min(len(trace['post_qpos'])-1,max(0,round(t/native['dt_s'])-1));q=q0 if i==0 else trace['post_qpos'][index]
                image=canvas(1280,'实际 MuJoCo 轨迹：杯子未保持稳定','记录的关节反馈 + 探索控制；非学习 Actor；0.25× 慢放（4倍慢放）',
                    [scene.draw(q,250),scene.draw(q,100)],['视角 1：杯柄与接触区域','视角 2：手掌与物体'],f'原始物理时间 {t:.3f} / {duration:.3f} 秒',
                    [f'固定案例：CHECK seed {native["seed"]}，未按效果筛选；无物体外力辅助。','轨迹在姿态角超过 45° 时终止；本视频不代表成功持握或 ego 重定向。'])
                if i==0:
                    for _ in range(18):video.append_data(image);frames+=1
                    Image.fromarray(image).save(out/'native_start.png')
                video.append_data(image);frames+=1;last_frame=image
            final=canvas(1280,'实际 MuJoCo 轨迹：已触发终止','记录控制已结束；以下为末帧静止展示，不是继续仿真',
                [scene.draw(trace['post_qpos'][-1],250),scene.draw(trace['post_qpos'][-1],100)],['视角 1：杯柄与接触区域','视角 2：手掌与物体'],f'原始物理时间 {duration:.3f} 秒（末帧）',
                ['终止原因：物体姿态角超过 45°。','当前没有新的学习策略闭环持握成功视频。'],status='姿态超限：未通过')
            Image.fromarray(final).save(out/'native_end.png')
            for _ in range(30):video.append_data(final);frames+=1
    finally:scene.close()
    health_path=ROOT/'runs/mug_impulse_resume2000_v1/health_2000.json';health=json.loads(health_path.read_text());rows=health['recorded_control_rollouts']['IMPULSE']['rows']
    case_index=next(i for i,r in enumerate(rows) if r['seed']==8200 and r['dt']==.0003125 and r['start']==2 and r['horizon']==10)
    case=rows[case_index];collection=json.loads((ROOT/'runs/mug_interventions_v1/report.json').read_text());prefix=next(p for p in collection['prefixes'] if p['seed']==case['seed'] and p['dt']==case['dt']);actual=dict(np.load(verify_file_pin(prefix['transitions'])))
    native_states=np.concatenate([actual['source55'][2:3],actual['next55'][2:12]]);states={};prediction_pins={}
    for mode in ['DIRECT','IMPULSE']:
        p=ROOT/f'runs/mug_impulse_resume2000_v1/{mode}_rollouts.npz';prediction_pins[mode]=file_pin(p);z=np.load(p);a,b=z['trajectory_offsets'][case_index:case_index+2];states[mode]=z['predicted_states'][a:b]
        if not np.array_equal(states[mode][0],native_states[0]):raise ValueError('Predictions must share exact native source, not a favorable reset')
    if native_states.shape!=(11,55) or any(v.shape!=(11,55) for v in states.values()):raise ValueError('Exact200ms/10-step saved sequences')
    scene=Scene(model,512,550,center);comparison_path=out/'world_models_200ms_comparison.mp4';comparison_frames=0
    try:
        with writer(comparison_path) as video:
            for i in range(11):
                panels=[scene.draw(qpos_from_state(s,COM),250) for s in [native_states[i],states['DIRECT'][i],states['IMPULSE'][i]]]
                for s in [native_states[i],states['DIRECT'][i],states['IMPULSE'][i]]:
                    q=qpos_from_state(s,COM);scene.data.qpos[:]=q;mujoco.mj_forward(model,scene.data)
                    if np.max(abs(scene.data.xipos[binding['mug_body_id']]-s[40:43]))>1e-12:raise ValueError('COM-to-native-BODY rendering conversion changed pose')
                image=canvas(1536,'最新世界模型：同一段 0.20 秒记录控制的递推对比','CHECK seed 8200；10 个 20 ms 步；0.10× 慢放（10倍慢放）；显示原始离散状态点',panels,
                    ['原生仿真（已记录）','DIRECT：独立速度头预测','IMPULSE：冲量耦合头预测'],f'片段内物理时间 {i*.02:.2f} / 0.20 秒；原记录时间 {(.04+i*.02):.2f} 秒',
                    ['中、右：模型递推后的运动学绘制，不是新的动力学执行；三列使用相同相机与材质。','无额外训练、无策略优化、无未来真值重置。短时预测接近不等于完整持握成功。'])
                if i==0:
                    Image.fromarray(image).save(out/'prediction_start.png')
                    for _ in range(18):video.append_data(image);comparison_frames+=1
                for _ in range(6):video.append_data(image);comparison_frames+=1
                if i==10:
                    Image.fromarray(image).save(out/'prediction_end.png')
                    for _ in range(24):video.append_data(image);comparison_frames+=1
    finally:scene.close()
    verify_file_pin(binding['model']);report=dict(schema='mug_saved_progress_video:1',source=file_pin(__file__),model=binding['model'],binding=file_pin(binding_path),
        inputs=dict(native_trace=native['trace'],native_transitions=native['transitions'],latest_health=file_pin(health_path),prefix=prefix['transitions'],predictions=prediction_pins),
        native=dict(seed=native['seed'],role=native['role'],physical_duration_s=duration,dt_s=native['dt_s'],termination=native['termination'],slow_speed=.25,frames=frames,video=file_pin(path)),
        comparison=dict(case_index=case_index,case={k:case[k] for k in ['seed','dt','start','horizon']},physical_duration_s=.2,slow_speed=.1,frames=comparison_frames,video=file_pin(comparison_path)),
        display=dict(camera_distance=.33,camera_elevation=-30,camera_azimuths=[250,100],lookat=center.tolist(),initial_and_terminal_stills_explicit=True),
        simulation_integration_steps=0,neural_updates=0,new_policy_execution=False,scope='Saved native kinematic playback and latest saved WM prediction rendering. Forward calls refresh display only. Original data/model/weights unchanged; no holding success or ego RGB claim.')
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args())
