"""Compare saved 256-env final policy poses to the precise controlled reference."""
from pathlib import Path
import argparse,json,sys,tomllib
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation,Slerp
import imageio.v2 as imageio
import mujoco
from render_mug_progress_video import ROOT,Scene,writer,font
from supermimic_research.mechanics.pen_task_contracts import tracking_series
from supermimic_research.artifacts import verify_file_pin,file_pin

KEYS=['full_angle_rad','shaft_angle_rad','reference_axis_twist_rad']
COLORS=['#f5f5f5','#53d4df','#ffb753']
def rms(x): return np.sqrt(np.mean(x*x,axis=0))
def main(run,out):
    out.mkdir(exist_ok=False,parents=True)
    final=json.loads((run/'report.json').read_bytes())
    evaluation=json.loads(verify_file_pin(final['evaluations'][-1]).read_bytes())
    all_metrics=[]; cases=[]; records=[]
    for entry in evaluation['cases']:
        case=json.loads(verify_file_pin(entry['report']).read_bytes())
        trace=dict(np.load(verify_file_pin(case['trace'])))
        series=tracking_series(trace,.0025)
        angles=np.rad2deg(np.column_stack([series[k] for k in KEYS]))
        assert series['twist_defined'].all() and case['stop_reason']=='REFERENCE_COMPLETION'
        assert np.isclose(np.rad2deg(case['summary']['metrics']['angle_RMS_rad']),rms(angles)[0])
        c=np.cos(np.deg2rad(angles)/2)
        assert np.allclose(c[:,0],c[:,1]*c[:,2],atol=1e-8)
        t=series['nominal_time_s']
        records.append(dict(rms_deg=rms(angles).tolist(),peak_t=t[angles.argmax(0)].tolist(),
            peak_deg=angles.max(0).tolist(),quarters_deg=[rms(a).tolist() for a in np.split(angles,4)],
            quarter_full_sq_share=[float(np.square(a[:,0]).sum()/np.square(angles[:,0]).sum()) for a in np.split(angles,4)]))
        all_metrics.append(angles); cases.append((case,trace))
    print(json.dumps(records),flush=True)
    case,z=cases[0]; t=np.r_[0.,t]; n=len(t)
    root=ROOT/'configs'; base=tomllib.loads((root/'controlled_evaluation_v4.toml').read_text())
    clip=json.loads((root/base['pen_recipe']).read_text())['clips'][0]
    clip_path=(root/base['clips_root']/f'{clip}.npz').resolve()
    joints=np.load(clip_path)['reference_qpos']; times=z['time_s']
    assert joints.shape==(369,20) and np.allclose(times,np.arange(369)*.02)
    import torch
    for phase in (1,184,368):
        saved=torch.load(run/f"eval_{case['iteration']:05}"/'repeat_0'/f'P{phase:03}'/'control_source.pt',map_location='cpu',weights_only=False)
        assert np.allclose(joints[phase],saved['policy_observation'][0,:20],atol=1e-7)
    qa=case['model_layout']['object_qposadr']; qids=case['qids']
    assert np.allclose(z['post_qpos'][:,qa:qa+3],z['post_COM'],atol=1e-7)
    actual=np.vstack([z['initial_qpos'],z['post_qpos']])[::8]
    target=actual.copy(); target[:,qids]=joints
    target[:,qa:qa+3]=z['COM_world_m'];target[:,qa+3:qa+7]=z['quaternion_wxyz']
    ref=Rotation.from_quat(target[:,qa+3:qa+7][:,[1,2,3,0]])
    act=Rotation.from_quat(actual[:,qa+3:qa+7][:,[1,2,3,0]])
    rel=(ref.inv()*act).as_quat()
    twist=(np.degrees(2*np.arctan2(rel[:,0],rel[:,3]))+180)%360-180
    full=np.degrees((ref.inv()*act).magnitude())
    shaft=np.degrees(np.arccos(np.clip((ref.inv()*act).as_matrix()[:,0,0],-1,1)))
    a=np.column_stack([full,shaft,abs(twist)])
    assert np.allclose(a[1:],all_metrics[0][7::8],atol=1e-4)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(10,6),sharex=True,layout='constrained')
    for i,(ax,label) in enumerate(zip(axes,['Full orientation','Shaft direction','Twist about shaft'])):
        for rep,m in enumerate(all_metrics):
            ax.plot(t[1:],m[:,i],lw=1.1,label=f'Repeat {rep+1}')
        ax.set_ylabel(label+' (deg)');ax.grid(alpha=.22);ax.axvline(3.68,color='grey',ls='--',lw=1)
    axes[0].legend(ncol=3);axes[-1].set_xlabel('Reference time (s)');axes[0].set_title('Final policy | pen orientation error, all three evaluations')
    fig.savefig(out/'angles.png',dpi=150);plt.close(fig)
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(case['model'])))
    model.vis.global_.offwidth=640;model.vis.global_.offheight=480
    model.vis.headlight.ambient[:]=.45;model.vis.headlight.diffuse[:]=.75
    center=z['COM_world_m'][0]
    scene=Scene(model,640,440,center)
    scene.data.mocap_pos[:]=z['initial_mocap_pos'][0];scene.data.mocap_quat[:]=z['initial_mocap_quat'][0]
    assert np.allclose(z['post_mocap_pos'],scene.data.mocap_pos) and np.allclose(z['post_mocap_quat'],scene.data.mocap_quat)
    fonts=[font(i) for i in (18,21,25)]
    base_img=Image.new('RGB',(1280,680),'#121a24');d=ImageDraw.Draw(base_img)
    d.text((18,9),'参考动作',font=fonts[2],fill='white');d.text((658,9),'256 环境训练 · 第 2781 轮 · 第一次评测',font=fonts[2],fill='white')
    x=np.linspace(26,1250,369); y=661-a*1.65
    for j,color in enumerate(COLORS):
        d.line(list(zip(x,y[:,j])),fill=color,width=2)
    d.text((24,541),'姿态误差曲线 · 0 至 7.36 秒',font=fonts[0],fill='#b7c7d4')
    path=out/'reference_and_policy.mp4'
    try:
        with writer(path) as video:
            for i,time in enumerate(times):
                im=base_img.copy();d=ImageDraw.Draw(im)
                im.paste(scene.draw(target[i],250),(0,48));im.paste(scene.draw(actual[i],250),(640,48))
                d.text((20,501),f'{time:.2f} / 7.36 秒    0.3× 慢放',font=fonts[1],fill='white')
                for j,(x0,name) in enumerate(zip([460,715,970],['完整姿态','笔轴方向','绕轴自转'])):
                    d.text((x0,501),f'{name}  {a[i,j]:.1f}°',font=fonts[1],fill=COLORS[j])
                d.line((x[i],568,x[i],664),fill='#a4a9b2',width=1)
                for _ in range(2): video.append_data(np.asarray(im))
    finally:scene.close()
    reader=imageio.get_reader(path)
    try:
        assert reader.count_frames()==738
        for i in [0,368,736]:Image.fromarray(reader.get_data(i)).save(out/f'preview_{i}.png')
    finally:reader.close()
    np.savez_compressed(out/'errors.npz',seconds=t[1:],degrees=np.array(all_metrics),columns=KEYS)
    (out/'report.json').write_text(json.dumps(dict(columns=KEYS,repeats=records,clip=file_pin(clip_path),video=file_pin(path),evaluation=final['evaluations'][-1]),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('out',type=Path);a=p.parse_args();main(a.run,a.out)
