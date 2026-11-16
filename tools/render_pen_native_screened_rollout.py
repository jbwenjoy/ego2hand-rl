"""Fixed repeat0 R/S nine-way selected-native playback after full54 review."""
import json
import argparse
import time
import math
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.mechanics.pen_selected_rollout import selected_poses
from supermimic_research.strict_json import write_json
from render_mug_progress_video import Scene,writer,FONT


def read(p):return json.loads(Path(p).read_text())
def font(size):return ImageFont.truetype(FONT,size)


def producer_observation():
    try:return read(ROOT/'runs/pen_native_screened_rollout_v1/work.json')
    except (OSError,json.JSONDecodeError):return dict(observation_unavailable=True)


def grid_order(teacher):
    return [f'R0_{teacher}_'+n for n in ['UNMODIFIED_ACTOR','NATIVE_SHAM','NATIVE_RANDOM','RESPONSE_ONLY_OBSERVABLE','PATH_POINT_OBSERVABLE','PATH_POINT_RESPONSE_OBSERVABLE','RESPONSE_ONLY_FULL','PATH_POINT_FULL','PATH_POINT_RESPONSE_FULL']]


def main(args):
    started=time.perf_counter();producer_before=producer_observation()
    rp=ROOT/'runs/pen_native_screened_rollout_v1/report.json';vp=args.review_index.resolve() if args.review_index else ROOT/'runs/pen_native_screened_review_v3/report.json';review=read(vp);preview=args.preview_teacher is not None;run=None if preview else read(rp);out=ROOT/'runs'/('pen_native_screened_repeat0_preview_'+args.preview_teacher+'_v1' if preview else 'pen_native_screened_video_v1')
    teachers=[args.preview_teacher] if preview else ['RESIDUAL','SUPPORT']
    if out.exists():raise ValueError('Fresh selected playback output required')
    if not preview and (review['run']!=file_pin(rp) or len(review['reviewed_cases'])!=54 or any(r['passed']!=r['total'] or not all(r['checks'].values()) for r in [run,review])):raise ValueError('Full rendering requires all54 independent review')
    reviews={c['name']:c for c in review['reviewed_cases']};required=[n for t in teachers for n in grid_order(t)]
    if any(n not in reviews for n in required):raise ValueError('All9 fixed repeat0 controllers per displayed teacher must be independently reviewed')
    case_map={}
    for name in required:
        v=read(verify_file_pin(reviews[name]['report']))
        if v['passed']!=v['total'] or not all(v['checks'].values()):raise ValueError('Complete case review required for playback')
        case_map[name]=dict(report=v['producer_case'])
    for p in [file_pin(ROOT/'runs/pen_native_screened_rollout_v1/source_receipt.json'),review['source_receipt']]:verify_source_snapshot(read(verify_file_pin(p)),live=True)
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'tools/render_mug_progress_video.py'],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt);outputs={};checks={}
    labels=['原策略','相同动作对照','随机方向对照','RESPONSE_ONLY · OBS','PATH_POINT · OBS','PATH_POINT_RESPONSE · OBS','RESPONSE_ONLY · FULL','PATH_POINT · FULL','PATH_POINT_RESPONSE · FULL'];indices={}
    for teacher in teachers:
        cases=[];poses=[];scenes=[];names=grid_order(teacher);folder=out/teacher;folder.mkdir();display_forwards=0
        try:
            for name in names:
                c=read(verify_file_pin(case_map[name]['report']));v=read(verify_file_pin(reviews[name]['report']));p=selected_poses(c,v);cases.append(c);poses.append(p);m=mujoco.MjModel.from_binary_path(str(verify_file_pin(p['model'])));m.vis.global_.offwidth=768;m.vis.global_.offheight=560;m.vis.headlight.ambient[:]=.45;m.vis.headlight.diffuse[:]=.75
                d=mujoco.MjData(m);d.qpos[:]=p['initial_qpos'];d.mocap_pos[:]=p['post_mocap_pos'][0];d.mocap_quat[:]=p['post_mocap_quat'][0];mujoco.mj_forward(m,d);display_forwards+=1;center=d.xipos[p['model_layout']['object_body']].copy();scenes.append(Scene(m,640,316,center))
            durations=[p['duration_s'] for p in poses];times=np.minimum(np.arange(math.ceil(max(durations)/.05*30)+1)*(.05/30),max(durations));selected_indices=[];count=0;path=folder/'selected_native_grid.mp4'
            with writer(path) as video:
                for fi,t in enumerate(times):
                    im=Image.new('RGB',(1920,1296),(17,22,32));draw=ImageDraw.Draw(im);draw.text((18,10),('残差教师' if teacher=='RESIDUAL' else '接触支撑教师')+'：离线分支筛选后选中的原生仿真轨迹'+('（预览，整轮尚未结束）' if preview else ''),font=font(28),fill='white');draw.text((18,49),'固定第0次数值重复；上排原策略／相同动作／随机对照，中排OBS、下排FULL；20倍慢放',font=font(21),fill=(180,200,220));frame_indices=[]
                    for k,(case,p,s,label,duration) in enumerate(zip(cases,poses,scenes,labels,durations)):
                        j=int(np.searchsorted(np.arange(1,len(p['post_qpos'])+1)*.0025,min(float(t),duration)+1e-12,side='right')-1);frame_indices.append(j);q=p['initial_qpos'] if j<0 else p['post_qpos'][j];j0=max(0,j);s.data.mocap_pos[:]=p['post_mocap_pos'][j0];s.data.mocap_quat[:]=p['post_mocap_quat'][j0];panel=s.draw(q,250);display_forwards+=1;x=(k%3)*640;y=100+(k//3)*380;draw.text((x+12,y+5),label,font=font(23),fill=(100,215,230));done=t>=duration
                        status='实测选中子步' if not done else {'BASE_PHYSICAL_INVALID':'物理审计失败，末帧停留','TASK_FAILURE':'任务终止，末帧停留','REFERENCE_COMPLETION':'完整参考结束，末帧停留'}[case['stop_reason']];color=(220,100,100) if done and not case['full_task_success'] else (145,215,160)
                        draw.text((x+12,y+35),f'{min(float(t),duration):.3f}/{duration:.3f}s  '+status,font=font(18),fill=color);im.paste(panel,(x,y+64))
                    draw.text((18,1249),'保存的2.5ms姿态，无插值；离线查询与回滚，非世界模型预测动画／ego／真机；完整任务7.36s',font=font(21),fill=(210,218,230));selected_indices.append(frame_indices);array=np.asarray(im)
                    if fi==0:
                        im.save(folder/'start.png')
                        for _ in range(18):video.append_data(array);count+=1
                    if fi==len(times)//2:im.save(folder/'mid.png')
                    video.append_data(array);count+=1
                im.save(folder/'end.png')
                for _ in range(30):video.append_data(array);count+=1
            reader=imageio.get_reader(path)
            try:
                actual=reader.count_frames();checks[teacher+'_encoded_frames']=actual==count
                for tag,frame_index in [('start',0),('mid',count//2),('end',count-1)]:
                    frame=reader.get_data(frame_index);checks[teacher+'_decoded_'+tag]=frame.shape==(1296,1920,3);Image.fromarray(frame).save(folder/('decoded_'+tag+'.png'))
            finally:reader.close()
            indices[teacher]=np.asarray(selected_indices);outputs[teacher]=dict(video=file_pin(path),case_order=names,case_reports=[case_map[n]['report'] for n in names],reviews=[reviews[n]['report'] for n in names],durations_s=durations,frames=count,fps=30,speed=.05,display_forward_calls=display_forwards,selected_query_counts=[len(p['selected_queries']) for p in poses]);checks[teacher+'_fixed_repeat0']=all(c['repetition']==0 for c in cases)
        finally:
            for scene in scenes:scene.close()
    np.savez_compressed(out/'indices.npz',**indices);verify_source_snapshot(receipt,live=True);result=dict(schema='pen_selected_native_video:1',preview=preview,run=file_pin(rp) if not preview else None,registration=review['registration'],review=file_pin(vp),source_receipt=file_pin(out/'source_receipt.json'),outputs=outputs,indices=file_pin(out/'indices.npz'),font=file_pin(FONT),checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,wall_s=time.perf_counter()-started,render_started_perf_counter=started,producer_before=producer_before,producer_after=producer_observation(),timing_scope='Any overlap of this graphics export and CPU review with evaluation contributes to observed device/CPU load and case wall time. These are not isolated speed benchmarks.',
        scope='Fixed repeat0/all9 controllers per teacher,actual saved selected native qpos/mocap2.5ms poses,no interpolation or further dynamics. Full-state chain independently checked;only selected branches animated. Forward calls are display only. Physical/task failures retain distinct end-frame badges;not ego/hardware or a new Actor.')
    write_json(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],videos={k:v['video']['path'] for k,v in outputs.items()})),flush=True)
    if not all(checks.values()):raise ValueError('Encoded playback discrepancy retained')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preview-teacher',choices=['RESIDUAL','SUPPORT']);p.add_argument('--review-index',type=Path);a=p.parse_args()
    if a.preview_teacher and a.review_index is None:p.error('Preview requires explicit immutable completed-case index')
    main(a)
