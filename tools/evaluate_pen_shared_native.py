"""Strictly evaluate the same common parent and both learned model architectures on both clips."""
from pathlib import Path
import json,sys,shutil
from evaluate_pen_dev_clip import b,evaluate_cases,config,read,file_pin,verify_file_pin,write_json

def main(run,out):
    root=b.ROOT/'runs'
    train=file_pin(run/'report.json')
    result=read(train);actors=result['actors']
    if set(actors)!= {'shared','specialist_001','specialist_003'}:raise ValueError('Expected two specialists and one shared actor')
    parent=file_pin(root/'pen_scaled_train_v1/actor_02781.pt')
    actors={'parent':parent,**actors}
    for p in actors.values():verify_file_pin(p)
    out.mkdir(exist_ok=False)
    write_json(out/'intent.json',dict(train=train,common_parent=parent,actors=actors,clips=['clip_001','clip_003'],repeats=2,
        scope='Warm-start paired residual PPO. Shared and both specialists use identical native eval and zero support environment weight; other-clip specialist assessed as transfer. All four weights on each clip. Previously inspected development setting.'))
    gpu=read(file_pin(root/'pen_gpu_catalog_v1/report.json'))
    legacy=read(file_pin(root/'pen_catalog_screen_v2/report.json'))
    summary={}
    for clip in ('clip_001','clip_003'):
        folder=out/clip;folder.mkdir()
        if clip=='clip_003':
            obj,item=config(folder,clip,root/'pen_refit_dev_preload_v1/report.json')
        else:
            obj=b.OBJECTIVE;item=legacy['results'][clip]
            shutil.copyfile(b.ROOT/'configs/controlled_evaluation_v4.toml',folder/'controlled_evaluation_v4.toml')
        write_json(folder/'intent.json',dict(parent=file_pin(out/'intent.json'),reference=item['source'],frames=item['frames'],
            gpu_ref=gpu['results'][clip]['series'],reset=('canonical' if clip=='clip_001' else 'refit_preloaded'),refit_cpu_export=True))
        rows=evaluate_cases(folder,obj,item,actors,dict.fromkeys(actors,0.),2,custom_start=(clip!='clip_001'),refit_export=True)
        cases={}
        for key in actors:
            subset=[r for r in rows if r['arm']==key];cases[key]=dict(valid_s=[r['valid_s'] for r in subset],
                full=sum(r['stop']=='REFERENCE_COMPLETION' for r in subset),strict=sum(r['summary']['qualified'] for r in subset),
                angle_deg=[r['summary']['metrics']['angle_RMS_rad']*180/3.141592653589793 if r['stop']=='REFERENCE_COMPLETION' else None for r in subset],
                stops=[r['stop'] for r in subset])
        summary[clip]=dict(report=file_pin(folder/'report.json'),actors=cases)
        write_json(out/'progress.json',dict(clips=summary))
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),results=summary));print(summary)
if __name__=='__main__':main(*map(lambda s:Path(s).resolve(),sys.argv[1:]))
