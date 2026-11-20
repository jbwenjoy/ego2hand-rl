"""Strictly evaluate all trained multi-reference shared-policy ablations on both clips."""
import json,sys,shutil
from pathlib import Path
from evaluate_pen_dev_clip import b,evaluate_cases,config,read,file_pin,verify_file_pin,write_json

def main(partial,mixed,out):
    root=b.ROOT/'runs'
    pins=[file_pin(p/'report.json') for p in (partial,mixed)]
    actors={'parent':file_pin(root/'pen_scaled_train_v1/actor_02781.pt')}
    for pin,expected in zip(pins,[{'shared_adam','local_adam'},{'balanced','balanced_kl'}]):
        outputs=read(pin)['actors']
        if set(outputs)!=expected:raise ValueError('Wrong trained endpoints')
        actors.update(outputs)
    for p in actors.values():verify_file_pin(p)
    out.mkdir(exist_ok=False)
    write_json(out/'intent.json',dict(train=pins,actors=actors,clips=['clip_001','clip_003'],repeats=2,
        scope='Common native no-aid evaluation of four shared-policy interventions and parent. Compare also the published complete-sharing and separate policy baselines, matched continuation budget.'))
    summary={}
    for clip in ('clip_001','clip_003'):
        folder=out/clip;folder.mkdir()
        if clip=='clip_003':
            obj,item=config(folder,clip,root/'pen_refit_dev_preload_v1/report.json')
        else:
            obj=b.OBJECTIVE;item=read(file_pin(root/'pen_catalog_screen_v2/report.json'))['results'][clip]
            shutil.copyfile(b.ROOT/'configs/controlled_evaluation_v4.toml',folder/'controlled_evaluation_v4.toml')
        write_json(folder/'intent.json',dict(source=file_pin(out/'intent.json'),reference=item['source']))
        rows=evaluate_cases(folder,obj,item,actors,dict.fromkeys(actors,0.),2,custom_start=(clip!='clip_001'),refit_export=True)
        summary[clip]=dict(report=file_pin(folder/'report.json'),outcomes={k:[dict(valid_s=r['valid_s'],stop=r['stop'],strict=r['summary']['qualified']) for r in rows if r['arm']==k] for k in actors})
        write_json(out/'progress.json',dict(clips=summary))
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),results=summary))
if __name__=='__main__':main(*map(lambda s:Path(s).resolve(),sys.argv[1:]))
