"""Disentangle one input to the frozen Actor from the physical servo initialization."""
import json,sys,tomllib
from pathlib import Path
import numpy as np
from evaluate_pen_dev_clip import config,evaluate_cases,b,file_pin,verify_file_pin,write_json


def history(obj):
    adapter=tomllib.loads((obj.parent/tomllib.loads(obj.read_text())['adapter_config']).read_text())
    with np.load(verify_file_pin(adapter['control_pin'])) as f:u=f['ctrl']
    with np.load(verify_file_pin(adapter['clip_pin'])) as f:q=f['reference_qpos'][0]
    return 2*(u-q.astype(np.float32)).astype(np.float32)


def main(out):
    r=b.ROOT/'runs';out.mkdir(exist_ok=False)
    conditions={'old':r/'pen_dev_static_preload_v1/report.json','new':r/'pen_refit_dev_preload_v1/report.json'}
    configs={}; histories={};allrows=[]
    for k,source in conditions.items():
        folder=out/k;folder.mkdir();configs[k]=config(folder,'clip_003',source);histories[k]=history(configs[k][0])
    actors={k:file_pin(r/d/'actor_02781.pt') for k,d in [('residual','pen_scaled_train_v1'),('support','pen_scaled_support_matched_v1')]}
    write_json(out/'intent.json',dict(preloads={k:file_pin(v) for k,v in conditions.items()},actors=actors,design='2 physical preload initializations x 2 first observed RAW histories x 2 frozen actors x 3 numerical repeats. Override only last 20 columns of the first Actor input. Subsequent observations and servo actions are native. Crossed cells are diagnostics.'))
    for k,(obj,item) in configs.items():
        folder=obj.parent
        weights={};pins={};overrides={}
        for actor,pin in actors.items():
            for h in conditions:
                key=actor+'_'+h;pins[key]=pin;weights[key]=.2 if actor=='support' else 0.;overrides[key]=histories[h]
        write_json(folder/'intent.json',dict(parent=file_pin(out/'intent.json'),physical=k))
        rows=evaluate_cases(folder,obj,item,pins,weights,3,initial_histories=overrides)
        for row in rows:
            actor,h=row['arm'].split('_');row.update(physical=k,history=h,policy=actor)
            with np.load(verify_file_pin(row['first_action'])) as f:
                np.testing.assert_allclose(f['native_obs'][-20:],histories[k],atol=1e-6)
                np.testing.assert_array_equal(f['native_obs'][:-20],f['actor_obs'][:-20])
                np.testing.assert_allclose(f['actor_obs'][-20:],histories[h],atol=1e-7)
            allrows.append(row)
    out_rows={}
    for policy in actors:
        out_rows[policy]={}
        for k in conditions:
            for h in conditions:
                select=[r for r in allrows if (r['physical'],r['history'],r['policy'])==(k,h,policy)]
                out_rows[policy][k+'_'+h]=[r['valid_s'] for r in select]
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),summary=out_rows,rows=allrows));print(out_rows)
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
