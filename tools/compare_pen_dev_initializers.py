"""Frozen control comparison with original and refitted-geometry preload bundles."""
import json,sys
from pathlib import Path
import numpy as np
from evaluate_pen_dev_clip import config,evaluate_cases,b,file_pin,verify_file_pin,write_json


def main(out):
    root=b.ROOT/'runs';out.mkdir(exist_ok=False)
    actors={k:file_pin(root/run/'actor_02781.pt') for k,run in [('residual','pen_scaled_train_v1'),('support','pen_scaled_support_matched_v1')]}
    old=root/'pen_dev_static_preload_v1/report.json';new=root/'pen_refit_dev_preload_v1/report.json'
    cases=[('legacy003','clip_003',old),('refit003','clip_003',new),('refit002','clip_002',new)]
    write_json(out/'intent.json',dict(actors=actors,preloads=[file_pin(old),file_pin(new)],order=[c[0] for c in cases],repeats=3,scope='Development clips only, original time and phase-one reset. Same native guarded factory, weights, termination and physical gates. The comparison is between initialization bundles (joint pose plus servo preload).'))
    results={};old_reference=None
    for code,name,pre in cases:
        folder=out/code;folder.mkdir()
        obj,item=config(folder,name,pre)
        if code=='legacy003':old_reference=item
        elif code=='refit003':
            assert old_reference['source']==item['source']
            with np.load(verify_file_pin(item['series'])) as x,np.load(verify_file_pin(old_reference['series'])) as y:
                for f in ('quat_wxyz','COM'):np.testing.assert_array_equal(x[f],y[f])
        write_json(folder/'intent.json',dict(master=file_pin(out/'intent.json'),preload=file_pin(pre),clip=item['source']))
        rows=evaluate_cases(folder,obj,item,actors,{'residual':0.,'support':.2},3)
        results[code]=dict(report=file_pin(folder/'report.json'),rows=[dict(arm=r['arm'],rep=r['rep'],valid_s=r['valid_s'],stop=r['stop'],success=r['summary']['qualified']) for r in rows])
        write_json(out/'progress.json',dict(cases=results))
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),cases=results))
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
