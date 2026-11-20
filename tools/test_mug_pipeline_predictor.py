"""Avoid the extra native lead window by transporting the observed error to the next phase."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc
import torch
import test_mug_one_tick_pipeline as pipeline
from test_mug_delayed_feedback import selected_rule
from test_strict_mug_feedback import tangent_errors


def cheap_preview(predictor):
    def preview(screen,s,i,cmd,mode,submitted):
        P=screen.P;fit=screen.fit;n=P.m.nv
        error=tangent_errors(P.m,fit['state'][i],s[None])[0]
        delta=cmd-fit['cmd'][i]
        if predictor!='constant':
            B=fit['B_native' if predictor=='local' else 'B_shared']
            error=fit['A'][i]@error+B[i]@delta
        predicted=fit['state'][i+1].copy()
        mujoco.mj_integratePos(P.m,predicted[1:1+P.m.nq],error[:n].copy(),1.)
        predicted[1+P.m.nq:]+=error[n:]
        command,idx,ticks=screen.choose(mode,predicted,i+1,delta)
        np.testing.assert_array_equal(command,selected_rule(P,fit,predicted,i+1,delta,idx))
        return dict(idx=idx,finished=perf_counter(),submitted=submitted,ticks=ticks,pred=predicted)
    return preview


def run(out):
    torch.set_num_threads(1);root,_,models,asset,bind,old,lib,_=pipeline.setup()
    source=pipeline.h3.file_pin(root/'mug_tick_pipeline_v1/report.json')
    head=pipeline.h3.physics.read_json(pipeline.h3.physics.read_json(source)['intent'])['source']
    fitpin=pipeline.h3.physics.read_json(pipeline.h3.physics.read_json(head)['intent'])['fit']
    weights=pipeline.h3.physics.read_json(fitpin)['model'];ck=torch.load(pipeline.h3.verify_file_pin(weights),map_location='cpu',weights_only=True)
    net=pipeline.MugHorizonHead(ck['n']);net.load_state_dict(ck['state']);net.eval()
    samples=qmc.scale(qmc.LatinHypercube(d=4,seed=8601).random(8),[.95,.95,.90,-.001],[1.05,1.05,1.10,.001])
    conditions={'nominal':[1.,1.,1.,0.],**{f'new_{i:02}':v.tolist() for i,v in enumerate(samples)}}
    out.mkdir(exist_ok=False)
    pipeline.h3.write_json(out/'intent.json',dict(source=source,code=pipeline.h3.file_pin(__file__),conditions=conditions,seed=8601,
        scope='Fresh 8 paired disturbance combinations. Reuse boundary-checked one-slot asynchronous delivery and identical centered four-option screening, vary only the prediction used to start next-slot planning. Native prediction costs 64 extra ticks; constant nominal-relative error and local Jacobians require no native lead ticks. Commands are re-evaluated at current state. Nominal completion and 20ms misses, timely job share, actual time and paired survival are separate outcomes.'))
    rows=[]
    native=pipeline.preview
    with ThreadPoolExecutor(max_workers=4) as ex,ThreadPoolExecutor(max_workers=1) as worker:
        for clip,group in ((1884,'mug_annotated_windows_original_v1'),(1888,'mug_unilateral_interval_v1')):
            fit=pipeline.h3.physics.load_arrays(models['clips'][str(clip)]['data'])
            item,=pipeline.h3.physics.read_json(pipeline.h3.file_pin(root/group/'report.json'))['rows']
            T=pipeline.h3.physics.load_arrays(item['reference'])['body'];pos=T[:,:3,3]+T[:,:3,:3]@old['COM_BODY_m'];rot=Rotation.from_matrix(T[:,:3,:3])
            m0=pipeline.plant(asset,bind,1,1,1);Pn,_=pipeline.context(m0,bind,pos,rot);cheap,_=pipeline.context(m0,bind,pos,rot)
            screen=pipeline.CenteredScreen(Pn,pipeline.NativeStrictWindow(Pn,lib),fit,net,ex)
            for number,(case,(mass,mu,tau,offset)) in enumerate(conditions.items()):
                m=pipeline.plant(asset,bind,mass,mu,tau);P,py=pipeline.context(m,bind,pos,rot);fast=pipeline.NativeStrictWindow(P,lib)
                modes=['open','instant','native','constant','local','shared'];modes=modes[number%6:]+modes[:number%6]
                for mode in modes:
                    route={'open':'open','instant':'sync_centered'}.get(mode,'pipe_centered')
                    with patch.object(pipeline,'preview',native if mode in ('open','instant','native') else cheap_preview(mode)):
                        r,tr=pipeline.run_case(P,fast,cheap,screen,worker,old,fit,case,offset,route)
                    replay=pipeline.initial(m,P,old,offset)
                    for i,cmd in enumerate(tr['ctrl']):
                        np.testing.assert_array_equal(py.step(replay,i,cmd),tr['metrics'][i]);np.testing.assert_array_equal(replay.qpos,tr['qpos'][i])
                    if case=='nominal':np.testing.assert_array_equal(tr['qpos'],fit['state'][1:,1:1+m.nq])
                    f=out/f'{clip}_{case}_{mode}.npz';np.savez_compressed(f,**tr)
                    rows.append(dict(**r,clip=clip,predictor=mode,trace=pipeline.h3.file_pin(f)))
                pipeline.h3.write_json(out/'progress.json',rows);print(clip,case,[r['passing'] for r in rows[-6:]],flush=True)
    pipeline.h3.write_json(out/'report.json',dict(intent=pipeline.h3.file_pin(out/'intent.json'),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);run(p.parse_args().out.resolve())
