"""Evaluate frozen causal search-timing heads with the same one-slot budget on new disturbances."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc
import torch
from collect_mug_search_timing import block_input
from fit_mug_search_schedule import MugSlotHead
from test_mug_event_budget import setup,initial,Selector,NativeStrictWindow,context,plant,run_case,h3


def run(out):
    torch.set_num_threads(1)
    root,parent,models,asset,bind,old,lib,nets=setup()
    fitpin=h3.file_pin(root/'mug_search_timing_fit_v1/report.json')
    fitreport=h3.physics.read_json(fitpin);pins=h3.physics.read_json(fitreport['frozen'])['weights'];heads={}
    for arm,pin in pins.items():
        ck=torch.load(h3.verify_file_pin(pin),map_location='cpu',weights_only=True)
        n=MugSlotHead(ck['width'],ck['error']);n.load_state_dict(ck['model']);heads[arm]=n.eval()
    samples=qmc.scale(qmc.LatinHypercube(d=4,seed=8571).random(12),[.95,.95,.9,-.001],[1.05,1.05,1.1,.001])
    conditions={'nominal':[1.,1.,1.,0.],**{f'new_{i:02}':v.tolist() for i,v in enumerate(samples)}}
    out.mkdir(exist_ok=False)
    h3.write_json(out/'intent.json',dict(fit=fitpin,code=h3.file_pin(__file__),source=parent,conditions=conditions,seed=8571,
        scope='12 new matched conditions per clip plus nominal. Block length four, primary measured vs periodic median paired benefit >=1 control for each clip, complete nominal. Same risk, reference-only head and open controls; head slot choice fixed at block start with >=0.5 predicted control margin over periodic.'))
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as ex:
        for clip,group in ((1884,'mug_annotated_windows_original_v1'),(1888,'mug_unilateral_interval_v1')):
            fit=h3.physics.load_arrays(models['clips'][str(clip)]['data'])
            item,=h3.physics.read_json(h3.file_pin(root/group/'report.json'))['rows'];ref=h3.physics.load_arrays(item['reference']);T=ref['body']
            pos=T[:,:3,3]+T[:,:3,:3]@old['COM_BODY_m'];rot=Rotation.from_matrix(T[:,:3,:3])
            m0=plant(asset,bind,1,1,1);Pn,_=context(m0,bind,pos,rot)
            selector=Selector(Pn,NativeStrictWindow(Pn,lib),fit,nets,ex)
            for number,(condition,(mass,mu,tau,offset)) in enumerate(conditions.items()):
                m=plant(asset,bind,mass,mu,tau);P,py=context(m,bind,pos,rot);checker=NativeStrictWindow(P,lib)
                for mode in ('open','periodic','event','reference','measured'):
                    slot=None;blocks=[]
                    def trigger(s,i,last,risk,index):
                        nonlocal slot
                        if index==0:
                            x=block_input(Pn,s,fit,i,last)
                            with torch.inference_mode():y=heads[mode](torch.from_numpy(x)).numpy()
                            base=(number+i//4)%4;winner=int(y.argmax())
                            slot=winner if y[winner]-y[base]>=.5 else base
                            blocks.append(dict(start=i,slot=slot,periodic=base))
                        return index==slot
                    result,trace=run_case(P,checker,Pn,selector,old,fit,condition,offset,mode,number,trigger)
                    if condition=='nominal':np.testing.assert_array_equal(trace['qpos'],fit['state'][1:,1:1+m.nq])
                    replay=initial(m,P,old,offset)
                    for i,cmd in enumerate(trace['ctrl']):
                        np.testing.assert_array_equal(py.step(replay,i,cmd),trace['metrics'][i]);np.testing.assert_array_equal(replay.qpos,trace['qpos'][i])
                    flags=[r['requested'] for r in result['decisions']]
                    if mode!='open':
                        for j in range(0,len(flags)-3,4):assert sum(flags[j:j+4])==1
                    p=out/f'{clip}_{condition}_{mode}.npz';np.savez_compressed(p,**trace)
                    rows.append(dict(clip=clip,blocks=blocks,trace=h3.file_pin(p),**result))
                h3.write_json(out/'progress.json',rows);print(clip,condition,[r['passing'] for r in rows[-5:]],flush=True)
    h3.write_json(out/'report.json',dict(intent=h3.file_pin(out/'intent.json'),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);run(p.parse_args().out.resolve())
