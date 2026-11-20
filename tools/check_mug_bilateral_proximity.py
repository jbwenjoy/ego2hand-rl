"""Check whether each annotated hand approaches the BOP9 surface during the selected windows."""
import argparse
from pathlib import Path
import numpy as np
import torch
from retarget_adjacent_mugs import read_hand_window,reconstruct,transform_points,physics,file_pin,verify_file_pin,write_json
from supermimic_research.assets.proximity import SurfaceQuery
from supermimic_research.ego.clip_inventory import read_clip


def main(out):
    root=Path(__file__).resolve().parents[1]/'runs'
    sources=[file_pin(root/name/'report.json') for name in ('mug_annotated_windows_original_v1','mug_screened_transfer_v1')]
    mesh=physics.read_json(file_pin(root/'mug_mesh_v1/report.json'))['raw']
    with np.load(verify_file_pin(mesh)) as f:query=SurfaceQuery(f['vertices_model_m'],f['faces'])
    out.mkdir(exist_ok=False)
    write_json(out/'intent.json',dict(sources=sources,mesh=mesh,code=file_pin(__file__),
        rule='All available annotated hand vertices, both hands, thresholds 1/3/5mm; sign and ground-truth load unknown. Skip only unavailable hand poses.'))
    results=[];torch.set_num_threads(2)
    for pin in sources:
        case,=physics.read_json(pin)['rows']
        win=case['window'];tar=verify_file_pin(case['archive'])
        inventory=read_clip(tar,[9]);n=win['end_frame']-win['start_frame']+1
        counts=np.full((n,2,3),-1,dtype=int)
        for col,side in enumerate(('right','left')):
            ids=[i for i in range(win['start_frame'],win['end_frame']+1) if inventory[i]['hands'][side]['pose'] is not None]
            if not ids:continue
            data=read_hand_window(tar,ids,side,9,win['object_uid'])
            vertices=reconstruct(data['profile'],data['angles22'],data['wrist'],side)['vertices']
            for idx,T,v in zip(ids,data['objects'],vertices):
                points=transform_points(np.linalg.inv(T),v)
                near=query.center_tree.query(points)[0]<=.005+query.maximum_radius+1e-10
                gaps=query.nearest(points[near])[1] if near.any() else np.empty(0)
                counts[idx-win['start_frame'],col]=(gaps[:,None]<=np.array([.001,.003,.005])).sum(0)
        valid=(counts[:,:,0]>=0).all(1)
        name=out/f"{case['clip']}_counts.npz";np.savez_compressed(name,counts=counts)
        results.append(dict(clip=case['clip'],frames=n,both_annotated=int(valid.sum()),
            right_near=(counts[:,0]>0).sum(0).tolist(),left_near=(counts[:,1]>0).sum(0).tolist(),
            both_near=((counts[valid]>0).all(1)).sum(0).tolist(),first_three=counts[:3].tolist(),counts=file_pin(name)))
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=results))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);main(p.parse_args().out.resolve())
