"""Read bounded annotation samples before downloading full HOT3D mug candidates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import tarfile
import numpy as np
import requests
from intake_adjacent_hot3d_mugs import REV,file_pin,write_json
from resume_hot3d_intake_ranges import finish
from supermimic_research.ego.hand_frames import pose_matrix


CLIPS=list(range(1874,1883))+list(range(1886,1895))
SUFFIX=re.compile(r'^(\d{6})\.(objects|hands|info)\.json$')


def sample(clip):
    url=f'https://huggingface.co/datasets/bop-benchmark/hot3d/resolve/{REV}/train_aria/clip-{clip:06}.tar'
    h=requests.head(url,allow_redirects=False,timeout=20);h.raise_for_status()
    size=int(h.headers['x-linked-size'])
    rows=[];total=0
    for ratio in (.20,.58):
        low=int(size*ratio)//512*512
        high=low+1024*1024-1
        response=requests.get(url,headers={'Range':f'bytes={low}-{high}'},timeout=(20,50))
        response.raise_for_status()
        if response.headers.get('Content-Range')!=f'bytes {low}-{high}/{size}':raise ValueError('Range mismatch')
        buf=response.content;assert len(buf)==high-low+1
        total+=len(buf);at={}
        for i in range(0,len(buf)-512,512):
            if not buf[i:i+6].isdigit():continue
            try:m=tarfile.TarInfo.frombuf(buf[i:i+512],'utf8','strict')
            except (tarfile.HeaderError,UnicodeError):continue
            match=SUFFIX.fullmatch(m.name)
            if match and i+512+m.size<=len(buf):
                at.setdefault(int(match[1]),{})[match[2]]=json.loads(buf[i+512:i+512+m.size])
        for frame,items in sorted(at.items()):
            if not {'hands','objects','info'}<=items.keys():continue
            hand=(items['hands'].get('right') or {})
            wp=(hand.get('umetrack_pose') or {}).get('T_world_from_wrist')
            for ob in items['objects'].get('9',[]):
                transform=ob.get('T_world_from_object')
                if not (wp and transform):continue
                p=pose_matrix(transform)[:3,3];w=pose_matrix(wp)[:3,3]
                rows.append(dict(frame=frame,uid=str(ob['object_uid']),position=p.tolist(),
                    distance_m=float(np.linalg.norm(p-w)),co_visible=bool(
                        (hand.get('visibilities_modeled',{}).get('214-1') or 0)>0 and
                        (ob.get('visibilities_predicted',{}).get('214-1') or 0)>0)))
    good=None
    for uid in sorted({x['uid'] for x in rows}):
        frames=[x for x in rows if x['uid']==uid]
        xyz=np.array([x['position'] for x in frames])
        delta=np.linalg.norm(xyz[:,None]-xyz[None,:],axis=-1).max()
        near=min((x['distance_m'] for x in frames if x['co_visible']),default=100.)
        if delta>=.01 and near<=.30:
            score=(float(delta),-near)
            if good is None or score>tuple(good):good=list(score)
    return dict(clip=clip,bytes_read=total,observations=rows,prefilter_score=good)


def main(folder):
    folder.mkdir(exist_ok=False)
    write_json(folder/'intent.json',dict(code=file_pin(__file__),clips=CLIPS,revision=REV,
        rule='Read two fixed 1 MiB aligned spans per tar. If a common BOP9 UID moves >=10mm, and a co-visible right wrist lies within 0.30m on any sampled frame, download up to two passing tars, ordered by motion then proximity. Apply original full-window intake independently. Negative sparse screen cannot establish absence.'))
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows=list(pool.map(sample,CLIPS))
    write_json(folder/'sparse.json',rows)
    selected=sorted((r for r in rows if r['prefilter_score']),key=lambda r:(-r['prefilter_score'][0],-r['prefilter_score'][1],r['clip']))[:2]
    full=[]
    for r in selected:
        (folder/f"clip-{r['clip']:06}.part").touch(exist_ok=False)
        full.append(finish(folder,r['clip']))
    write_json(folder/'report.json',dict(intent=file_pin(folder/'intent.json'),sparse=file_pin(folder/'sparse.json'),rows=full))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    main(p.parse_args().out.resolve())
