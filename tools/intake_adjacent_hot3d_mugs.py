"""Check both immediate HOT3D neighbours of the development clip before ranking by simulation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path

import requests
from supermimic_research.artifacts import file_pin
from supermimic_research.ego.clip_inventory import read_clip,inventory
from supermimic_research.strict_json import write_json


REV='30fe9674782f32e1e5edba98476b6ff4300132c5'


def fetch(out,clip):
    name=f'clip-{clip:06}.tar'
    url=f'https://huggingface.co/datasets/bop-benchmark/hot3d/resolve/{REV}/train_aria/{name}'
    response=requests.head(url,allow_redirects=False,timeout=20)
    response.raise_for_status()
    expected=response.headers['x-linked-etag'].strip('"')
    target=out/name
    tmp=target.with_suffix('.part')
    if not target.exists():
        with requests.get(url,stream=True,timeout=(20,60)) as stream:
            stream.raise_for_status()
            with tmp.open('wb') as f:
                for chunk in stream.iter_content(1024*1024):
                    f.write(chunk)
        pin=file_pin(tmp)
        if pin['sha256']!=expected or pin['bytes']!=int(response.headers['x-linked-size']):
            raise ValueError('HOT3D download integrity failure')
        tmp.replace(target)
    pin=file_pin(target)
    if pin['sha256']!=expected:
        raise ValueError('Incorrect local clip')
    frames=read_clip(target,[9])
    data=inventory(frames)
    options=[v for v in data['candidates'] if v['hand']=='right']
    return dict(clip=clip,url=url,archive=pin,source_sequences=sorted({x['sequence_id'] for x in frames}),
        participants=sorted({x['participant_id'] for x in frames}),time_ns=[frames[0]['timestamp_ns'],frames[-1]['timestamp_ns']],
        all_windows=len(data['windows']),right_eligible=len(options),selected=options[0] if options else None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    out=p.parse_args().out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():raise FileExistsError(out/'report.json')
    write_json(out/'intent.json',dict(revision=REV,clips=[1883,1885],code=file_pin(__file__),selection=
        'Apply the existing 61-frame intake and ranking to right-hand BOP9 windows on both immediate neighbours of 1884. Retain all eligible clips before any new simulation.'))
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(lambda c:fetch(out,c),[1883,1885]))
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=rows))
