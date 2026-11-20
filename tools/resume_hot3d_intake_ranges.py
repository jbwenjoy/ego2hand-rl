"""Complete previously streamed HOT3D tar prefixes with validated HTTP byte ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from intake_adjacent_hot3d_mugs import REV,fetch,file_pin,write_json


def finish(folder,clip):
    url=f'https://huggingface.co/datasets/bop-benchmark/hot3d/resolve/{REV}/train_aria/clip-{clip:06}.tar'
    head=requests.head(url,timeout=20,allow_redirects=False)
    head.raise_for_status()
    size=int(head.headers['x-linked-size'])
    digest=head.headers['x-linked-etag'].strip('"')
    part=folder/f'clip-{clip:06}.part'
    final=part.with_suffix('.tar')
    if final.exists():return fetch(folder,clip)
    start=part.stat().st_size
    def segment(low):
        high=min(low+1024*1024,size)-1
        for attempt in range(4):
            try:
                response=requests.get(url,headers={'Range':f'bytes={low}-{high}'},timeout=(15,40))
                response.raise_for_status()
                if response.status_code!=206 or response.headers.get('content-range')!=f'bytes {low}-{high}/{size}' or len(response.content)!=high-low+1:
                    raise ValueError('Unexpected range response')
                return response.content
            except (requests.RequestException,ValueError):
                if attempt==3:raise
    with ThreadPoolExecutor(max_workers=8) as pool,part.open('ab') as out:
        for block in pool.map(segment,range(start,size,1024*1024)):
            out.write(block)
    pin=file_pin(part)
    if pin['sha256']!=digest or pin['bytes']!=size:raise ValueError('Assembled archive hash mismatch')
    part.replace(final)
    return fetch(folder,clip)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    folder=p.parse_args().folder.resolve()
    if (folder/'report.json').exists():raise FileExistsError(folder/'report.json')
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(lambda c:finish(folder,c),[1883,1885]))
    write_json(folder/'report.json',dict(intent=file_pin(folder/'intent.json'),recovery=file_pin(__file__),rows=rows))
