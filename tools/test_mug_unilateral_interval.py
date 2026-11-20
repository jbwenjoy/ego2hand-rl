"""Repeat the fixed-wrist experiment within the longest right-only geometric interval."""
import argparse
from itertools import groupby
from pathlib import Path
import numpy as np
import torch
from retarget_adjacent_mugs import trial,physics,verify_file_pin,file_pin,write_json


def run(out):
    root=Path(__file__).resolve().parents[1]/'runs';torch.set_num_threads(2)
    source=file_pin(root/'mug_bilateral_proximity_v1/report.json')
    contacts=physics.read_json(source)
    parent=file_pin(root/'mug_screened_transfer_v1/report.json')
    original=physics.read_json(parent)['rows'][0]
    inputs=physics.read_json(file_pin(root/'mug_annotated_windows_original_v1/intent.json'))['inputs']
    row=next(v for v in contacts['rows'] if v['clip']==original['clip'])
    with np.load(verify_file_pin(row['counts'])) as f:counts=f['counts'][:,:,1]
    eligible=(counts[:,0]>0)&(counts[:,1]==0)
    blocks=[list(g) for flag,g in groupby(range(len(eligible)),key=lambda i:eligible[i]) if flag]
    block=max(blocks,key=lambda b:(len(b),-b[0]))
    offset=original['window']['start_frame'];win=dict(start_frame=offset+block[0],end_frame=offset+block[-1],
        object_uid=original['window']['object_uid'])
    out.mkdir(exist_ok=False)
    write_json(out/'intent.json',dict(proximity=source,parent=parent,window=win,code=file_pin(__file__),
        rule='Longest contiguous set of frames with at least one right vertex within 3mm and no left vertex within 3mm. Both hands annotated; criterion based solely on annotations within already examined interval. Same original mechanical entry pose and nominal dynamics; interval-specific anchored target.'))
    result=trial(out,dict(archive=original['archive'],clip=original['clip'],selected=win),inputs,physics.load_arrays(inputs['reference']),False)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=[result]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);run(p.parse_args().out.resolve())
