"""Run the same validated retargeter on every fully eligible sparse-screened clip."""
import argparse
from pathlib import Path
import torch
from retarget_adjacent_mugs import trial,physics,file_pin,write_json


def run(intake,out):
    root=Path(__file__).resolve().parents[1]/'runs'
    torch.set_num_threads(2)
    protocol=file_pin(root/'mug_annotated_windows_original_v1/intent.json')
    spec=physics.read_json(protocol)
    inputs=spec['inputs']
    old=physics.load_arrays(inputs['reference'])
    archive=file_pin(intake);records=physics.read_json(archive)['rows']
    out.mkdir(exist_ok=False)
    write_json(out/'intent.json',dict(source=archive,original_protocol=protocol,code=file_pin(__file__),
        scope='Identical original-window solver and nominal mug asset on every eligible selected right-hand window; no per-window tuning.'))
    rows=[]
    for r in records:
        rows.append(trial(out,r,inputs,old,False) if r['selected'] else dict(clip=r['clip'],status='failed_full_intake'))
        write_json(out/'progress.json',rows)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('intake',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args();run(args.intake,args.out.resolve())
