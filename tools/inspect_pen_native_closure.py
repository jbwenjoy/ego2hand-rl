"""Inspect the actual controlled adapter closure before declaring portable support."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.artifacts import file_pin
from supermimic_research.runtime.portable_inventory import runtime_inventory,describe_inventory
from supermimic_research.runtime.pen_native_policy import NativePenPolicyAdapter
from supermimic_research.runtime.gpu_model_snapshot import model_leaves


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')


def main(a):
    out=a.output.resolve();cp=a.config.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh inspection output')
    out.mkdir()
    paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),cp]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    write(out/'source_receipt.json',receipt)
    with torch.jit.optimized_execution(False):
        adapter=NativePenPolicyAdapter(cp,1)
        try:
            native=[adapter.env.sim.wp_model,adapter.env.sim.wp_data]
            inv=runtime_inventory(adapter,native_roots=native)
            desc=describe_inventory(inv);write(out/'inventory.json',desc)
            n=[model_leaves(v) for v in native]
            result=dict(schema='pen_native_mutable_closure_inventory:1',inventory=file_pin(out/'inventory.json'),
                source_receipt=file_pin(out/'source_receipt.json'),config=file_pin(cp),
                tensors=len(inv['tensors']),arrays=len(inv['arrays']),scalars=len(inv['scalars']),
                opaque=len(inv['opaque']),aliases=len(inv['aliases']),objects=len(inv['objects']),
                native_model_arrays=len(n[0][0]),native_data_arrays=len(n[1][0]),
                native_integrations=adapter.physics_integrations,
                scope='Read-only mutable closure inventory at fresh reset; opaque resources not yet classified. Not portable resume qualification.')
            verify_source_snapshot(receipt,live=True);write(out/'report.json',result);print(json.dumps(result))
        finally:adapter.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
