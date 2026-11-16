"""All36 frozen-source compression,exact bytes/trees and conservative storage."""
import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.snapshot_transport import pack_bytes,unpack_bytes
from supermimic_research.learning.world_model import raw_tree_equal
from supermimic_research.strict_json import write_json


def read(p):return json.loads(Path(p).read_text())


def main():
    started=time.perf_counter();bp=ROOT/'runs/pen_native_suggestions_bank_v1/report.json';bank=read(bp);out=ROOT/'runs/pen_snapshot_transport_v1'
    if out.exists() or bank['passed']!=bank['total'] or not all(bank['checks'].values()):raise ValueError('Fresh admitted snapshot transport check required')
    verify_source_snapshot(read(verify_file_pin(bank['source_receipt'])),live=True);out.mkdir();rows=[]
    receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__)],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt)
    for s in bank['sources']:
        p=verify_file_pin(s['original_source']['source']);payload=p.read_bytes();t=time.perf_counter();record=pack_bytes(out/(s['name']+'.pt.gz'),payload);pack_s=time.perf_counter()-t
        t=time.perf_counter();decoded=unpack_bytes(record);unpack_s=time.perf_counter()-t
        a=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False);b=torch.load(io.BytesIO(decoded),map_location='cpu',weights_only=False)
        rows.append(dict(name=s['name'],source=s['original_source']['source'],transport=record,bytes_equal=payload==decoded,tree_equal=raw_tree_equal(a,b),pack_s=pack_s,unpack_s=unpack_s))
    raw=sum(r['source']['bytes'] for r in rows);compressed=sum(r['transport']['compressed']['bytes'] for r in rows);max_compressed=max(r['transport']['compressed']['bytes'] for r in rows)
    # All3 snapshots at every screened node plus2 per unmodified node. Estimate,
    # not a worst-case compression guarantee for future live/terminal states.
    snapshots=(16*3*368*3)+(2*3*368*2);free=shutil.disk_usage(ROOT).free;reserve=32*1024**3;estimate=2*max_compressed*snapshots+5*1024**3
    checks=dict(all36=len(rows)==36,all_serialized_bytes=all(r['bytes_equal'] for r in rows),all_tensor_array_trees=all(r['tree_equal'] for r in rows),no_overwrite_originals=all(verify_file_pin(r['source']) for r in rows))
    result=dict(schema='pen_owned_snapshot_transport_qualification:1',bank=file_pin(bp),source_receipt=file_pin(out/'source_receipt.json'),rows=rows,checks=checks,passed=sum(checks.values()),total=len(checks),raw_bytes=raw,compressed_bytes=compressed,compressed_to_raw=compressed/raw,
        max_sample_compressed_bytes=max_compressed,pack_wall_s=sum(r['pack_s'] for r in rows),unpack_wall_s=sum(r['unpack_s'] for r in rows),
        projected_full54_snapshot_count=snapshots,projected_2x_sample_max_plus5GiB_bytes=estimate,free_bytes_observed=free,reserve_bytes=reserve,estimate_fits_current_free=estimate+reserve<free,
        actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,wall_s=time.perf_counter()-started,
        scope='Historical36-source serialized bytes/CPU trees exact. Compression is transport only,no source/native metadata conversion. Storage estimate is empirical,not a guarantee;live endpoint coverage/runtime reserve guard still required. No native capture/rollback or trajectory admission.')
    write_json(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],raw_MiB=raw/2**20,compressed_MiB=compressed/2**20,estimate_GiB=estimate/2**30,fits=result['estimate_fits_current_free'])),flush=True)
    if not all(checks.values()):raise ValueError('Snapshot transport mismatch retained')


if __name__=='__main__':main()
