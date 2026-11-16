"""Lossless immutable-field partition of all36 historical owned native4 states."""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.shared_snapshot_transport import SharedImmutableTransport,load_shared,tree_digest
from supermimic_research.learning.world_model import raw_tree_equal
from supermimic_research.strict_json import write_json


def read(p):return json.loads(Path(p).read_text())


def main():
    started=time.perf_counter();bp=ROOT/'runs/pen_native_suggestions_bank_v1/report.json';bank=read(bp);out=ROOT/'runs/pen_shared_snapshot_transport_v1'
    if out.exists() or bank['passed']!=bank['total'] or not all(bank['checks'].values()):raise ValueError('Fresh admitted shared transport qualification required')
    verify_source_snapshot(read(verify_file_pin(bank['source_receipt'])),live=True);out.mkdir();rows=[];stores={}
    receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+[Path(__file__)],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt)
    try:
        for source in bank['sources']:
            original=source['original_source'];path=verify_file_pin(original['source']);value=torch.load(path,map_location='cpu',weights_only=False)['native'];episode=original['episode']
            if episode not in stores:stores[episode]=SharedImmutableTransport(out/(episode+'.shared.pt.gz'))
            t=time.perf_counter();record=stores[episode].save(out/(source['name']+'.delta.pt.gz'),value);save_s=time.perf_counter()-t
            t=time.perf_counter();decoded=load_shared(record);load_s=time.perf_counter()-t
            row=dict(name=source['name'],episode=episode,source=original['source'],transport=record,tree_equal=raw_tree_equal(value,decoded),typed_layout_digest_equal=tree_digest(value)==tree_digest(decoded),save_s=save_s,load_s=load_s)
            rows.append(row);write_json(out/(source['name']+'.transport.json'),record)
        shared=sum(s.record['compressed']['bytes'] for s in stores.values());delta=sum(r['transport']['payload']['compressed']['bytes'] for r in rows);maximum=max(r['transport']['payload']['compressed']['bytes'] for r in rows);max_static=max(s.record['compressed']['bytes'] for s in stores.values())
        snapshots=(16*3*368*3)+(2*3*368*2);free=shutil.disk_usage(ROOT).free;reserve=32*1024**3;estimate=2*maximum*snapshots+2*max_static*54+5*1024**3
        checks=dict(all36=len(rows)==36,all_owned_trees=all(r['tree_equal'] for r in rows),all_types_layouts=all(r['typed_layout_digest_equal'] for r in rows),shared_fields_per_episode=len(stores)==8,original_files_unchanged=all(verify_file_pin(r['source']) for r in rows))
        verify_source_snapshot(receipt,live=True)
        result=dict(schema='pen_shared_owned_snapshot_transport_qualification:1',bank=file_pin(bp),source_receipt=file_pin(out/'source_receipt.json'),rows=rows,checks=checks,passed=sum(checks.values()),total=len(checks),shared_bytes=shared,delta_bytes=delta,total_compressed_bytes=shared+delta,max_sample_delta_bytes=maximum,max_sample_static_bytes=max_static,
            save_wall_s=sum(r['save_s'] for r in rows),load_wall_s=sum(r['load_s'] for r in rows),projected_full54_snapshot_count=snapshots,projected_2x_sample_max_plus5GiB_bytes=estimate,free_bytes_observed=free,reserve_bytes=reserve,estimate_fits_current_free=estimate+reserve<free,
            actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,wall_s=time.perf_counter()-started,
            scope='Native4 object reconstruction,not original torch zip bytes. Full model and declared immutable fields shared only within an exact-checked episode. All dynamic fields/config/RNG/sources/layout manifests preserved. Empirical storage estimate only,not future worst-case or GPU rollback qualification;rolling free-space guard/live endpoint coverage still required.')
        write_json(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],shared_MiB=shared/2**20,delta_MiB=delta/2**20,estimate_GiB=estimate/2**30,fits=result['estimate_fits_current_free'])),flush=True)
        if not all(checks.values()):raise ValueError('Shared transport discrepancy retained')
    except Exception as e:
        write_json(out/'failure.json',dict(type=type(e).__name__,message=str(e),completed_rows=rows,actual_native_integrations=0,optimizer_steps=0));raise


if __name__=='__main__':main()
