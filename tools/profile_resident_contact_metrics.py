"""Measure the qualified resident graph on a fixed real native contact state."""
import argparse
import json
import statistics
import subprocess
import sys
import time
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import mujoco
import numpy as np
import torch
import warp as wp
from qualify_resident_contact_replay import Buffer
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh metric profile')
    out.mkdir();paths=sorted((ROOT/'src/supermimic_research').rglob('*.py'))+[Path(__file__).resolve(),ROOT/'tools/qualify_resident_contact_replay.py',cp]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    rp=(cp.parent/cfg['native_report']).resolve();report=json.loads(rp.read_text())
    if report['passed']!=report['total']:raise ValueError('Live metrics must qualify before profiling')
    lp=(cp.parent/cfg['evaluation_ledger']).resolve();ledger=json.loads(lp.read_text());evaluation=json.loads(verify_file_pin(ledger['roots']['evaluation']).read_text());mp=evaluation['cases'][0]['model']
    m=mujoco.MjModel.from_binary_path(str(verify_file_pin(mp)));body=m.body('object/cube').id;cases=[]
    for c in report['cases']:
        records=torch.load(verify_file_pin(c['records']),map_location='cpu',weights_only=False);sources=torch.load(verify_file_pin(c['full_sources']),map_location='cpu',weights_only=False)
        # Choose the densest real contact packet, rather than an empty fast case.
        index=max(range(len(records)),key=lambda i:len(records[i]['packet']['contact']['dim']));packet=records[index]['packet']
        arrays=sources['model']['arrays']
        if not np.array_equal(arrays['geom_bodyid'].reshape(-1),m.geom_bodyid):raise ValueError('Profile topology mismatch')
        capacity=next(iter(sources['data_sources'].values()))['before_physics']['scalars']['naconmax'] if cfg['use_native_capacity'] else None
        b=Buffer(m,body,arrays['jnt_range'],[packet],cfg['device'],capacity=capacity);metric=b.metrics
        before=b.fingerprint();measurements=[]
        for _ in range(cfg['warmup']):metric.sample()
        wp.synchronize_stream(metric.stream)
        for repeat in range(cfg['repeats']):
            metric.reset_window();wp.synchronize_stream(metric.stream)
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            allocated=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved()
            start.record(metric.torch_stream);t=time.perf_counter()
            for _ in range(cfg['samples_per_repeat']):metric.sample()
            dispatch=time.perf_counter()-t;end.record(metric.torch_stream);end.synchronize();wall=time.perf_counter()-t
            measurements.append(dict(repeat=repeat,GPU_interval_ms=start.elapsed_time(end),host_dispatch_ms=dispatch*1000,host_complete_ms=wall*1000,
                torch_allocated_delta_bytes=torch.cuda.memory_allocated()-allocated,torch_reserved_delta_bytes=torch.cuda.memory_reserved()-reserved))
        result=metric.numpy();checks=dict(input_bytes_unchanged=before==b.fingerprint(),sample_count=bool(np.all(result['window']['counts'][:,0]==cfg['samples_per_repeat'])),
            finite_active=not result['instant']['errors'].any() and not result['instant']['global_error'].any(),captured_graph=metric.graph is not None,
            no_retained_torch_allocation=all(x['torch_allocated_delta_bytes']==0 and x['torch_reserved_delta_bytes']==0 for x in measurements))
        cases.append(dict(worlds=c['worlds'],source_record=c['records'],source_full_model=c['full_sources'],selected_packet=index,
            global_active_contacts=len(packet['contact']['dim']),launched_contact_capacity=b.data.naconmax,measurements=measurements,checks=checks,
            median_GPU_us_per_sample=statistics.median(x['GPU_interval_ms'] for x in measurements)*1000/cfg['samples_per_repeat'],
            median_host_dispatch_us_per_sample=statistics.median(x['host_dispatch_ms'] for x in measurements)*1000/cfg['samples_per_repeat'],
            median_host_complete_us_per_sample=statistics.median(x['host_complete_ms'] for x in measurements)*1000/cfg['samples_per_repeat']))
    verify_source_snapshot(receipt,live=True);checks={f"W{c['worlds']}_"+k:bool(v) for c in cases for k,v in c['checks'].items()}
    result=dict(schema='resident_contact_fixed_input_profile:1',config=file_pin(cp),source_receipt=file_pin(out/'source_receipt.json'),native_report=file_pin(rp),host_model=mp,
        cases=cases,checks=checks,passed=sum(checks.values()),total=len(checks),native_integrations=0,optimizer_steps=0,
        timed_metric_samples=sum(cfg['samples_per_repeat']*cfg['repeats'] for _ in cases),untimed_warmup_samples=cfg['warmup']*len(cases),
        synchronization='Before each measured repeat and final event only; no numpy/item/host contact reads in the sample loop. Stream wait events and Python dispatch are included.',
        limits='Fixed packet with declared source native contact capacity; not a physics/collection benchmark. Torch allocation delta excludes Warp/driver pools and transient events.',
        scope=cfg['scope'])
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],GPU_us=[c['median_GPU_us_per_sample'] for c in cases])))
    if not all(checks.values()):raise ValueError('Preserve profile/metric discrepancy')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
