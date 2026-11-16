"""Independent raw-trace counts and forward-frequency arithmetic,not speedup."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    run=json.loads(a.timeline.read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or run['passed']!=run['total']:raise ValueError('Fresh complete raw GPU timeline review')
    out.mkdir();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve()]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    events=json.loads(verify_file_pin(run['trace']).read_text())['traceEvents'];kernels=[e for e in events if e.get('cat')=='kernel' and 'dur' in e];checks=dict(kernel_count=len(kernels)==run['kernel_count']);phase_records={}
    for name,phase in run['phases'].items():
        label=next(e for e in events if e.get('name')==name and e.get('ph')=='X');selected=[e for e in kernels if label['ts']<=e['ts'] and e['ts']+e['dur']<=label['ts']+label['dur']]
        checks[name+'_kernel_count']=len(selected)==phase['GPU_kernel_count'];checks[name+'_duration']=bool(np.isclose(sum(e['dur'] for e in selected),phase['GPU_kernel_sum_us'],atol=1e-3,rtol=1e-12));phase_records[name]=dict(kernel_count=len(selected),sum_kernel_ms=sum(e['dur'] for e in selected)/1000,label_wall_ms=label['dur']/1000)
    memory=torch.load(verify_file_pin(run['rollout']),map_location='cpu',weights_only=False);done=memory['terminated']|memory['truncated'];reset_calls=int(done.any(-1).sum());reset_worlds=int(done.sum());expected_full_forward_sequences=512+64+2*reset_calls
    ccd=[g for g in run['kernel_groups'] if g['name'].startswith('ccd_kernel_builder')];ccd_us=sum(g['sum_us'] for g in ccd)
    for i,g in enumerate(ccd):
        selected=[e for e in kernels if e['name']==g['name']];checks[f'CCD{i}_count']=len(selected)==g['count']==expected_full_forward_sequences;checks[f'CCD{i}_duration']=bool(np.isclose(sum(e['dur'] for e in selected),g['sum_us'],atol=1e-3,rtol=1e-12))
    checks['CCD_groups_present']=len(ccd)>0;checks['all_kernels_in_labels']=sum(p['kernel_count'] for p in phase_records.values())==len(kernels)
    verify_source_snapshot(receipt,live=True)
    result=dict(schema='ordinary_pen_GPU_timeline_saved_review:1',timeline=file_pin(a.timeline),source_receipt=file_pin(out/'source_receipt.json'),phases=phase_records,reset_calls=reset_calls,reset_worlds=reset_worlds,
        forward_frequency=dict(native_steps=512,post_control_forward=64,upstream_reset_forward=reset_calls,adapter_reset_forward=reset_calls,total=expected_full_forward_sequences),CCD_groups=ccd,CCD_sum_ms=ccd_us/1000,
        CCD_fraction_of_collect_kernel_sum=ccd_us/run['phases']['collect64']['GPU_kernel_sum_us'],checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,
        scope='Independent raw-trace duration/count and saved-reset arithmetic. Kernel sums may overlap,not wall/occupancy;profiler overhead is material. Matching698-forward frequencies identifies repeated reset/control geometry work,not proof any refresh can be removed without epoch/state qualification.')
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],reset_calls=reset_calls,forward_sequences=expected_full_forward_sequences,CCD_sum_ms=ccd_us/1000,CCD_kernel_sum_fraction=result['CCD_fraction_of_collect_kernel_sum'])),flush=True)
    if not all(checks.values()):raise ValueError('Retain raw timeline arithmetic discrepancy')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--timeline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
