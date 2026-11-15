"""Balanced whole-batch source restore time including all epoch boundaries."""
import argparse,json,subprocess,sys,time,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_support_runtime import SupportPenRuntime
from supermimic_research.runtime.pen_owned_probe_recorder import restore_source
from supermimic_research.runtime.source_validation_epoch import SourceValidationEpoch


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    run=json.loads(a.benchmark.read_text());cfg=tomllib.loads(verify_file_pin(run['config']).read_text());op=(ROOT/'configs'/cfg['objective_config']).resolve();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or run['passed']!=run['total']:raise ValueError('Fresh qualified batch timing')
    out.mkdir();protocol=ROOT/'docs/source-epoch-amortized-protocol-v1.md';paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),protocol]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    states=[torch.load(verify_file_pin(s['source']),map_location='cpu',weights_only=False) for s in run['sources']]
    from mjlab.sim.sim import Simulation
    original=Simulation.step;observed=0;runtime=None;checks={};blocks=[]
    def step(sim):
        nonlocal observed
        original(sim);observed+=sim.num_envs
    Simulation.step=step
    try:
        with torch.no_grad(),torch.jit.optimized_execution(False):
            runtime=SupportPenRuntime(op,1)
            for s in states:restore_source(runtime,s)
            def batch(token,block):
                for i,source in enumerate(states):
                    for repeat in range(3):
                        audit=restore_source(runtime,source,source_epoch=token)
                        checks[f'B{block}_S{i}_R{repeat}']=audit['all_array_bytes_equal'] and audit['all_scalar_values_equal'] and all(torch.equal(v.cpu(),source['observation'][key]) for key,v in runtime.observation().items())
            for block,mode in enumerate(['STRICT','READ_LOCKED_EPOCH','READ_LOCKED_EPOCH','STRICT']):
                torch.cuda.synchronize();start=time.perf_counter();epoch=None
                if mode=='STRICT':batch(None,block)
                else:
                    first=states[0]['native']
                    with SourceValidationEpoch(first['sources'],first['engine_sources']) as epoch:batch(epoch,block)
                torch.cuda.synchronize();wall=time.perf_counter()-start
                if epoch is not None:checks[f'B{block}_committed']=epoch.validated
                blocks.append(dict(block=block,mode=mode,wall_s=wall,restores=18,epoch=None if epoch is None else epoch.audit()))
                print(json.dumps(blocks[-1]),flush=True)
    finally:
        if runtime:runtime.close()
        Simulation.step=original
    checks['zero_native']=observed==0;checks['registered_population']=len(states)==6 and len(blocks)==4
    total={mode:sum(b['wall_s'] for b in blocks if b['mode']==mode) for mode in ['STRICT','READ_LOCKED_EPOCH']};verify_source_snapshot(receipt,live=True)
    result=dict(schema='source_epoch_whole_batch_timing:1',benchmark=file_pin(a.benchmark),protocol=file_pin(protocol),source_receipt=file_pin(out/'source_receipt.json'),blocks=blocks,
        total_s=total,pooled_ratio=total['STRICT']/total['READ_LOCKED_EPOCH'],checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=observed,optimizer_steps=0,
        scope='Whole measured18-restore blocks including epoch construction/lock/hash/identity/cleanup. Warmup and runtime construction excluded; no ordinary PPO or end-to-end speedup claim.')
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],total_s=total,ratio=result['pooled_ratio'])),flush=True)
    if not all(checks.values()):raise ValueError('Retain whole-batch byte/equivalence discrepancy')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
