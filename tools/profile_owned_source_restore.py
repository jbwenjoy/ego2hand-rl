"""Read-only cProfile of all declared source restores; no physics or updates."""
import argparse,cProfile,json,pstats,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_support_runtime import SupportPenRuntime
from supermimic_research.runtime.pen_owned_probe_recorder import restore_source


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    import tomllib
    run=json.loads(a.probe.read_text());cfg=tomllib.loads(verify_file_pin(run['config']).read_text());op=(ROOT/'configs'/cfg['objective_config']).resolve();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or a.repeats!=3:raise ValueError('Fresh fixed3-repeat12-source restore profile')
    out.mkdir();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve()]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    from mjlab.sim.sim import Simulation
    original=Simulation.step;observed=0;runtime=None;timings=[];checks={};profile=cProfile.Profile()
    def step(sim):
        nonlocal observed
        original(sim);observed+=sim.num_envs
    Simulation.step=step
    try:
        with torch.no_grad(),torch.jit.optimized_execution(False):
            for actor in ['RESIDUAL','SUPPORT']:
                runtime=SupportPenRuntime(op,1);runtime.support_weight=float(runtime.objective_cfg['support_weights'][actor])
                for entry in [s for s in run['sources'] if s['actor']==actor and s['available']]:
                    state=torch.load(verify_file_pin(entry['source']),map_location='cpu',weights_only=False)
                    warmup=restore_source(runtime,state);checks[entry['name']+'_warmup']=warmup['all_array_bytes_equal'] and warmup['all_scalar_values_equal']
                    for repeat in range(3):
                        torch.cuda.synchronize();start=time.perf_counter();profile.enable()
                        audit=restore_source(runtime,state)
                        profile.disable();torch.cuda.synchronize();wall=time.perf_counter()-start
                        checks[f'{entry["name"]}_R{repeat}']=audit['all_array_bytes_equal'] and audit['all_scalar_values_equal']
                        timings.append(dict(source=entry['name'],repeat=repeat,wall_s=wall))
                runtime.close();runtime=None
    finally:
        profile.disable()
        if runtime:runtime.close()
        Simulation.step=original
    profile.dump_stats(str(out/'restore.prof'));stats=pstats.Stats(profile);top=[]
    for (filename,line,function),(primitive,calls,total,cumulative,_) in sorted(stats.stats.items(),key=lambda item:item[1][3],reverse=True)[:30]:
        top.append(dict(file=filename,line=line,function=function,primitive_calls=primitive,calls=calls,self_s=total,cumulative_s=cumulative))
    checks.update(zero_native=observed==0,declared_measurements=len(timings)==36)
    verify_source_snapshot(receipt,live=True)
    result=dict(schema='owned_source_restore_cprofile:1',probe=file_pin(a.probe),source_receipt=file_pin(out/'source_receipt.json'),profile=file_pin(out/'restore.prof'),
        timings=timings,top_cumulative=top,total_measured_wall_s=sum(x['wall_s'] for x in timings),checks=checks,passed=sum(checks.values()),total=len(checks),
        actual_native_integrations=observed,optimizer_steps=0,
        scope='Instrumented full-source restore only,12 sources x3 after one warmup each. cProfile affects time; overlapping cumulative entries must not be added. No end-to-end training/collection speedup or measured uninstrumented GPU saturation claim.')
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],wall_s=result['total_measured_wall_s'],top=top[:12])),flush=True)
    if not all(checks.values()):raise ValueError('Preserve restore profile qualification failure')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--probe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=3);main(p.parse_args())
