"""Default/default/epoch negative controls for the retained impulse anomaly."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import mujoco
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_support_runtime import SupportPenRuntime
from supermimic_research.runtime.pen_owned_probe_recorder import OwnedProbeRecorder,restore_source
from supermimic_research.runtime.source_validation_epoch import SourceValidationEpoch
from supermimic_research.runtime.gpu_model_snapshot import model_leaves
from supermimic_research.mechanics.warp_packet_decode import decode_saved_case


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


class AnomalyObserver(OwnedProbeRecorder):
    def after_step(self):
        super().after_step();t=self.trace;mass=self.mass;dt=.0025
        residual=mass*(t['post_qvel'][-1][:3]-t['pre_qvel'][-1][:3])-dt*(t['post_qfrc_constraint'][-1][:3]+mass*self.gravity)
        if np.linalg.norm(residual)>5e-7:
            arrays,_=model_leaves(self.sim.wp_data);path=self.folder/f'anomaly_tick{len(t["post_qpos"])-1}.npz'
            np.savez_compressed(path,**{k:v.numpy().copy() for k,v in arrays.items()})
            self.anomalies.append(dict(tick=len(t['post_qpos'])-1,generalized_impulse_residual_N_s=residual.tolist(),data=file_pin(path)))


def main(a):
    import tomllib
    run=json.loads(a.benchmark.read_text());original_case=run['cases'][69];out=a.output.resolve();cfg=tomllib.loads(verify_file_pin(run['config']).read_text());op=(ROOT/'configs'/cfg['objective_config']).resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or original_case['name']!='RESIDUAL_P030_READ_LOCKED_EPOCH_D2_E0.005_S1_R0':raise ValueError('Exact retained counterexample and fresh output required')
    out.mkdir();protocol=ROOT/'docs/source-epoch-impulse-replay-protocol-v1.md';paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),protocol]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    source=torch.load(verify_file_pin(original_case['source']),map_location='cpu',weights_only=False);old=dict(np.load(verify_file_pin(original_case['trace'])));raw=old['actions20'][0]
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(original_case['model'])));physical=dict(np.load(verify_file_pin(original_case['physical_context'])))
    from mjlab.sim.sim import Simulation
    native_step=Simulation.step;observed=0;recorder=None;runtime=None;cases=[];checks={}
    def step(sim):
        nonlocal observed
        if recorder is not None and recorder.enabled:recorder.before_step()
        native_step(sim);observed+=sim.num_envs
        if recorder is not None and recorder.enabled:recorder.after_step()
    Simulation.step=step
    try:
        with torch.no_grad(),torch.jit.optimized_execution(False):
            runtime=SupportPenRuntime(op,1);recorder=AnomalyObserver(runtime);recorder.mass=original_case['physics']['mass'];recorder.gravity=np.array(original_case['physics']['gravity']);restore_source(runtime,source)
            with SourceValidationEpoch(source['native']['sources'],source['native']['engine_sources']) as epoch:
                modes=['STRICT_A','STRICT_B','READ_LOCKED_EPOCH']
                for repeat in range(30):
                    order=modes[repeat%3:]+modes[:repeat%3]
                    for mode in order:
                        audit=restore_source(runtime,source,source_epoch=epoch if mode=='READ_LOCKED_EPOCH' else None)
                        name=f'{mode}_R{repeat:02d}';folder=out/name;folder.mkdir();recorder.folder=folder;recorder.anomalies=[];recorder.begin(raw)
                        runtime.step(torch.tensor(raw,device=runtime.device)[None]);packet=recorder.finish();np.savez_compressed(folder/'trace.npz',**packet)
                        metrics,decoded=decode_saved_case(original_case,packet,model,physical)
                        cases.append(dict(name=name,mode=mode,repeat=repeat,trace=file_pin(folder/'trace.npz'),metrics=metrics,anomalies=recorder.anomalies))
                        checks[name+'_source']=audit['all_array_bytes_equal'] and audit['all_scalar_values_equal']
                    if repeat%5==4:print(json.dumps(dict(repeat=repeat,cases=len(cases),impulse_failures=sum(not c['metrics']['checks']['linear_impulse'] for c in cases))),flush=True)
            checks.update(epoch_committed=epoch.validated,registered_cases=len(cases)==90,native_count=observed==720)
            summaries={mode:dict(cases=sum(c['mode']==mode for c in cases),impulse_failures=sum(c['mode']==mode and not c['metrics']['checks']['linear_impulse'] for c in cases),
                maximum_impulse_residual_N_s=max(c['metrics']['max_linear_impulse_residual_N_s'] for c in cases if c['mode']==mode)) for mode in modes}
            verify_source_snapshot(receipt,live=True)
            result=dict(schema='source_epoch_impulse_counterexample_replay:1',benchmark=file_pin(a.benchmark),protocol=file_pin(protocol),source_receipt=file_pin(out/'source_receipt.json'),
                original_case=original_case,source=original_case['source'],cases=cases,summaries=summaries,epoch=epoch.audit(),checks=checks,passed=sum(checks.values()),total=len(checks),
                actual_native_integrations=observed,optimizer_steps=0,scope='Selected counterexample negative controls;physical failures retained,not passed by contract checks. Does not establish failure rates,cache causality,solver root cause or successful equivalence.')
            write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],summaries=summaries)),flush=True)
    finally:
        if runtime:runtime.close()
        Simulation.step=native_step


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
