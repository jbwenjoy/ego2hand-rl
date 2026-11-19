"""Six fresh W1 actor/model/observation admissions with a zero-step budget."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_frozen_cost_long_design_v1 import settings,read
from supermimic_research.strict_json import write_json


def main(a):
    start=time.perf_counter(); cp=a.config.resolve(); cfg,paths,inputs,actors,contexts,q=settings(cp); out=paths['admission'].parent
    if out.exists(): raise ValueError('Fresh zero-step admission required')
    out.mkdir(); sources=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,paths['protocol']]
    receipt=freeze_sources(sources,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()); write_json(out/'source_receipt.json',receipt)
    write_json(out/'intent.json',dict(inputs=inputs,actors=actors,contexts=contexts,cases=6,native_cap=0,optimizer_budget=0,normalizer_fit_budget=0))
    import numpy as np
    import torch
    from mjlab.sim.sim import Simulation
    from supermimic_research.runtime.explicit_collision_backend import backend_scope
    from supermimic_research.runtime.pen_stable_friction_profile_v2 import CostProfile
    from supermimic_research.runtime import owned_pen_state as native
    from supermimic_research.runtime.pen_continuous_math_view_v1 import record_dynamic,policy,context
    from supermimic_research.learning.pen_frozen_cost_actor_v1 import initialize
    manifest=read(verify_file_pin(q['factory_manifest'])); original=Simulation.step; runtime=None; rows=[]; stage='setup'
    def reject(sim): raise AssertionError('No native steps in zero-step actor admission')
    Simulation.step=reject
    try:
        torch.set_num_threads(4); torch.use_deterministic_algorithms(True); torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        with backend_scope(paths['backend_manifest']),torch.no_grad(),torch.jit.optimized_execution(False):
            for algorithm in cfg['algorithms']:
                with CostProfile(q['identities'][algorithm],manifest) as profile:
                    for label in cfg['policies']:
                        stage=algorithm+'_'+label; folder=out/stage; folder.mkdir(); runtime=profile.create_runtime(paths['objective_config'],1)
                        try:
                            state=torch.load(verify_file_pin(actors[label]),map_location='cpu',weights_only=False)
                            actor,audit=initialize(runtime,state,actors[label],contexts['SUPPORT' if label=='SUPPORT' else 'RESIDUAL'],label,profile)
                            before=native.capture(runtime,receipt); dynamic=record_dynamic(runtime)
                            case=dict(aids=runtime.action._position_ctrl_ids.cpu().tolist(),qids=runtime.features.qids.cpu().tolist(),dofs=runtime.features.dofs.cpu().tolist(),model_layout=dict(object_qposadr=runtime.features.qa))
                            cpu=policy(before,dynamic,case).numpy(); actual=runtime.observation()['policy'].cpu().numpy(); error=float(np.linalg.norm(cpu.astype(float)-actual.astype(float))/(1+np.linalg.norm(actual.astype(float))))
                            raw=actor.distribution(runtime.observation()['policy'],validate_args=False).mean.cpu().numpy(); ctx=context(before,dynamic,case)
                            checks=dict(full_model=audit['full_W1_model_bytes_equal'],actor_norm=audit['actor_normalizer_bytes_equal'],no_native=runtime.owned_native_integrations==runtime.physics_integrations==0,
                                fresh_phase=dynamic['phase']==1 and float(dynamic['arrays']['time'][0])==0.,observation161=actual.shape==(1,161) and np.isfinite(actual).all() and error<=1e-5,
                                finite_RAW=raw.shape==(1,20) and np.isfinite(raw).all(),no_norm_update=all(torch.equal(v.cpu(),state['modules']['actor'][k]) for k,v in actor.state_dict().items()),profile=profile.verify())
                            torch.save(dict(initial=before,dynamic=dynamic,case=case,CPU_policy=cpu,GPU_policy=actual,RAW=raw,context=ctx),folder/'admission.pt')
                            row=dict(name=stage,algorithm=algorithm,policy=label,actor=actors[label],audit=audit,checks={k:bool(v) for k,v in checks.items()},passed=sum(bool(v) for v in checks.values()),total=len(checks),observation_scaled_L2=error,record=file_pin(folder/'admission.pt'))
                            write_json(folder/'report.json',row); rows.append(file_pin(folder/'report.json'))
                            if not all(checks.values()): raise ValueError('Actor admission mismatch')
                        finally: runtime.close(); runtime=None
        checks=dict(all6=len(rows)==6,all_rows=all(read(verify_file_pin(p))['passed']==read(verify_file_pin(p))['total'] for p in rows),zero_native=True)
        verify_source_snapshot(receipt,live=True); write_json(paths['admission'],dict(schema='pen_frozen_cost_actor_zero_step_admission:1',config=file_pin(cp),inputs=inputs,actors=actors,contexts=contexts,rows=rows,source_receipt=file_pin(out/'source_receipt.json'),checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,wall_s=time.perf_counter()-start,scope='Actor/norm-only new-backend transfer,full W1 Model witness and initial observation audit;no native state import,policy execution or W32 admission.'))
        print(json.dumps(dict(report=file_pin(paths['admission']),checks=[sum(checks.values()),len(checks)],rows=6,native=0)),flush=True)
    except BaseException as e:
        write_json(out/'failure.json',dict(stage=stage,type=type(e).__name__,message=str(e),rows=rows,actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,resumable=False)); raise
    finally: Simulation.step=original


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True); main(p.parse_args())
