"""One complete, zero-update frozen collector rollout on both simultaneously loaded clips."""
from pathlib import Path
import json,sys
import torch
from benchmark_pen_scaled import ROOT,OBJECTIVE,COLLISION,backend_scope,factory,pilot_settings,BackendSupportPenPPO
from evaluate_pen_dev_clip import config
from supermimic_research.runtime.pen_multiclip_batch import multiclip_w32
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json


def main(out):
    out.mkdir(exist_ok=False);clips=ROOT/'runs';obj,_=config(out,'clip_003',clips/'pen_refit_dev_preload_v1/report.json')
    src=file_pin(clips/'pen_scaled_train_v1/latest_full.pt');data=torch.load(verify_file_pin(src),weights_only=False,map_location='cpu')
    assert data['iteration']==2781
    write_json(out/'intent.json',dict(parent=src,clip003_objective=file_pin(obj),clip001_objective=file_pin(OBJECTIVE),controls=64,seed=7411,
        scope='First fresh collector batch per clip, same residual PPO Actor and Critic including fixed empirical normalization state. No optimizer calls and no loading any old native state.'))
    results={}
    with torch.jit.optimized_execution(False),backend_scope(COLLISION),multiclip_w32(OBJECTIVE,obj,factory()) as tasks:
        for key,rt in tasks.items():
            cfg=pilot_settings(OBJECTIVE)[1]
            learner=BackendSupportPenPPO(rt,cfg,'RESIDUAL');learner.actor.load_state_dict(data['modules']['actor']);learner.critic.load_state_dict(data['modules']['critic'])
            learner.normalizers_ready=True;torch.manual_seed(7411)
            m=learner.collect(64);health=learner.consistency()
            write_json(out/(key+'_collector.json'),health)
            if not all(health['checks'].values()):raise ValueError('Collector failed: '+key)
            phase=m['phase_used'];geometry=(m['contact_geometry_max']<=rt.objective_constants['geometry_limits']).all(-1)
            valid=geometry & ~m['engine_overflow'] & m['no_applied_force'] & (m['contact_errors']==0)&(m['contact_nonhand_max_sum'][...,0]<=.1)
            results[key]=dict(phase_min=int(phase.min()),phase_max=int(phase.max()),phase_hist=torch.bincount(phase.flatten()).cpu().tolist(),
                valid=int(valid.sum()),rows=phase.numel(),geom_failed=int((~geometry).sum()),task_terminated=int(m['terminated'].sum()),
                truncated=int(m['truncated'].sum()),updates=learner.optimizer_steps,integrations=rt.owned_native_integrations)
            assert learner.optimizer_steps==0 and rt.owned_native_integrations==16384
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),results=results))
    print({k:{a:b for a,b in v.items() if a!='phase_hist'} for k,v in results.items()})
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
