"""Controlled inference-only action ablation in the two loaded fingers."""
import sys
import json
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import torch
import evaluate_pen_scaled as evaluate
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.strict_json import write_json
from supermimic_research.mechanics.pen_contact_groups import normal_group


def main(run,out):
    run=run.resolve();out.mkdir(exist_ok=False)
    actor=run/'actor_02781.pt'
    original=evaluate.run_case
    write_json(out/'intent.json',dict(actor=file_pin(actor),role='W1 saved frozen Actor inference intervention',controls=[206,222],
        schedule=['SHAM','HALF','HALF','SHAM','SHAM','HALF'],repeats=3,scale={'SHAM':1.,'HALF':.5},
        endpoints=['shaft error at 4.44s if observed','strict full task','valid horizon and censoring','full-horizon SO3 RMS when complete']))
    results=[]
    for idx,label in enumerate(['SHAM','HALF','HALF','SHAM','SHAM','HALF']):
        changed=[]
        def case(runtime,policy,decl,*args,**kwargs):
            m=runtime.env.sim.mj_model
            actuators=runtime.action._position_ctrl_ids.cpu().tolist()
            groups=[normal_group(m.joint(int(m.actuator_trnid[i,0])).name) for i in actuators]
            selected=[j for j,g in enumerate(groups) if g in (1,2)]
            if len(selected)!=8:raise ValueError('Expected 8 joint actuators for fingers 2 and 3')
            distribution=policy.distribution
            def intervention(obs,**options):
                d=distribution(obs,**options)
                phase=int(runtime.motion.time_steps[0])
                if 206<=phase<=222:
                    raw=d.mean.clone();raw[:,selected]*=(.5 if label=='HALF' else 1.)
                    changed.append(dict(phase=phase,raw_delta_norm=float((raw-d.mean).norm()),indices=selected))
                    return SimpleNamespace(mean=raw)
                return d
            with patch.object(policy,'distribution',intervention):
                return original(runtime,policy,decl,*args,**kwargs)
        folder=out/f'{idx}_{label}'
        with patch.object(evaluate,'run_case',case):
            evaluate.main(actor,folder,1)
        report=json.loads((folder/'report.json').read_bytes())
        c=json.loads(verify_file_pin(report['cases'][0]['report']).read_bytes())
        import numpy as np
        from supermimic_research.mechanics.pen_task_contracts import tracking_series
        with np.load(verify_file_pin(c['trace'])) as z: series=tracking_series(z,.0025)
        angles=np.rad2deg(series['shaft_angle_rad'])
        index=lambda t:round(t/.0025)-1
        results.append(dict(label=label,trial=idx,observed_controls=changed,case=report['cases'][0],
            shaft_before=float(angles[index(4.1)]) if len(angles)>index(4.1) else None,
            shaft_peak_time=float(angles[index(4.44)]) if len(angles)>index(4.44) else None,
            full_RMS_deg=float(np.rad2deg(c['summary']['metrics']['angle_RMS_rad'])) if c['stop_reason']=='REFERENCE_COMPLETION' else None))
        write_json(out/'progress.json',dict(completed=len(results),rows=results))
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=results))
if __name__=='__main__': main(Path(sys.argv[1]),Path(sys.argv[2]))
