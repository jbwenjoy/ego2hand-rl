"""Closed-loop local deployment of the frozen M3 probe on a separate residual Actor."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json,sys
import numpy as np
import torch
import evaluate_pen_scaled as e
from check_pen_native_load_suggestions import suggest,LoadForecast
from supermimic_research.artifacts import verify_file_pin,file_pin
from supermimic_research.strict_json import write_json
from supermimic_research.mechanics.pen_task_contracts import tracking_series


def run(out):
    out.mkdir(exist_ok=False);root=e.ROOT/'runs'
    read=lambda pin:json.loads(verify_file_pin(pin).read_bytes())
    checkfile=root/'pen_load_native_suggestions_v1/report.json';study=json.loads(checkfile.read_bytes());verified=read(study['arms']['residual'])
    # Same-source test is a go/no-go diagnostic only. Do not adjust scale/window or pick a best trial.
    successes=0
    for phase in [160,180,192,200,206,210,214,218,222,226]:
        cs=[c for c in verified['cases'] if c['phase']==phase]
        if len(cs)!=6 or not all(len(c['steps'])==3 and all(s['valid'] and not s['terminal'] for s in c['steps']) for c in cs):continue
        v={k:[np.mean([s['loss'] for s in c['steps']]) for c in cs if c['kind']==k] for k in ('BASE','ASCEND','DESCEND')}
        if max(v['DESCEND'])+np.ptp(v['BASE'])<min(min(v['BASE']),min(v['ASCEND'])):successes+=1
    if successes<8:raise ValueError(f'Only {successes}/10 sources passed fixed noise-aware direction gate')
    intent=read(study['intent']);model=intent['model'];st=torch.load(verify_file_pin(model),map_location='cpu',weights_only=True)
    net=LoadForecast(torch.zeros(2,len(st['center'])),torch.zeros(2,len(st['out_center'])));net.load_state_dict(st);net=net.double().eval().requires_grad_(False)
    target=root/'pen_batch_updates_v1/W2048_U32/final_actor.pt'
    schedule=['BASE','DESCEND','ASCEND','ASCEND','BASE','DESCEND','DESCEND','ASCEND','BASE']
    write_json(out/'intent.json',dict(gate_source=file_pin(checkfile),gate=successes,actor=file_pin(target),model=model,schedule=schedule,
        active_phases=[206,222],scope='New suggestion application on a distinct previously inspected Actor. Always recompute original actor output, three-step model gradient and feasible axes at each invoked phase. Zero update. No native lookahead.'))
    original=e.run_case;results=[]
    for idx,kind in enumerate(schedule):
        calls=[]
        def case(rt,actor,decl,*args,**kwargs):
            base=actor.distribution
            m=rt.env.sim.mj_model;body=rt.features.body
            geom=dict(qids=rt.features.qids.cpu().tolist(),dofs=rt.features.dofs.cpu().tolist(),aids=rt.action._position_ctrl_ids.cpu().tolist(),model_layout=dict(object_qposadr=rt.features.qa,object_dofadr=int(m.jnt_dofadr[m.body_jntadr[body]])))
            # Use the native live model constants (same as save_physics).
            dm=rt.env.sim.wp_model
            inertia=dm.body_inertia.numpy()[0,body]
            from scipy.spatial.transform import Rotation
            Q=Rotation.from_quat(dm.body_iquat.numpy()[0,body,[1,2,3,0]]).as_matrix()
            ph=dict(mass=float(dm.body_mass.numpy()[0,body]),inertia_BODY=Q@np.diag(inertia)@Q.T,gravity=dm.opt.gravity.numpy()[0])
            def guided(obs,**opts):
                d=base(obs,**opts);phase=int(rt.motion.time_steps[0])
                if phase<206 or phase>222:return d
                prop=suggest(rt,net,ph,geom,d.mean)
                calls.append(prop)
                if 'skip' in prop:return d
                return SimpleNamespace(mean=torch.as_tensor(prop['proposals'][kind]['raw'],device=d.mean.device,dtype=d.mean.dtype)[None])
            with patch.object(actor,'distribution',guided):
                return original(rt,actor,decl,*args,**kwargs)
        folder=out/f'{idx}_{kind}'
        with patch.object(e,'run_case',case):e.main(target,folder,1)
        ev=json.loads((folder/'report.json').read_bytes());row=ev['cases'][0];c=read(row['report'])
        with np.load(verify_file_pin(c['trace'])) as z:s=tracking_series(z,.0025)
        shaft=np.rad2deg(s['shaft_angle_rad'])
        dst=folder/'interventions.json';write_json(dst,calls)
        results.append(dict(kind=kind,trial=idx,calls=file_pin(dst),case=row,shaft_at_4p10=float(shaft[1639]) if len(shaft)>=1640 else None,
            shaft_at_4p44=float(shaft[1775]) if len(shaft)>=1776 else None,full_so3_rms_deg=float(np.rad2deg(c['summary']['metrics']['angle_RMS_rad'])) if c['stop_reason']=='REFERENCE_COMPLETION' else None))
        write_json(out/'progress.json',dict(completed=len(results),rows=results))
        print(kind,row['valid_s'],results[-1]['shaft_at_4p44'],results[-1]['full_so3_rms_deg'],flush=True)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=results))
if __name__=='__main__':run(Path(sys.argv[1]).resolve())
