"""Choose from declared proposals by CPU physics before reading their native GPU outcomes."""
import json,sys
from pathlib import Path
import numpy as np
import mujoco
from check_pen_capacity_geometry_oracle import geometry
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.pen_wrench_capacity import balanced_directions
from supermimic_research.strict_json import write_json
read=lambda pin:json.loads(verify_file_pin(pin).read_bytes())


def main(root,out):
    out.mkdir(exist_ok=False);report=read(file_pin(root/'report.json'));intent=read(report['intent'])
    dirs=balanced_directions()
    with np.load(verify_file_pin(intent['targets'])) as f:targets=f['capacity']
    write_json(out/'intent.json',dict(source=file_pin(root/'report.json'),design='Offline exploratory CPU exact-dynamics reference, native chooser restricted to BASE or either signed suggestion from both models. Only slot-zero pre-state and all registered candidate PD vectors supplied before selecting the minimum CPU-predicted capacity mismatch. No future native branch data used by selector. Strict success requires both selected repeats better than both BASE repeats by more than BASE spread.'))
    result={}
    for arm,entry in report['results'].items():
        ar=read(entry['report']);proposals=read(ar['proposals'])
        from_file=Path(entry['report']['path']).parent
        model=mujoco.MjModel.from_binary_path(str(from_file/'model.mjb'));body=model.body('object/cube').id
        rows=[]
        for src in proposals['entries']:
            p=src['phase'];replays=[r for r in ar['observations'] if r['phase']==p]
            with np.load(verify_file_pin(next(r for r in replays if r['slot']==0)['trace'])) as f:state=f['initial_native_state'].copy()
            scores={}
            for name,act in src['actions'].items():
                d=mujoco.MjData(model);mujoco.mj_setState(model,d,state,mujoco.mjtState.mjSTATE_INTEGRATION)
                # Use the float32 precision of the transmitted GPU joint target.
                d.ctrl[:]=np.array(act['pd'],dtype=np.float32);assert not d.qfrc_applied.any() and not d.xfrc_applied.any()
                geom=mujoco.MjData(model);geom.mocap_pos[:]=d.mocap_pos;geom.mocap_quat[:]=d.mocap_quat
                cap=[]
                for _ in range(8):
                    cap.append(geometry(model,geom,d.qpos,body,dirs));mujoco.mj_step(model,d)
                scores[name]=float(np.mean(np.abs(np.mean(cap,0)-targets[p-1])))
            pick=min(scores,key=scores.get)
            by_action={key:np.array([r['mismatch'] for r in replays if r['name']==key]) for key in scores}
            valid=all(r['valid'] and not r['terminal'] for r in replays) and all(v.shape==(2,) for v in by_action.values())
            baseline=by_action['BASE'];gain=float(baseline.mean()-by_action[pick].mean())
            rows.append(dict(phase=p,chosen=pick,cpu_scores=scores,native_gain=gain,admitted=valid,
                improves=bool(valid and by_action[pick].max()+np.ptp(baseline)<baseline.min()),
                worsens=bool(valid and by_action[pick].min()-np.ptp(baseline)>baseline.max())))
        clean=[s for s in rows if s['admitted']]
        result[arm]=dict(rows=rows,complete=len(clean),improves=sum(s['improves'] for s in clean),worsens=sum(s['worsens'] for s in clean),base_choices=sum(s['chosen']=='BASE' for s in clean),median_gain=float(np.median([s['native_gain'] for s in clean])))
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),result=result))
    print({a:{k:v for k,v in r.items() if k!='rows'} for a,r in result.items()})
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
