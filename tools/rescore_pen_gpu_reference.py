"""Rescore the identical executed A1/A2 transitions against CPU and GPU geometry references."""
from pathlib import Path
import json,sys
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.mechanics.pen_capacity_trace import packet_capacity
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json
read=lambda pin:json.loads(verify_file_pin(pin).read_bytes())


def main(out):
    out.mkdir(exist_ok=False);root=ROOT/'runs';pilot=file_pin(root/'pen_paired_imagination_v1/report.json')
    cpupin=read(file_pin(root/'pen_capacity_targets_v1/report.json'))['refs']['clip_001']['series']
    gpupin=read(file_pin(root/'pen_gpu_catalog_v1/report.json'))['results']['clip_001']['series']
    refs={}
    for mode,pin in [('cpu',cpupin),('gpu',gpupin)]:
        with np.load(verify_file_pin(pin)) as f:refs[mode]=(f['capacity'][:-1],f['eligible'])
    write_json(out/'intent.json',dict(original=pilot,cpu=cpupin,gpu=gpupin,design='Post-hoc metric sensitivity only. Frozen original actors/decisions/native outputs. Compare true eight-tick mean capacity vs each source-frame target. Pair A1/A2 by training seed on the intersection of both repetitions valid horizons. Do not infer changed training outcome.'))
    results={}
    for entry in read(pilot)['evaluated']:
        if entry['key']=='parent':continue
        trials=[]
        for ev in read(entry['evaluation'])['cases']:
            c=read(ev['report']);m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['model'])))
            with np.load(verify_file_pin(c['physical_context'])) as f:mu=f['geom_friction'][0]
            rows=[]
            for i,recpin in enumerate(c['controls']):
                rc=read(recpin)
                if not rc['physical_audit']['valid']:break
                if not any(ref[1][i] for ref in refs.values()):continue
                with np.load(verify_file_pin(rc['trace'])) as z:
                    values=packet_capacity(z,m,c['model_layout'],mu)
                if values['maximum_containment_excess']>2e-5:raise ValueError('Projection bound violated')
                h=values['capacity'].mean(0)
                rows.append(dict(index=i,**{name:float(np.abs(h-ref[0][i]).mean()) for name,ref in refs.items()}))
            trials.append(dict(case=ev['report'],valid_controls=round(ev['valid_s']/.02),rows=rows))
        results[entry['key']]=trials
    summary={}
    for seed in (7411,7412,7413):
        groups=[results[f'{arm}_{seed}'] for arm in ('A1','A2')]
        cutoff=min(r['valid_controls'] for g in groups for r in g);delta={}
        for name,ref in refs.items():
            ids=set(np.flatnonzero(ref[1][:cutoff]).tolist())
            means=[[np.mean([a[name] for a in r['rows'] if a['index'] in ids]) for r in g] for g in groups] if ids else None
            delta[name]=dict(phases=len(ids),A2_minus_A1=float(np.mean(means[1])-np.mean(means[0])) if means else None)
        summary[seed]=dict(cutoff=cutoff,delta=delta)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),summary=summary,results=results));print(summary)
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
