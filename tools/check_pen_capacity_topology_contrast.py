"""Sensitivity of the observed capacity contrast to changing contact multiplicity."""
from collections import Counter
from pathlib import Path
import sys,json
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import verify_file_pin,file_pin
from supermimic_research.strict_json import write_json
from supermimic_research.mechanics.pen_capacity_trace import packet_capacity


def main(root,out):
    out.mkdir(exist_ok=False)
    master=json.loads((root/'report.json').read_bytes());result={}
    for role,r in master['roles'].items():
        m=mujoco.MjModel.from_binary_path(str(root/'native'/role/'model.mjb'))
        body=m.body('object/cube').id
        layout=dict(object_body=body,object_qposadr=m.jnt_qposadr[m.body_jntadr[body]])
        with np.load(root/'native'/role/'physical_context.npz') as f:friction=f['geom_friction']
        groups={}
        for row in r['rows']:groups.setdefault(row['phase'],{})[row['name']]=row
        evidence=[]
        for phase,cases in groups.items():
            for key in ['mixed']+[f'j{j}' for j in r['selected']]:
                records=[cases[key+s] for s in ('_minus','_plus')]
                if not all(a['valid'] and not a['terminal'] for a in records):continue
                packets=[];keys=[];counts=[];caps=[]
                for a in records:
                    with np.load(verify_file_pin(a['trace'])) as f:
                        z={k:f[k] for k in ('pre_qpos','pre_object_COM','raw_contact_tick_plus23')}
                    akeys=[]
                    for row in z['raw_contact_tick_plus23']:
                        t,g,h=map(int,row[:3]);on=m.geom_bodyid[[g,h]]==body
                        akeys.append((t,g,h) if row[5]<=0 and on.sum()==1 and (m.geom(h if on[0] else g).name or '').startswith('robot/') else None)
                    keys.append(akeys);counts.append(Counter(k for k in akeys if k is not None));packets.append(z)
                    with np.load(verify_file_pin(a['packet'])) as f:caps.append(f['capacity'])
                changed={k for k in set(counts[0])|set(counts[1]) if counts[0][k]!=counts[1][k]}
                delta=caps[1]-caps[0];ablated=[];peak=0.
                if changed:
                    for z,ids in zip(packets,keys):
                        keep=np.array([k not in changed for k in ids],dtype=bool)
                        removed=z['raw_contact_tick_plus23'][~keep]
                        if len(removed):peak=max(peak,float(removed[:,18].max()))
                        zz=dict(z,raw_contact_tick_plus23=z['raw_contact_tick_plus23'][keep])
                        ablated.append(packet_capacity(zz,m,layout,friction)['capacity'].mean(0))
                    common=ablated[1]-ablated[0]
                else:common=delta
                evidence.append(dict(kind=key,phase=phase,delta_L2=float(np.linalg.norm(delta)),common_L2=float(np.linalg.norm(common)),changed_keys=[list(k) for k in sorted(changed)],changed_peak_N=peak))
        summary={}
        for name in ('mixed','basis'):
            rows=[x for x in evidence if (x['kind']=='mixed')==(name=='mixed')]
            diff=[x for x in rows if x['changed_keys']];total=sum(x['delta_L2']**2 for x in rows)
            summary[name]=dict(pairs=len(rows),changed_pairs=len(diff),changed_fraction_squared_energy=sum(x['delta_L2']**2 for x in diff)/total,
                RMS_ratio_after_symmetric_group_removal=np.sqrt(sum(x['common_L2']**2 for x in rows)/total))
        result[role]=dict(rows=evidence,summary=summary)
    write_json(out/'report.json',dict(source=file_pin(root/'report.json'),roles=result,scope='At each tick remove from BOTH packets every hand/object geom pair with different contact multiplicities before re-evaluating the capacity. Post-hoc descriptor sensitivity, not an intervention in physical forces.'))
    print(result['TEST']['summary'])
if __name__=='__main__':main(*map(Path,sys.argv[1:]))

