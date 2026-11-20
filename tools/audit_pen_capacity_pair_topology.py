"""Audit paired 72D labels and distinguish geometry-mode changes from smooth responses."""
from collections import Counter
from pathlib import Path
import json,sys
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from check_pen_native_load_suggestions import native_state55
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json


def run(root,out):
    out.mkdir(exist_ok=False)
    report=json.loads((root/'report.json').read_bytes());rows=[];summary={}
    for role,arm in report['roles'].items():
        folder=root/'native'/role
        m=mujoco.MjModel.from_binary_path(str(folder/'model.mjb'))
        body=m.body('object/cube').id;qa=m.jnt_qposadr[m.body_jntadr[body]];da=m.jnt_dofadr[m.body_jntadr[body]]
        joints=[m.joint(f'robot/right_finger{i}_joint{j}').id for i in range(1,6) for j in range(1,5)]
        qids=m.jnt_qposadr[joints];dofs=m.jnt_dofadr[joints]
        # Actuator order must identify the same hand-joint order as native PD features.
        np.testing.assert_array_equal(m.actuator_trnid[:,0],joints)
        entries={}
        for row in arm['rows']:
            with np.load(verify_file_pin(row['packet'])) as f:x,y,pd=f['x'],f['capacity'],f['pd']
            with np.load(verify_file_pin(row['trace'])) as f:
                ticks=f['raw_contact_tick_plus23'];q=f['pre_qpos'][0];v=f['pre_qvel'][0]
                np.testing.assert_array_equal(native_state55(q,v,qids,dofs,qa,da),x[:55])
                np.testing.assert_array_equal(x[55:81],f['initial_warmstart'])
                np.testing.assert_array_equal(f['post_ctrl'],np.tile(pd,(8,1)))
                ids=[[] for _ in range(8)]
                for rec in ticks:
                    tick,g,h=map(int,rec[:3]);on=m.geom_bodyid[[g,h]]==body
                    if rec[5]<=0 and on.sum()==1:
                        pen,hand=(g,h) if on[0] else (h,g)
                        if (m.geom(hand).name or '').startswith('robot/'):ids[tick].append((pen,hand))
                top=[tuple(sorted(items)) for items in ids]
            entries.setdefault(row['phase'],{})[row['name']]=(x[:81],y,pd,top,row['valid'] and not row['terminal'])
        for phase,d in entries.items():
            src,y0,_,_,_=d['sham']
            for e in d.values():np.testing.assert_array_equal(e[0],src)
            noise=np.linalg.norm(y0-d['sham_repeat'][1])
            for name in ['mixed']+[f'j{j}' for j in arm['selected']]:
                _,ym,pm,tm,vm=d[name+'_minus'];_,yp,pp,tp,vp=d[name+'_plus'];dist=float(np.linalg.norm(pp-pm))
                response=float(np.linalg.norm(yp-ym))
                rows.append(dict(role=role,phase=phase,kind=name,pd_norm=dist,delta_capacity_l2=response,
                    differs_ticks=sum(a!=b for a,b in zip(tm,tp)),sham_difference_l2=float(noise),above_sham=response>noise,valid=bool(vm and vp)))
        for part in ('basis','mixed'):
            chosen=[r for r in rows if r['role']==role and (r['kind']=='mixed')==(part=='mixed') and r['valid'] and r['pd_norm']>1e-7]
            for subset,selected in [('same',[r for r in chosen if r['differs_ticks']==0]),('changed',[r for r in chosen if r['differs_ticks']>0])]:
                summary[f'{role}_{part}_{subset}']=dict(n=len(selected),median_response_l2=float(np.median([r['delta_capacity_l2'] for r in selected])) if selected else None,
                    energy=sum(r['delta_capacity_l2']**2 for r in selected),above_sham=sum(r['above_sham'] for r in selected))
    write_json(out/'report.json',dict(source=file_pin(root/'report.json'),scope='On every native tick compare multiset of nonpositive-gap hand/object geom ID pairs. Stable IDs may still move, rotate, change active mesh face or friction direction. Check all row source and delivered PD identities.',summary=summary,rows=rows))
    print(summary)
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
