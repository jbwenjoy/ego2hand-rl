"""Resolve the 4.10-4.50 s pen tracking divergence from saved contact substeps."""
import argparse
import json
import sys
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from matplotlib import pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import verify_file_pin,file_pin
from supermimic_research.mechanics.pen_contact_groups import normal_group


def arc(a,b):
    return np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a,b),axis=-1),np.sum(a*b,axis=-1)))


def analyse(case):
    z=np.load(verify_file_pin(case['trace']))
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(case['model'])))
    body=case['model_layout']['object_body']
    forces=[]
    for pin in case['controls'][205:225]:
        step=json.loads(verify_file_pin(pin).read_bytes())
        raw=np.load(verify_file_pin(step['trace']))['raw_contact_tick_plus23']
        fn=np.zeros((8,6))
        for row in raw:
            i,g,h=map(int,row[:3])
            on=model.geom_bodyid[[g,h]]==body
            if sum(on)!=1: continue
            group=normal_group(model.geom(h if on[0] else g).name or '')
            if group>=0: fn[i,group]+=max(row[18],0.)
        window=np.load(verify_file_pin(step['window']))
        np.testing.assert_allclose(fn.sum(0),window['Fn6_sum'],atol=1e-9,rtol=1e-9)
        forces.extend(fn)
    forces=np.array(forces)
    ticks=np.arange(1640,1801)
    t=ticks*.0025
    actual=Rotation.from_quat(z['post_quaternion_wxyz'][ticks-1][:,[1,2,3,0]])
    ref=Slerp(z['time_s'],Rotation.from_quat(z['quaternion_wxyz'][:,[1,2,3,0]]))(t)
    x=np.array([1.,0.,0.])
    a,b=actual.apply(x),ref.apply(x)
    shaft=arc(a,b)
    def displacement(i):
        return dict(time_s=float(t[i]),shaft_error_deg=float(shaft[i]),
                    actual_axis_travel_from_start_deg=float(arc(a[0],a[i])),ref_axis_travel_from_start_deg=float(arc(b[0],b[i])))
    return dict(source_trace=case['trace'],milestones=[displacement(i) for i in (0,64,136,160)],
        mean_digit_normal_N=forces.mean(0).tolist(),peak_digit_normal_N=forces.max(0).tolist(),
        active_tick_count=(forces>.1).sum(0).tolist(),all_ticks=160),t,forces,shaft


def main(evaluation,out):
    out.mkdir(parents=True,exist_ok=False)
    root=json.loads(evaluation.read_bytes())
    fig,(top,bottom)=plt.subplots(2,1,sharex=True,figsize=(9,6),layout='constrained')
    rows=[]
    for rep,entry in enumerate(root['cases']):
        case=json.loads(verify_file_pin(entry['report']).read_bytes())
        data,t,forces,shaft=analyse(case)
        rows.append(data)
        top.plot(t,shaft,label=f'Repeat {rep+1}')
        if rep==0:
            for i,col in enumerate(forces.T):
                if col.max()>.01: bottom.plot(t[1:],col,label=f'Finger {i+1}' if i<5 else 'Other hand')
    top.set_ylabel('Shaft error (deg)');top.legend()
    bottom.set(xlabel='Reference time (s)',ylabel='Object contact normal force (N)')
    bottom.legend();fig.suptitle('Pen axis divergence and per-finger contact | forces from first repeat')
    fig.savefig(out/'contacts.png',dpi=150);plt.close(fig)
    result=dict(evaluation=file_pin(evaluation),window='(4.10,4.50] s for forces; axis displacement relative to 4.10 s',rows=rows)
    (out/'report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(rows))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('evaluation',type=Path);p.add_argument('out',type=Path);args=p.parse_args();main(args.evaluation,args.out)
