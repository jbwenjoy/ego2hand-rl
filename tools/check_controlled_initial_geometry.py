"""Native forward-only geometry audit of actual saved initial GPU states."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin


def main(args):
    if args.output.exists():raise ValueError('Fresh geometry audit output')
    r=json.loads(args.report.read_text(encoding='utf8'));cases=[]
    for task in ['hand2-reorient','hand2-pen-spin']:
        c=next(c for c in r['cases'] if c['task']==task and c['controller']=='ZERO')
        m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])))
        z=np.load(verify_file_pin(c['trace']));p=np.load(verify_file_pin(c['GPU_physical_context']))
        for key in p.files:
            value=p[key][0];dest=getattr(m,key)
            if dest.shape!=value.shape:raise ValueError('Explicit GPU parameter shape mismatch '+key)
            dest[:]=value
        d=mujoco.MjData(m);mujoco.mj_setConst(m,d)
        mujoco.mj_setState(m,d,z['initial_native_state'],mujoco.mjtState.mjSTATE_INTEGRATION)
        q=d.qpos.copy();v=d.qvel.copy();mujoco.mj_forward(m,d)
        if not np.array_equal(q,d.qpos) or not np.array_equal(v,d.qvel):raise ValueError('Forward changed actual initial state')
        contacts=[]
        for con in d.contact:
            contacts.append(dict(geoms=[m.geom(int(g)).name for g in con.geom],distance_m=float(con.dist),
                                 position_world_m=con.pos.tolist()))
        contacts.sort(key=lambda row:row['distance_m'])
        cases.append(dict(task=task,actual_initial_source=c['trace'],GPU_context=c['GPU_physical_context'],
            max_penetration_m=max([0.]+[-row['distance_m'] for row in contacts]),contacts=contacts,
            no_state_change=True,contact_geometry_only=True))
    result=dict(schema='controlled_actual_initial_native_geometry:1',source=file_pin(__file__),run=file_pin(args.report),
        cases=cases,neural_updates=0,native_integration_steps=0,
        scope='Fresh native mj_forward geometry at saved initial GPU qpos/qvel/mocap, with captured GPU physical tables. Not GPU solver force truth, physical rollout, or calibrated geometry certificate.')
    args.output.mkdir();(args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps([dict(task=c['task'],max_penetration_m=c['max_penetration_m'],deepest=c['contacts'][:1]) for c in cases]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
