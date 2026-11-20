"""Recompute the registered window capacity and compare using only observed common controls."""
import json,sys
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from check_pen_capacity_geometry_oracle import geometry,balanced_directions
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.pen_capacity_trace import packet_capacity
from supermimic_research.strict_json import write_json

def main(run,out):
    out.mkdir(exist_ok=False);source=file_pin(run/'report.json');report=json.loads(verify_file_pin(source).read_bytes())
    intent=json.loads(verify_file_pin(report['intent']).read_bytes())
    with np.load(verify_file_pin(intent['target'])) as f:target=f['capacity'];eligible=f['eligible']
    phases=[p for p in range(206,223) if eligible[p-1]];dirs=balanced_directions()
    rows=[]
    for trial,item in enumerate(report['rows']):
        case_path=verify_file_pin(item['case']);folder=case_path.parent;case=json.loads(case_path.read_bytes())
        m=mujoco.MjModel.from_binary_path(str(folder/'model.mjb'));body=m.body('object/cube').id;qa=m.jnt_qposadr[m.body_jntadr[body]]
        layout=dict(object_body=body,object_qposadr=qa)
        with np.load(folder/'physical_context.npz') as f:fr=f['geom_friction']
        measured={}
        for r in case['records']:
            p=r['phase']
            if p not in phases:continue
            key=r['selected']
            if r['scores'] and key!=min(r['scores'],key=r['scores'].get):raise ValueError('Selected candidate differs')
            with np.load(folder/f'raw_{p}.npz') as f:
                z=dict(f)
            cap=packet_capacity(z,m,layout,fr)
            if cap['maximum_containment_excess']>2e-5:raise ValueError('Invalid measured cone')
            loss=float(np.abs(cap['capacity'].mean(0)-target[p-1]).mean());np.testing.assert_allclose(loss,r['mismatch'],atol=1e-12,rtol=0)
            if item['mode']=='CPU':
                d=mujoco.MjData(m);mujoco.mj_setState(m,d,z['initial_native_state'],mujoco.mjtState.mjSTATE_INTEGRATION)
                d.ctrl[:]=z['post_ctrl'][0]
                tmp=mujoco.MjData(m);tmp.mocap_pos[:]=d.mocap_pos;tmp.mocap_quat[:]=d.mocap_quat
                hs=[]
                for _ in range(8):
                    hs.append(geometry(m,tmp,d.qpos,body,dirs));mujoco.mj_step(m,d)
                pred=np.mean(np.abs(np.mean(hs,axis=0)-target[p-1]))
                # CPU chooser used float32 conversion of its computed PD; native readback may differ by ~1 ulp.
                measured.setdefault('_pd_quantization_score_diff',[]).append(float(pred-r['scores'][key]))
            if r['valid']:measured[p]=loss
        cpu=measured.pop('_pd_quantization_score_diff',[])
        with np.load(verify_file_pin(case['trace'])) as z:
            j=round(4.44/.0025)-1
            shaft=float(np.rad2deg(np.arccos(np.clip((Rotation.from_quat(z['quaternion_wxyz'][222,[1,2,3,0]]).inv()*Rotation.from_quat(z['post_quaternion_wxyz'][j,[1,2,3,0]])).as_matrix()[0,0],-1,1)))) if len(z['post_qpos'])>j and item['valid_s']>=4.44 else None
        rows.append(dict(trial=trial,mode=item['mode'],stop=item['stop'],valid_s=item['valid_s'],strict_success=item['success'],
                         errors=measured,full_angle_deg=float(np.rad2deg(case['summary']['metrics']['angle_RMS_rad'])) if item['stop']=='REFERENCE_COMPLETION' else None,
                         shaft_at_4p44=shaft,cpu_replayed_score_max_abs=max(abs(v) for v in cpu) if cpu else None))
    common=sorted(set.intersection(*(set(r['errors']) for r in rows)));summary={}
    for mode in dict.fromkeys(intent['schedule']):
        rs=[r for r in rows if r['mode']==mode]
        summary[mode]=dict(strict=sum(r['strict_success'] for r in rs),valid_s=[r['valid_s'] for r in rs],full_deg=[r['full_angle_deg'] for r in rs],
            shaft_deg=[r['shaft_at_4p44'] for r in rs],common_MAE=[np.mean([r['errors'][p] for p in common]) if common else None for r in rs])
    write_json(out/'report.json',dict(source=source,intervention_phases=phases,common=common,summary=summary,rows=rows))
    print(summary)
if __name__=='__main__':main(*map(Path,sys.argv[1:]))

