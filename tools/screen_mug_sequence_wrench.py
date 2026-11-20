"""Reference-frame contact feasibility for static and articulated HOT3D mug references."""
import json,sys
from pathlib import Path
import mujoco
import numpy as np
from track_mug_tip_sequence import ROOT
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.mean_reference_wrench import mean_reference_wrench
from supermimic_research.mechanics.multiclip_wrench_screen import contact_rays,min_normal
from supermimic_research.strict_json import write_json


def main(source,out):
    out.mkdir(exist_ok=False)
    experiment=json.loads((source/'report.json').read_bytes());intent=json.loads(verify_file_pin(experiment['intent']).read_bytes())
    binding=json.loads(verify_file_pin(intent['binding']).read_bytes());ref=json.loads(verify_file_pin(intent['reference']).read_bytes())
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(ref['model'])))
    with np.load(verify_file_pin(ref['output'])) as f: z=dict(f)
    with np.load(verify_file_pin(experiment['trajectory'])) as f:q=f['q20']
    body=binding['mug_body_id'];qa=binding['mug_free_qpos'];hand={model.geom(n).id for n in binding['hand_collision_names']}
    from scipy.spatial.transform import Rotation
    axes=Rotation.from_quat(model.body_iquat[body,[1,2,3,0]]).as_matrix()
    inertia=axes@np.diag(model.body_inertia[body])@axes.T
    gravity=model.opt.gravity.copy();mass=model.body_mass[body];weight=np.r_[-mass*gravity,np.zeros(3)]
    prior=mean_reference_wrench(z['COM_world_m'],z['quaternion_wxyz'],mass,inertia,gravity)
    rows=[]
    for name,angles in [('constant',z['grasp_q20_reference']),('retarget',q)]:
        d=mujoco.MjData(model)
        for i,frame in enumerate(angles):
            d.qpos[:20]=frame;d.qpos[qa:qa+7]=np.r_[z['T_world_from_BODY'][i,:3,3],z['quaternion_wxyz'][i]]
            mujoco.mj_forward(model,d)
            np.testing.assert_allclose(d.xipos[body],z['COM_world_m'][i],atol=1e-8)
            rays=[];geoms=set();penetration=0.
            for c in d.contact:
                g,h=map(int,c.geom);on=model.geom_bodyid[[g,h]]==body
                if on.sum()!=1 or c.dist>0:continue
                other=h if on[0] else g
                if other not in hand:continue
                geoms.add(model.geom(other).name);penetration=max(penetration,-c.dist)
                rays.extend(contact_rays(c,1 if on[1] else -1,d.xipos[body]))
            result={}
            for kind,wrench in [('gravity',weight),('nominal',prior['wrench_world'][i] if i<len(q)-1 and prior['valid_interval'][i] else None)]:
                if wrench is None:result[kind]='undefined';continue
                try:load=min_normal(rays,wrench);result[kind]='infeasible' if load is None else float(load)
                except (RuntimeError,ValueError):result[kind]='numerically_unresolved'
            rows.append(dict(arm=name,frame=i,hand_geoms=sorted(geoms),max_hand_pen_mm=1000*penetration,**result))
    report={}
    for arm in ('constant','retarget'):
        group=[r for r in rows if r['arm']==arm]
        report[arm]=dict(gravity_solutions=sum(isinstance(r['gravity'],float) for r in group),
            nominal_solutions=sum(isinstance(r['nominal'],float) for r in group),nominal_defined=sum(r['nominal']!='undefined' for r in group),
            nominal_within_pad_band=sum(isinstance(r['nominal'],float) and r['max_hand_pen_mm']<=.1 for r in group),
            max_hand_pen_mm=max(r['max_hand_pen_mm'] for r in group),frames_without_hand_contacts=sum(not r['hand_geoms'] for r in group),
            unresolved=sum('numerically_unresolved' in (r['gravity'],r['nominal']) for r in group))
    write_json(out/'report.json',dict(parent=file_pin(source/'report.json'),model=ref['model'],model_cone_type=int(model.opt.cone),rows=rows,summary=report,
        scope='Pyramidal friction rays at nonpositive-gap hand/cup contact points with unlimited normal load. Nominal simulator mass/gravity. A geometric support screen, not measured ego contact or inverse dynamics certification.'))
    print(report)
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
