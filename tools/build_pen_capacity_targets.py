"""Finite geometry-only BODY support targets and measured-contact containment checks."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.mechanics.pen_wrench_capacity import balanced_directions,body_wrench_rays,directional_support
from supermimic_research.mechanics.pen_contact_groups import normal_group
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json

L=.10

def read(pin):return json.loads(verify_file_pin(pin).read_bytes())

def compute(con,sign,com,R,dirs):
    rays=body_wrench_rays(con,sign,com_world=com,world_from_body=R,length_scale_m=L)
    return directional_support(rays,dirs)

def targets(root,out,dirs):
    report=json.loads((root/'report.json').read_bytes())
    screen_intent=read(report['intent']);c=read(screen_intent['case'])
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(screen_intent.get('cpu_model',c['model']))))
    d=mujoco.MjData(model);body=c['model_layout']['object_body'];qa=c['model_layout']['object_qposadr']
    with np.load(verify_file_pin(c['trace'])) as z:
        d.mocap_pos[:]=z['initial_mocap_pos'][0];d.mocap_quat[:]=z['initial_mocap_quat'][0]
    refs={}
    for name,clip in report['results'].items():
        with np.load(verify_file_pin(clip['source'])) as f:q=f['reference_qpos']
        with np.load(verify_file_pin(clip['series'])) as f:
            com=f['COM'];quats=f['quat_wxyz'];me=f['metrics']
        rotation=Rotation.from_quat(quats[:,[1,2,3,0]])
        omega=np.linalg.norm((rotation[1:]*rotation[:-1].inv()).as_rotvec(),axis=1)/.02
        masks=[];values=[]
        for i in range(len(q)):
            d.qpos[c['qids']]=q[i];d.qpos[qa:qa+7]=np.r_[com[i],quats[i]]
            mujoco.mj_forward(model,d)
            contacts=[];signs=[];mask=0
            for contact in d.contact:
                g,h=map(int,contact.geom);on=model.geom_bodyid[[g,h]]==body
                if on.sum()!=1 or contact.dist>0:continue
                digit=normal_group(model.geom(h if on[0] else g).name or '')
                if digit<0:continue
                mask|=1<<digit;contacts.append(contact);signs.append(1 if on[1] else -1)
            values.append(compute(contacts,signs,d.xipos[body],rotation[i].as_matrix(),dirs));masks.append(mask)
        masks=np.asarray(masks);values=np.asarray(values)
        geometric=me[:,5].astype(bool)
        # Advisory phases of the *reference* only, with transitions omitted from positive supervision.
        switch=masks[:-1]!=masks[1:]
        eligible=geometric[:-1]&geometric[1:]&~switch&(masks[:-1]!=0)
        stage=np.where(switch,'switch',np.where(masks[:-1]==0,'no_contact',np.where(omega>=.5,'turn','slow')))
        dst=out/(name+'.npz');np.savez_compressed(dst,capacity=values,directions=dirs,mask=masks,eligible=eligible,stage=stage,omega_rad_s=omega)
        refs[name]=dict(source=clip['source'],series=file_pin(dst),intervals=len(stage),eligible=int(eligible.sum()),stages={k:int(((stage==k)&eligible).sum()) for k in ('turn','slow')},transition_intervals=int(switch.sum()),noncontact=int((masks[:-1]==0).sum()))
    return refs


def actual(run,clip,dirs):
    ev=json.loads((run/'report.json').read_bytes());total=0;loaded=0;max_violation=0.;max_gap=0.;errors=[]
    with np.load(verify_file_pin(clip['series'])) as f:elig=f['eligible'];ref=f['capacity']
    for entry in ev['cases']:
        c=read(entry['report']);m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['model'])));layout=c['model_layout'];qa=layout['object_qposadr'];body=layout['object_body']
        with np.load(verify_file_pin(c['physical_context'])) as f:friction=f['geom_friction'][0]
        assert m.npair==0 and int(m.opt.cone)==int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
        for i,pin in enumerate(c['controls']):
            rec=read(pin)
            if not rec['physical_audit']['valid']:continue
            with np.load(verify_file_pin(rec['trace'])) as f:
                # One consistently selected sample per control, pre-step geometry and same-step force.
                raw=f['raw_contact_tick_plus23'];rows=raw[raw[:,0]==0];com=f['pre_object_COM'][0];R=Rotation.from_quat(f['pre_qpos'][0,qa+np.array([4,5,6,3])]).as_matrix()
            contacts=[];sign=[];load=np.zeros(6);fn=0.
            for row in rows:
                g,h=map(int,row[1:3]);on=m.geom_bodyid[[g,h]]==body
                if on.sum()!=1 or normal_group(m.geom(h if on[0] else g).name or '')<0:continue
                if row[5]>0:
                    max_gap=max(max_gap,float(abs(row[18])));continue
                fg,fh=friction[[g,h]]
                mu=fg if m.geom_priority[g]>m.geom_priority[h] else fh if m.geom_priority[h]>m.geom_priority[g] else np.maximum(fg,fh)
                frame=row[9:18].reshape(3,3);s=1 if on[1] else -1
                contacts.append(SimpleNamespace(frame=frame,pos=row[6:9],dim=int(row[3]),friction=mu[[0,0,1,2,2]]));sign.append(s)
                F=s*frame.T@row[18:21];T=s*frame.T@row[21:24]+np.cross(row[6:9]-com,F)
                fn+=row[18];load+=np.r_[R.T@F,R.T@T/L]
            H=compute(contacts,sign,com,R,dirs);total+=1
            if fn>1e-8:
                violation=float(np.max(dirs@(load/fn)-H));max_violation=max(max_violation,violation);loaded+=1
            if elig[i]:errors.append(float(np.mean(np.abs(H-ref[i]))))
    return dict(samples=total,loaded=loaded,maximum_containment_excess=max_violation,positive_gap_skipped_max_load_N=max_gap,
                eligible_comparisons=len(errors),mean_directional_L1=float(np.mean(errors)) if errors else None)

def main(out):
    out.mkdir(exist_ok=False);root=ROOT/'runs';dirs=balanced_directions()
    write_json(out/'intent.json',dict(source=file_pin(root/'pen_catalog_screen_v2/report.json'),length_m=L,
        scope='Geometry-only proxy: unit shared normal budget, moment about moving COM in BODY. Low angular speed below 0.5rad/s, contact switch when nonpositive-gap hand group IDs change; unknown or invalid frames masked. Not force or actual handover ground truth. Reserve clips not policy-scored.'))
    refs=targets(root/'pen_catalog_screen_v2',out,dirs)
    native={k:actual(root/path/'eval_02781',refs['clip_001'],dirs) for k,path in [('residual','pen_scaled_train_v1'),('support','pen_scaled_support_matched_v1')]}
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),directions=len(dirs),refs=refs,native=native))
    print(refs['clip_001'],native)
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
