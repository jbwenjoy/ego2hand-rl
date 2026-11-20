"""Audit native-vs-CPU contact geometry and friction-frame sensitivity on new switch bank."""
from pathlib import Path
from types import SimpleNamespace
import sys,json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from build_pen_capacity_targets import compute,balanced_directions
from supermimic_research.mechanics.pen_elliptic_capacity import elliptic_support
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json

def main(root,out):
    out.mkdir(exist_ok=False);src=file_pin(root/'report.json');report=json.loads(verify_file_pin(src).read_bytes())
    dirs=balanced_directions();rows=[];matched=0;count=0;max_positive=0.;min_margin=0.
    write_json(out/'intent.json',dict(source=src,scope='Diagnostic elliptic contact sets vs actual pyramids on same contacts; no change to native physics. At each recorded pre-step qpos use separate CPU contact geometry. Both mean shape and signed branch response compared per matched-source pair.'))
    for role,meta in report['actors'].items():
        m=mujoco.MjModel.from_binary_path(str(verify_file_pin(meta['model'])));geo=mujoco.MjData(m)
        body=meta['layout']['object_body'];qa=meta['layout']['object_qposadr'];hand={i for i in range(m.ngeom) if (m.geom(i).name or '').startswith('robot/')}
        with np.load(verify_file_pin(meta['physical_context'])) as f:mu=f['geom_friction'][0]
        for s in [r for r in report['rows'] if r['actor']==role]:
            branches={r['name']:r for r in s['branches']};results={}
            if not all(r['valid'] and not r['terminal'] for r in branches.values()):continue
            if np.linalg.norm(np.subtract(branches['plus']['pd'],branches['minus']['pd']))<1e-7:continue
            for side in ['plus','minus']:
                r=branches[side];values=[];membership=[]
                with np.load(verify_file_pin(r['trace'])) as f:
                    q=f['pre_qpos'];com=f['pre_object_COM'];raw=f['raw_contact_tick_plus23']
                    geo.mocap_pos[:]=f['post_mocap_pos'][0];geo.mocap_quat[:]=f['post_mocap_quat'][0]
                for t in range(8):
                    cs=[];sign=[];ids=[];rot=Rotation.from_quat(q[t,qa+np.array([4,5,6,3])]).as_matrix()
                    for z in raw[raw[:,0]==t]:
                        g,h=map(int,z[1:3]);on=m.geom_bodyid[[g,h]]==body
                        if on.sum()!=1 or (h if on[0] else g) not in hand or z[5]>0:continue
                        coef=mu[g] if m.geom_priority[g]>m.geom_priority[h] else mu[h] if m.geom_priority[h]>m.geom_priority[g] else np.maximum(mu[g],mu[h])
                        cs.append(SimpleNamespace(frame=z[9:18],pos=z[6:9],friction=coef[[0,0,1,2,2]],dim=int(z[3])));sign.append(-1 if on[0] else 1);ids.append(tuple(sorted((g,h))))
                    hp=compute(cs,sign,com[t],rot,dirs);he=elliptic_support(cs,sign,com[t],rot,.1,dirs)
                    geo.qpos[:]=q[t];mujoco.mj_forward(m,geo);qs=[];ss=[];cpu_ids=[]
                    for c in geo.contact:
                        g,h=map(int,c.geom);on=m.geom_bodyid[[g,h]]==body
                        if on.sum()!=1 or (h if on[0] else g) not in hand or c.dist>0:continue
                        qs.append(c);ss.append(-1 if on[0] else 1);cpu_ids.append(tuple(sorted((g,h))))
                    cp=compute(qs,ss,geo.xipos[body],geo.xmat[body].reshape(3,3),dirs)
                    ce=elliptic_support(qs,ss,geo.xipos[body],geo.xmat[body].reshape(3,3),.1,dirs)
                    min_margin=min(min_margin,float((he-hp).min()),float((ce-cp).min()))
                    values.append([hp,he,cp,ce]); membership.append(sorted(ids)==sorted(cpu_ids))
                mean=np.mean(values,axis=0);np.testing.assert_allclose(mean[0],r['capacity'],atol=2e-7,rtol=0)
                matched+=sum(membership);count+=8;results[side]=mean
            delta=results['plus']-results['minus']
            rows.append(dict(actor=role,phase=s['phase'],risk=s['risk_selected'],uniform=s['uniform_selected'],delta=delta.tolist()))
    summary={}
    for actor in report['actors']:
        for kind in ('risk','uniform','all'):
            data=np.array([r['delta'] for r in rows if r['actor']==actor and (kind=='all' or r[kind])])
            p,e,cp,ce=data.transpose(1,0,2)
            ratio=lambda x,y:float(np.linalg.norm(x)/np.linalg.norm(y)) if np.linalg.norm(y) else None
            summary[actor+'_'+kind]=dict(pairs=len(data),elliptic_to_pyramid_response=ratio(e,p),cpu_exact_pose_pyramid_error=ratio(cp-p,p),cpu_exact_pose_elliptic_error=ratio(ce-e,e),
                normalized_shape_change_disagreement=ratio(e-p,p))
    assert min_margin>=-1e-6
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),summary=summary,matching_tick_contact_multisets=matched,total_ticks=count,minimum_elliptic_minus_pyramidal=min_margin,rows=rows))
    print({k:v for k,v in summary.items() if k.endswith('_all')})
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
