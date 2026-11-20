"""At measured poses replace only CPU contact groups whose membership disagrees with Warp."""
from collections import defaultdict
import json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from build_pen_capacity_targets import compute,balanced_directions
from supermimic_research.artifacts import verify_file_pin,file_pin
from supermimic_research.strict_json import write_json


def main(root,out,disable_midphase=False):
    out.mkdir(exist_ok=False);data=json.loads((root/'report.json').read_bytes());dirs=balanced_directions()
    write_json(out/'intent.json',dict(parent=file_pin(root/'report.json'),disable_cpu_midphase=disable_midphase,plan='At true source pose each substep, replace CPU rays with native rays for absent/extra hand-object geom groups, then for different row multiplicities. Use maximum over groups with one shared unit normal budget. Privileged data diagnose model errors, no action selection or simulation.'))
    rows=[]
    for role,meta in data['actors'].items():
        m=mujoco.MjModel.from_binary_path(str(verify_file_pin(meta['model'])));d=mujoco.MjData(m)
        if disable_midphase:m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
        qa=meta['layout']['object_qposadr'];body=meta['layout']['object_body']
        with np.load(verify_file_pin(meta['physical_context'])) as f:fr=f['geom_friction'][0]
        for r in [z for z in data['rows'] if z['actor']==role]:
            br={b['name']:b for b in r['branches']}
            if not all(b['valid'] and not b['terminal'] for b in br.values()) or np.linalg.norm(np.subtract(br['plus']['pd'],br['minus']['pd']))<1e-7:continue
            ends={}
            for side in ['minus','plus']:
                sums=[]
                with np.load(verify_file_pin(br[side]['trace'])) as z:
                    raw=z['raw_contact_tick_plus23'];qs=z['pre_qpos'];COM=z['pre_object_COM'];d.mocap_pos[:]=z['post_mocap_pos'][0];d.mocap_quat[:]=z['post_mocap_quat'][0]
                for i in range(8):
                    d.qpos[:]=qs[i];mujoco.mj_forward(m,d)
                    R=Rotation.from_quat(qs[i,qa+np.array([4,5,6,3])]).as_matrix();geom=[defaultdict(list),defaultdict(list)]
                    def add(store,g,h,contact):
                        on=m.geom_bodyid[[g,h]]==body
                        if on.sum()!=1 or contact.dist>0:return
                        other=h if on[0] else g
                        if (m.geom(other).name or '').startswith('robot/'):
                            store[tuple(sorted((g,h)))].append((contact,-1 if on[0] else 1))
                    for c in d.contact:add(geom[0],*map(int,c.geom),c)
                    for row in raw[raw[:,0]==i]:
                        g,h=map(int,row[1:3]);mu=fr[g] if m.geom_priority[g]>m.geom_priority[h] else fr[h] if m.geom_priority[h]>m.geom_priority[g] else np.maximum(fr[g],fr[h])
                        add(geom[1],g,h,SimpleNamespace(dist=row[5],dim=int(row[3]),frame=row[9:18],pos=row[6:9],friction=mu[[0,0,1,2,2]]))
                    H=np.zeros((4,len(dirs)))
                    for k in geom[0].keys()|geom[1].keys():
                        supports=[]
                        for j,groups in enumerate(geom):
                            group=groups.get(k,[])
                            supports.append(compute([v[0] for v in group],[v[1] for v in group],COM[i] if j else d.xipos[body],R if j else d.xmat[body].reshape(3,3),dirs))
                        n0=len(geom[0].get(k,[]));n1=len(geom[1].get(k,[]));cpu,native=supports
                        H=np.maximum(H,[cpu,native if bool(n0)!=bool(n1) else cpu,native if n0!=n1 else cpu,native])
                    sums.append(H)
                mean=np.mean(sums,axis=0);np.testing.assert_allclose(mean[-1],br[side]['capacity'],atol=2e-7,rtol=0);ends[side]=mean
            delta=ends['plus']-ends['minus'];err=np.linalg.norm(delta[:-1]-delta[-1],axis=1)
            rows.append(dict(role=role,phase=r['phase'],risk=r['risk_selected'],uniform=r['uniform_selected'],magnitude=np.linalg.norm(delta[-1]),errors=err.tolist()))
    summary={}
    for role in data['actors']:
        rs=[r for r in rows if r['role']==role];signal=sum(r['magnitude']**2 for r in rs)
        summary[role]=np.sqrt(np.square([r['errors'] for r in rs]).sum(0)/signal).tolist()
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),columns=['original_CPU','match_group_existence','match_group_multiplicity'],summary=summary,rows=rows));print(summary)
if __name__=='__main__':main(*map(Path,sys.argv[1:3]),disable_midphase='--disable-midphase' in sys.argv[3:])
