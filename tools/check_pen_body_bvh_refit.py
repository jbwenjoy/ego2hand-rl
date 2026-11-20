"""Source-specific pen body BVH consistency test at measured poses; CPU copies only."""
from pathlib import Path
import sys
import json
import mujoco
import numpy as np
from audit_pen_geom_membership_gaps import active, members, file_pin, verify_file_pin, write_json


from supermimic_research.mechanics.pen_body_bvh import copy_refit_pen_bvh


def main(root,out):
    out.mkdir(exist_ok=False);report=json.loads((root/'report.json').read_bytes())
    write_json(out/'intent.json',dict(source=file_pin(root/'report.json'),scope='At each measured pre-step pose, compare original, repeated mj_setConst, refit only the three pen geom leaves and two internal pen BODY BVH nodes, and disable-midphase CPU controls. Readback of raw GPU group membership; no native integration.'))
    modes=['original','mj_setConst','pen_refit','no_midphase'];rows=[];models={}
    for role,meta in report['actors'].items():
        ms=[mujoco.MjModel.from_binary_path(str(verify_file_pin(meta['model']))) for _ in modes]
        body=meta['layout']['object_body'];adr=ms[0].body_bvhadr[body];count=ms[0].body_bvhnum[body]
        mujoco.mj_setConst(ms[1],mujoco.MjData(ms[1]))
        np.testing.assert_array_equal(ms[1].bvh_aabb,ms[0].bvh_aabb)
        ms[2],proof=copy_refit_pen_bvh(ms[0],body)
        np.testing.assert_array_equal(ms[2].bvh_aabb[:adr],ms[0].bvh_aabb[:adr])
        np.testing.assert_array_equal(ms[2].bvh_aabb[adr+count:],ms[0].bvh_aabb[adr+count:])
        ms[3].opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
        ds=[mujoco.MjData(m) for m in ms];models[role]=proof
        for src in [r for r in report['rows'] if r['actor']==role]:
            for br in src['branches']:
                with np.load(verify_file_pin(br['trace'])) as z:
                    poses=z['pre_qpos'];mp=z['post_mocap_pos'];mq=z['post_mocap_quat'];raw=z['raw_contact_tick_plus23']
                for i in range(8):
                    gpu=members(((r[1],r[2],r[5]) for r in raw[raw[:,0]==i]),ms[0],body)
                    gpu_set=active(gpu);record=dict(role=role,phase=src['phase'],branch=br['name'],tick=i); sets=[]
                    for mode,m,d in zip(modes,ms,ds):
                        d.qpos[:]=poses[i];d.mocap_pos[:]=mp[i];d.mocap_quat[:]=mq[i];mujoco.mj_forward(m,d)
                        g=members(((c.geom[0],c.geom[1],c.dist) for c in d.contact),m,body)
                        cpu=active(g);sets.append(cpu)
                        count_error=sum(sum(z<=0 for z in g.get(k,[]))!=sum(z<=0 for z in gpu.get(k,[])) for k in cpu|gpu_set)
                        record[mode]=[len(gpu_set-cpu),len(cpu-gpu_set),count_error]
                    record['refit_vs_no_midphase']=len(sets[2]^sets[3])
                    record['setconst_vs_original']=len(sets[0]^sets[1])
                    rows.append(record)
    def summarize(rs):
        return {**{m:np.sum([r[m] for r in rs],axis=0).tolist() for m in modes},
                'setconst_vs_original':sum(r['setconst_vs_original'] for r in rs),
                'refit_vs_no_midphase':sum(r['refit_vs_no_midphase'] for r in rs)}
    summary=summarize(rows)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),ticks=len(rows),models=models,columns=['warp_only','cpu_only','different_multiplicity'],
        summary=summary,roles={role:summarize([r for r in rows if r['role']==role]) for role in models},rows=rows))
    print('summary',summary);print('original/refit root',models['TEST_SUPPORT']['old'][0],models['TEST_SUPPORT']['refit'][0])
if __name__=='__main__':main(*map(lambda x:Path(x).resolve(),sys.argv[1:]))
