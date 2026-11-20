"""One-bit CPU midphase ablation and cutoff sensitivity on the same recorded poses."""
from collections import Counter
import json
import sys
from pathlib import Path
import mujoco
import numpy as np
from audit_pen_geom_membership_gaps import active, members, file_pin, verify_file_pin, write_json


def read(pin):return json.loads(verify_file_pin(pin).read_bytes())


def run(audit,out):
    out.mkdir(exist_ok=False)
    original=read(file_pin(audit/'report.json'))
    source=read(read(original['intent'])['source'])
    write_json(out/'intent.json',dict(audit=file_pin(audit/'report.json'),
        scope='CPU post-hoc offline exact-pose intervention. Compare original MJB default flags and only disable MIDPHASE on a separate in-memory CPU copy. All recorded branches and ticks, no native integration. Cutoff sensitivity on original disagreement instances only.'))
    lookup={(r['role'],r['phase'],r['branch'],r['tick'],r['hand'],r['obj']):r for r in original['records']}
    cases=[];queries=[];comparison={}
    for role,meta in source['actors'].items():
        models=[mujoco.MjModel.from_binary_path(str(verify_file_pin(meta['model']))) for _ in range(2)]
        flag=int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
        if models[0].opt.disableflags&flag:raise ValueError('Midphase already disabled')
        models[1].opt.disableflags |= flag
        data=[mujoco.MjData(m) for m in models];body=meta['layout']['object_body']
        m=models[0]
        for row in [r for r in source['rows'] if r['actor']==role]:
            for branch in row['branches']:
                with np.load(verify_file_pin(branch['trace'])) as f:
                    pre=f['pre_qpos'];mp=f['post_mocap_pos'];mq=f['post_mocap_quat'];raw=f['raw_contact_tick_plus23']
                for tick in range(8):
                    native=members(((r[1],r[2],r[5]) for r in raw[raw[:,0]==tick]),m,body)
                    gpu=active(native)
                    cpu=[]
                    for md,d in zip(models,data):
                        d.qpos[:]=pre[tick];d.mocap_pos[:]=mp[tick];d.mocap_quat[:]=mq[tick]
                        mujoco.mj_forward(md,d)
                        cpu.append(members(((c.geom[0],c.geom[1],c.dist) for c in d.contact),md,body))
                    old,new=map(active,cpu)
                    for h,o in old^gpu:
                        key=(role,row['phase'],branch['name'],tick,m.geom(h).name,m.geom(o).name)
                        vals=[float(mujoco.mj_geomDistance(m,data[0],h,o,cutoff,None))*1e6 for cutoff in [0.,1e-6,.03]]
                        np.testing.assert_allclose(vals[-1],lookup[key]['query_um'],atol=1e-8)
                        queries.append(dict(role=role,phase=row['phase'],branch=branch['name'],tick=tick,kind=lookup[key]['kind'],
                            hand=key[-2],obj=key[-1],gaps_um=vals,reversed_default_um=float(mujoco.mj_geomDistance(m,data[0],o,h,.03,None))*1e6))
                    def multiplicity(x):
                        return {k:sum(gap<=0 for gap in gaps) for k,gaps in x.items() if min(gaps)<=0}
                    gm=multiplicity(native)
                    def count_diff(x):
                        cm=multiplicity(x)
                        return sum(cm.get(k,0)!=gm.get(k,0) for k in gm.keys()|cm.keys())
                    cases.append(dict(role=role,phase=row['phase'],branch=branch['name'],tick=tick,
                        old_warp_only=len(gpu-old),new_warp_only=len(gpu-new),old_cpu_only=len(old-gpu),new_cpu_only=len(new-gpu),
                        old_count_diff=count_diff(cpu[0]),new_count_diff=count_diff(cpu[1]),
                        recovered=len((gpu-old)&new),new_extra=len(new-(old|gpu)),
                        native_groups=len(gpu),old_cpu_groups=len(old),new_cpu_groups=len(new)))
    def group(rs):
        return {k:sum(row[k] for row in rs) for k in cases[0] if k not in ('role','phase','branch','tick')}
    for role in source['actors']:comparison[role]=group([r for r in cases if r['role']==role])
    if len(queries)!=len(lookup):raise ValueError('Mismatch audit lost records')
    qs={}
    for kind in ('warp_only','cpu_only'):
        rs=[r for r in queries if r['kind']==kind]
        qs[kind]=dict(n=len(rs),negative_per_cutoff=np.sum(np.asarray([r['gaps_um'] for r in rs])<0,axis=0).tolist(),
            max_order_swap_difference_um=max(abs(r['gaps_um'][-1]-r['reversed_default_um']) for r in rs))
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),ticks=len(cases),pooled=group(cases),roles=comparison,
        cutoffs_m=[0,1e-6,.03],query_summary=qs,cases=cases,queries=queries))
    print('pooled',group(cases),'by_role',comparison,'cutoffs',qs)

if __name__=='__main__':run(*map(lambda s:Path(s).resolve(),sys.argv[1:]))
