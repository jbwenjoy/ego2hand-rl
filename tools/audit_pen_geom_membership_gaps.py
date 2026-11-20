"""CPU/guarded-Warp contact membership at the same saved substep poses."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.strict_json import write_json


def members(records, model, body):
    by_pair = defaultdict(list)
    for g, h, distance in records:
        g, h = int(g), int(h)
        og, oh = model.geom_bodyid[[g,h]] == body
        if og == oh:
            continue
        hand, obj = (h,g) if og else (g,h)
        if (model.geom(hand).name or '').startswith('robot/'):
            by_pair[hand,obj].append(float(distance))
    return by_pair


def active(groups):
    return {pair for pair, gaps in groups.items() if min(gaps) <= 0.}


def compare_model(m, native):
    # Compare at the precision and storage layout used by the actual GPU arrays.
    keys = ('geom_type geom_bodyid geom_contype geom_conaffinity geom_priority geom_condim '
            'geom_dataid geom_margin geom_gap geom_pos geom_quat geom_size geom_aabb geom_rbound '
            'mesh_vert mesh_face mesh_graph mesh_vertadr mesh_vertnum mesh_faceadr mesh_graphadr '
            'mesh_pos mesh_quat mesh_polyadr mesh_polynum mesh_polyvert mesh_polyvertnum '
            'body_pos body_quat jnt_pos jnt_axis exclude_signature geom_friction').split()
    result = {}
    for key in keys:
        gpu = native['arrays'][key]
        cpu = np.asarray(getattr(m,key))
        if cpu.size!=gpu.size:
            raise ValueError(f'Incompatible layout of {key}')
        gpu=gpu.reshape(cpu.shape)
        diff = np.max(np.abs(cpu.astype(float)-gpu), initial=0.)
        result[key] = {'max_abs':float(diff), 'equal_after_cast':bool(np.array_equal(gpu,cpu.astype(gpu.dtype)))}
    for key in ('ccd_iterations','ccd_tolerance','cone','disableflags','enableflags'):
        value=native['arrays'].get('opt.'+key,native['scalars'].get('opt.'+key))
        if value is None:raise ValueError(key)
        cpu=float(getattr(m.opt,key));gpu=np.asarray(value)
        result['opt.'+key]=dict(max_abs=float(np.max(abs(gpu.astype(float)-cpu))),
                                equal_after_cast=bool((gpu==np.asarray(cpu,dtype=gpu.dtype)).all()))
    return result


def q(values):
    return np.quantile(values,[0,.05,.25,.5,.75,.95,1]).tolist() if values else []


EDGES=[-1000.,-100.,-10.,-1.,0.,1.,10.,100.,1000.]


def stats(rows):
    gap=[row['query_um'] for row in rows]
    bins=Counter(int(np.searchsorted(EDGES,g,side='left')) for g in gap)
    return dict(records=len(rows),unique_sources=len({(r['role'],r['phase']) for r in rows}),
                unique_source_branches=len({(r['role'],r['phase'],r['branch']) for r in rows}),
                unique_ticks=len({(r['role'],r['phase'],r['branch'],r['tick']) for r in rows}),
                quantiles_um=q(gap),bins_um=[bins[i] for i in range(len(EDGES)+1)],
                positive=sum(g>0 for g in gap),nonpositive=sum(g<=0 for g in gap),
                abs_le_1um=sum(abs(g)<=1 for g in gap),abs_le_10um=sum(abs(g)<=10 for g in gap),
                native_min_gap_quantiles_um=q([r['warp_min_um'] for r in rows if r['warp_min_um'] is not None]),
                max_native_normal_N=max((r['warp_normal_N'] for r in rows), default=0.))


def main(root,out):
    out.mkdir(exist_ok=False)
    pin=file_pin(root/'report.json');report=json.loads(verify_file_pin(pin).read_bytes())
    write_json(out/'intent.json',dict(source=pin,scope='All four branches and all eight paired substeps at every source. Replay measured pre_qpos/postfixed mount in CPU; count complete hand/object geom membership for signed dist<=0. Also replay post_qpos only as timing diagnostic. CPU mj_geomDistance for every disagreeing pair. No new integration.'))
    results={};all_gaps=[];all_phases=[]
    for role,meta in report['actors'].items():
        m=mujoco.MjModel.from_binary_path(str(verify_file_pin(meta['model'])))
        pre=mujoco.MjData(m);post=mujoco.MjData(m);body=meta['layout']['object_body'];qa=meta['layout']['object_qposadr']
        addresses=report['actors'][role]['layout'];assert addresses['COM_BODY']==[0.,0.,0.]
        rows=[r for r in report['rows'] if r['actor']==role]
        source_models=[];n=0;com_err=0.;qtime_err=0.;mount_change=0.;pre_dis=0;post_dis=0;pre_events=0
        for row in rows:
            saved=torch.load(verify_file_pin(row['source']),map_location='cpu',weights_only=False)
            source_models.append(compare_model(m,saved['native']['model']))
            source=saved['native']['data']['arrays']
            counts={}
            for br in row['branches']:
                branch=br['name'];phase_count=Counter()
                with np.load(verify_file_pin(br['trace'])) as f:
                    rec=f['raw_contact_tick_plus23'];preq=f['pre_qpos'];postq=f['post_qpos']
                    poses=f['post_mocap_pos'];quats=f['post_mocap_quat'];com=f['pre_object_COM']
                    clocks=f['post_GPU_time'].reshape(-1)
                    np.testing.assert_array_equal(source['qpos'][0],preq[0]);np.testing.assert_array_equal(preq[1:],postq[:-1])
                    for a,ref in [(poses,source['mocap_pos'][0]),(quats,source['mocap_quat'][0])]:
                        mount_change=max(mount_change,float(np.max(abs(a-ref))))
                    qtime_err=max(qtime_err,float(np.max(abs(clocks-(float(np.asarray(source['time']).ravel()[0])+np.arange(1,9)*m.opt.timestep)))))
                    for tick in range(8):
                        for d,qs in [(pre,preq[tick]),(post,postq[tick])]:
                            d.qpos[:]=qs;d.mocap_pos[:]=poses[tick];d.mocap_quat[:]=quats[tick];mujoco.mj_forward(m,d)
                        com_err=max(com_err,float(np.max(abs(pre.xipos[body]-com[tick]))))
                        raw=rec[rec[:,0]==tick]
                        gpu=members(((a[1],a[2],a[5]) for a in raw),m,body)
                        cpu=members(((c.geom[0],c.geom[1],c.dist) for c in pre.contact),m,body)
                        after=members(((c.geom[0],c.geom[1],c.dist) for c in post.contact),m,body)
                        yes,no=active(gpu),active(cpu);n+=1
                        pre_dis+=len(yes^no);post_dis+=len(yes^active(after));pre_events+=bool(yes^no)
                        for hand,obj in sorted(yes^no):
                            kind='warp_only' if (hand,obj) in yes else 'cpu_only'
                            dist=float(mujoco.mj_geomDistance(m,pre,hand,obj,.03,None))
                            norms=[a[18] for a in raw if a[5]<=0 and {int(a[1]),int(a[2])}=={hand,obj}]
                            gmins=gpu.get((hand,obj),[]);cmins=cpu.get((hand,obj),[])
                            all_gaps.append(dict(role=role,phase=row['phase'],branch=branch,tick=tick,kind=kind,
                                hand=m.geom(hand).name,obj=m.geom(obj).name,hand_type=int(m.geom_type[hand]),object_type=int(m.geom_type[obj]),
                                query_um=dist*1e6,query_censored=dist>=.03,
                                cpu_min_um=min(cmins)*1e6 if cmins else None,warp_min_um=min(gmins)*1e6 if gmins else None,
                                warp_normal_N=float(sum(norms)),risk_selected=row['risk_selected'],uniform_selected=row['uniform_selected']))
                            phase_count[kind]+=1
                counts[branch]=dict(phase_count)
            all_phases.append(dict(role=role,phase=row['phase'],risk=row['risk_selected'],uniform=row['uniform_selected'],branches=counts))
        model_evidence={k:dict(max_abs=max(d[k]['max_abs'] for d in source_models),
            equal_after_cast=all(d[k]['equal_after_cast'] for d in source_models)) for k in source_models[0]}
        results[role]=dict(source_count=len(rows),ticks=n,ticks_with_disagreement=pre_events,pre_pose_differences=pre_dis,post_pose_differences=post_dis,
            source_models=model_evidence,npair=m.npair,max_geom_margin_m=float(np.max(np.abs(m.geom_margin))),max_geom_gap_m=float(np.max(np.abs(m.geom_gap))),
            mount_max_diff=mount_change,pre_com_max_component_error_m=com_err,clock_max_error_s=qtime_err,
            kinds={k:stats([r for r in all_gaps if r['kind']==k and r['role']==role]) for k in ['cpu_only','warp_only']})
    summary={k:stats([r for r in all_gaps if r['kind']==k]) for k in ('cpu_only','warp_only')}
    by_branch={br:{k:stats([r for r in all_gaps if r['kind']==k and r['branch']==br]) for k in summary} for br in ('minus','plus','sham','sham_repeat')}
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),scope='Membership mismatch records are pair/substep incidences, not independent errors. Native buffers and fixed geometry are checked at every saved source. CPU and Warp have different collision implementations.',
        ticks=sum(v['ticks'] for v in results.values()),summary=summary,bin_right_inclusive_edges_um=EDGES,quantile_prob=[0,.05,.25,.5,.75,.95,1],roles=results,by_branch=by_branch,by_source=all_phases,records=all_gaps))
    print(json.dumps(summary));print([(k,v['ticks'],v['pre_pose_differences'],v['post_pose_differences'],v['pre_com_max_component_error_m'],v['clock_max_error_s']) for k,v in results.items()])
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('out',type=Path);a=parser.parse_args();main(a.root.resolve(),a.out.resolve())
