"""One predeclared joint-only contact projection; source object/wrist unchanged."""
from pathlib import Path
import argparse,json,sys,tomllib
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.controlled_snapshot import native_controlled_snapshot
from supermimic_research.mechanics.bounded_contact_projection import BoundedContactProjection


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf8')


def main(args):
    cp=args.config.resolve();cfg=tomllib.loads(cp.read_text());source=(cp.parent/cfg['source_report']).resolve();r=json.loads(source.read_text());case=next(c for c in r['cases'] if c['task']=='hand2-pen-spin' and c['controller']=='ZERO')
    if args.output.exists():raise ValueError('Fresh proposal output')
    m,d,trace,index=native_controlled_snapshot(case);q0=d.qpos[index['qpos']].copy();j=index['joints'];a=index['actuators']
    lower=np.maximum(q0-cfg['max_joint_delta_rad'],np.maximum(m.jnt_range[j,0],m.actuator_ctrlrange[a,0])+1e-6)
    upper=np.minimum(q0+cfg['max_joint_delta_rad'],np.minimum(m.jnt_range[j,1],m.actuator_ctrlrange[a,1])-1e-6)
    body=case['model_layout']['object_body'];parts=[g for g in range(m.ngeom) if m.geom_bodyid[g]==body and (m.geom_contype[g] or m.geom_conaffinity[g])]
    hand=[g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('robot/') and (m.geom_contype[g] or m.geom_conaffinity[g])]
    groups={i:[m.geom(f'robot/right_finger{i}_{suffix}').id for suffix in cfg['contact_suffixes']] for i in cfg['contact_fingers']}
    args.output.mkdir();write(args.output/'intent.json',dict(config=file_pin(cp),source=file_pin(source),case=case,neural_updates=0,
        scope='One bounded joint-only geometry proposal; object and wrist fixed. Original reference and failed baselines preserved. No geometry-to-dynamics success substitution.'))
    engine=BoundedContactProjection(m,d,index['qpos'],lower,upper,parts,hand,groups,cfg)
    initial=engine.gaps(q0)
    with (args.output/'search.jsonl').open('w',encoding='utf8') as log:result=engine.solve(log)
    q=d.qpos.copy();q[index['qpos']]=result['q'];d.qpos[:]=q;mujoco.mj_forward(m,d)
    contacts=[dict(geoms=[m.geom(int(g)).name for g in con.geom],dist_m=float(con.dist)) for con in d.contact]
    # Re-query every eligible pair without sphere pruning for final native readback.
    gaps=[]
    for g,h in engine.pairs:
        witness=np.zeros(6);gaps.append(float(mujoco.mj_geomDistance(m,d,g,h,cfg['distance_limit_m'],witness)))
    gaps=np.array(gaps);n=len(engine.object_pairs);group_gaps=[float(min(gaps[v])) for v in engine.group_indices.values()]
    qualification=dict(all_object_pairs=bool(gaps[:n].min()>=-cfg['qualification_penetration_m']),
        all_self_pairs=bool(np.all(gaps[n:]>=-cfg['qualification_self_penetration_m'])),
        all_contact_groups=bool(all(-cfg['qualification_penetration_m']<=g<=cfg['qualification_contact_gap_m'] for g in group_gaps)),
        forward_contacts=bool(all(con['dist_m']>=-cfg['qualification_penetration_m'] for con in contacts)),
        raw_joint_delta=bool(np.max(abs(result['delta']))<=cfg['max_joint_delta_rad']+1e-12),
        nonjoint_qpos_unchanged=bool(np.array_equal(np.delete(q,index['qpos']),np.delete(trace['initial_qpos'],index['qpos']))))
    np.savez_compressed(args.output/'proposal.npz',qpos=q,q20=result['q'],delta20=result['delta'],qpos_indices=index['qpos'],actuator_indices=index['actuators'],
                        all_pairs=np.array(engine.pairs),initial_gaps=initial,final_gaps=gaps)
    maximum_delta=float(abs(result['delta']).max())
    result={k:v for k,v in result.items() if k not in ['q','delta','gaps','group_gaps']}
    report=dict(schema='controlled_pen_joint_only_projection:1',config=file_pin(cp),source=file_pin(source),case=case,
        sources=[file_pin(__file__),file_pin(ROOT/'src/supermimic_research/mechanics/bounded_contact_projection.py'),file_pin(ROOT/'src/supermimic_research/mechanics/controlled_snapshot.py'),file_pin(ROOT/'src/supermimic_research/mechanics/compound_collision.py')],
        proposal=file_pin(args.output/'proposal.npz'),result=result,qualification=qualification,geometry_qualified=all(qualification.values()),
        initial_max_all_pair_object_penetration_m=float(max(0,-initial[:n].min())),final_max_all_pair_object_penetration_m=float(max(0,-gaps[:n].min())),
        final_group_gaps_m=group_gaps,max_joint_delta_rad=maximum_delta,
        native_forward_contacts=contacts,object_pairs=n,self_pairs=len(engine.self_pairs),neural_updates=0,
        scope='Native all-pair distance plus native forward readback, not independent calibrated CAD or dynamic support qualification.')
    write(args.output/'report.json',report);print(json.dumps({k:report[k] for k in ['geometry_qualified','initial_max_all_pair_object_penetration_m','final_max_all_pair_object_penetration_m','final_group_gaps_m','max_joint_delta_rad']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
