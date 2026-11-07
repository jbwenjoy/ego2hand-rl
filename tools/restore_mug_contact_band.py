"""Bounded local exact-pair secants, followed by full compound qualification."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.compound_collision import CompoundCollisionAudit


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.tolist())+'\n',encoding='utf8')


def main(args):
    import mujoco
    reference,parent,out=args.reference.resolve(),args.parent.resolve(),args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    bind=json.loads((reference/'binding.json').read_text());previous=json.loads((parent/'report.json').read_text())
    for item in previous['results']:verify_file_pin(item['candidate']);verify_file_pin(item['geometry'])
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(bind['model'])));data=mujoco.MjData(model)
    hand=[model.geom(n).id for n in bind['hand_collision_names']]
    permitted={i:[model.geom(f'right_finger{i}_{s}').id for s in ['link3_col','link4_col','tip_sensor_col']] for i in range(1,4)}
    audit=CompoundCollisionAudit(model,bind['mug_part_geom_ids'],hand,model.geom('mug_task_table').id,permitted)
    sources=[file_pin(p) for p in [Path(__file__),ROOT/'src/supermimic_research/mechanics/compound_collision.py',
        ROOT/'src/supermimic_research/mechanics/collision.py',ROOT/'src/supermimic_research/mechanics/convex.py']]
    out.mkdir(parents=True);write(out/'intent.json',dict(schema='mug_exact_contact_restoration_intent:1',sources=sources,parent=file_pin(parent/'report.json'),
        independent_band_m=[-.0001,.0002],target_gap_m=.00005,target_tolerance_m=.00001,
        central_secant_rad=.00001,max_iterations_per_bad_group=8,max_step_Linf_rad=.02,
        line_search=[1.,.5,.25],modify_only_initially_missing_groups=True,all_piece_final_audit=True,
        neural_updates=0,integration_steps=0))
    results=[];tick=time.perf_counter()
    for seed,item in enumerate(previous['results']):
        initial=np.load(item['candidate']['path']);q=initial['qpos27'].copy();start_q=q.copy()
        before=json.loads(Path(item['geometry']['path']).read_text());folder=out/f'seed{seed:02d}';folder.mkdir()
        counts=dict(kinematics=0,exact_pair_queries=0,full_audits=0);log=[]
        lo=np.maximum(model.jnt_range[:20,0],model.actuator_ctrlrange[:,0])+1e-6
        hi=np.minimum(model.jnt_range[:20,1],model.actuator_ctrlrange[:,1])-1e-6
        def gap(proposal,pair):
            if np.any(proposal[:20]<lo) or np.any(proposal[:20]>hi):raise ValueError('Illegal local proposal, never clamp')
            data.qpos[:]=proposal;mujoco.mj_kinematics(model,data);counts['kinematics']+=1
            a,b=(audit.shape(data,g) for g in pair)
            answer=audit.query(a,b,force_exact=True);counts['exact_pair_queries']+=1
            if answer['gap_m'] is None:raise ValueError('Unknown exact selected-pair gap')
            return answer['gap_m']
        for group,pads in permitted.items():
            if before['canonical_contact_candidates'][str(group)]:continue
            candidates=[row for row in before['rows'] if row['kind']=='object' and row['geom_ids'][1] in pads and row.get('gap_m') is not None]
            if not candidates:
                log.append(dict(group=group,status='NO_EXACT_NEAR_COMPONENT_PAIR'));continue
            selected=min(candidates,key=lambda row:row['gap_m']);pair=selected['geom_ids'];indices=np.arange((group-1)*4,group*4)
            for iteration in range(8):
                current=gap(q,pair);error=current-.00005
                if abs(error)<=.00001:break
                gradient=np.zeros(4)
                for j,k in enumerate(indices):
                    step=min(.00001,(hi[k]-q[k])/2,(q[k]-lo[k])/2)
                    if step<1e-9:continue
                    plus=q.copy();minus=q.copy();plus[k]+=step;minus[k]-=step
                    gradient[j]=(gap(plus,pair)-gap(minus,pair))/(2*step)
                if np.linalg.norm(gradient)<1e-10:
                    log.append(dict(group=group,iteration=iteration,status='SINGULAR_SELECTED_PAIR'));break
                delta=-error*gradient/(gradient@gradient)
                # Explicit bounded step scaling, not target/state projection.
                if np.max(np.abs(delta))>.02:delta*=.02/np.max(np.abs(delta))
                accepted=False
                for alpha in [1.,.5,.25]:
                    proposal=q.copy();proposal[indices]+=alpha*delta
                    if np.any(proposal[:20]<lo) or np.any(proposal[:20]>hi):continue
                    after=gap(proposal,pair)
                    record=dict(group=group,iteration=iteration,pair=pair,before_gap_m=current,after_gap_m=after,
                        delta4=(alpha*delta).tolist(),q20=proposal[:20].tolist(),accepted=bool(abs(after-.00005)<abs(error)))
                    log.append(record)
                    if record['accepted']:q=proposal;accepted=True;break
                if not accepted:break
        data.qpos[:]=q;mujoco.mj_kinematics(model,data);counts['kinematics']+=1
        geometry=audit.evaluate(data);counts['full_audits']+=1;write(folder/'geometry.json',geometry)
        write(folder/'proposals.json',log);np.savez_compressed(folder/'candidate.npz',qpos27=q,ctrl20=q[:20],targets_world=initial['targets_world'])
        result=dict(seed=seed,geometry_pass=geometry['passed'],all_pairs_clear=geometry['all_pairs_clear_with_declared_pad_band'],
            canonical_contact_candidates=geometry['canonical_contact_candidates'],unknown=geometry['unknown'],counts=counts,
            max_q_change_rad=float(np.max(np.abs(q[:20]-start_q[:20]))),
            candidate=file_pin(folder/'candidate.npz'),geometry=file_pin(folder/'geometry.json'),proposals=file_pin(folder/'proposals.json'))
        write(folder/'result.json',result);results.append(result);print(json.dumps(result),flush=True)
    final=dict(schema='mug_exact_contact_restoration:1',sources=sources,parent=file_pin(parent/'report.json'),results=results,
        geometry_pass_count=sum(r['geometry_pass'] for r in results),neural_updates=0,integration_steps=0,
        native_physical_holding_qualified=False,wall_s=time.perf_counter()-tick,
        scope='Local static pair-band restoration and full2816 object/189 self/21 hand-table component audit. Not source contact truth, global search, native support force, dynamic holding or independent ego reconstruction.')
    for pin in sources:verify_file_pin(pin)
    write(out/'report.json',final);print(json.dumps(dict(report=file_pin(out/'report.json'),geometry_pass_count=final['geometry_pass_count'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
