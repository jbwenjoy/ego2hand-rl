"""Predeclared first qualified mug pose with five physical PD overclosures."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.compound_collision import CompoundCollisionAudit
from supermimic_research.mechanics.compound_holding import evaluate_hold
from supermimic_research.runtime.native_evaluation import GATES,DTS,PAIR_GATES


def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False,default=lambda x:x.tolist())+'\n',encoding='utf8')


def main(args):
    import mujoco
    reference,parent,out=args.reference.resolve(),args.parent.resolve(),args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    bind=json.loads((reference/'binding.json').read_text());previous=json.loads((parent/'report.json').read_text())
    qualified=[(i,r) for i,r in enumerate(previous['results']) if r['geometry_pass']]
    if not qualified:raise ValueError('No static geometry-qualified candidate')
    seed,record=qualified[0];verify_file_pin(record['candidate']);verify_file_pin(record['geometry'])
    candidate=np.load(record['candidate']['path']);q=candidate['qpos27'].copy()
    geometry=json.loads(Path(record['geometry']['path']).read_text())
    if not geometry['passed']:raise ValueError('Declared complete static geometry gate')
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(bind['model'])));data=mujoco.MjData(model)
    hand=[model.geom(n).id for n in bind['hand_collision_names']]
    permitted={i:[model.geom(f'right_finger{i}_{s}').id for s in ['link3_col','link4_col','tip_sensor_col']] for i in range(1,4)}
    audit=CompoundCollisionAudit(model,bind['mug_part_geom_ids'],hand,model.geom('mug_task_table').id,permitted)
    gradients=[];pairs=[]
    for group,pads in permitted.items():
        rows=[r for r in geometry['rows'] if r['kind']=='object' and r['geom_ids'][1] in pads and r.get('gap_m') is not None]
        pair=min(rows,key=lambda r:r['gap_m'])['geom_ids'];pairs.append(pair);gradient=np.zeros(20)
        for j in range((group-1)*4,group*4):
            values=[]
            for sign in [1,-1]:
                proposal=q.copy();proposal[j]+=sign*1e-5;data.qpos[:]=proposal;mujoco.mj_kinematics(model,data)
                result=audit.query(audit.shape(data,pair[0]),audit.shape(data,pair[1]),force_exact=True);values.append(result['gap_m'])
            gradient[j]=(values[0]-values[1])/2e-5
        if np.linalg.norm(gradient)<1e-10:raise ValueError('No supported closure direction')
        gradients.append(gradient)
    offsets=[0.,.00025,.0005,.001,.002]
    targets=[q[:20]-sum((depth*g/(g@g) for g in gradients),start=np.zeros(20)) for depth in offsets]
    legal=[bool(np.all(u>=model.actuator_ctrlrange[:,0]) and np.all(u<=model.actuator_ctrlrange[:,1])) for u in targets]
    if not all(legal):raise ValueError('Predeclared target outside hard controls, never clamp')
    sources=[file_pin(p) for p in [Path(__file__),ROOT/'src/supermimic_research/mechanics/compound_holding.py',
        ROOT/'src/supermimic_research/mechanics/compound_contacts.py',ROOT/'src/supermimic_research/mechanics/compound_collision.py',
        ROOT/'src/supermimic_research/mechanics/collision.py',ROOT/'src/supermimic_research/mechanics/convex.py',
        ROOT/'src/supermimic_research/runtime/native_evaluation.py']]
    out.mkdir(parents=True);write(out/'intent.json',dict(schema='mug_static_hold_preload_intent:1',sources=sources,
        parent=file_pin(parent/'report.json'),reference=file_pin(reference/'report.json'),selected_seed=seed,
        qualified_candidate=record['candidate'],qualified_geometry=record['geometry'],selected_component_pairs=pairs,
        geometric_overclosure_m=offsets,targets=targets,dts=DTS,duration_s=2.,gates=GATES,dt_pair_gates=PAIR_GATES,
        control='constant legal20 absolute PD; secant selected-pair closure is target construction only, not imposed fingertip displacement',
        max_integration_steps=96000,pad_groups=permitted,no_object_assistance=True,neural_updates=0))
    cases=[];tick=time.perf_counter()
    for index,(depth,target) in enumerate(zip(offsets,targets)):
        for dt in DTS:
            folder=out/f'offset{index:02d}_dt{dt}'
            result=evaluate_hold(model,q,target,bind,permitted,dt,folder)
            cases.append(dict(overclosure_m=depth,case=file_pin(folder/'report.json'),trace=file_pin(folder/'trace.npz'),summary=result))
            print(json.dumps(dict(offset_m=depth,dt=dt,duration_s=result['actual_duration_s'],termination=result['termination'],
                native_holding_gates_pass=result['native_holding_gates_pass'],COM_drift_m=result['metrics']['final_COM_drift_m'],
                force_wrench_error=result['max_sampled_native_external_wrench_error'],elapsed_s=time.perf_counter()-tick)),flush=True)
    pairs=[]
    for depth in offsets:
        selected=[c for c in cases if c['overclosure_m']==depth];endpoints=[np.load(c['trace']['path']) for c in selected]
        delta=float(np.linalg.norm(endpoints[0]['post_COM'][-1]-endpoints[1]['post_COM'][-1]))
        angle=float(2*np.arccos(np.clip(abs(endpoints[0]['post_quat'][-1]@endpoints[1]['post_quat'][-1]),0,1)))
        complete=all(c['summary']['gates']['duration_complete'] for c in selected)
        consistent=complete and delta<=.002 and angle<=PAIR_GATES['final_orientation_difference_rad']
        pairs.append(dict(overclosure_m=depth,both_complete=complete,COM_endpoint_difference_m=delta,
            orientation_endpoint_difference_rad=angle,dt_pair_consistency_pass=consistent,
            qualified=bool(consistent and all(c['summary']['native_holding_gates_pass'] for c in selected))))
    steps=sum(c['summary']['counts']['integration_steps'] for c in cases)
    if steps>96000:raise ValueError('Declared native budget')
    final=dict(schema='mug_static_hold_preload_evaluation:1',sources=sources,cases=cases,dt_pairs=pairs,
        qualified_pair_count=sum(p['qualified'] for p in pairs),integration_steps=steps,neural_updates=0,
        wall_s=time.perf_counter()-tick,source_trajectory_replayed=False,source_gravity_preserved=False,
        scope='Five constant physical PD commands, same statically qualified first pose and two original timesteps. No policy, learning, source trajectory correction, independently estimated ego poses or measured dynamics. Partial cases are not2s endpoint agreement.')
    for pin in sources:verify_file_pin(pin)
    write(out/'report.json',final);print(json.dumps(dict(report=file_pin(out/'report.json'),qualified_pairs=final['qualified_pair_count'],integration_steps=steps)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
