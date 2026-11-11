"""Independent raw contact/control/state readback; pack explicit new paired roles."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.compound_transition import observe55
from supermimic_research.mechanics.transition_records import state_rotation
from supermimic_research.mechanics.residual_control import residual_ema
from supermimic_research.mechanics.intervention_pairs import intervention_pairs


def main(args):
    import mujoco
    run,out=args.run.resolve(),args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    collection=json.loads((run/'report.json').read_text());binding=json.loads(verify_file_pin(collection['binding']).read_text());reference=json.loads(verify_file_pin(collection['reference']).read_text());ref=np.load(verify_file_pin(reference['output']))
    for pin in collection['sources']:verify_file_pin(pin)
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(binding['model'])));parts=np.array(binding['mug_part_geom_ids']);base=ref['feedforward_PD_target20'][0]
    neg=np.minimum(.05,base-model.actuator_ctrlrange[:,0]);pos=np.minimum(.05,model.actuator_ctrlrange[:,1]-base);COM=model.body_ipos[binding['mug_body_id']]
    geom_groups=np.full(model.ngeom,-1,dtype=int)
    for finger in [1,2,3]:
        for suffix in ['link3_col','link4_col','tip_sensor_col']:geom_groups[model.geom(f'right_finger{finger}_{suffix}').id]=finger-1
    packs={r:[] for r in ['TRAIN','CHECK']};maxima=dict(controller=0.,world_wrench=0.,BODY_wrench=0.,Fn=0.,observer=0.,target_chart=0.);raw_count=0
    for g in collection['groups']:
        a=np.load(verify_file_pin(g['transitions']));z=np.load(verify_file_pin(g['trace']));raw=z['raw_contact_tick_plus23'];n=round(.02/g['dt']);preq,prev,postq,postv=z['pre_qpos'],z['pre_qvel'],z['post_qpos'],z['post_qvel'];preCOM=z['pre_object_COM']
        if a['x'].shape!=(80,102) or a['y'].shape!=(80,87) or not np.all(a['complete_H']) or np.any(a['terminated']) or not np.all(a['actual_duration_s']==.02):raise ValueError('Complete H group contract')
        for key in a.files:
            if a[key].dtype.kind in 'fc' and not np.isfinite(a[key]).all():raise ValueError('Nonfinite saved group')
        expected=residual_ema(a['action20'],a['EMA_before20'],neg,pos);maxima['controller']=max(maxima['controller'],float(abs(expected-a['EMA_after20']).max()),float(abs(base+expected-a['actual_control20']).max()))
        if np.any(abs(a['action20'])>1) or np.any(a['actual_control20']<model.actuator_ctrlrange[:,0]) or np.any(a['actual_control20']>model.actuator_ctrlrange[:,1]):raise ValueError('Illegal executed actions or controls')
        for i,((ta,tb),(ra,rb)) in enumerate(zip(z['tick_ranges'],z['contact_ranges'])):
            if tb-ta!=n:raise ValueError('Not actual full H tick count')
            rows=raw[ra:rb];raw_count+=len(rows);o1,o2=np.isin(rows[:,1],parts),np.isin(rows[:,2],parts);rows=rows[o1^o2];o2=o2[o1^o2]
            g1,g2=rows[:,1].astype(int),rows[:,2].astype(int);sign=np.where(o2,1.,-1.);frame=rows[:,9:18].reshape(-1,3,3)
            F=sign[:,None]*np.einsum('nji,nj->ni',frame,rows[:,18:21]);T=sign[:,None]*np.einsum('nji,nj->ni',frame,rows[:,21:24])
            tick=rows[:,0].astype(int)-int(a['k'][i])*n
            if np.any(tick<0) or np.any(tick>=n):raise ValueError('Raw contact tick outside branch')
            T+=np.cross(rows[:,6:9]-preCOM[ta+tick],F);w=np.r_[F.sum(0),T.sum(0)]/n;normal=np.zeros(3);groups=geom_groups[np.where(o2,g1,g2)]
            for finger in range(3):normal[finger]=np.maximum(rows[groups==finger,18],0).sum()/n
            R=state_rotation(a['source55'][i]);body=np.r_[R.T@w[:3],R.T@w[3:]]
            maxima['world_wrench']=max(maxima['world_wrench'],float(abs(w-a['mean_native_wrench_world_COM'][i]).max()));maxima['BODY_wrench']=max(maxima['BODY_wrench'],float(abs(body-a['y'][i,78:84]).max()));maxima['Fn']=max(maxima['Fn'],float(abs(normal-a['y'][i,84:]).max()))
            for state,q,v in [(a['source55'][i],preq[ta],prev[ta]),(a['next55'][i],postq[tb-1],postv[tb-1])]:maxima['observer']=max(maxima['observer'],float(abs(observe55(q,v,COM)-state).max()))
            from scipy.spatial.transform import Rotation
            s,t=a['source55'][i],a['next55'][i];delta=np.r_[t[:43]-s[:43],Rotation.from_matrix(state_rotation(t)@state_rotation(s).T).as_rotvec(),t[49:]-s[49:]]
            maxima['target_chart']=max(maxima['target_chart'],float(abs(a['x'][i]-np.r_[s,a['warmstart26_before'][i],a['actual_control20'][i],g['dt']]).max()),float(abs(a['y'][i,:78]-np.r_[delta,a['warmstart26_after'][i]]).max()))
        pairs=intervention_pairs(a)
        if len(pairs)!=40:raise ValueError('All20 joints x2 amplitudes')
        packs[g['role']].append({key:a[key] for key in ['x','y','seed','dt','k','joint','amplitude','sign','source_native_state','center_action20']})
    packed={role:{key:np.concatenate([a[key] for a in groups]) for key in groups[0]} for role,groups in packs.items()}
    source_keys={role:{r[:81].tobytes() for r in p['x']} for role,p in packed.items()};source_overlap=len(source_keys['TRAIN']&source_keys['CHECK'])
    old=json.loads((ROOT/'runs/mug_H20_packed_v1/report.json').read_text());old_check={row[:81].tobytes() for role in ['CHECK','RESPONSE'] for row in np.load(verify_file_pin(old['outputs'][role]))['x']};old_probe=json.loads((ROOT/'runs/mug_action_probes_v1/report.json').read_text());old_check|={row[:81].tobytes() for row in np.load(verify_file_pin(old_probe['probes']))['x']}
    old_overlap=len(source_keys['TRAIN']&old_check)
    checks=dict(full_replay_barriers=len(collection['barriers'])==60 and all(b['passed'] for b in collection['barriers']),complete_H_counts=len(packed['TRAIN']['x'])==2880 and len(packed['CHECK']['x'])==1920,
        raw_control_reconstruction=maxima['controller']<1e-14,raw_wrench_and_group_Fn=max(maxima[k] for k in ['world_wrench','BODY_wrench','Fn'])<1e-8,
        actual_observer_and_target_chart=max(maxima[k] for k in ['observer','target_chart'])<1e-12,TRAIN_CHECK_source_disjoint=source_overlap==0,old_diagnostic_sources_excluded=old_overlap==0,
        declared_seed_roles=set(packed['TRAIN']['seed'])==set(range(200,212)) and set(packed['CHECK']['seed'])==set(range(8200,8204)))
    out.mkdir();outputs={}
    for role,arrays in packed.items():
        arrays['pairs']=intervention_pairs(arrays);np.savez_compressed(out/(role+'.npz'),**arrays);outputs[role]=file_pin(out/(role+'.npz'))
    result=dict(schema='mug_paired_interventions_review:1',collection=file_pin(run/'report.json'),sources=[file_pin(__file__),file_pin(ROOT/'src/supermimic_research/mechanics/intervention_pairs.py')],
        checks={k:bool(v) for k,v in checks.items()},passed=sum(checks.values()),total=len(checks),max_errors=maxima,raw_contacts_reviewed=raw_count,source_states={r:len(k) for r,k in source_keys.items()},
        TRAIN_CHECK_source_overlap=source_overlap,old_diagnostic_source_overlap=old_overlap,outputs=outputs,neural_updates=0,
        scope='Independent raw force/couple/COM/sign, actual control and endpoint chart reconstruction; all20 joint pairs share complete native states within pairs. Same task/prefix distribution, not new trajectories, source contact truth or analytic native gradients.')
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps(result))
    if not all(checks.values()):raise ValueError('Retain failed review, do not fit')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
