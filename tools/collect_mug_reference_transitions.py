"""New role-separated native H20 data; no NN fitting, source-label fabrication or aid."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.compound_transition import CompoundTransitionReader
from supermimic_research.runtime.native_evaluation import DTS


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.tolist())+'\n',encoding='utf8')


def save_episode(folder,transitions,meta):
    folder.mkdir();arrays={};trace={k:np.concatenate([t['trace'][k] for t in transitions]) for k in transitions[0]['trace']}
    raw=np.concatenate([t['raw_contact_tick_plus23'] for t in transitions]);ranges=[];start=0
    for t in transitions:ranges.append([start,start+t['executed_native_steps']]);start+=t['executed_native_steps']
    for key in ['source55','next55','warmstart26_before','warmstart26_after','source_native_state','next_native_state','mean_native_wrench_world_COM','mean_native_wrench_source_BODY_COM','mean_native_group_Fn_N']:
        arrays[key]=np.array([t[key] for t in transitions])
    for key in ['action20','EMA_before20','EMA_after20','actual_control20']:arrays[key]=np.array([t['decision'][key] for t in transitions])
    arrays.update(complete_H=np.array([t['complete_H'] for t in transitions]),terminated=np.array([t['terminated'] for t in transitions]),
        actual_duration_s=np.array([t['actual_duration_s'] for t in transitions]),k=np.array([t['k'] for t in transitions]),tick_ranges=np.array(ranges))
    np.savez_compressed(folder/'transitions.npz',**arrays);np.savez_compressed(folder/'trace.npz',**trace,raw_contact_tick_plus23=raw)
    item=dict(**meta,transitions=file_pin(folder/'transitions.npz'),trace=file_pin(folder/'trace.npz'),intervals=len(transitions),complete_intervals=sum(t['complete_H'] for t in transitions),
        steps=start,termination=transitions[-1]['termination'],source_contact_truth_known=False,native_simulated_contact_labels_known=True)
    write(folder/'report.json',item);return item


def profile(seed,k,source,grasp):
    rng=np.random.default_rng(seed);phase=rng.uniform(-np.pi,np.pi,20);frequency=rng.uniform(.3,2.,20)
    signal=.6*np.sin(2*np.pi*frequency*(k+.5)*.02+phase)
    feedback=np.tanh((grasp-source[:20])/.05-.02*source[20:40])
    return .75*signal+.25*feedback


def main(args):
    binding_path,reference,out=map(lambda p:p.resolve(),[args.binding,args.reference,args.output])
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    binding=json.loads(binding_path.read_text());reference_report=json.loads((reference/'report.json').read_text());ref=np.load(verify_file_pin(reference_report['output']))
    q,base,grasp=ref['initial_qpos27'],ref['feedforward_PD_target20'][0],ref['grasp_q20_reference'][0]
    sources=[file_pin(ROOT/p) for p in ['tools/collect_mug_reference_transitions.py','src/supermimic_research/mechanics/compound_transition.py',
        'src/supermimic_research/mechanics/residual_control.py','src/supermimic_research/mechanics/compound_contacts.py','src/supermimic_research/mechanics/compound_holding.py']]
    out.mkdir(parents=True);write(out/'intent.json',dict(schema='mug_H20_reference_collection_intent:1',sources=sources,binding=file_pin(binding_path),reference=file_pin(reference/'report.json'),model=binding['model'],
        controller=dict(limit_rad=.05,EMA_beta=.5,memory_units='radian residual',control_H_s=.02,asymmetric_room=True,no_action_or_control_clamp=True),
        state55=dict(q20=[0,20],dq20=[20,40],COM_world3=[40,43],BODY_R_world_first_two_columns6=[43,49],COM_velocity_world3=[49,52],omega_world3=[52,55]),
        solver_memory='Full native integration state for replay;warmstart26 input/output retained; state55 alone is not asserted exact numerical Markov',
        native_contact_labels='Actual interval mean simulated forces/moving-COM wrench; source human contacts UNKNOWN',
        TRAIN_seeds=list(range(16)),CHECK_seeds=list(range(1000,1004)),TRAIN_dt=[DTS[0]],CHECK_dt=DTS,
        action_rule='a=.75*(.6 multisine)+.25*tanh((q_grasp-q)/.05-.02*dq); fixed seeded phases/frequencies. Genuine joint-feedback, not learned object controller.',
        barrier='Two zero-action baseline full traces must match historical constant LP targets byte for byte before TRAIN/CHECK collection',
        response_branch_role='CHECK_RESPONSE_ONLY_NEVER_FIT',branch_prefix_k=[0,3,6],branch_joint_directions=[0,4,8,9],branch_action_amplitude=.5,
        max_intervals_per_episode=100,max_episodes=24,max_native_integration_steps=600000,partial_H_fitting=False,
        reference_role='New anchored source-informed object goal; not source gravity preservation, human joint retargeting or independent RGB',neural_updates=0))
    episodes=[];barriers=[];readers={};snapshots={};counts=0;tick=time.perf_counter()
    parent=json.loads((ROOT/'runs/mug_support_fit_v1/report.json').read_text())['results'][0]
    for dt in DTS:
        reader=CompoundTransitionReader(binding,q,base,dt_s=dt);readers[dt]=reader;transitions=[];saved={}
        while not reader.terminated and reader.k<100:
            if reader.k in [0,3,6]:saved[reader.k]=reader.snapshot()
            transitions.append(reader.step(np.zeros(20)))
        reader.verify_physics();item=save_episode(out/f'baseline_{dt}',transitions,dict(role='BARRIER_ONLY_NOT_FIT',dt_s=dt,seed=None))
        old_case=next(c for c in parent['native_cases'] if c['control']=='inverse_PD_strict_LP' and c['dt_s']==dt);old=np.load(verify_file_pin(old_case['trace']));new=np.load(item['trace']['path'])
        parity={key:old[key].dtype==new[key].dtype and old[key].shape==new[key].shape and old[key].tobytes()==new[key].tobytes() for key in transitions[0]['trace']}
        barrier=dict(dt_s=dt,passed=all(parity.values()),arrays=parity,historical_trace=old_case['trace'],current=item['trace'],episode=item)
        barriers.append(barrier);write(out/'barrier.json',barriers)
        if not barrier['passed']:raise ValueError('Physical baseline parity failed; no new data fitting/collection allowed')
        snapshots[dt]=saved;counts+=item['steps'];print(json.dumps(dict(baseline_dt=dt,passed=True,steps=item['steps'])),flush=True)
    resume=[];branches=[]
    for dt in DTS:
        reader=readers[dt]
        for prefix,state in snapshots[dt].items():
            reader.restore(state);first=reader.step(np.zeros(20));reader.restore(state);second=reader.step(np.zeros(20))
            equal=all(first[k].tobytes()==second[k].tobytes() for k in ['source55','next55','source_native_state','next_native_state','warmstart26_before','warmstart26_after']) and all(first['trace'][k].tobytes()==second['trace'][k].tobytes() for k in first['trace'])
            resume.append(dict(dt_s=dt,k=prefix,passed=equal));counts+=first['executed_native_steps']+second['executed_native_steps']
            if not equal:raise ValueError('Full native/controller restore parity failed')
            for joint in [0,4,8,9]:
                for sign in [-1,1]:
                    reader.restore(state);action=np.zeros(20);action[joint]=sign*.5;t=reader.step(action);counts+=t['executed_native_steps']
                    branches.append(save_episode(out/f'branch_{len(branches):03d}',[t],dict(role='CHECK_RESPONSE_ONLY_NEVER_FIT',dt_s=dt,seed=None,prefix_k=prefix,joint=joint,sign=sign)))
        reader.verify_physics()
    write(out/'resume.json',resume)
    for role,seeds,dts in [('TRAIN',range(16),[DTS[0]]),('CHECK',range(1000,1004),DTS)]:
        for seed in seeds:
            for dt in dts:
                reader=CompoundTransitionReader(binding,q,base,dt_s=dt);transitions=[]
                while not reader.terminated and reader.k<100:
                    action=profile(seed,reader.k,reader.observe(),grasp);transitions.append(reader.step(action))
                reader.verify_physics();item=save_episode(out/f'episode_{len(episodes):03d}',transitions,dict(role=role,dt_s=dt,seed=seed));episodes.append(item);counts+=item['steps']
                print(json.dumps(dict(role=role,seed=seed,dt=dt,complete=item['complete_intervals'],intervals=item['intervals'],termination=item['termination'],elapsed_s=time.perf_counter()-tick)),flush=True)
    if counts>600000:raise ValueError('Predeclared total native integration budget exceeded')
    for pin in sources:verify_file_pin(pin)
    verify_file_pin(binding['model'])
    report=dict(schema='mug_H20_reference_native_transitions:1',sources=sources,binding=file_pin(binding_path),reference=file_pin(reference/'report.json'),barriers=barriers,resume_checks=resume,branches=branches,episodes=episodes,
        total_integration_steps=counts,TRAIN_complete_rows=sum(e['complete_intervals'] for e in episodes if e['role']=='TRAIN'),CHECK_complete_rows=sum(e['complete_intervals'] for e in episodes if e['role']=='CHECK'),
        branch_cases=len(branches),neural_updates=0,wall_s=time.perf_counter()-tick,
        scope='Bounded feedback/exploration transitions and heldout paired actions in one anchored source-informed mug task. Partial H preserved but excluded from fixed-H fitting. No old phone heldouts used, new policy, independent RGB, full source trajectory success or generalization claim.')
    write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),steps=counts,TRAIN=report['TRAIN_complete_rows'],CHECK=report['CHECK_complete_rows'],branches=len(branches))))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['binding','reference','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
