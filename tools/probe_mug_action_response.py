"""Fixed CHECK-only native central differences; no fitting or physics changes."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.compound_transition import CompoundTransitionReader
from supermimic_research.mechanics.transition_records import transition_xy


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')


def main(args):
    collection_path=args.collection.resolve();report=json.loads(collection_path.read_text());out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified probe output')
    binding=json.loads(verify_file_pin(report['binding']).read_text());reference=json.loads(verify_file_pin(report['reference']).read_text());ref=np.load(verify_file_pin(reference['output']))
    sources=[file_pin(ROOT/p) for p in ['tools/probe_mug_action_response.py','src/supermimic_research/mechanics/transition_records.py',
        'src/supermimic_research/mechanics/compound_transition.py','src/supermimic_research/mechanics/residual_control.py',
        'src/supermimic_research/mechanics/compound_contacts.py','src/supermimic_research/mechanics/compound_holding.py']]
    out.mkdir(parents=True);write(out/'intent.json',dict(schema='mug_native_action_CHECK_probes:1',collection=file_pin(collection_path),sources=sources,
        k=[0,3,6],joints=[0,4,8,9],amplitudes=[.01,.05,.2],signs=[-1,1],dt=[b['dt_s'] for b in report['barriers']],
        role='CHECK_ONLY_NEVER_FIT_OR_SELECT',paired_cases=72,neural_updates=0,physics_changes=False,
        derivative_scope='Three finite central secants; contact/solver nonsmoothness and dt disagreement retained; not analytic native derivatives'))
    start=time.perf_counter();rows=[];barriers=[];steps=0;traces={};ranges=[];raw=[];full=[];raw_start=0
    for barrier in report['barriers']:
        dt=barrier['dt_s'];saved=np.load(verify_file_pin(barrier['episode']['transitions']));reader=CompoundTransitionReader(binding,ref['initial_qpos27'],ref['feedforward_PD_target20'][0],dt_s=dt)
        for k in [0,3,6]:
            i=int(np.flatnonzero(saved['k']==k)[0]);state=reader.snapshot();state.update(integration_state=saved['source_native_state'][i].copy(),tick=k*reader.n,k=k,terminated=False)
            state['controller'].update(memory=saved['EMA_before20'][i].copy(),updates=k)
            reader.restore(state);t=reader.step(np.zeros(20));steps+=t['executed_native_steps']
            equal=all(np.array_equal(t[key],saved[key][i]) for key in ['source55','next55','source_native_state','next_native_state','warmstart26_before','warmstart26_after','mean_native_wrench_source_BODY_COM','mean_native_group_Fn_N'])
            barriers.append(dict(k=k,dt=dt,passed=equal))
            if not equal:raise ValueError('Stored full state/zero-action native replay changed')
            for joint in [0,4,8,9]:
                if reader.controller.negative[joint]!=reader.controller.positive[joint]:raise ValueError('Central action derivative at asymmetric room kink is not declared here')
                for amplitude in [.01,.05,.2]:
                    for sign in [-1,1]:
                        reader.restore(state);action=np.zeros(20);action[joint]=sign*amplitude;t=reader.step(action);steps+=t['executed_native_steps']
                        if not t['complete_H'] or t['terminated']:raise ValueError('Incomplete/terminal probe is not a full H derivative label')
                        x,y=transition_xy(t,dt)
                        pairs=np.unique(t['raw_contact_tick_plus23'][:,1:3].astype(np.int64),axis=0)
                        rows.append(dict(x=x,y=y,k=k,dt=dt,joint=joint,amplitude=amplitude,sign=sign,
                            action20=action,EMA_before20=t['decision']['EMA_before20'],actual_control20=t['decision']['actual_control20'],
                            source_native_state=t['source_native_state'],next_native_state=t['next_native_state'],unique_contact_pairs=len(pairs),
                            du_da=reader.controller.beta*reader.controller.positive[joint]))
                        full.append(pairs);raw.append(t['raw_contact_tick_plus23']);ranges.append([raw_start,raw_start+len(raw[-1])]);raw_start+=len(raw[-1])
                        for name,value in t['trace'].items():traces.setdefault(name,[]).append(value)
        reader.verify_physics()
        print(json.dumps(dict(dt=dt,cases=len(rows),steps=steps)),flush=True)
    arrays={key:np.array([r[key] for r in rows]) for key in rows[0]};np.savez_compressed(out/'probes.npz',**arrays)
    np.savez_compressed(out/'raw_trace.npz',**{k:np.concatenate(v) for k,v in traces.items()},raw_contacts=np.concatenate(raw),raw_ranges=np.array(ranges),
        unique_pairs=np.concatenate(full),pair_ranges=np.cumsum([0]+[len(p) for p in full]))
    for pin in sources:verify_file_pin(pin)
    result=dict(schema='mug_native_action_CHECK_probes:1',intent=file_pin(out/'intent.json'),collection=file_pin(collection_path),sources=sources,
        probes=file_pin(out/'probes.npz'),raw_trace=file_pin(out/'raw_trace.npz'),barriers=barriers,cases=len(rows),pairs=len(rows)//2,
        total_native_steps=steps,wall_s=time.perf_counter()-start,role='CHECK_ONLY_NEVER_FIT_OR_SELECT',neural_updates=0,
        scope='Same baseline states at k0/3/6 and two native dt; central finite action responses at amplitudes .01/.05/.2. No new trajectories, independent RGB, policy or analytic native gradient.')
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),cases=len(rows),steps=steps,barriers=all(x['passed'] for x in barriers))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
