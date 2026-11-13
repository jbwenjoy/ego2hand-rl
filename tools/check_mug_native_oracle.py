"""Prove batched native branches equal the frozen observed step1/step2 plant."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.compound_transition import CompoundTransitionReader
from supermimic_research.mechanics.native_oracle import NativeBranchOracle


def main(args):
    out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    collection_path=ROOT/'runs/mug_H20_collection_v1/report.json';collection=json.loads(collection_path.read_text());binding=json.loads(verify_file_pin(collection['binding']).read_text());reference=json.loads(verify_file_pin(collection['reference']).read_text());ref=np.load(verify_file_pin(reference['output']))
    out.mkdir();checks=[];oracle_wall=0.;observed_wall=0.;source_steps=0;saved_arrays={};native_steps=0
    for barrier in collection['barriers']:
        reader=CompoundTransitionReader(binding,ref['initial_qpos27'],ref['feedforward_PD_target20'][0],dt_s=barrier['dt_s']);oracle=NativeBranchOracle(reader)
        data=np.load(verify_file_pin(barrier['episode']['transitions']))
        for k in [0,3,6]:
            i=int(np.flatnonzero(data['k']==k)[0]);snapshot=reader.snapshot();snapshot.update(integration_state=data['source_native_state'][i].copy(),tick=k*reader.n,k=k,terminated=False);snapshot['controller'].update(memory=data['EMA_before20'][i].copy(),updates=k)
            for action_id,action in enumerate([np.zeros(20),np.r_[.2,np.zeros(19)],np.r_[.3*np.sin(np.arange(12)+.2),np.zeros(8)]]):
                reader.restore(snapshot);before=reader.snapshot();start=time.perf_counter();pred=oracle.rollout(snapshot,action,3);oracle_wall+=time.perf_counter()-start;after=reader.snapshot()
                unchanged=np.array_equal(before['integration_state'],after['integration_state']) and np.array_equal(before['controller']['memory'],after['controller']['memory']) and before['k']==after['k']
                start=time.perf_counter();actual=[reader.step(action) for _ in range(3)];observed_wall+=time.perf_counter()-start;source_steps+=sum(t['executed_native_steps'] for t in actual)
                equal={key:np.array_equal(pred[key],np.array([t[field] for t in actual])) for key,field in [('states55','next55'),('native_states','next_native_state')]}
                equal['EMA_exact']=np.array_equal(pred['EMA_after20'],actual[-1]['decision']['EMA_after20']);equal['controls_exact']=np.array_equal(pred['controls20'],np.array([t['decision']['actual_control20'] for t in actual]));equal['active_plant_unchanged_by_prediction']=unchanged
                checks.append(dict(dt=reader.dt,k=k,action_id=action_id,checks=equal,passed=all(equal.values())));saved_arrays[f'case{len(checks)-1}_oracle']=pred['native_states'];saved_arrays[f'case{len(checks)-1}_observed']=np.array([t['next_native_state'] for t in actual])
                if not all(equal.values()):raise ValueError('Oracle/observed native parity failed; no planning allowed')
        reader.verify_physics();native_steps+=oracle.steps
    np.savez_compressed(out/'state_parity.npz',**saved_arrays)
    result=dict(schema='native_branch_vs_observed_compound_parity:1',sources=[file_pin(__file__),file_pin(ROOT/'src/supermimic_research/mechanics/native_oracle.py'),*collection['sources']],collection=file_pin(collection_path),
        model=binding['model'],checks=checks,passed=sum(x['passed'] for x in checks),total=len(checks),state_parity=file_pin(out/'state_parity.npz'),oracle_native_steps=native_steps,observed_native_steps=source_steps,
        oracle_wall_s=oracle_wall,observed_wall_s=observed_wall,neural_updates=0,
        official_source='https://raw.githubusercontent.com/google-deepmind/mujoco/3.11.0/src/engine/engine_forward.c',scope='Exact54H endpoint full numerical state/control/EMA parity across two dt, three prefixes and three actions. Batched mj_step uses same implicitfast as split stepping in pinned3.11.0. Endpoint contacts for prediction are sampled, not full execution gates.')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args())
