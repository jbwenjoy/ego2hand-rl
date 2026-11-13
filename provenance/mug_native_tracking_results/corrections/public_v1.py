"""Persist diagnostic public packets; failed MPC actions are never expert labels."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.control_observation import actor_observation,SCHEMA,OFFSETS


def main(args):
    run,out=args.run.resolve(),args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified diagnostic output')
    report=json.loads((run/'report.json').read_text());ref_report=json.loads(verify_file_pin(report['inputs']['reference']).read_text());ref=dict(np.load(verify_file_pin(ref_report['output'])))
    rows=[];aliases={}
    for case_id,case in enumerate(report['cases']):
        data=np.load(verify_file_pin(case['execution']['transitions']));previous=np.zeros(20)
        for i,k in enumerate(data['k']):
            obs=actor_observation(data['source55'][i],data['EMA_before20'][i],previous,int(k),ref,task=case['task']);row=dict(observation144=obs,recorded_action20=data['action20'][i],case=case_id,k=int(k),dt=case['dt'],complete_H=bool(data['complete_H'][i]))
            rows.append(row);aliases.setdefault(obs.tobytes(),[]).append(len(rows)-1);previous=data['action20'][i]
    alias_groups=[ids for ids in aliases.values() if len(ids)>1];conflicts=[]
    for ids in alias_groups:
        actions=np.array([rows[i]['recorded_action20'] for i in ids]);delta=float(abs(actions-actions[0]).max())
        if delta>0:conflicts.append(dict(rows=ids,max_action_difference=delta))
    out.mkdir();np.savez_compressed(out/'diagnostic_public_packets.npz',**{k:np.array([r[k] for r in rows]) for k in rows[0]})
    result=dict(schema='mug_public_actor_observation_diagnostic:1',observation_schema=SCHEMA,source=file_pin(__file__),codec=file_pin(ROOT/'src/supermimic_research/mechanics/control_observation.py'),run=file_pin(run/'report.json'),reference=ref_report['output'],
        output=file_pin(out/'diagnostic_public_packets.npz'),rows=len(rows),future_offsets_H=list(OFFSETS),terminal_padding='Repeat declared endpoint goal with validity0 beyond2s; no extrapolation.',
        future_frame='Current BODY axes: relative COM3,relative R first2 columns6,relative COM velocity3/omega3; known reference only.',
        EMA_units='radian residual, not raw policy action',solver_state_included=False,native_dt_included=False,future_actual_states_or_forces_included=False,
        exact_public_alias_groups=len(alias_groups),conflicting_recorded_action_groups=conflicts,data_role='DIAGNOSTIC_ONLY_NOT_EXPERT_NOT_FIT',all_control_cases_qualified=False,
        neural_updates=0,native_integration_steps=0,scope='Public packet contract and privileged-teacher alias diagnosis only. Actions came from failed baseline/MPC rollouts, not demonstrations. Native dt/warmstart excluded; public Markov sufficiency and hardware object observation remain unproven.')
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
