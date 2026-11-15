"""Saved feasible action envelopes and a declared velocity-score diagnostic only."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.runtime.controlled_evaluation import write


def main(a):
    r=json.loads(a.report.read_text());review=json.loads(a.review.read_text())
    if verify_file_pin(review['run'])!=a.report.resolve() or a.output.exists():raise ValueError('Exact reviewed run and fresh diagnostics')
    cases=[]
    for c in r['cases']:
        ref=np.load(verify_file_pin(c['initialization']['nominal_mean_wrench']))
        qa,da=c['model_layout']['object_qposadr'],c['model_layout']['object_dofadr'];rows=[]
        for decision in c['planning']:
            k,index=decision['k'],decision['selected'];d=np.load(verify_file_pin(decision['prediction']))
            feasible=d['score_feasible'].astype(bool);ids=np.flatnonzero(feasible)
            if not len(ids):raise ValueError('This diagnostic requires feasible candidates; never silently filter fallback cases')
            p=d['Q'][:,-1,qa:qa+3].astype(float);v=d['V'][:,-1,da:da+3].astype(float)
            poserr=np.linalg.norm(p-ref['COM_world_m'][k+1],axis=1)
            force=d['W'].mean(1)[:,:3].astype(float);target=d['support_target_world'][:3]
            forceerr=np.linalg.norm(force-target,axis=1)
            row=dict(k=k,source_s=k*.02,feasible_candidates=len(ids),selected=index,
                feasible_COM_envelope_diagonal_m=float(np.linalg.norm(np.ptp(p[ids],axis=0))),
                feasible_velocity_envelope_diagonal_m_s=float(np.linalg.norm(np.ptp(v[ids],axis=0))),
                selected_terminal_COM_error_m=float(poserr[index]),best_sampled_terminal_COM_error_m=float(poserr[ids].min()),
                selected_target_force_error_N=float(forceerr[index]),best_sampled_target_force_error_N=float(forceerr[ids].min()),
                target_force_norm_N=float(np.linalg.norm(target)))
            # Endpoint velocity exists iff its original reference derivative is valid.
            valid=k+1<len(ref['velocity_world'])
            row['endpoint_velocity_valid']=valid
            if valid:
                verr=np.linalg.norm(v-ref['velocity_world'][k+1],axis=1)
                # v scale = original COM tolerance / native prediction horizon.
                alternative=d['score_total'].astype(float)+(verr/.25)**2
                j=int(ids[np.argmin(alternative[ids])])
                row.update(selected_terminal_velocity_error_m_s=float(verr[index]),best_sampled_terminal_velocity_error_m_s=float(verr[ids].min()),
                    diagnostic_velocity_choice=j,diagnostic_choice_differs=j!=index,
                    diagnostic_terminal_velocity_error_m_s=float(verr[j]),diagnostic_terminal_COM_error_m=float(poserr[j]))
            rows.append(row)
        cases.append(dict(mode=c['name'],dt=c['dt'],rows=rows))
    a.output.mkdir();result=dict(schema='native_pen_saved_candidate_authority:1',run=file_pin(a.report),review=file_pin(a.review),source=file_pin(__file__),cases=cases,
        diagnostic_velocity_scale_m_s=.25,diagnostic_velocity_weight=1.,native_integrations=0,neural_updates=0,
        scope='All saved feasible57-candidate endpoint envelopes and independently rescored velocity diagnostic. Sampled minima are not global reachability/controllability bounds; rescoring is not a new executed controller or proven gain. Current failed replay/task gates remain.')
    write(a.output/'report.json',result)
    print(json.dumps(dict(report=file_pin(a.output/'report.json'),cases=[dict(mode=c['mode'],dt=c['dt'],decisions=len(c['rows']),
        velocity_choices_changed=sum(d.get('diagnostic_choice_differs',False) for d in c['rows']),
        early_choices_changed=sum(d.get('diagnostic_choice_differs',False) for d in c['rows'][:5]),
        first=c['rows'][0],last=c['rows'][-1]) for c in cases])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
