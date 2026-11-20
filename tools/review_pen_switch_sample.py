"""Independent contact-switch audit on source-only stratified selections and native replays."""
from collections import Counter
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.mechanics.pen_capacity_trace import packet_capacity
from supermimic_research.mechanics.controlled_pen_records import pack_case
from supermimic_research.mechanics.warp_packet_decode import decode_saved_case
from supermimic_research.strict_json import write_json


def read(pin):
    return json.loads(verify_file_pin(pin).read_bytes())


def compare(a, b):
    """Compare a baseline with an alternative for the same tick."""
    before, after = set(a), set(b)
    return dict(add=sorted(after - before), remove=sorted(before - after),
                counts_changed=sorted(k for k in (before & after) if a[k] != b[k]))


def changes(left, right):
    if len(left) != len(right):
        raise ValueError('Missing comparison tick')
    details = [compare(a,b) for a,b in zip(left,right)]
    membership = [i for i,d in enumerate(details) if d['add'] or d['remove']]
    multiplicity = [i for i,d in enumerate(details) if d['counts_changed']]
    return dict(add_or_remove=bool(membership), ticks_membership=membership,
                any_count_change=bool(multiplicity), ticks_shared_count_change=multiplicity,
                multiplicity_only=bool(multiplicity and not membership),
                ticks=[dict(index=i,**d) for i,d in enumerate(details) if any(d.values())])


def topology(z, m, body):
    ticks = [Counter() for _ in range(8)]
    for row in z['raw_contact_tick_plus23']:
        if row[5] > 0:
            continue
        tick,g,h = map(int,row[:3])
        on = m.geom_bodyid[[g,h]] == body
        if on.sum() == 1:
            obj,hand = (g,h) if on[0] else (h,g)
            if (m.geom(hand).name or '').startswith('robot/'):
                ticks[tick][obj,hand] += 1
    return ticks


def screen_times(raw):
    if len(raw) != 8:
        raise ValueError('Expected eight pre-step contact lists')
    return [Counter(tuple(map(int,p)) for p in tick) for tick in raw]


def summarize(rows):
    admitted = [r for r in rows if r['admitted']]
    noisy = [r for r in admitted if r['sham_topology']['add_or_remove']]
    switches = [r for r in admitted if r['response_topology']['add_or_remove']]
    return dict(selected=len(rows), admitted=len(admitted), excluded=len(rows)-len(admitted),
                pulse_membership=len(switches),pulse_d23_membership=sum(r['response_d23']['add_or_remove'] for r in admitted),
                pulse_membership_without_sham_membership=sum(not r['sham_topology']['add_or_remove'] for r in switches),
                pulse_multiplicity_only=sum(r['response_topology']['multiplicity_only'] for r in admitted),
                sham_membership=len(noisy),
                pulse_capacity_above_sham=sum(r['capacity_l2']>r['sham_l2'] for r in admitted),
                natural_sham_membership=sum(r['sham_temporal']['add_or_remove'] for r in admitted),
                median_capacity_l2=float(np.median([r['capacity_l2'] for r in admitted])) if admitted else None)


def stratified_candidates(candidates, selected, uniform_phases, first_phase, width, n_bins, hand_geoms):
    """The full nominal stream is observed, whereas only selected pulse branches exist."""
    if len(uniform_phases)!=n_bins:raise ValueError('Incomplete uniform draw')
    by_phase = {s['phase']:s for s in selected}
    if len(by_phase) != len(selected):
        raise ValueError('Duplicate selected phase for this actor')
    labels=[]
    for c in candidates:
        ticks=screen_times(c['nominal_ticks'])
        event=changes(ticks[:-1],ticks[1:])
        phase=c['phase']
        if c['bin']!=(phase-first_phase)//width or not first_phase<=phase<first_phase+n_bins*width:
            raise ValueError('Candidate outside declared source stratum')
        scoped=[Counter({k:n for k,n in t.items() if k[1] in hand_geoms}) for t in ticks]
        scoped_event=changes(scoped[:-1],scoped[1:])
        labels.append(dict(phase=phase,bin=c['bin'],score=c['score'],
                           any_membership=event['add_or_remove'],d23_membership=scoped_event['add_or_remove'],multiplicity_only=event['multiplicity_only'],
                           first_pre_time_ms=(min(event['ticks_membership'])+1)*2.5 if event['ticks_membership'] else None,
                           risk=bool(by_phase.get(phase,{}).get('risk_selected')),
                           uniform=bool(by_phase.get(phase,{}).get('uniform_selected'))))
    if len({c['phase'] for c in labels})!=len(labels) or not by_phase.keys()<={c['phase'] for c in labels}:
        raise ValueError('Invalid phase universe')
    strata=[]
    for index in sorted({c['bin'] for c in labels}):
        bins=[c for c in labels if c['bin']==index]
        risk=[c for c in bins if c['risk']];uniform=[c for c in bins if c['uniform']]
        if len(bins)<width and not risk and not uniform:
            continue
        if len(bins)!=width or len(risk)!=1 or len(uniform)!=1:
            raise ValueError(f'Expected one selection per arm in observed bin {index}')
        if risk[0]['phase']!=max(bins,key=lambda c:c['score'])['phase'] or uniform[0]['phase']!=uniform_phases[index]:
            raise ValueError(f'Selected risk row is not a maximum in bin {index}')
        strata.append(dict(bin=index,observed=len(bins),membership=sum(c['any_membership'] for c in bins),
                           risk_phase=risk[0]['phase'],uniform_phase=uniform[0]['phase'],
                           risk_hit=risk[0]['any_membership'],uniform_hit=uniform[0]['any_membership'],
                           d23_membership=sum(c['d23_membership'] for c in bins),risk_d23=risk[0]['d23_membership'],uniform_d23=uniform[0]['d23_membership']))
    probabilities=[s['membership']/s['observed'] for s in strata]
    return dict(observed=len(labels),natural_membership=sum(c['any_membership'] for c in labels),
                natural_multiplicity_only=sum(c['multiplicity_only'] for c in labels),strata=strata,
                expected_uniform_membership=sum(probabilities),expected_uniform_d23=sum(s['d23_membership']/s['observed'] for s in strata),
                natural_d23=sum(c['d23_membership'] for c in labels),risk_d23_hits=sum(s['risk_d23'] for s in strata),uniform_d23_hits=sum(s['uniform_d23'] for s in strata),
                risk_hits=sum(s['risk_hit'] for s in strata),uniform_hits=sum(s['uniform_hit'] for s in strata),
                labels=labels)


def inspect(row,model,layout,physical,hand_geoms):
    results={};initial=None
    for branch in row['branches']:
        name=branch['name']
        if name in results:raise ValueError(f'Duplicate branch {name}')
        with np.load(verify_file_pin(branch['trace'])) as f:
            z=dict(f)
        now=(z['pre_qpos'][0], z['pre_qvel'][0], z['initial_warmstart'], z['initial_native_state'])
        if initial is None:initial=now
        else:
            for expected,actual in zip(initial,now):np.testing.assert_array_equal(expected,actual)
        if int(z['pre_action_phase'][0])!=row['phase']:raise ValueError('Source phase differs')
        pd=z['post_ctrl'][0]
        np.testing.assert_array_equal(z['post_ctrl'],np.tile(pd,(8,1)))
        if branch['valid'] and (np.any(z['overflow']) or np.any(z['post_qfrc_applied']) or np.any(z['post_xfrc_applied'])):
            raise ValueError('Marked valid despite overflow or assistance')
        if 'pd' in branch:np.testing.assert_array_equal(pd,branch['pd'])
        audit=packet_capacity(z,model,layout,physical['geom_friction'])
        np.testing.assert_allclose(branch['capacity'],audit['capacity'].mean(0),atol=1e-10,rtol=0)
        case=dict(model_layout=layout,dt=.0025); metrics,decoded=decode_saved_case(case,z,model,physical)
        body=layout['object_body']
        R=np.array(model.body_iquat[body])
        from scipy.spatial.transform import Rotation
        R=Rotation.from_quat(R[[1,2,3,0]]).as_matrix()
        physics=dict(mass=float(physical['body_mass'][0,body]),inertia_BODY=(R@np.diag(physical['body_inertia'][0,body])@R.T).tolist(),gravity=physical['gravity'][0].tolist())
        j=model.actuator_trnid[:,0]
        native=pack_case(model,case,z,decoded,model.jnt_qposadr[j],model.jnt_dofadr[j],np.arange(20),z['initial_warmstart'],physics)
        valid=bool(all(metrics['checks'].values()) and native['valid'][0] and np.all(native['geometry_flags'][0]<=[.002,.0002,.001]) and audit['maximum_containment_excess']<2e-5)
        if valid!=branch['valid']:raise ValueError('Save-only physical qualification differs')
        bounded=audit['maximum_containment_excess']<=2e-5 and audit['positive_gap_skipped_max_load_N']<=1e-5
        results[name]=dict(ticks=topology(z,model,layout['object_body']),pd=pd,cap=audit['capacity'].mean(0),
                           admitted=bool(branch['valid'] and not branch.get('terminal',False) and bounded))
    if set(results)!={'sham','minus','plus','sham_repeat'}:raise ValueError('Incomplete declared branches')
    m,p,b,repeat=(results[k] for k in ('minus','plus','sham','sham_repeat'))
    np.testing.assert_array_equal(b['pd'],repeat['pd'])
    pd_delta=float(np.linalg.norm(p['pd']-m['pd']))
    t=b['ticks']
    scoped=lambda branch:[Counter({key:n for key,n in tick.items() if key[1] in hand_geoms}) for tick in branch['ticks']]
    return dict(actor=row['actor'],phase=row['phase'],bin=row['bin'],risk_selected=row['risk_selected'],uniform_selected=row['uniform_selected'],
                admitted=all(x['admitted'] for x in results.values()) and pd_delta>1e-7,pd_delta=pd_delta,
                branches_admitted={k:v['admitted'] for k,v in results.items()},
                capacity_l2=float(np.linalg.norm(p['cap']-m['cap'])),sham_l2=float(np.linalg.norm(repeat['cap']-b['cap'])),
                response_topology=changes(m['ticks'],p['ticks']),response_d23=changes(scoped(m),scoped(p)),sham_topology=changes(b['ticks'],repeat['ticks']),
                sham_temporal=changes(t[:-1],t[1:]))


def matched_strata(rows):
    """Enrichment cannot be assessed on a one-arm-valid bin. Retain these exclusions."""
    result = []
    for actor,index in sorted({(r['actor'],r['bin']) for r in rows}):
        group=[r for r in rows if r['actor']==actor and r['bin']==index]
        risk=[r for r in group if r['risk_selected']]; uniform=[r for r in group if r['uniform_selected']]
        if len(risk)!=1 or len(uniform)!=1:raise ValueError('Invalid matched stratum')
        a,b=risk[0],uniform[0]
        result.append(dict(actor=actor,bin=index,risk_phase=a['phase'],uniform_phase=b['phase'],
                           overlap=a['phase']==b['phase'],both_admitted=a['admitted'] and b['admitted'],
                           risk_membership=a['response_topology']['add_or_remove'],uniform_membership=b['response_topology']['add_or_remove'],
                           risk_d23=a['response_d23']['add_or_remove'],uniform_d23=b['response_d23']['add_or_remove']))
    admitted=[b for b in result if b['both_admitted']]
    return dict(declared=len(result),both_admitted=len(admitted),same_phase=sum(b['overlap'] for b in admitted),
                risk_only_positive=sum(b['risk_membership'] and not b['uniform_membership'] for b in admitted),
                uniform_only_positive=sum(b['uniform_membership'] and not b['risk_membership'] for b in admitted),
                both_positive=sum(b['uniform_membership'] and b['risk_membership'] for b in admitted),
                neither_positive=sum(not (b['uniform_membership'] or b['risk_membership']) for b in admitted),rows=result)


def main(root,out):
    src=file_pin(root/'report.json'); report=read(src);rows=[];screens={}; per_actor={}
    intent=read(report['intent'])
    for actor,metadata in report['actors'].items():
        m=mujoco.MjModel.from_binary_path(str(verify_file_pin(metadata['model'])))
        with np.load(verify_file_pin(metadata['physical_context'])) as f:physical=dict(f)
        hand_geoms={k for k in range(m.ngeom) if (m.geom(k).name or '').startswith(('robot/right_finger2_','robot/right_finger3_'))}
        c=[r for r in report['rows'] if r['actor']==actor]
        rows_actor=[inspect(r,m,metadata['layout'],physical,hand_geoms) for r in c]
        scr=read(metadata['screen'])
        if scr['uniform']!=intent['uniform'][actor]:raise ValueError('Uniform draw changed')
        screens[actor]=stratified_candidates(scr['candidates'],rows_actor,scr['uniform'],intent['first_phase'],intent['stratum_width'],intent['strata'],hand_geoms)
        rows.extend(rows_actor)
        per_actor[actor]={k:summarize([r for r in rows_actor if r[k+'_selected']]) for k in ['risk','uniform']}
    if len(rows)!=len(report['rows']) or not all(r['risk_selected'] or r['uniform_selected'] for r in rows):
        raise ValueError('Unassigned or unselected row in study')
    for kinds in [['risk','uniform'],['risk_only','uniform_only','overlap']]:
        for k in kinds:
            within=lambda r: {'risk':r['risk_selected'],'uniform':r['uniform_selected'],
                                'risk_only':r['risk_selected'] and not r['uniform_selected'],
                                'uniform_only':r['uniform_selected'] and not r['risk_selected'],
                                'overlap':r['risk_selected'] and r['uniform_selected']}[k]
            per_actor.setdefault('pooled',{})[k]=summarize([r for r in rows if within(r)])
    strata=matched_strata(rows)
    document=dict(matched_strata=strata,source=src,reviewer=file_pin(Path(__file__)),summary=per_actor,candidate_coverage=screens,rows=rows,
                  convention='One sample is one actor and source phase, never a contact point or substep. Shared risk/uniform selections are counted in each arm and explicitly reported. Native valid, nonterminal, nondegenerate actual PD and four admitted branches required for paired numerators. Natural events in nominal prefix use seven changes between eight pre-step rows; not last post-step geometry. Topology membership canonicalized object first, hand second and gap <=0. Multiplicity-only means counts change with all pair IDs preserved. Strictly greater L2 mean-capacity contrast than the sham-repeat difference; two shams do not estimate a noise distribution. Population event prevalence applies only to nominal prefix, not unselected plus/minus pairs.')
    out.mkdir(exist_ok=False,parents=True)
    write_json(out/'report.json',document)
    print('pooled interventions',json.dumps({k:{n:v[n] for n in ('selected','admitted','pulse_membership','pulse_multiplicity_only','pulse_capacity_above_sham')} for k,v in per_actor['pooled'].items()}))
    print('nominal streams',json.dumps({k:{n:v[n] for n in ('observed','natural_membership','risk_hits','uniform_hits','expected_uniform_membership')} for k,v in screens.items()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('out',type=Path)
    args=parser.parse_args();main(args.run,args.out)
