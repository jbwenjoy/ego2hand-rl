"""Descriptive full-declaration summaries;no ranking or independent-seed inference."""
from collections import Counter
import numpy as np


def finite_stats(values):
    a=np.asarray(values,np.float64)
    if len(a)==0:return dict(n=0,min=None,median=None,mean=None,max=None)
    if not np.isfinite(a).all():raise ValueError('Finite recorded outcomes required')
    return dict(n=len(a),min=float(a.min()),median=float(np.median(a)),mean=float(a.mean()),max=float(a.max()))


def case_metrics(case,review):
    if review['passed']!=review['total'] or not all(review['checks'].values()) or review['summary']['name']!=case['name']:raise ValueError('Complete exact case review required')
    packets={p['name']:p for p in review['packets']}
    if len(packets)!=case['native_queries']:raise ValueError('Unique reviewed query count differs')
    pairs=[];invalid=Counter();base_invalid=alt_invalid=0
    for node in review['nodes']:
        stem=case['name']+'_P'+str(node['phase']).zfill(3);base=packets[stem+'_BASE'];alt=packets.get(stem+'_ALT')
        for p,label in [(base,'BASE'),(alt,'ALT')]:
            if p is None or p['physical']['valid']:continue
            if label=='BASE':base_invalid+=1
            else:alt_invalid+=1
            reasons=[k for k,v in p['physical']['checks'].items() if not v]
            if not p['physical']['geometry_legal']:reasons.append('GEOMETRY_GATE')
            if not p['physical']['online_quality']:reasons.append('ONLINE_QUALITY')
            if not reasons:reasons=['PACKET_QUALITY_OTHER']
            for reason in reasons:invalid[reason]+=1
        if alt is not None:
            pairs.append(dict(phase=node['phase'],gain=alt['reward_value']-base['reward_value'],positive_margin=alt['reward_value']>base['reward_value']+.0001,both_physical=base['physical']['valid'] and alt['physical']['valid'],candidate_nonfailure=not alt['terminated'],accepted=node['accepted'],duplicate_BASE=next(p['fallback'] is not None for p in review['proposals'] if p['phase']==node['phase'])))
    eligible=[p for p in pairs if p['both_physical'] and p['candidate_nonfailure']];fallbacks=Counter(p['fallback'] for p in review['proposals'] if p['fallback'] is not None);s=review['summary']
    accepted=sum(p['accepted'] for p in pairs)
    if accepted!=s['accepted_proposals'] or any(p['accepted']!=(p['both_physical'] and p['candidate_nonfailure'] and p['positive_margin']) for p in pairs):raise ValueError('Predeclared native acceptance reconstruction differs')
    return dict(s,case_wall_s=case['wall_s'],actor_wall_s=case['actor_wall_s'],proposal_wall_s=case['proposal_wall_s'],restore_wall_s=sum(r['wall_s'] for r in case['restores']),capture_wall_s=sum(r['wall_s'] for r in case['captures']),native_integrations=8*s['native_queries'],source_capture_count=len(case['captures']),restore_count=len(case['restores']),
        invalid_BASE_queries=base_invalid,invalid_ALT_queries=alt_invalid,invalid_reasons_nonexclusive=dict(invalid),proposals_computed=len(review['proposals']),fallback_counts=dict(fallbacks),paired_queries=len(pairs),physically_valid_nonfailure_pairs=len(eligible),accepted_duplicate_BASE=sum(p['accepted'] and p['duplicate_BASE'] for p in pairs),
        all_queried_pair_gain=finite_stats([p['gain'] for p in pairs]),eligible_pair_gain=finite_stats([p['gain'] for p in eligible]),accepted_pair_gain=finite_stats([p['gain'] for p in pairs if p['accepted']]))


def aggregate(declared,metrics):
    if len(declared)!=54 or len({d['name'] for d in declared})!=54:raise ValueError('All54 declared cases required')
    observed={m['name']:m for m in metrics}
    if len(observed)!=len(metrics) or any(n not in {d['name'] for d in declared} for n in observed):raise ValueError('Unique declared reviewed outcomes required')
    groups=[];seen=set();paired=[]
    for d in declared:
        key=tuple(d[k] for k in ['teacher','controller','model','score_mode'])
        if key in seen:continue
        seen.add(key);planned=[p for p in declared if tuple(p[k] for k in ['teacher','controller','model','score_mode'])==key];rows=[observed[p['name']] for p in planned if p['name'] in observed]
        groups.append(dict(zip(['teacher','controller','model','score_mode'],key),declared_repetitions=3,reviewed_repetitions=len(rows),unreviewed_case_names=[p['name'] for p in planned if p['name'] not in observed],full_task_successes=sum(r['full_task_success'] for r in rows),full_task_success_declared_denominator=3,
            stop_counts=dict(Counter(r['stop_reason'] for r in rows)),executed_selected_duration_s=[observed[p['name']]['executed_selected_duration_s'] if p['name'] in observed else None for p in planned],valid_prefix_s=[observed[p['name']]['valid_selected_prefix_s'] if p['name'] in observed else None for p in planned],native_integrations=sum(r['native_integrations'] for r in rows),native_queries=sum(r['native_queries'] for r in rows),accepted_proposals=sum(r['accepted_proposals'] for r in rows),accepted_duplicate_BASE=sum(r['accepted_duplicate_BASE'] for r in rows),observed_case_wall_s=sum(r['case_wall_s'] for r in rows)))
    for m in metrics:
        if m['controller']!='WM':continue
        for control in ['UNMODIFIED_ACTOR','NATIVE_SHAM','NATIVE_RANDOM']:
            target=next(d for d in declared if d['teacher']==m['teacher'] and d['repetition']==m['repetition'] and d['controller']==control)
            if target['name'] not in observed:continue
            b=observed[target['name']];paired.append(dict(case=m['name'],control=target['name'],executed_duration_difference_s=m['executed_selected_duration_s']-b['executed_selected_duration_s'],valid_prefix_difference_s=m['valid_selected_prefix_s']-b['valid_selected_prefix_s'],case_stop=m['stop_reason'],control_stop=b['stop_reason'],both_full_task_success=m['full_task_success'] and b['full_task_success'],scope='Descriptive paired recorded stop times;physical stopping and task failure remain distinct,no retention/generalization causal claim.'))
    return dict(declared_cases=54,reviewed_cases=len(metrics),unreviewed_case_names=[d['name'] for d in declared if d['name'] not in observed],groups=groups,paired_descriptions=paired,full_task_successes=sum(m['full_task_success'] for m in metrics),full_task_success_declared_denominator=54,
        stop_counts=dict(Counter(m['stop_reason'] for m in metrics)),native_queries=sum(m['native_queries'] for m in metrics),native_integrations=sum(m['native_integrations'] for m in metrics),observed_case_wall_s=sum(m['case_wall_s'] for m in metrics),
        timing_scope='Observed case times include the registered audit/storage pipeline and concurrent review/graphics activity. Categories overlap;no isolated speed benchmark or total adaptation acceleration claim.',scope='Frozen models/single controlled reference,three numerical repetitions,not independent training seeds. All54 denominators retained;no best-case selection,ranking,confidence interval or ego/hardware inference.')
