"""Compare pose and twist on identical valid time windows across proposal seeds."""
import argparse
from pathlib import Path
import numpy as np
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.mechanics.pen_task_contracts import tracking_series
from supermimic_research.strict_json import write_json
import json


def read(pin):
    return json.loads(verify_file_pin(pin).read_bytes())


def summarize(runs,out):
    if out.exists():
        raise FileExistsError(out)
    sources={s:file_pin(runs/s/'report.json') for s in ('pen_receding_support_v1','pen_wrench_permutation_v1','pen_receding_seeds_v1')}
    cases=[]
    for group,source in sources.items():
        result=read(source)
        if 'rows' in result:
            cases.extend(result['rows'])
        else:
            for clip,rows in result['clips'].items():
                for case in rows:
                    mode='shifted' if group=='pen_wrench_permutation_v1' else case['arm']
                    cases.append(dict(clip=clip,seed=112025,mode=mode,case=case))
    records=[]
    for trial in cases:
        meta=trial['case']
        report=read(meta['native_case'])
        ticks=8*min(r['case']['passing'] for r in cases if r['clip']==trial['clip'])
        assert ticks>0
        with np.load(verify_file_pin(report['trace'])) as trace:
            series=tracking_series(trace,.0025)
            if not series['twist_defined'][:ticks].all():
                raise ValueError('Undefined axial twist in comparison window')
            metrics={}
            for source,key,scale in [('COM_error_m','COM_mm',1000.),('full_angle_rad','SO3_deg',180/np.pi),
                                     ('shaft_angle_rad','shaft_deg',180/np.pi),('reference_axis_twist_rad','twist_deg',180/np.pi)]:
                metrics[key]=float(scale*np.sqrt(np.mean(series[source][:ticks]**2)))
        records.append(dict(clip=trial['clip'],seed=trial['seed'],mode=trial['mode'],rep=meta['rep'],
            valid_s=meta['passing_s'],common_s=ticks*.0025,rms=metrics,half=meta['passed_halfway'],
            complete=meta['complete'],strict=meta['strict'],n_controls=meta['controls'],
            ticks=meta['cpu_planning_ticks'],planning_wall_s=meta['planning_wall_s'],pd_error=meta['max_pd_error_rad']))
    assert len(records)==30
    write_json(out,dict(sources=sources,rows=records))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',type=Path)
    p.add_argument('out',type=Path)
    args=p.parse_args()
    summarize(args.runs,args.out)
