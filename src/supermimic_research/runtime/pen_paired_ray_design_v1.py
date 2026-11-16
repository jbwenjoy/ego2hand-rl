"""Complete two-algorithm/two-phase-scope schedule, with separate observation ID."""
import hashlib
import json
from pathlib import Path

from ..artifacts import file_pin,verify_file_pin
from .pen_termination_design import schedule as old_schedule

SCOPES=['SOLVE_EXIT_ONLY','SOLVE_EXIT_PLUS_RAY_TRACE']


def schedule(rows,replica):
    mapping=dict(PACKET_BASELINE=SCOPES[0],PACKET_PLUS_SOLVE_EXIT=SCOPES[1])
    result=[]
    for old in old_schedule(rows,replica):
        scope=mapping[old['scope']]
        result.append(dict(old,scope=scope,name='RAY_'+old['name'].replace(old['scope'],scope)))
    return result


def observation_identity(qualification_path):
    root=Path(__file__).resolve().parents[1]
    qpin=file_pin(qualification_path); q=json.loads(verify_file_pin(qpin).read_text())
    if q['passed']!=q['total'] or q['total']!=8 or not all(q['checks'].values()): raise ValueError('Exact CPU prototype qualification required')
    verify_file_pin(q['prototype_identity']['generated_manifest'])
    source=[root/'runtime'/n for n in ['pen_paired_ray_design_v1.py','pen_paired_ray_observer_v1.py','pen_ray_bindings_v1.py','pen_ray_post_kernels_v1.py']]
    source += [root/'mechanics'/n for n in ['pen_ray_trace_review_v1.py','pen_ray_reference_v1.py']]
    identity=dict(schema='paired_ray_observation_implementation:1',qualification=qpin,scopes=SCOPES,sources=[file_pin(p) for p in source],
        interpretation='Endpoint-only versus same endpoint plus aligned trace;neither assumes neutrality against unobserved physics.')
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(observation_id=digest,identity=identity)
