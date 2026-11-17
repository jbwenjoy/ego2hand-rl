"""Exact old scope order with independently identified original/cost algorithms."""
from .pen_solve_exit_design import schedule as scope_schedule
from .pen_stable_friction_profile_v2 import ALGORITHMS


def schedule(rows,replica):
    original=scope_schedule(rows,replica); algorithms=ALGORITHMS if replica=='A' else list(reversed(ALGORITHMS))
    result=[]
    for teacher in ['RESIDUAL','SUPPORT']:
        for algorithm in algorithms:
            for item in original:
                if item['teacher']==teacher: result.append(dict(item,algorithm=algorithm,name=algorithm+'_'+item['name']))
    if len(result)!=120 or len({r['name'] for r in result})!=120: raise ValueError('Complete120-query cost process schedule required')
    return result
