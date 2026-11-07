"""Two source/ABI fixtures only; never reads actual task-gradient output."""
import ast
import csv
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'scripts/audit_component_task_gradient_saved_cpu_v1.py'
OUT=ROOT/'outputs/full151105_component_task_gradient_saved_adapter_preparation_v1'


def main():
    OUT.mkdir(exist_ok=False);ast.parse(SRC.read_bytes())
    spec=importlib.util.spec_from_file_location('saved_gradient_adapter',SRC);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    sources={'auditor':m.pin(SRC),'checker':m.pin(__file__),'independent_math':m.pin(m.MATH_PATH),'prior_independent_helper':m.pin(m.BASE_PATH)}
    for key,path in [('auditor',SRC),('checker',Path(__file__)),('math',m.MATH_PATH)]: (OUT/(key+'_before_execution.py')).write_bytes(path.read_bytes())
    m.dump(OUT/'source_freeze.json',{'source_pins':sources,'actual_taskscore_or_gradient_values_read':False})
    math=m.module(m.MATH_PATH,m.MATH_SHA);x=np.zeros((2,58),dtype='f4');x[:,43:52]=np.eye(3,dtype='f4').reshape(1,9)
    p=np.zeros((2,3),dtype='f4');R=np.repeat(np.eye(3,dtype='f4')[None],2,0)
    checks={'samepost20_fixedgoal_closedform_2_and_8_no_native_import':m.bits(m.score_terms64(x,p,R),np.array([[2.,8.],[2.,8.]],dtype='f8'))}
    _,pairs=m.BASE.symbolic_order();u=np.zeros((152,20),dtype='f4');scores=np.ones(152,dtype='f4')
    for row in pairs:
        q=row['joint_index'];eps=np.float32(row['requested_epsilon_rad']);mi,pl=row['minus_branch_id'],row['plus_branch_id']
        u[mi,q]=-eps;u[pl,q]=eps;scores[mi]=np.float32(1)-np.float32(4)*eps;scores[pl]=np.float32(1)+np.float32(4)*eps
    a={'branch_u':u,'branch_object_score_k':scores,'donor_object_score_k':np.ones(8,dtype='f4'),
      'branch_object_terms_k':np.column_stack((scores,np.zeros(152,dtype='f4'))),'branch_object_terms_k_CPU_secondary':np.zeros((152,2),dtype='f4'),
      'direction__center_offset':np.zeros((8,4,2,20),dtype='f8')}
    groups,rows=m.physical_rows(a,pairs,math)
    table=OUT/'synthetic_all64.csv';fields=list(rows[0])
    with table.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
    wanted_order=[(s,j,ei) for s in range(8) for j in range(4) for ei in range(2)]
    wrong=rows.copy();wrong[0],wrong[2]=wrong[2],wrong[0]
    broken={k:v.copy() for k,v in a.items()};broken['branch_u'][pairs[0]['plus_branch_id']]=broken['branch_u'][pairs[0]['minus_branch_id']]
    try:m.physical_rows(broken,pairs,math);zeroaccepted=True
    except ValueError:zeroaccepted=False
    checks['all32_64_symbolicorder_CSV_types_swappedrows_and_zeroactualden_block']=len(groups)==32 and len(rows)==64 and [(r['snapshot_id'],r['j'],r['epsilon_index']) for r in rows]==wanted_order and all(r['actual_coordinate_defined'] for r in rows) and m.csv_matches(table,rows) and not m.csv_matches(table,wrong) and not zeroaccepted
    checks={k:bool(v) for k,v in checks.items()}
    require_no_backend=not any(k.split('.')[0] in ('torch','warp','mujoco','mujoco_warp','mjlab','dex_supermimic') for k in sys.modules)
    if not require_no_backend:raise ValueError('Saved-only NumPy fixture imported backend')
    report={'schema':'full151105_component_task_gradient_saved_adapter_CPU_proof:1','source_pins':sources,'passed':sum(checks.values()),'total':len(checks),'checks':checks,
      'actual_taskscore_or_gradient_values_read':0,'model_FW_AD_Env_physics_GPU':0,'backend_imports':False}
    m.dump(OUT/'report.json',report);ready={'source_pins':sources,'cpu_report':m.pin(OUT/'report.json'),'cpu_checks':{'passed':report['passed'],'total':report['total']},
      'api':{'entry':str(SRC)},'actual_saved_audit_completed':False}
    m.dump(OUT/'READY.json',ready);print(json.dumps({'READY':m.pin(OUT/'READY.json'),**ready},indent=2));return 0 if all(checks.values()) else 2


if __name__=='__main__':raise SystemExit(main())
