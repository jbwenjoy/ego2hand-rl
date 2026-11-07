"""Minimal units API delta; preserve original23-proof module and source."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
import ast,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151.contact_row_oracle import geometry_rows as old
from full151.contact_row_oracle_v2 import geometry_rows as new

def run(out):
    if out.exists():raise FileExistsError(out)
    a=ROOT/'src/full151/contact_row_oracle.py';b=ROOT/'src/full151/contact_row_oracle_v2.py';s=a.read_text();t=b.read_text();normalized=t.replace(",'units':'m'",'')
    B=np.zeros((601,151));B[3,7]=.4;si=np.array([3]);d=np.array([-.0031]);J=np.ones((1,20))*.02
    x=old(B,si,d,J);y=new(B,si,d,J)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    checks={'old_executed_source92cc_preserved':sha(a)=='92cc3736251a8d408d43974dfa3669453cad0d303b2dc45d490edd6823dd932b',
            'only_units_field_text_delta':normalized==s,
            'only_units_field_AST_delta':ast.dump(ast.parse(normalized),include_attributes=False)==ast.dump(ast.parse(s),include_attributes=False),
            'top_level_units_m_for_globalbuilder':y['units']=='m' and y['column_order']=='joint-major:j*151+c',
            'native_units_numbers_and_CSR_unchanged':np.array_equal(x['rhs_m'],y['rhs_m']) and (x['A_step']!=y['A_step']).nnz==0,
            'no_native_or_WM_import':all(k not in sys.modules for k in ['mujoco','warp','torch'])}
    out.mkdir(parents=True,exist_ok=False);pin=lambda p:{'path':str(p),'sha256':sha(p)}
    result={'schema':'full151_contact_row_oracle_units_delta_CPU:1','status':'PASS' if all(checks.values()) else 'FAIL','passed':sum(checks.values()),'total':len(checks),'checks':checks,'module':pin(b),'checker':pin(Path(__file__).resolve()),'retained23math_proof':pin(ROOT/'outputs/contact_row_oracle_CPU_v1/report.json'),'real_query_solver_Env_WM_physics_GPU_calls':0}
    (out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.output);print(json.dumps(r,indent=2));sys.exit(0 if r['status']=='PASS' else 1)
