"""Retain the two source-bounded smaller FD steps without changing nominal subset."""
from pathlib import Path
import hashlib
import json
import numpy as np

R = Path(__file__).resolve().parent
PARENT = R/'dataset_v1/report.json'
SHA = '5faa9e251f30b32d3a7e1132d091440054b13d274d629f8005e6d5aa7f685efe'

def pin(p):
    p = Path(p).resolve();b = p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def save(p, v):
    with p.open('x',encoding='utf8') as f:
        json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

def main():
    assert pin(PARENT)['sha256'] == SHA
    report = json.loads(PARENT.read_text())
    p = report['outputs']['pairs112.json'];assert pin(p['path']) == p
    pairs = json.loads(Path(p['path']).read_text())
    operand = report['outputs']['secants112_F64.npz'];assert pin(operand['path']) == operand
    with np.load(operand['path'],allow_pickle=False) as z:
        response = z['finite_response52_F64'].copy();du = z['actual_du_rad_F64'].copy()
        nominal = z['defined_mask'].copy()
    matching = np.array([p['FD_sample_id'] is not None and all(v for k,v in p['guards'].items()
         if k != 'selected_coordinate_is_literal_base_plus_signed_0p01') for p in pairs],dtype=bool)
    assert len(pairs) == 112 and matching.all() and np.isfinite(du).all() and (du != 0).all()
    secants = response/du[:,None]
    assert np.isfinite(secants).all()
    out = R/'bounded_step_supplement_v1';out.mkdir()
    np.savez_compressed(out/'all112_actual_signed_step_secants_F64.npz',
        finite_response52_F64=response,actual_du_rad_F64=du,secant52_per_actual_rad_F64=secants,
        exact_captured_state_and_nonj_control_match_mask=matching,
        original_literal_signed_0p01_mask=nominal,source_bounded_smaller_step_mask=~nominal)
    smaller = [{'pair_id':p['pair_id'],'boundary':p['source_boundary_index'],
        'GN_iteration':p['GN_iteration'],'actuator':p['actuator_index'],'actual_du_rad':p['actual_denominator_rad'],
        'source_declared_du_rad':p['declared_original_signed_increment_rad']} for p in pairs if not p['defined']]
    save(out/'report.json',{'schema':'phone_H1_actual_bounded_step_secant_supplement:1',
        'status':'COMPLETE_ALL112_CAPTURED_STATE_MATCHED_ACTUAL_DENOMINATORS_SEPARATE_STEP_GROUPS',
        'source':pin(__file__),'parent_nominal_report':pin(PARENT),'parent_pairs':p if False else report['outputs']['pairs112.json'],
        'parent_raw_operands':operand,'output':pin(out/'all112_actual_signed_step_secants_F64.npz'),
        'counts':{'all_state_and_control_matched':112,'literal_signed_0p01':int(nominal.sum()),
                  'source_bounded_smaller_steps':len(smaller),'new_saved_NPZ_reads':1,'new_model_FK_Log_solver_physics_NN_training_GPU':0},
        'smaller_steps':smaller,'parent_files_modified':False,
        'scope':'The smaller source-recorded steps are genuine finite secants at their actual denominators, not .01rad samples. Keep step groups visible. Same physical world-delta52 chart/units as parent; no full Jacobian, infinitesimal/force gradient or independent episodes.'})
    print(json.dumps(pin(out/'report.json')))

if __name__ == '__main__':main()
