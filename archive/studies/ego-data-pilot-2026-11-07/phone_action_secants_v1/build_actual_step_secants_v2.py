"""All112 finite secants at actual recorded increments; nominal subset kept separate."""
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
    prec = report['outputs']['pairs112.json'];assert pin(prec['path']) == prec
    pairs = json.loads(Path(prec['path']).read_text())
    operand = report['outputs']['secants112_F64.npz'];assert pin(operand['path']) == operand
    with np.load(operand['path'],allow_pickle=False) as z:
        response = z['finite_response52_F64'].copy();du = z['actual_du_rad_F64'].copy()
        nominal = z['defined_mask'].copy()
        base_d = z['baseline_physical_delta52_F64'].copy();fd_d = z['FD_physical_delta52_F64'].copy()
    matching = np.array([p['FD_sample_id'] is not None and all(v for k,v in p['guards'].items()
         if k != 'selected_coordinate_is_literal_base_plus_signed_0p01') for p in pairs],dtype=bool)
    assert len(pairs) == 112 and matching.all() and np.isfinite(du).all() and (du != 0).all()
    assert response.shape == (112,52) and np.array_equal(response,fd_d-base_d)
    assert np.array_equal(du,np.array([p['declared_original_signed_increment_rad'] for p in pairs]))
    secants = response/du[:,None]
    scalar = np.array([[float(response[i,j])/float(du[i]) for j in range(52)] for i in range(112)])
    assert np.isfinite(secants).all() and np.array_equal(secants,scalar)
    out = R/'actual_step_secants_v2';out.mkdir()
    destination = out/'actual_step_secants112_F64.npz'
    np.savez_compressed(destination,
        baseline_physical_delta52_F64=base_d,FD_physical_delta52_F64=fd_d,
        finite_response52_F64=response,actual_du_rad_F64=du,secant52_per_actual_rad_F64=secants,
        exact_captured_state_and_nonj_control_match_mask=matching,
        original_literal_signed_0p01_mask=nominal,source_bounded_smaller_step_mask=~nominal,
        pair_id=np.arange(112,dtype=np.int64),
        source_boundary_index=np.array([p['source_boundary_index'] for p in pairs],dtype=np.int64),
        GN_iteration=np.array([p['GN_iteration'] for p in pairs],dtype=np.int64),
        actuator_index=np.array([p['actuator_index'] for p in pairs],dtype=np.int64),
        baseline_sample_id=np.array([p['baseline_sample_id'] for p in pairs],dtype=np.int64),
        FD_sample_id=np.array([p['FD_sample_id'] for p in pairs],dtype=np.int64))
    with np.load(destination,allow_pickle=False) as z:
        readback = z['secant52_per_actual_rad_F64'].copy()
        checks = {'all112_actual_steps_finite_nonzero_and_original_GN_exact':bool(np.isfinite(du).all() and (du!=0).all()),
                  'all112_complete_input_receipts_X58_and_nonj19_control_bits_pass':bool(matching.all()),
                  'all5824_scalar_divisions_equal_F64_array_divisions':bool(np.array_equal(secants,scalar)),
                  'all5824_saved_secant_components_bit_equal_readback':secants.tobytes()==readback.tobytes(),
                  'all112_actual_denominators_bit_equal_readback':du.tobytes()==z['actual_du_rad_F64'].tobytes(),
                  'original110_nominal_mask_unchanged':bool(np.array_equal(z['original_literal_signed_0p01_mask'],nominal) and nominal.sum()==110),
                  'all112_source_pair_ids_and_rows_retained':bool(np.array_equal(z['pair_id'],np.arange(112)) and len(z['FD_sample_id'])==112)}
    assert all(checks.values())
    links = [{'pair_id':p['pair_id'],'source_boundary_index':p['source_boundary_index'],
              'GN_iteration':p['GN_iteration'],'actuator_index':p['actuator_index'],
              'actuator_joint_name':p['actuator_joint_name'],
              'baseline_identity':p['baseline_identity'],'FD_identity':p['FD_identity'],
              'input_complete_integration_receipt':p['input_complete_integration_receipt'],
              'actual_denominator_rad':p['actual_denominator_rad'],
              'literal_signed_0p01_comparable':p['defined'],
              'actual_step_secant_defined':True} for p in pairs]
    save(out/'identities112.json',links)
    smaller = [{'pair_id':p['pair_id'],'boundary':p['source_boundary_index'],
        'GN_iteration':p['GN_iteration'],'actuator':p['actuator_index'],'actual_du_rad':p['actual_denominator_rad'],
        'source_declared_du_rad':p['declared_original_signed_increment_rad'],
        'reciprocal_step_per_rad':1/abs(p['actual_denominator_rad'])} for p in pairs if not p['defined']]
    save(out/'report.json',{'schema':'phone_H1_actual_step_action_secant_report:2',
        'status':'COMPLETE_ALL112_ACTUAL_DENOMINATOR_FINITE_ACTION_SECANTS',
        'source':pin(__file__),'parent_nominal_report':pin(PARENT),'parent_pairs':prec,
        'parent_raw_operands':operand,'parent_units_schema':report['outputs']['schema.json'],
        'outputs':{'arrays':pin(destination),'identities':pin(out/'identities112.json')},
        'counts':{'all_state_and_control_matched':112,'literal_signed_0p01':int(nominal.sum()),
                  'source_bounded_smaller_steps':len(smaller),'components_checked':112*52,
                  'saved_NPZ_reads':2,'new_model_FK_Log_solver_physics_NN_training_GPU':0},
        'checks':checks,'passed':sum(checks.values()),'total':len(checks),
        'smaller_steps':smaller,'parent_files_modified':False,
        'nominal_mask_semantics':'False denotes nominal0.01rad comparability only; the two bounded smaller nonzero increments remain mathematically defined captured-state finite secants.',
        'conditioning':'Step4.770366794443204e-5rad divides any response rounding/noise by about20962.75/rad. This is a conditioning caution, not a measured noise floor, accuracy gate or rejection. Report step sizes separately.',
        'formula':'(saved physical_world_delta52_FD - saved physical_world_delta52_baseline)/actual_signed_du_rad; rotation components share the common input world Log chart. No new Log, R projection or native call.',
        'coverage':'First8 actuators of20 only;14 finite directions each from7 consecutive boundaries and2 original GN bases. No20D coverage or fullJacobian.',
        'scope':'Author saved arithmetic/readback, not independent investigator review. Finite simulator-planning action secants from one failed-hold lineage, not infinitesimal/force/contact gradients, independent episodes, heldout ego truth or learned/holding qualification.'})
    print(json.dumps(pin(out/'report.json')))

if __name__ == '__main__':main()
