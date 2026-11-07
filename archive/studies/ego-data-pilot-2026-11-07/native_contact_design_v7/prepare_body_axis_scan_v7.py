"""Freeze three finite reference-relocation hypotheses; zero native calls."""
from pathlib import Path
import json
from scan_body_axis_reference_v7 import pin,write

r=Path(__file__).resolve().parent;p=r.parent
j=json.loads((p/'native_equilibrium_v6/completed_v1/report.json').read_text(encoding='utf8'))
c=next(s for s in j['seed_results'] if s['seed']==0)['shortlist'][0]
b=json.loads((p/'hand2_holding_v1/compiled_nominal_v2/binding.json').read_text(encoding='utf8'))
record={'schema':'fixed_wrist_bodyZ_contact_relocation_protocol:7','CPU_run_authorized':True,
 'training_GPU_or_steps_authorized':False,'bodyZ_offsets_m':[0.,.0245,.049],
 'phone_rotation_and_robot_q20':'Original v6seed0rank0 unchanged',
 'hypothesis':'Current native pad contacts are near phone-body z49mm; move phone along +bodyZ by0/half/full49mm to change COM lever while reporting human material-location error. Finite geometry hypotheses, no faithful ego success claim.',
 'source_pins':{'runner':pin(r/'scan_body_axis_reference_v7.py'),
   'native_wrapper':pin(r/'evaluator_contact_design_v7.py'),'LP':pin(r/'native_wrench_lp_v7.py')},
 'inputs':{'native_J_contract':pin(r/'contract_v1/report.json'),
   'analytic_LP_contract':pin(r/'analytic_contract_v1/report.json'),
   'native_search_report':pin(p/'native_equilibrium_v6/completed_v1/report.json'),
   'initial_state':c['state'],'model_mjb':b['model'],
   'fixed_human_material_targets':pin(p/'retarget_geometry_v2/fixed_anchors_and_targets.json')},
 'output_path':str(r/'body_axis_scan_v1'),
 'budget':{'initial_forward_max':3,'additional_inverse_control_forward_max':3,
   'independent_final_geometries_max':3,'LP_max':3,'steps_training_GPU':0},
 'geometry':{'phone_pairs':22,'self_pairs':189,'pad_gap_window_m':[-.0001,.0002],
    'nonpad_self_buffer_m':0.,'unknown_fails':True},
 'ideal_force_moment_gates':{'force_infinity_N':1e-8,'moment_infinity_Nm':1e-4,
   'normal_cap_per_finger_N':3.,'true_unsaturated_inverse_PD_and_force_bounds':True},
 'automatic_retry':False,'holding_or_policy_or_ego_qualification':False}
write(r/'body_axis_scan_protocol_v1.json',record)
print(json.dumps({'protocol':pin(r/'body_axis_scan_protocol_v1.json'),'runner':record['source_pins']['runner']}))
