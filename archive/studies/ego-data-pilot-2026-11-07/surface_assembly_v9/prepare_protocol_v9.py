"""Freeze staged assembly/support exploration, with no native execution."""
from pathlib import Path
from solve_assembly_support_v9 import CFG,pin,read,write

r=Path(__file__).resolve().parent;p=r.parent
b=read(p/'hand2_holding_v1/compiled_nominal_v2/binding.json')
protocol={'schema':'assembly_then_hard_geometry_support_protocol:9','CPU_run_authorized':True,
 'config':CFG,'output_path':str(r/'completed_v1'),'automatic_retry':False,
 'source_pins':{'runner':pin(r/'solve_assembly_support_v9.py'),'cheap_FK':pin(r/'assembly_kinematics_v9.py'),
  'native_wrapper':pin(p/'native_contact_design_v7/evaluator_contact_design_v7.py'),
  'native_base':pin(p/'native_equilibrium_v6/evaluator_native_v6.py'),
  'static_geometry':pin(p/'retarget_gravity_v3/evaluator_v3.py'),
  'scaled_support_LP':pin(p/'wrench_residual_v6/scaled_wrench_residual_lp_v6.py'),
  'native_wrench_LP':pin(p/'native_contact_design_v7/native_wrench_lp_v7.py'),
  'strict_native_support_LP':pin(p/'native_contact_design_v7/strict_support_lp_guarded_v7.py'),
  'native_LP_adapter':pin(p/'native_contact_design_v7/scan_body_axis_reference_v7.py')},
 'inputs':{'old_native_report':pin(p/'native_equilibrium_v6/completed_v1/report.json'),
  'surface_anchors':pin(p/'contact_patch_v7/paired_surface_anchors_v1.json'),
  'reference_design':pin(p/'reference_seed_design_v7/report.json'),
  'cheap_FK_contract':pin(r/'original_pose_contract_v1/report.json'),
  'native_J_contract':pin(p/'native_contact_design_v7/contract_v1/report.json'),
  'native_LP_contract':pin(p/'native_contact_design_v7/analytic_contract_v1/report.json'),
  'original_ego_material_targets':pin(p/'retarget_geometry_v2/fixed_anchors_and_targets.json'),
  'model_binding':pin(p/'hand2_holding_v1/compiled_nominal_v2/binding.json'),'model_mjb':b['model']},
 'A0':'Cheap FK10eq+opposite branch and weak pose regularizer; no force or collision qualification. On budget stop use lowest finite visited residual including FD, with saved last requested point.',
 'A1':'SLSQP10eq plus opposite branch and full collision-window constraints; objective stay close to A0. Pool/ALREADY_PASSED additionally require max(abs(eq10_scaled))<=1e-5, not only coarse closure/angle gate.',
 'B':'SLSQP under the same10eq+collision constraints. Objective conditional LP scaled_Linf squared plus1e-7 normalized pose norm; pool ranks primary scaled_Linf. Common A is real hand-triangle point, conditionally inside phone preload, not actual native con.pos.',
 'collision_inequalities':'Designated2pad signed gaps in[-0.1,+0.2]mm; other20phone pairs and189eligible self pairs nonpenetrating. Nonnegative constraint values do not mean designated pads have positive signed gap.',
 'opposed_normal_equations':'Each actual phone triangle provides two fixed body tangent rows rotated to world; tangent dot hand outward normal gives2eq perpair, with opposite-normal branch.',
 'model_and_reference':'Original fixed wrist, geometry, scale, gains, mass, inertia, friction and solver. Remaining12q and phone material points fixed; original ego targets separately retained. No adjacent-face substitution in this run.',
 'budget_scope':'Stage wall limits are cooperative checks before cache-miss/full queries; cached and cheap work plus finite library iterations/load/save overhead are not an absolute process deadline. Attempts are flushed with requestedx before full geometry; no restart on observation timeout.',
 'final':'Up to2candidates/seed, original complete22/189 geometry and newly generated native multipt/J/friction/PD. Minimax guidance followed by strict force-hard/moment-box witness if geometry and assembly pass; then2fresh command probes/repeat. No integration here.',
 'initial_load_budget':'Ideal per-finger3N remains an LP design cap. Actual initial Fn and its3N flag are descriptive for this new diagnostic enrollment; original2s dynamic gates have no instantaneous3N cap. Do not claim actual load meets3N unless measured flag does.',
 'qualification':'Candidate assembly/ideal support/repeated preload do not establish calibrated physical pad, faithful ego trajectory, learned benefit, native equilibrium or2s holding.',
 'new_training_GPU_or_dynamics_steps_authorized':False}
write(r/'protocol_v1.json',protocol)
print({'protocol':pin(r/'protocol_v1.json'),'source':protocol['source_pins']['runner']})
