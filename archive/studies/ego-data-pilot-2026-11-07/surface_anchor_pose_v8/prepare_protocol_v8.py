"""Freeze a bounded surface-pair search without native model construction."""
from pathlib import Path
import json
from solve_surface_anchor_pose_v8 import CONFIG,pin,write

r=Path(__file__).resolve().parent;p=r.parent
binding=json.loads((p/'hand2_holding_v1/compiled_nominal_v2/binding.json').read_text(encoding='utf8'))
record={'schema':'fixed_surface_material_q8_phone6_search_protocol:8','CPU_run_authorized':True,
 'config':CONFIG,'output_path':str(r/'completed_v1'),'automatic_retry':False,
 'source_pins':{'runner':pin(r/'solve_surface_anchor_pose_v8.py'),
  'native_wrapper':pin(p/'native_contact_design_v7/evaluator_contact_design_v7.py'),
  'native_base':pin(p/'native_equilibrium_v6/evaluator_native_v6.py'),
  'static_geometry':pin(p/'retarget_gravity_v3/evaluator_v3.py'),
  'scaled_wrench_LP':pin(p/'wrench_residual_v6/scaled_wrench_residual_lp_v6.py'),
  'native_wrench_LP':pin(p/'native_contact_design_v7/native_wrench_lp_v7.py'),
  'strict_native_witness_LP':pin(p/'native_contact_design_v7/strict_support_lp_guarded_v7.py'),
  'native_LP_adapter':pin(p/'native_contact_design_v7/scan_body_axis_reference_v7.py')},
 'inputs':{'surface_anchors':pin(p/'contact_patch_v7/paired_surface_anchors_v1.json'),
  'native_search_report':pin(p/'native_equilibrium_v6/completed_v1/report.json'),
  'reference_design':pin(p/'reference_seed_design_v7/report.json'),
  'native_J_contract':pin(p/'native_contact_design_v7/contract_v1/report.json'),
  'native_LP_analytic_contract':pin(p/'native_contact_design_v7/analytic_contract_v1/report.json'),
  'model_binding':pin(p/'hand2_holding_v1/compiled_nominal_v2/binding.json'),'model_mjb':binding['model']},
 'surface_model':'Fixed projected points on actual filled triangles. hand points/normals in compiled geom frame; phone points/normals in body frame. Functional pad/finite pressure area UNKNOWN.',
 'conditional_force_model':'Object force applied at phone material B, hand reaction at fixed hand material A; closure target B−A=−25um*nH and normal opposition. This conditional model can have a small tangent-force couple while A!=B. No direct torsion; conservative8ray requires equal actual normal for subset interpretation.',
 'final_validation':'At most3shortlist/seed: complete22+189 independent geometry, new actual native multi-contact/J, native4ray minimax and force-hard/moment-box LP; requested moment1e−4Nm gate unchanged, solver box inset1e−10Nm. Possible2extra fresh native command probes each for preload/repeat. No actual dynamics steps.',
 'budget_scope':'Perseed static search at most1000evals/6000heavy/120s; final cooperative geometry budget3000heavy/60s, plus bounded18total native forwards and12final LPs. Report all actual counters and budget stops. Load/save overhead separately included in measured wall time.',
 'original_goal_and_qualification':'Same fixed physical model/wrist/scale/gains/materials; original ego material targets retained. Search success does not qualify faithful ego tracking, real parameters, learned method, native equilibrium or2s holding.',
 'training_GPU_or_physics_steps_authorized':False}
write(r/'protocol_v1.json',record)
print(json.dumps({'protocol':pin(r/'protocol_v1.json'),'source':record['source_pins']['runner']}))
