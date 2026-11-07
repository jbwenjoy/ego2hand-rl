"""JSON/source pins only, no model/FK/query/solver execution."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent;P=H.parent
def pin(p):
    p=p.resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
cfg=dict(schema='functional_tripod_geometry_protocol:1',CPU_static_run_authorized=False,integration_or_training_authorized=False,
    output=str(H/'completed_v1'),designated_geom_ids=[15,28,41],
    budget=dict(global_wall_s=360,phaseA_wall_s=15,phaseB_wall_s=60,final_wall_s=8,cheap_FK_per_phase=1200,
        full_queries_per_entry=250,final_full_query_reserve=2,max_total_heavy_queries=20000,heavy_queries_per_entry=4500),
    inputs={k:pin(P/f) for k,f in dict(
        exact_source='surface_assembly_v11/exact_source_original_hand_completed_v1/seed1_rank0.json',
        source1mm='surface_assembly_v11/source1mm_opposed_side_original_hand_completed_v1/seed1_rank0.json',
        exact_anchors='source_surface_mapping_v11/exact_source_original_hand_anchors_v1.json',
        source1mm_anchors='source_surface_mapping_v11/source1mm_opposed_side_original_hand_anchors_v1.json',
        model_binding='hand2_holding_v1/compiled_nominal_v2/binding.json',model_mjb='hand2_holding_v1/compiled_nominal_v2/model.mjb',
        memo='functional_support_review_v1/memo_v1.json').items()},
    source_pins={k:pin(p) for k,p in dict(solver=H/'solve_tripod_geometry_v2.py',builder=H/'build_protocol_v2.py',
        static_geometry=P/'retarget_gravity_v3/evaluator_v3.py',convex_query=P/'geometry_query_v2/convex_query.py',
        feature_interface=P/'geometry_query_v2/query_interface_v2.py').items()},
    entries=[dict(name=source+'__'+rule,source_input=source,anchor_input=anchor,third_target_rule=rule) for source,anchor in
        [('exact_source','exact_anchors'),('source1mm','source1mm_anchors')] for rule in ['source_middle_anchor_nearest_phone','index_material_toward_COM_30mm']],
    protocol_scope='New functional mapping, not old2pad success or measured functionalpad truth. All q0..11 free within true limits; q12..19 fixed. Phone35mm/30deg trusts are algorithmic concessions, not ego uncertainty. Own214geometry constraints explicitly3designatedpads; idealLP/zeroacc never entrygate. No static/dynamic support promise.',
    candidate_interface='Report.candidates contains candidate pin/state_npz pin. Candidate schema functional_tripod_geometric_reference_candidate:1; stateNPZ qpos27_F64/qvel26_F64. ctrl20None/control_qualifiedFalse; future native preload controls independently verified.')
out=H/'protocol_CPUFalse_v2.json'
with out.open('x') as f:json.dump(cfg,f,indent=2,allow_nan=False)
print(json.dumps(pin(out)))
