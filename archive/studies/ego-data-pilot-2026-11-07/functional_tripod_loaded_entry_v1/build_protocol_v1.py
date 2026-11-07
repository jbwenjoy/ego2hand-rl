"""Freeze compression-search metadata only; never execute geometry/FK/native."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent;P=H.parent
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
parent=P/'functional_tripod_projection_v1/completed_run1/report.json';r=read(parent)
selected=[rec['candidate'] for rec in r['candidates'] if rec['geometry_pass']]
assert len(selected)==2
cfg=dict(schema='functional_tripod_loaded_entry_geometry_protocol:1',CPU_static_run_authorized=False,
    neural_training_or_integration_authorized=False,output=str(H/'completed_run1'),
    compression_target_window_m=[-.00009,-.00006],compression_target_center_m=-.000075,
    selected_parent_candidates=selected,
    sources=dict(runner=pin(H/'solve_compression_geometry_v1.py'),protocol_builder=pin(Path(__file__)),
        frozen_geometry_solver=pin(P/'functional_tripod_support_v1/solve_tripod_geometry_v2.py'),
        plane_helper=pin(P/'functional_tripod_support_v1/refinement_v1/separating_plane_refinement_v1.py'),
        original_projection_runner=pin(P/'functional_tripod_projection_v1/project_saved_candidates_v1.py')),
    inputs=dict(original_geometry_protocol=r['protocol'] if False else r['parent_report'],
        original_geometry_report=pin(parent)),
    budget=dict(global_wall_s=180,entry_wall_s=75,final_wall_s=8,iterations_per_entry=20,
        full_queries_per_entry=80,final_full_queries_reserved=2,heavy_queries_per_entry=2000),
    proposal=dict(joint_step_trust_rad=.02,interior_target_margin_m=0.,center_objective_weight=100.,
        line_search_fractions=[1.,.5,.25,.125,.0625]),
    hard_geometry=dict(designated_geom_ids=[15,28,41],pad_window_m=[-.0001,.0002],nonpad_minimum_gap_m=0.,self_minimum_gap_m=0.,
        phone_pair_count=22,self_pair_count=189,table_complete_required=True,unknown_forbidden=True,
        frozen_phone_and_q12_19=True,hardware_joint_bounds_unchanged=True),
    scope='Compression[-90,-60]um and center-75um are separate search targets, not a changed original geometry gate or proof of native loaded/holding. No if geometrypassed break: require targetpass AND originalgeometrypass. All2 finals/failures kept; full fresh final mandatory. No idealpointLP or zeroacc necessary gate.',
    candidate_ABI='Existing functional_tripod_geometric_reference_candidate:1/report:1 retained. Additional compression_target,compression_target_pass,geometry_and_compression_target_pass fields. Existing geometry.passed remains original214+trust result.')
# The Tripod constructor needs its original4-entry source protocol, not the projection protocol/report.
original_report=read(Path(r['parent_report']['path']))
cfg['inputs']['original_geometry_protocol']=original_report['protocol']
out=H/'protocol_CPUFalse_v1.json'
with out.open('x') as f:json.dump(cfg,f,indent=2,allow_nan=False)
print(json.dumps(pin(out)))
