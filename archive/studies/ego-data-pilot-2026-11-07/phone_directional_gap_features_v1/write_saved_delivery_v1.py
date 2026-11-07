"""Derive masks/summary from the completed saved check; no Tensor FK replay."""
from pathlib import Path
import hashlib,json
import numpy as np
H=Path(__file__).resolve().parent
def pin(p):
    b=p.read_bytes();return dict(path=str(p.resolve()),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(p.read_text())
def write(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def main():
    O=H/'completed_v2';r=read(O/'report.json');rows=read(O/'pair_comparison44.json')
    assert r['passed']==r['total']==8 and pin(O/'report.json')['sha256']=='d74363c7c6fbb0cd355e39a200a3a4a8ddd1cee058436ecc13f1121cde7e946e'
    with np.load(O/'descriptor_arrays.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    masks={};summary={}
    for dtype in ['F64','F32']:
        vals=np.stack([arrays[x['state']+'_directional_gap_'+dtype] for x in rows])
        known=np.isfinite(vals);assert known.shape==(2,22) and known.all()
        sourceknown=np.array([[v['source_exact_gap_m'] is not None for v in x['pairs22'][dtype]] for x in rows],dtype=bool)
        sourceexact=np.array([[v['source_exact_gap_m'] if v['source_exact_gap_m'] is not None else float('nan') for v in x['pairs22'][dtype]] for x in rows])
        masks['directional_gap_known_'+dtype]=known
        masks['pair_pose_known_'+dtype]=np.ones((2,22),dtype=bool)
        masks['pair_geometry_known_'+dtype]=np.ones((2,22),dtype=bool)
        for label in ['pair_exact_signed_gap_known','closest_contact_normal_known','normal_unique_known','formal_floating_certificate_known','full_H_zero_contact_certificate_known']:
            masks[label+'_'+dtype]=np.zeros((2,22),dtype=bool)
        masks['comparison_source_exact_row_known_'+dtype]=sourceknown
        masks['comparison_source_exact_signed_gap_'+dtype]=sourceexact
        errors=(vals-sourceexact)[sourceknown]
        summary[dtype]=dict(pair_cases=44,source_exact_comparisons=int(sourceknown.sum()),maximum_descriptor_minus_exact_m=float(errors.max()),maximum_looseness_m=float((-errors).max()),
            guard_range_m=[min(v['guard_m'] for x in rows for v in x['pairs22'][dtype]),max(v['guard_m'] for x in rows for v in x['pairs22'][dtype])])
    fp=O/'saved_knowledge_masks_v1.npz'
    with fp.open('xb') as f:np.savez_compressed(f,**masks)
    scope=dict(schema='phone_directional_gap_saved_mask_derivation:1',producer_report=pin(O/'report.json'),saved_values=pin(O/'descriptor_arrays.npz'),saved_source_comparisons=pin(O/'pair_comparison44.json'),output=pin(fp),
        derivation='Descriptor known masks reconstructed from saved finite values; pose/binding/all-false exact/normal/formal/fullH masks follow successfully validated frozen output contract. They were not separately captured by the checker and are not a new FK/geometry replay.',
        exact_reference_masks='Only saved EXACT_NUMERICAL_IDEAL_CONVEX rows are comparisons; AABB-only rows are not exact truth.',new_Tensor_FK_native_NN_physics_calls=0)
    write(O/'saved_mask_derivation_v1.json',scope)
    delivery=dict(schema='phone_directional_gap_delivery:1',status='DELIVERED_BOUNDED_DIFFERENTIABLE_DIRECTIONAL_GEOMETRY_NOT_SDF',
        provider=pin(H/'directional_gap_provider.py'),catalog=pin(H/'geometry_catalog_v1.json'),protocol=pin(H/'protocol_v2.json'),report=pin(O/'report.json'),terminal=pin(H/'process_v2/terminal.json'),
        known_mask_derivation=pin(O/'saved_mask_derivation_v1.json'),mask_arrays=pin(fp),array_values=pin(O/'descriptor_arrays.npz'),all_pair_comparisons=pin(O/'pair_comparison44.json'),
        retained_failed_v1=dict(source=pin(H/'check_directional_gap_v1.py'),failure=pin(H/'failure_v1.json'),terminal=pin(H/'process_v1/terminal.json')),checker_revision=pin(H/'source_revision_v2.json'),
        numerical_summary=summary,descriptor_derivative=r['descriptor_derivative'],counts=r['counts'],
        geometry_catalog_scope='All22 compiled boundary pairs; original semantic pad IDs15/28 stay source metadata. Geometry ID41 is supplied without changing functional/contact labels.',
        certificate_scope='Ideal convex real-arithmetic directional lower-bound; finite direction set can be loose. No certified total float/FK envelope, exact SDF/normal, native force, whole-H/contact-dynamics, real-ego or learning claim.',
        new_calls_for_this_delivery_writer=0,delivery_writer=pin(Path(__file__)))
    write(H/'delivery_v1.json',delivery);print(json.dumps(pin(H/'delivery_v1.json')))
if __name__=='__main__':main()
