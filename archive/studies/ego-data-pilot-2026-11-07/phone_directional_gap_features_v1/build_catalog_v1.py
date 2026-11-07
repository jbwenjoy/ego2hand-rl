"""Pure saved geometry preparation; no native, Torch, search or model load."""
from pathlib import Path
import ast, hashlib, json
import numpy as np

H=Path(__file__).resolve().parent;P=H.parent
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def main():
    meta=P/'phone_geometry_features_v1/completed_v1/compiled_constants.json'
    provider=P/'phone_geometry_features_v1/geometry_provider.py'
    m=json.loads(meta.read_text());b=next(b for b in m['geometry_bindings'] if b['geom_id']==71)
    assert pin(meta)['sha256']=='330243d3b85abb1e996c2658b9216fe89df9ede53b0c4e035af63deb651f8fd9'
    assert pin(provider)['sha256']=='f00da9a2c5deb19261a3ff4e168d79f8aea9784736e0fdd0897ffe9599a3b07b'
    with np.load(m['geometry_arrays']['path'],allow_pickle=False) as z:
        V=z[b['vertices_key']].astype(np.float64);F=z[b['faces_key']].astype(np.int64)
    R=np.array(m['geom_R'][71]);pos=np.array(m['geom_pos'][71]);ipos=np.array(m['phone_body_ipos'])
    bodyCOM=V@R.T+pos-ipos;tri=bodyCOM[F]
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);norm=np.linalg.norm(cross,axis=1)
    assert len(F)==700 and np.all(norm>1e-16)
    normals=cross/norm[:,None];raw=np.r_[normals,-normals,np.eye(3),-np.eye(3)]
    # Deterministic direction deduplication only; no facet/normal uniqueness claim.
    chosen=[];seen=set()
    for d in raw:
        key=tuple(np.round(d,12))
        if key not in seen:seen.add(key);chosen.append(d/np.linalg.norm(d))
    directions=np.array(chosen,dtype=np.float64)
    arrays=H/'direction_arrays_v1.npz'
    with arrays.open('xb') as f:np.savez_compressed(f,directions_BODY_F64=directions,phone_source_triangles_i64=F)
    catalog=dict(schema='phone_directional_gap_geometry_catalog:1',provider_source=pin(provider),provider_metadata=pin(meta),direction_arrays=pin(arrays),
        phone_geometry_arrays=m['geometry_arrays'],model=m['model'],pair_geom_ids=[x['other_geom_id'] for x in m['pairs']],
        direction_count=len(directions),direction_recipe='Every nondegenerate compiled phone source-triangle normal and its negative, plus +/-BODY axes; dedup rounded12 decimal only',
        direction_frame='phone BODY axes; all projected supports use phone COM origin',
        phone_transform='compiled geom vertices @ geom_R71.T + geom_pos71 - body_ipos; no second mesh_pos/quat',
        true_signed_gap_known=False,formal_floating_point_certificate=False,full_H_zero_contact_certificate=False,
        numeric_guard_recipe='256*eps(dtype)*(1m+norm(pair COM-relative center)+geom radius bound+max phone COM vertex radius), heuristic not certified FK error')
    write(H/'geometry_catalog_v1.json',catalog)
    # Exact name is read from the frozen result instead of a guessed alias.
    report=json.loads((P/'functional_tripod_loaded_entry_v1/completed_run1/report.json').read_text())
    states=[Path(next(x['candidate']['path'] for x in report['candidates'] if x['geometry_pass'])),P/'functional_tripod_loaded_entry_v1/completed_run2/source1mm__index_material_toward_COM_30mm_candidate.json']
    protocol=dict(schema='phone_directional_gap_bounded_tensor_contract:1',pure_tensor_geometry_authorized=True,native_NN_training_physics_authorized=False,
        output=str(H/'completed_v1'),sources=dict(provider=pin(H/'directional_gap_provider.py'),checker=pin(H/'check_directional_gap_v1.py'),catalog_builder=pin(Path(__file__))),
        inputs=dict(catalog=pin(H/'geometry_catalog_v1.json'),provider_report=pin(P/'phone_geometry_features_v1/completed_v1/report.json'),state_exact=pin(states[0]),state_source1mm=pin(states[1]),reference_compiled_geometry=pin(P/'functional_tripod_loaded_entry_v1/completed_run2/compiled_geometry_and_limits.npz')),
        budget=dict(saved_states=2,canonical_pairs=22,dtypes=['float64','float32'],pure_Torch_descriptor_FK_calls=6,pure_geometry_AD_calls=2,analytic_fixture_cases=4,new_native_model_loads=0,new_native_FK=0,new_native_forward=0,new_physics_steps=0,new_NN_calls=0),
        reference_comparison=dict(exact_source_rows_only=True,F64_arithmetic_allowance_m=1e-9,F32_diagnostic_allowance_m=5e-6,allowances_are_not_geometry_gates=True),
        AD=dict(state='exact',pair_geom_id=13,q_index=0,FD_h_rad=1e-7,scope='descriptor arithmetic only, no physics gradient claim'),
        masks=dict(direction_descriptor_known='finite implemented support arithmetic',pose_known='frozen compiled Torch FK',exact_gap_known=False,closest_normal_known=False,formal_float_certificate_known=False,full_H_causal_zero_known=False))
    for f in [H/'directional_gap_provider.py',H/'check_directional_gap_v1.py']:ast.parse(f.read_text())
    write(H/'protocol_v1.json',protocol)
    write(H/'SOURCE_FREEZE_v1.json',dict(schema='phone_directional_gap_source_freeze:1',sources=protocol['sources'],protocol=pin(H/'protocol_v1.json'),catalog=pin(H/'geometry_catalog_v1.json')))
    print(json.dumps(dict(protocol=pin(H/'protocol_v1.json'),catalog=pin(H/'geometry_catalog_v1.json'),directions=len(directions))))
if __name__=='__main__':main()
