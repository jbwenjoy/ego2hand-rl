"""Static saved-pose probe: one passive MJB read, no MjData/FK/forward/distance.

Compiled collider vertices/sizes come from the exact stage1 binary. World poses
come exclusively from the pinned seed1/2 NPZ captures. No candidate is optimized.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from convex_query import ConvexPolytope,jsonable

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent


def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def main():
    fixture=json.loads((HERE/'fixture_results_v1.json').read_text())
    if fixture['passed'] is not True:raise ValueError('Analytic fixtures must pass first')
    report=json.loads((ROOT/'hand2_stage1/completed_v2/report.json').read_text())
    modelpin=report['output_pins']['hand2_phone_scene.mjb']
    if pin(modelpin['path'])!=modelpin:raise ValueError('Changed compiled scene')
    parent=json.loads((ROOT/'retarget_geometry_v1/report.json').read_text())
    import mujoco  # passive metadata only, no MjData/forward/FK/collision
    model=mujoco.MjModel.from_binary_path(modelpin['path'])
    phone=model.geom('phone24_collision_convex').id
    mids=int(model.geom_dataid[phone]);vstart=int(model.mesh_vertadr[mids]);nv=int(model.mesh_vertnum[mids])
    localphone=model.mesh_vert[vstart:vstart+nv].astype(np.float64).copy()
    assert nv==352
    rows=[];inputs={};saved={};started=time.perf_counter()
    names=[f'right_finger{f}_{part}' for f in (1,2) for part in ('link2_col','link3_col','link4_col','tip_sensor_col')]
    for seed in (1,2):
        source=Path(parent['all_seed_results'][seed]['snapshot']['path'])
        assert pin(source)==parent['all_seed_results'][seed]['snapshot']
        inputs[f'seed{seed}']=pin(source)
        with np.load(source,allow_pickle=False) as z:
            gp=z['geom_xpos'].copy();gr=z['geom_xmat'].reshape(-1,3,3).copy()
        phonev=localphone@gr[phone].T+gp[phone]
        poly=ConvexPolytope(phonev)
        saved[f'seed{seed}_phone_world_vertices_m']=phonev
        native={v['geom_name']:v['phone_signed_distance_m'] for v in parent['all_seed_results'][seed]['pair_distance_table']}
        for name in names:
            gid=model.geom(name).id;gtype=int(model.geom_type[gid]);geominfo={'id':gid,'name':name,'native_type':gtype,'contype':int(model.geom_contype[gid]),'conaffinity':int(model.geom_conaffinity[gid])}
            assert geominfo['contype'] or geominfo['conaffinity']
            if gtype==int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                radius,half=map(float,model.geom_size[gid,:2]);a=gp[gid]-half*gr[gid,:,2];b=gp[gid]+half*gr[gid,:,2]
                query=poly.capsule(a,b,radius)
                geominfo.update(radius_m=radius,half_centerline_length_m=half,a_world_m=a.tolist(),b_world_m=b.tolist())
                saved[f'seed{seed}_geom{gid}_capsule_a_b_r']=np.r_[a,b,radius]
            elif gtype==int(mujoco.mjtGeom.mjGEOM_MESH):
                mid=int(model.geom_dataid[gid]);s=int(model.mesh_vertadr[mid]);n=int(model.mesh_vertnum[mid])
                world=model.mesh_vert[s:s+n].astype(np.float64)@gr[gid].T+gp[gid]
                query=poly.convex(world)
                geominfo.update(native_mesh_id=mid,compiled_mesh_vertices=n)
                saved[f'seed{seed}_geom{gid}_world_mesh_vertices_m']=world
            else:raise ValueError('Explicit supported capsule/convex-mesh only')
            rows.append({'seed':seed,'geom':geominfo,'query':jsonable(query),
                         'previous_native_distance_for_description_only_m':native[name],
                         'previous_native_value_is_not_used_as_truth_or_query_input':True})
    elapsed=time.perf_counter()-started
    np.savez_compressed(HERE/'saved_probe_geometry_inputs_v1.npz',**saved)
    result={'schema':'independent_saved_capsule_convex_phone_query:1','source':pin(__file__),'query_source':pin(HERE/'convex_query.py'),
        'fixture_source':pin(HERE/'check_convex_query.py'),'fixture':pin(HERE/'fixture_results_v1.json'),
        'compiled_scene':modelpin,'scene_report':pin(ROOT/'hand2_stage1/completed_v2/report.json'),
        'parent_candidate_report':pin(ROOT/'retarget_geometry_v1/report.json'),'saved_candidates':inputs,
        'geometry_input_arrays':pin(HERE/'saved_probe_geometry_inputs_v1.npz'),'queries':rows,'query_count':len(rows),
        'elapsed_query_seconds':elapsed,'counts':{'passive_MJB_reads':1,'candidate_NPZ_reads':2,'MjData_FK_forward_nativeDistance_integration_training_GPU':0},
        'scope':'Ideal compiled convex hulls/capsules at saved world poses; all8 requested thumb/index colliders for each of seed1/2, no result-based selection. Qhull/double arithmetic is numerically checked, not formal interval certification. Positive margins are true convex separation distances; negative margins describe current-orientation pure-translation exit, not a scan SDF or observed contact/force.',
        'holding_or_contact_force_qualified':False}
    with (HERE/'saved_candidate_queries_v1.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'report':pin(HERE/'saved_candidate_queries_v1.json'),'query_count':len(rows),'elapsed_seconds':elapsed,
                     'rows':[{'seed':x['seed'],'geom':x['geom']['name'],'margin_mm':1000*x['query']['signed_translation_margin_m'],'classification':x['query']['classification']} for x in rows]},allow_nan=False))


if __name__=='__main__':main()
