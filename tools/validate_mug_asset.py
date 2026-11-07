"""Sampled source/union geometry, native gauge fixtures and isolated drop tests.

No Hand2 policy or learned world model is loaded. Probe fixtures are separate
zero-gravity geometry tests, not object assistance in a holding experiment.
"""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin
from supermimic_research.assets.mesh import union_membership
from supermimic_research.assets.validation import nearest_surface,plane_lower_bound,corridor_clearance
from supermimic_research.assets.compound import scene_xml


def write(path,value):
    with Path(path).open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def assert_prior(model,meta,mj):
    body=model.body('hot3d_mug9').id;prior=meta['physical_prior'];R=np.empty(9);mj.mju_quat2Mat(R,model.body_iquat[body]);R=R.reshape(3,3)
    if abs(model.body_mass[body]-prior['mass_kg'])>1e-15 or not np.allclose(model.body_ipos[body],prior['COM_model_m'],rtol=0,atol=1e-15):raise ValueError('Explicit source-frame mass/COM prior')
    if not np.allclose(R@np.diag(model.body_inertia[body])@R.T,prior['inertia_model_kg_m2'],rtol=0,atol=1e-12):raise ValueError('Full source-frame inertia prior')
    visual=model.geom('mug9_visual').id
    if model.geom_contype[visual] or model.geom_conaffinity[visual] or model.neq or model.nmocap or np.any(model.body_gravcomp):raise ValueError('Visual excluded, no aid')
    return body


def drop_test(model,dt,out,mj):
    model.opt.timestep=dt;data=mj.MjData(model);mj.mj_forward(model,data);body=model.body('hot3d_mug9').id;R=np.empty(9);N=round(2/dt)
    rows=[];contact_reads=0;maximum_depth=0.;warnings=0
    for tick in range(N):
        mj.mj_step(model,data)
        if np.any(data.qfrc_applied) or np.any(data.xfrc_applied) or not np.isfinite(np.r_[data.qpos,data.qvel,data.qacc]).all():raise ValueError('No-aid finite asset drop')
        Fn=0.
        for ci in range(data.ncon):
            force=np.empty(6);mj.mj_contactForce(model,data,ci,force);contact_reads+=1
            if not np.isfinite(force).all():raise ValueError('Finite native contact')
            Fn+=max(0.,float(force[0]));maximum_depth=max(maximum_depth,-float(data.contact[ci].dist))
        mj.mju_quat2Mat(R,data.qpos[3:7]);rotation=R.reshape(3,3);lever=rotation@model.body_ipos[body];COM=data.qpos[:3]+lever
        omega=rotation@data.qvel[3:6];vCOM=data.qvel[:3]+np.cross(omega,lever)
        rows.append(np.r_[tick+1,data.time,data.qpos.copy(),data.qvel.copy(),COM,vCOM,omega,Fn,data.ncon])
        warnings=int(data.warning.number.sum())
        if warnings:raise ValueError('Asset drop native warning')
    a=np.array(rows);tail=np.arange(N)>=round(1.5/dt)
    # Columns: tick,time,q7,v6,COM3,vCOM3,omega3,PRE-normal-force,ncon.
    metrics=dict(final_COM_world_m=a[-1,15:18].tolist(),tail_vCOM_RMS_m_s=float(np.sqrt(np.mean(np.sum(a[tail,18:21]**2,axis=1)))),
        tail_omega_RMS_rad_s=float(np.sqrt(np.mean(np.sum(a[tail,21:24]**2,axis=1)))),tail_mean_table_normal_N=float(a[tail,24].mean()),maximum_native_penetration_m=maximum_depth,warning_count=warnings)
    checks=dict(finite=bool(np.isfinite(a).all()),settled_linear=metrics['tail_vCOM_RMS_m_s']<.02,settled_angular=metrics['tail_omega_RMS_rad_s']<.5,
        native_depth=maximum_depth<.002,table_load=abs(metrics['tail_mean_table_normal_N']-.25*9.81)<.05,warning_free=warnings==0)
    path=out/f'drop_dt{dt}.npz'
    with path.open('xb') as stream:np.savez_compressed(stream,trace=a)
    return dict(dt_s=dt,steps=N,contact_force_reads=contact_reads,trace=file_pin(path),metrics=metrics,checks=checks,passed=all(checks.values()),final_quaternion_wxyz=data.qpos[3:7].tolist(),
        scope='Two-second isolated object/table drop; PRE-solver contacts and POST pose/COM. No hand/controller/retargeting/learning and no original holding-pass claim.')


def run(args):
    import trimesh,mujoco
    from scipy.spatial import cKDTree
    out=args.output.resolve();meta=json.loads(args.mesh_report.read_text());decomp=json.loads(args.decomposition.read_text());gauges=json.loads(args.gauges.read_text())
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh asset QA output')
    if decomp['mesh_report']!=file_pin(args.mesh_report) or gauges['mesh_report']!=file_pin(args.mesh_report):raise ValueError('Same pinned source')
    for item in (meta['raw'],decomp['raw'],gauges['raw']):
        if file_pin(item['path'])!=item:raise ValueError('Changed source/component/probe array')
    out.mkdir(parents=True);tick=time.perf_counter()
    write(out/'intent.json',dict(mesh_report=file_pin(args.mesh_report),decomposition=file_pin(args.decomposition),gauges=file_pin(args.gauges),
        sampled_limits=gauges['sampled_error_limits_m'],primary_radius_m=.001,positive_control_radius_m=.003,source_samples=1024,exposed_face_samples_max=1024,
        drop_dt_s=[.0003125,.00015625],maximum_drop_steps=19200,contact_time_prior_s=args.contact_time,automatic_retry=False,
        sources={name:file_pin(ROOT/name) for name in ('tools/validate_mug_asset.py','src/supermimic_research/assets/mesh.py','src/supermimic_research/assets/validation.py','src/supermimic_research/assets/compound.py')}))
    with np.load(meta['raw']['path'],allow_pickle=False) as z:source=trimesh.Trimesh(vertices=z['vertices_model_m'],faces=z['faces'],process=False)
    with np.load(decomp['raw']['path'],allow_pickle=False) as z:parts=[(z[f'part{i:03d}_vertices'].copy(),z[f'part{i:03d}_faces'].copy(),z[f'part{i:03d}_halfspaces'].copy()) for i in range(len(decomp['components']))]
    halfspaces=[p[2] for p in parts];pieces=[trimesh.Trimesh(vertices=v,faces=f,process=False) for v,f,_ in parts];combined=trimesh.util.concatenate(pieces)
    rng=np.random.default_rng(20261108);source_points=np.r_[source.vertices[rng.choice(len(source.vertices),512,replace=False)],source.triangles_center[rng.choice(len(source.faces),512,replace=False)]]
    inside=union_membership(source_points,halfspaces);outside_distance=np.zeros(len(source_points))
    if (~inside).any():_,outside_distance[~inside]=nearest_surface(combined,source_points[~inside])
    candidate=np.concatenate([p.triangles_center for p in pieces]);normals=np.concatenate([p.face_normals for p in pieces]);exposed=~union_membership(candidate+normals*1e-6,halfspaces)
    selected=np.flatnonzero(exposed)
    if len(selected)>1024:selected=np.sort(rng.choice(selected,1024,replace=False))
    exposed_points=candidate[selected];_,reverse_distance=nearest_surface(source,exposed_points)
    geometry=dict(source_samples=len(source_points),source_samples_outside_union=int((~inside).sum()),source_outside_max_m=float(outside_distance.max()),source_outside_RMS_m=float(np.sqrt(np.mean(outside_distance**2))),
        exposed_face_samples=len(exposed_points),exposed_surface_to_source_max_m=float(reverse_distance.max()),exposed_surface_to_source_RMS_m=float(np.sqrt(np.mean(reverse_distance**2))),
        scope='Finite deterministic source vertices/face centroids and exposed piece-face centroids, exterior1um classification. Not global Hausdorff/volume error or ground-truth material accuracy.')
    with np.load(gauges['raw']['path'],allow_pickle=False) as z:g={k:z[k].copy() for k in z.files}
    corridors={}
    for name in ('handle','cavity'):
        points=g[name+'_points'];bound=plane_lower_bound(points,halfspaces)
        corridors[name]={str(r):dict(source=corridor_clearance(g[name+'_source_distance'],points,r),compound=corridor_clearance(bound,points,r)) for r in gauges['radii_m']}
    (out/'mug_asset.xml').write_text(scene_xml(meta,decomp,out,contact_time=args.contact_time),encoding='utf8');model=mujoco.MjModel.from_xml_path(str(out/'mug_asset.xml'));assert_prior(model,meta,mujoco)
    if (model.nq,model.nv,model.nu)!=(7,6,0):raise ValueError('Isolated free object only')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data);compiled_error=[]
    for record,(v,f,eq) in zip(decomp['components'],parts):
        gid=model.geom(f"mug9_collision_{record['index']:03d}").id;mid=model.geom_dataid[gid];start=model.mesh_vertadr[mid];n=model.mesh_vertnum[mid]
        world=model.mesh_vert[start:start+n].astype(float)@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid]
        error=max(cKDTree(v).query(world)[0].max(),cKDTree(world).query(v)[0].max());compiled_error.append(float(error))
    if max(compiled_error)>1e-7:raise ValueError('Compiled collider/model-frame vertex mismatch')
    mujoco.mj_saveModel(model,str(out/'mug_asset.mjb'),None)
    reloaded=mujoco.MjModel.from_binary_path(str(out/'mug_asset.mjb'))
    binary_arrays=('qpos0','body_mass','body_ipos','body_iquat','body_inertia','geom_type','geom_dataid','geom_pos','geom_quat','mesh_vert','pair_friction')
    binary_exact=all(getattr(model,k).tobytes()==getattr(reloaded,k).tobytes() for k in binary_arrays)
    if not binary_exact:raise ValueError('Native binary asset roundtrip')
    native=[]
    for radius in gauges['radii_m']:
        fixture=out/f'probe_{radius}.xml';fixture.write_text(scene_xml(meta,decomp,out,probe_radius=radius,contact_time=args.contact_time),encoding='utf8');m=mujoco.MjModel.from_xml_path(str(fixture));assert_prior(m,meta,mujoco);d=mujoco.MjData(m);probe=m.geom('probe').id
        for name,points in [('handle',g['handle_points']),('cavity',g['cavity_points']),('source_surface',g['source_surface_controls'])]:
            for index,p in enumerate(points):
                mujoco.mj_resetData(m,d);d.qpos[:3]=p;d.qpos[3:7]=[1,0,0,0];mujoco.mj_forward(m,d)
                contacts=[ci for ci in range(d.ncon) if probe in map(int,d.contact[ci].geom)]
                native.append(dict(radius_m=radius,role=name,index=index,point_model_m=p.tolist(),contact_count=len(contacts)))
    (out/'drop_scene.xml').write_text(scene_xml(meta,decomp,out,drop=True,contact_time=args.contact_time),encoding='utf8');drop_model=mujoco.MjModel.from_xml_path(str(out/'drop_scene.xml'));assert_prior(drop_model,meta,mujoco)
    drops=[drop_test(drop_model,dt,out,mujoco) for dt in (.0003125,.00015625)]
    checks=dict(sampled_source_coverage=geometry['source_outside_max_m']<=gauges['sampled_error_limits_m']['source_outside_union_max'],
        sampled_exposed_surface=geometry['exposed_surface_to_source_max_m']<=gauges['sampled_error_limits_m']['exposed_union_surface_to_source_max'],
        primary_source_corridors=all(corridors[n]['0.001']['source']['clear'] for n in corridors),primary_compound_corridors=all(corridors[n]['0.001']['compound']['clear'] for n in corridors),
        primary_native_void=all(x['contact_count']==0 for x in native if x['radius_m']==.001 and x['role']!='source_surface'),
        native_surface_controls=all(x['contact_count']>0 for x in native if x['radius_m']==.003 and x['role']=='source_surface'),
        compiled_frame=max(compiled_error)<=1e-7,binary_asset_roundtrip=binary_exact,isolated_drop=all(x['passed'] for x in drops))
    with (out/'geometry_samples.npz').open('xb') as stream:np.savez_compressed(stream,source_points=source_points,source_outside_distance=outside_distance,exposed_points=exposed_points,exposed_distance_to_source=reverse_distance)
    report=dict(schema='supermimic_research.mug_compound_validation:1',intent=file_pin(out/'intent.json'),checks=checks,passed=sum(checks.values()),total=len(checks),geometry=geometry,corridors=corridors,native_gauges=native,
        compiled_vertex_max_errors_m=compiled_error,asset_mjb=file_pin(out/'mug_asset.mjb'),drops=drops,physical_prior=meta['physical_prior'],contact_time_prior_s=args.contact_time,elapsed_s=time.perf_counter()-tick,
        qualified_numerical_asset_prototype=all(checks.values()),Hand2_loaded=False,policy_or_retargeting_qualified=False,learning_updates=0,
        counts=dict(native_models=6,binary_model_reads=1,native_gauge_forwards=len(native),isolated_drop_steps=sum(x['steps'] for x in drops),NN_FW=0,GPU_calls=0),
        scope='Numerical compound-asset prototype under unmeasured priors. Source surface/void sample checks and native fixture/drop tests, not robot holding, calibrated physical truth or full collision error certification.')
    write(out/'report.json',report);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=checks,geometry=geometry)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for flag in ('mesh-report','decomposition','gauges','output'):p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--contact-time',type=float,default=.02)
    run(p.parse_args())
