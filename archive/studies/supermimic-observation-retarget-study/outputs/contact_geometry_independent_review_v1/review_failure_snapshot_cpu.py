"""Reproduce one failed distance witness; geometry/collision queries only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import json,hashlib,sys
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path('D:/workspace/supermimic-observation-retarget-study');OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from contact_penetration_geometry import ContactGeometry
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
def main():
    model=Path('D:/workspace/supermimic-chord-study/outputs/rotating_reference_warp_a0/model.mjb');context=Path('D:/workspace/supermimic-chord-study/outputs/a1l_paired_v2_clip007_2000iter_seed1/compiled/fixed_physics_context.npz');snap=ROOT/'outputs/contact_refined_geometry_debug_v1/query_failure_snapshot.npz'
    before={str(p):sha(p) for p in [model,context,ROOT/'scripts/contact_penetration_geometry.py']}
    z=np.load(snap,allow_pickle=False);q=z['q'].copy();pose=z['object_pose'].copy();g=ContactGeometry(model,context);g.set_object_pose(pose);g.robot.forward(q);m,d=g.model,g.data;h,o=map(int,g.pairs[12]);baseq=d.qpos.copy();v=d.qvel.copy();clock=d.time
    try:g.distance_and_jacobian(q)
    except ValueError as error:guard={'rejected':True,'error':str(error)}
    else:guard={'rejected':False}
    rows=[];arrays={};base_options=(int(m.opt.disableflags),float(m.opt.ccd_tolerance),int(m.opt.ccd_iterations))
    for variant in ['native_frozen_query_options','native_original_MJB_options','legacy_MPR_diagnostic_only']:
        m.opt.disableflags=base_options[0];m.opt.ccd_tolerance=base_options[1];m.opt.ccd_iterations=base_options[2]
        if variant=='native_original_MJB_options':m.opt.ccd_tolerance=1e-6;m.opt.ccd_iterations=50
        if variant=='legacy_MPR_diagnostic_only':m.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        g.robot.forward(q)
        for first,second in [(h,o),(o,h)]:
            for cutoff in [0.,1.]:
                segment=np.zeros(6);distance=mujoco.mj_geomDistance(m,d,first,second,cutoff,segment);length=np.linalg.norm(segment[3:]-segment[:3]);normal=(segment[3:]-segment[:3])/distance if distance!=0 else None
                rows.append({'variant':variant,'geom_first':g.names[first],'geom_second':g.names[second],'distmax':cutoff,'distance_m':float(distance),'segment_length_m':float(length),'witness_error_m':float(abs(length-abs(distance))),'normal_if_divided_by_signed_distance':normal.tolist() if normal is not None else None,'normal_norm':float(np.linalg.norm(normal)) if normal is not None else None,'segment':segment.tolist()})
        mujoco.mj_collision(m,d)
        hits=[{'dist_m':float(c.dist),'geom':list(map(int,c.geom)),'pos_w':c.pos.copy().tolist(),'normal_geom0_to_geom1':c.frame[:3].copy().tolist()} for c in d.contact if set(map(int,c.geom))=={h,o}]
        rows.append({'variant':variant,'native_collision_pair_contacts':hits,'all_contact_count':int(d.ncon)})
    # Restore only the declared in-memory query configuration. No model file or
    # trained/evaluated model is rewritten; legacy/default variants are probes.
    m.opt.disableflags=base_options[0];m.opt.ccd_tolerance=base_options[1];m.opt.ccd_iterations=base_options[2];g.robot.forward(q)
    raw=np.zeros(6);dist=mujoco.mj_geomDistance(m,d,h,o,1.,raw);normraw=(raw[3:]-raw[:3])/dist
    jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,jr,raw[:3],int(m.geom_bodyid[h]));badjac=-normraw@jp[:,g.robot.va]
    # A normalized witness is also tested diagnostically; it is not adopted as
    # a fix, since normalizing an inconsistent witness does not validate depth.
    normalizedjac=-(normraw/np.linalg.norm(normraw))@jp[:,g.robot.va]
    fdrows=[]
    for eps in [1e-5,2e-5,5e-5]:
        fd=np.empty(20)
        for j in range(20):
            values=[]
            for sign in [1,-1]:
                qj=q.copy();qj[j]+=sign*eps;g.robot.forward(qj);values.append(mujoco.mj_geomDistance(m,d,h,o,1.,None))
            fd[j]=(values[0]-values[1])/(2*eps)
        fdrows.append({'epsilon_rad':eps,'max_abs_raw_witness_vs_scalarFD_m_per_rad':float(np.max(np.abs(badjac-fd))),'max_abs_normalized_witness_vs_scalarFD_m_per_rad':float(np.max(np.abs(normalizedjac-fd))),'max_abs_scalarFD_m_per_rad':float(np.max(np.abs(fd)))})
        arrays['FD_'+str(eps)]=fd
    g.robot.forward(q);arrays.update(q=q,object_pose=pose,raw_witness=raw,raw_witness_Jac=badjac,normalized_witness_Jac=normalizedjac)
    np.savez_compressed(OUT/'failure_snapshot_independent.npz',**arrays)
    report={'status':'REPRODUCED_QUERY_WITNESS_FAILURE_NOT_A_QUALIFIED_PENETRATION_ORACLE','frame':int(z['frame']),'pair_index':12,'pair_names':[g.names[h],g.names[o]],'geom_types':[int(m.geom_type[h]),int(m.geom_type[o])],'geom_sizes':[m.geom_size[h].tolist(),m.geom_size[o].tolist()],'guard':guard,'queries_and_native_collision':rows,'scalar_FD_comparison':fdrows,'snapshot':pin(snap),'model':pin(model),'context':pin(context),'geometry_source':pin(ROOT/'scripts/contact_penetration_geometry.py'),'code':pin(__file__),'arrays':pin(OUT/'failure_snapshot_independent.npz'),'MuJoCo_version':mujoco.__version__,'state_qpos_qvel_time_preserved_by_geometric_queries':bool(np.array_equal(d.qpos,baseq) and np.array_equal(d.qvel,v) and d.time==clock),'input_files_unchanged':before=={p:sha(p) for p in before},'dynamics_steps':0,'physical_reference_qualified':False,'contacts_scope':'mj_collision creates geometric contact candidates only; no forward dynamics, solver, forces, rollout or known observed-contact labels.','interpretation':'Scalar distance and witness consistency failure is reproduced at frozen q/pose. Do not relax the guard, silently normalize, reverse one bad pair ad hoc, or treat legacyMPR as validated. Further query replacement requires predeclared all-state regression and feature-switch/FD checks.','primary_sources':[{'url':'https://github.com/google-deepmind/mujoco/blob/3.11.0/src/engine/engine_support.c#L487-L565','finding':'Native distance helper returns mjc_ccd scalar and separately copied status.x1/x2; alternate path synthesizes fromto from collision pos/normal/dist. This distinction can explain consistent synthetic legacy witnesses without proving its scalar accuracy.'},{'url':'https://github.com/google-deepmind/mujoco/blob/3.11.0/src/engine/engine_collision_convex.c#L81-L124','finding':'Native collision normal is normalized witness difference; normalization is not an independent accuracy certificate.'},{'url':'https://mujoco.readthedocs.io/en/stable/computation/#geom-distance','finding':'Official documentation discourages legacy MPR for general geom-distance values.'}]}
    (OUT/'failure_snapshot_review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'guard':guard,'queries':rows,'FD':fdrows,'state_preserved':report['state_qpos_qvel_time_preserved_by_geometric_queries'],'inputs_unchanged':report['input_files_unchanged']},indent=2))
if __name__=='__main__':main()
