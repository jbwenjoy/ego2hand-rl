"""Measure actual versus reference hand geometry against the same simulated carton pose."""

import argparse
import json
from pathlib import Path

import mujoco
import newton
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import warp as wp

from supermimic_research.artifacts import file_pin
from supermimic_research.runtime.flash_wuji import WujiHand2
from inspect_hocap_grasp_load import part


def wxyz(xyzw):
    return xyzw[..., [3,0,1,2]]


def certified_gap(model,data,geom,normals,box_position,offset):
    """Separating plane lower bound in metres. Zero means this check is inconclusive."""
    axes=data.geom_xmat[geom].reshape(3,3)
    local_dirs=normals@axes
    projected=normals@(data.geom_xpos[geom]-box_position)+offset
    if model.geom_type[geom]==mujoco.mjtGeom.mjGEOM_CAPSULE:
        radius,half_length=model.geom_size[geom,:2]
        separation=projected-radius-half_length*np.abs(local_dirs[:,2])
    else:
        centre,extent=np.split(model.geom_aabb[geom],2)
        separation=projected+local_dirs@centre-np.abs(local_dirs)@extent
    return float(max(0.,separation.max()))


def measure(args):
    wp.set_device("cpu")
    builder=newton.ModelBuilder()
    hand=WujiHand2(args.assets).build(builder).hand("right")
    spec=mujoco.MjSpec.from_file(str(args.assets/"wuji_right.xml"))
    box=spec.worldbody.add_body(name="measured_object")
    box.add_freejoint(name="measured_free")
    spec.add_mesh(name="clearance_object",file=str(args.hull.resolve()))
    box.add_geom(name="object_collision",type=mujoco.mjtGeom.mjGEOM_MESH,meshname="clearance_object",mass=.15)
    vertices=np.array([[float(x) for x in line.split()[1:4]] for line in args.hull.read_text().splitlines()
                       if line.startswith("v ")])
    planes=ConvexHull(vertices).equations
    planes/=np.linalg.norm(planes[:,:3],axis=1)[:,None]
    model=spec.compile()
    data=mujoco.MjData(model)
    gid=model.geom("object_collision").id
    groups={model.geom(i).id:part(model.geom(i).name) for i in range(model.ngeom) if i!=gid}
    hand_bodies=[str(n).split("/")[-1] for n in builder.body_label]
    matched=[(i,model.body(n).id) for i,n in enumerate(hand_bodies)]
    finger_idx=model.jnt_qposadr[[model.joint(n).id for n in hand.finger_joint_names]]
    wrist_xyz=model.jnt_qposadr[[model.joint("right_wrist_"+c).id for c in "xyz"]]
    wrist_rot=model.jnt_qposadr[model.joint("right_wrist_rotvec").id]
    obj_idx=model.jnt_qposadr[model.joint("measured_free").id]
    with np.load(args.reference,allow_pickle=False) as a:
        wrist=a["target_wrist"]
        q_ref=a["target_finger_q"]
        raw_box=a["world_from_carton_source"]
    refquat=Rotation.from_matrix(wrist[:,:3,:3]).as_quat(scalar_first=True)
    rows={}
    pins=[]
    arrays={}
    for segment in json.loads(args.ledger.read_bytes()):
        for checked in segment["checks"]:
            if checked["sequence"]!="subject1_G04_2":continue
            path=Path(checked["report"]).with_name("trace.npz")
            with np.load(path,allow_pickle=False) as a:
                bodies=a["body_q"]
                q=a["joint_q"]
                frames=a["source_frames"]
                np.testing.assert_allclose(raw_box,a["raw_object_pose"],atol=1e-8,rtol=0)
            object_ids=np.flatnonzero(np.linalg.norm(bodies[0,:,:3]-raw_box[0,:3,3],axis=1)<1e-6)
            if len(object_ids)!=1:raise ValueError("Ambiguous original carton body")
            obj=bodies[:,int(object_ids[0])]
            np.testing.assert_array_equal(frames,np.arange(len(frames)))
            key=f'{segment["stream"]}_r{segment["round"]}'
            rows[key]={}
            pins.append(file_pin(path))
            bound_pose_err=0.
            palm=bodies[:,hand_bodies.index("right_palm_link")]
            wrist_delta=palm[:,:3]-wrist[frames,:3,3]
            wrist_deg=np.rad2deg((Rotation.from_quat(palm[:,3:]).inv()*Rotation.from_matrix(wrist[frames,:3,:3])).magnitude())
            rows[key]["wrist_pre_lift_26_35"]=dict(
                position_rms_mm=float(1000*np.sqrt(np.mean(np.sum(wrist_delta[26:36]**2,axis=1)))),
                mean_world_xyz_error_mm=(1000*wrist_delta[26:36].mean(0)).tolist(),
                rotation_rms_deg=float(np.sqrt(np.mean(wrist_deg[26:36]**2)))) if len(frames)>26 else None
            for wrist_source,finger_source in [(False,False),(True,False),(False,True),(True,True)]:
                label=("refW" if wrist_source else "simW")+("_refQ" if finger_source else "_simQ")
                names=list(dict.fromkeys(groups.values()))
                minimum=np.full((len(q),len(names)), np.inf)
                certified=np.full_like(minimum,np.inf)
                for i in range(len(q)):
                    data.qpos[wrist_xyz] = wrist[i,:3,3] if wrist_source else q[i,list(hand.wrist_pos_q_ids)]
                    data.qpos[wrist_rot:wrist_rot+4] = (refquat[i] if wrist_source else
                                                       wxyz(q[i,hand.wrist_orient_q_id:hand.wrist_orient_q_id+4]))
                    data.qpos[finger_idx] = q_ref[i] if finger_source else q[i,list(hand.finger_q_ids)]
                    data.qpos[obj_idx:obj_idx+7] = np.r_[obj[i,:3],wxyz(obj[i,3:])]
                    mujoco.mj_kinematics(model,data)
                    mujoco.mj_comPos(model,data)
                    face_normals=planes[:,:3]@Rotation.from_quat(obj[i,3:]).as_matrix().T
                    if not wrist_source and not finger_source:
                        for newton_id,native_id in matched:
                            err=np.linalg.norm(data.xpos[native_id]-bodies[i,newton_id,:3])
                            rot_err=np.max(np.abs(Rotation.from_quat(bodies[i,newton_id,3:]).as_matrix()
                                                  -data.xmat[native_id].reshape(3,3)))
                            if err>1e-5 or rot_err>2e-5:
                                raise ValueError(f"Recorded body FK mismatch {key}, {i}, {native_id}: {err}, {rot_err}")
                            bound_pose_err=max(bound_pose_err,float(err))
                    for geom,finger in groups.items():
                        distance=mujoco.mj_geomDistance(model,data,geom,gid,2.,None)*1000.
                        j=names.index(finger)
                        minimum[i,j]=min(minimum[i,j],distance)
                        lower=1000*certified_gap(model,data,geom,face_normals,obj[i,:3],planes[:,3])
                        certified[i,j]=min(certified[i,j],lower)
                arrays[key+"_"+label]=minimum
                arrays[key+"_"+label+"_certified_gap"]=certified
                by_phase={}
                for start,end in [(26,35),(36,46),(47,54)]:
                    sl=minimum[start:end+1]
                    bound=certified[start:end+1]
                    by_phase[f"{start}-{end}"]=dict(observed=len(sl), minimum_mm=dict(zip(names,sl.min(0).tolist())),
                        certified_minimum_mm=dict(zip(names,bound.min(0).tolist())),
                        certified_minimum_over_hand_mm=float(bound.min()),
                        any_within_1mm_frames=int(np.any(sl<=1.,axis=1).sum()),
                        thumb_within_1mm_frames=int((sl[:,names.index("thumb")]<=1.).sum())) if len(sl) else None
                rows[key][label]=by_phase
            rows[key]["max_recorded_all_hand_body_fk_error_mm"]=1000*bound_pose_err
    args.out.mkdir(parents=True,exist_ok=False)
    output=args.out/"signed_minimum_distances_mm.npz"
    np.savez_compressed(output,**arrays)
    (args.out/"report.json").write_text(json.dumps(dict(
        results=rows,columns=names,reference=file_pin(args.reference),mesh=file_pin(args.hull),traces=pins,
        distances=file_pin(output),code=file_pin(__file__),
        scope="MuJoCo signed surface query and separate conservative lower bound using source carton supporting planes "
              "against hand collision capsules and enclosing mesh boxes at saved control endpoints. "
              "Each of the four FK alternatives uses the same observed physical object pose, "
              "including alternatives with demo-derived target wrist or fingers. No counterfactual is integrated."
    ),indent=2,allow_nan=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("ledger","assets","hull","reference","out"):
        p.add_argument("--"+name,type=Path,required=True)
    measure(p.parse_args())
