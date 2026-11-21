"""Prepare Wuji direct-IK reference arrays and relocatable assets for FlashCHORD."""
import argparse
import json
from pathlib import Path
import shutil
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.assets.proximity import SurfaceQuery
from supermimic_research.strict_json import write_json


def wxyz(rotations):
    shape = rotations.shape[:-2]
    return Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_quat(scalar_first=True).reshape(*shape, 4)


def write_assets(asset_dir, object_id, mesh_path, hull_path, mass, com, inertia):
    asset_dir.mkdir(parents=True)
    shutil.copyfile(mesh_path, asset_dir / "visual.obj")
    shutil.copyfile(hull_path, asset_dir / "collision.obj")
    root = ET.Element("robot", name=object_id)
    link = ET.SubElement(root, "link", name=object_id)
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=" ".join(map(str, com)), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=str(mass))
    components = {(0,0):"ixx", (1,1):"iyy", (2,2):"izz", (0,1):"ixy", (0,2):"ixz", (1,2):"iyz"}
    ET.SubElement(inertial, "inertia", {name:str(inertia[ix]) for ix,name in components.items()})
    for name, path in (("visual", "visual.obj"), ("collision", "collision.obj")):
        node = ET.SubElement(link, name)
        ET.SubElement(ET.SubElement(node, "geometry"), "mesh", filename=path, scale="1 1 1")
    ET.indent(root)
    ET.ElementTree(root).write(asset_dir/"object.urdf", encoding="utf-8", xml_declaration=True)


def contact_geometry(mesh, world_obj, human, center, persisted):
    if not (mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0):
        raise ValueError("Oriented source scan required for inward contact normals")
    # Keep the original v2 operation order on exact shared-triangle edge ties.
    local = np.einsum("fti,fij->ftj", human[:, [4,8,12,16,20]]-world_obj[:,None,:3,3], world_obj[:,:3,:3])
    points, distances, triangles, _ = SurfaceQuery(mesh.vertices, mesh.faces).nearest(local.reshape(-1,3), chunk=24)
    points, normals = points.reshape(-1,5,3), -mesh.face_normals[triangles].reshape(-1,5,3)
    distances = distances.reshape(-1,5)
    active = distances <= .01
    if persisted:
        with np.load(persisted, allow_pickle=False) as stored:
            np.testing.assert_allclose(stored["object_origin_to_nominal_com_m"], center, atol=1e-12, rtol=0)
            np.testing.assert_allclose(stored["distance_to_scan_mm"], distances*1000, atol=1e-8, rtol=0)
            np.testing.assert_allclose(stored["projected_contact_comframe_m"]+center, points, atol=1e-10, rtol=0)
            np.testing.assert_allclose(stored["inward_normals_object"], normals, atol=1e-10, rtol=0)
            np.testing.assert_array_equal(stored["candidate_10mm"], active)
            points = stored["projected_contact_comframe_m"] + center
            normals, active = stored["inward_normals_object"], stored["candidate_10mm"]
    rot = world_obj[:,:3,:3].transpose(0,2,1)
    return points@rot+world_obj[:,None,:3,3], normals@rot, active, distances


def prepare(source_path, direct_path, root, stored=None):
    source, direct = (json.loads(p.read_bytes()) for p in (source_path,direct_path))
    assert direct["source_report"]["sha256"] == file_pin(source_path)["sha256"]
    model = mujoco.MjModel.from_binary_path(str(verify_file_pin(direct["mjb"])))
    mesh_path = verify_file_pin(source["geometry"]["source"])
    hull_path = verify_file_pin(source["geometry"]["hull"])
    mesh = trimesh.load_mesh(mesh_path, process=False)
    with np.load(verify_file_pin(source["source_poses"]), allow_pickle=False) as a:
        obj, human, frames = a["world_from_carton"], a["world_right_hand21"], a["frames"]
    with np.load(verify_file_pin(direct["trace"]), allow_pickle=False) as a:
        wrist, joint = a["target_wrist"], a["target_finger_q"]
        np.testing.assert_array_equal(a["ik_tip_targets"], human[:,[4,8,12,16,20]])
        target_fk = a["ik_tips"]
    palm, body = model.body("right_palm_link").id, model.body("carton").id
    fingers = [model.body(i).name for i in range(model.nbody) if model.body(i).name.startswith("right_finger")]
    nodes = [palm]+[model.body(name).id for name in fingers]
    hinges = model.actuator_trnid[:,0]
    qids = model.jnt_qposadr[hinges]
    names = [model.joint(i).name for i in hinges]
    assert names == [f"right_finger{i}_joint{j}" for i in range(1,6) for j in range(1,5)]
    d = mujoco.MjData(model)
    sites = [model.site(f"right_finger{i}_tip").id for i in range(1,6)]
    poses = []
    for i in range(len(frames)):
        d.qpos[:3] = wrist[i,:3,3]
        d.qpos[3:7] = wxyz(wrist[i,:3,:3])
        d.qpos[qids] = joint[i]
        mujoco.mj_kinematics(model,d)
        np.testing.assert_allclose(d.site_xpos[sites], target_fk[i], atol=1e-10, rtol=0)
        poses.append(np.column_stack((d.xpos[nodes], d.xquat[nodes])))
    poses = np.asarray(poses)
    com = model.body_ipos[body].copy()
    principal = Rotation.from_quat(model.body_iquat[body],scalar_first=True).as_matrix()
    inertia = principal @ np.diag(model.body_inertia[body]) @ principal.T
    mass = float(model.body_mass[body])
    cp, cn, ca, distances = contact_geometry(mesh, obj, human, com, stored)
    oid = source["object_id"]
    sequence = source["sequence"].split("/")[0].replace("_","")+"_"+oid
    relassets = Path("object_assets")/"hocap"/oid
    asset_dir = root/"assets"/relassets
    write_assets(asset_dir,oid,mesh_path,hull_path,mass,com,inertia)
    prefix = root/"assets/human_motion_data/hocap"
    stage = prefix/"reconstructed_stage"
    stage.mkdir(exist_ok=True, parents=True)
    assert direct["table"]["unit_normal"] == [0.0,0.0,1.0] and direct["table"]["z_intercept_m"] == 0
    stagefile=stage/f"{sequence}_support.usda"
    stagefile.write_text('#usda 1.0\n(\n    upAxis = "Z"\n    metersPerUnit = 1\n)\n'
                         'def Cube "Table"\n{\n    double size = 1.44\n'
                         '    double3 xformOp:translate = (0, 0, -0.72)\n'
                         '    uniform token[] xformOpOrder = ["xformOp:translate"]\n}\n',encoding="utf-8")
    motion = prefix/"processed"/f"sequence_id={sequence}"/"robot_name=wuji_hand2"
    motion.mkdir(parents=True)
    cache = motion/"reference_inputs.npz"
    np.savez_compressed(cache, frames=frames, robot_root_position=wrist[:,:3,3], robot_root_wxyz=wxyz(wrist[:,:3,:3]),
                        robot_joint_names=names, robot_joint_positions=joint, ee_link_names=["right_palm_link"],
                        ee_pose_w=poses[:,:1], hand_frame_names=fingers, hand_frames_w=poses[:,1:],
                        object_body_position=obj[:,None,:3,3], object_body_wxyz=wxyz(obj[:,None,:3,:3]),
                        contact_positions=cp, contact_normals=cn, contact_part_ids=ca.astype(np.int32),
                        hand_contact_active=ca.any(axis=1).astype(np.float32), tip_to_scan_m=distances)
    physical=dict(mass_kg=mass, com_body_m=com.tolist(), inertia_body_kg_m2=inertia.tolist(),
                  table=direct["table"], gravity_m_s2=direct["gravity_m_s2"],
                  material_hypotheses=direct["object_hypotheses"], independent_object_measurements=False,
                  support_source_scope="Flat z=0 table matching direct IK baseline. Effective contact friction and solver settings require scene configuration.")
    write_json(motion/"export.json",dict(
        sequence=sequence, original_sequence=source["sequence"], frame_range=frames[[0,-1]].tolist(), source_fps=30.,
        schema_version="motion_v1", motion_kind="single_robot", robot_name="wuji_hand2", side="right",
        object_id=oid, mesh_radius_m=float(np.linalg.norm(mesh.vertices-com,axis=1).max()),
        raw_source=file_pin(source_path), direct_ik=file_pin(direct_path), phase_scope="training/development",
        source_contact_support=file_pin(stored) if stored else None,
        contact_semantics="Five source tip joints on original scan, nearest triangles, inward normal, <=10 mm. Geometry proxy; neither human force nor certain contact label.",
        tool=file_pin(__file__), arrays=file_pin(cache), support=file_pin(stagefile), physics=physical,
        object_mesh=f"/producer/assets/{relassets.as_posix()}/visual.obj",
        object_urdf=f"/producer/assets/{relassets.as_posix()}/object.urdf",
        asset_pins=[file_pin(asset_dir/name) for name in ("visual.obj","collision.obj","object.urdf")],
    ))
    return motion


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("direct",type=Path)
    p.add_argument("root",type=Path)
    p.add_argument("--source-contact",type=Path)
    args=p.parse_args()
    print(prepare(args.source.resolve(), args.direct.resolve(), args.root.resolve(), args.source_contact))
