"""Write and check one-row motion_v1 parquet files in the official FlashCHORD environment."""
import argparse
import hashlib
import json
from pathlib import Path

import newton
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import warp as wp

from flash_chord.assets import registry
from flash_chord.data.loader import load_reference
from flash_chord.data.motion_v1 import MotionV1Reference
from flash_chord.objectives.wrench import compute_reference_supports
from flash_chord.scene.objects import spawn_objects
from flash_chord.scene.support import add_support_surfaces


def pin(path):
    return dict(filename=path.as_posix(), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def to_row(export, values):
    oid = export["object_id"]
    row = dict(schema_version="motion_v1", motion_kind="single_robot", source_dataset="hocap",
               sequence_id=export["sequence"], robot_name=export["robot_name"], coord_frame="world",
               raw_motion_file=f'{export["original_sequence"]} frames {export["frame_range"]}',
               fps=export["source_fps"], hand_sides=["right"], object_name=oid, object_body_names=[oid],
               object_mesh_paths=[export["object_mesh"]], object_urdf_paths=[export["object_urdf"]],
               object_mesh_radius=[export["mesh_radius_m"]], object_articulation=[],
               source_frames=values["frames"].tolist(), source_attribution="Wang et al., HO-Cap, CC BY 4.0")
    for field in ("robot_joint_names","robot_joint_positions","robot_root_position","robot_root_wxyz",
                  "ee_link_names","ee_pose_w","object_body_position","object_body_wxyz"):
        row[field] = values[field].tolist()
    for col, key in (("hand_frame_names","hand_frame_names"), ("hand_frames_w","hand_frames_w"),
                     ("hand_object_contact_positions","contact_positions"), ("hand_object_contact_normals","contact_normals"),
                     ("hand_object_contact_part_ids","contact_part_ids"), ("hand_contact_active","hand_contact_active")):
        row[col] = [values[key].tolist()]
    return row


def materialize(manifest):
    export = json.loads(manifest.read_bytes())
    work = manifest.parent
    cache = work / "reference_inputs.npz"
    assert pin(cache)["sha256"] == export["arrays"]["sha256"]
    with np.load(cache, allow_pickle=False) as files:
        values = dict(files)
    parquet = work / "data.parquet"
    row = to_row(export, values)
    table = pa.Table.from_pylist([row])
    if not parquet.exists():
        candidate = work / "data.parquet.tmp"
        pq.write_table(table, candidate, compression="zstd")
        candidate.replace(parquet)
    assert pq.ParquetFile(parquet).read().equals(table)
    ref = load_reference(parquet)
    assert isinstance(ref, MotionV1Reference) and ref.sides == ("right",)
    assert ref.num_frames == len(values["frames"]) and ref.fps == export["source_fps"]
    assert ref.metadata.sequence_id == export["sequence"] and ref.metadata.robot_name == "wuji_hand2"
    assert ref.metadata.coord_frame=="world" and not ref.metadata.is_resampled
    assert ref.robot_joint_names() == values["robot_joint_names"].tolist()
    assert ref.robot_frame_names() == values["ee_link_names"].tolist()+values["hand_frame_names"].tolist()
    assert ref.object_articulation().shape == (ref.num_frames,0)
    frame_pose = np.concatenate((values["ee_pose_w"], values["hand_frames_w"]),axis=1)
    checks = dict(joints=(ref.robot_joint_pos(),values["robot_joint_positions"]),
                  root_position=(ref.robot_root_pos_w(),values["robot_root_position"]),
                  root_wxyz=(ref.robot_root_quat_w(),values["robot_root_wxyz"]),
                  frame_position=(ref.robot_frame_pos_w(),frame_pose[:,:,:3]),
                  frame_wxyz=(ref.robot_frame_quat_w(),frame_pose[:,:,3:]),
                  object_position=(ref.object_body_pos_w(),values["object_body_position"]),
                  object_wxyz=(ref.object_body_quat_w(),values["object_body_wxyz"]),
                  contact_position=(ref.contact_pos_w("right"),values["contact_positions"]),
                  contact_normal=(ref.contact_normal_w("right"),values["contact_normals"]),
                  contact_ids=(ref.contact_part_ids("right"),values["contact_part_ids"]),
                  activity=(ref.contact_active("right"),values["hand_contact_active"]))
    errors = {}
    for name, (actual, expected) in checks.items():
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
        errors[name] = float(np.max(np.abs(actual-expected)))
    spec, = ref.object_assets()
    visual = Path(registry.resolve_asset_path(ref.object_mesh_paths()[0],reference_path=parquet))
    urdf = Path(spec.urdf_path)
    for p in (visual, urdf.parent/"collision.obj",urdf):
        match = next(v for v in export["asset_pins"] if v["path"].replace("\\","/").endswith("/"+p.name))
        assert p.is_file() and pin(p)["sha256"] == match["sha256"]
    support = registry.support_usda_for_reference(ref)
    assert support.is_file() and pin(support)["sha256"] == export["support"]["sha256"]
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    binding, = spawn_objects(builder, ref, decompose_collision=False)
    body = binding.root.body_id
    phy = export["physics"]
    np.testing.assert_allclose(builder.body_mass[body],phy["mass_kg"],rtol=1e-6,atol=1e-9)
    np.testing.assert_allclose(np.array(builder.body_com[body]),phy["com_body_m"],rtol=1e-6,atol=1e-8)
    tensor = np.asarray(builder.body_inertia[body]).reshape(3,3)
    np.testing.assert_allclose(tensor,phy["inertia_body_kg_m2"],rtol=1e-4,atol=2e-8)
    support_binding = add_support_surfaces(builder,support)
    assert len(support_binding.shapes)==1
    official_cws = compute_reference_supports(ref, sides=("right",), device="cpu")
    assert official_cws.shape == (ref.num_frames,1,1,512) and np.isfinite(official_cws).all()
    active = values["hand_contact_active"].astype(bool)
    assert np.all(official_cws[~active]==0) and np.all(official_cws[active].max(axis=(1,2,3))>0)
    cws_path = work / "official_source_cws.npy"
    if cws_path.exists():
        np.testing.assert_array_equal(np.load(cws_path),official_cws)
    else:
        np.save(cws_path,official_cws,allow_pickle=False)
    return dict(sequence=export["sequence"], parquet=pin(parquet), frames=ref.num_frames, loader_max_abs_error=errors,
                verified_object=pin(urdf), verified_support=pin(support), newton_mass_kg=float(builder.body_mass[body]),
                newton_com_m=np.array(builder.body_com[body]).tolist(), newton_inertia_kg_m2=tensor.tolist(),
                official_cws=pin(cws_path), source_proxy_active_frames=int(active.sum()),
                direct_target_scope="The right palm root and 25 right-hand frames are kinematic FK of direct IK only.",
                native_parity_scope="Verifies object and support import plus raw reference. Full robot/contact solver equivalence requires a separate rollout.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    args=p.parse_args()
    wp.set_device("cpu")
    root=args.root.resolve()
    reports=[materialize(m) for m in sorted(root.glob("assets/human_motion_data/hocap/processed/*/*/export.json"))]
    if len(reports)!=3:
        raise ValueError(f"Expected three development sequences, found {len(reports)}")
    result=dict(rows=reports, tool=pin(Path(__file__).resolve()))
    (root/"loader_verification.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    for row in reports: print(row["sequence"],row["frames"],max(row["loader_max_abs_error"].values()))
