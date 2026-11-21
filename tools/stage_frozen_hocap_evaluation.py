"""Open a reserved source window under a frozen recipe and export direct-IK evaluation input."""
import argparse
import io
import json
from pathlib import Path
import time
import zipfile


def prepare_eval(reservation, freeze, output):
    import mujoco
    import numpy as np
    import trimesh
    from prepare_hocap_flashchord import prepare
    from stage_hocap_development_window import static_collision
    from supermimic_research.artifacts import file_pin, verify_file_pin
    from supermimic_research.ego import hocap_carton_control as control
    from supermimic_research.strict_json import write_json

    start=time.perf_counter()
    freeze_pin=file_pin(freeze)
    record=json.loads(reservation.read_bytes())
    if record["role"]!="future unseen trajectory; excluded from development" or not record["right_hand_only"]:
        raise ValueError("Expected a reserved source-only right-hand demonstration")
    bundle=verify_file_pin(record["source_bundle"])
    first,last=record["frames"]
    root=output/"source"
    root.mkdir(parents=True,exist_ok=False)
    mesh_path=root/"scan.obj"
    obj_id=record["object"]
    if not obj_id.isidentifier():
        raise ValueError("Invalid object identifier")
    with zipfile.ZipFile(bundle) as z:
        # Use known named artifacts only. No extraction of arbitrary member paths.
        mesh_path.write_bytes(z.read(f"models/{obj_id}/cleaned_mesh_10000.obj"))
        data=z.read("source_window.npz")
        import yaml
        source_meta=yaml.safe_load(z.read(record["sequence"]+"/meta.yaml"))
        ego_camera=source_meta["hololens"]["serial"]
        for flag,i in (("first",first),("last",last)):
            (root/f"source_{flag}.jpg").write_bytes(z.read(f'{record["sequence"]}/{ego_camera}/color_{i:06}.jpg'))
    (root/"source_carton.npz").write_bytes(data)
    with np.load(io.BytesIO(data),allow_pickle=False) as a:
        frames=a["frames"]; hand=a["world_right_hand21"];obj=a["world_from_carton"]
    np.testing.assert_array_equal(frames,np.arange(first,last+1))
    if hand.shape!=(len(frames),21,3) or obj.shape!=(len(frames),4,4) or not np.isfinite(hand).all() or not np.isfinite(obj).all():
        raise ValueError("Invalid reserved source geometry")
    mesh=trimesh.load_mesh(mesh_path,process=False)
    assets=static_collision(mesh,obj_id,root)
    assets["source"]=file_pin(mesh_path)
    original=root/"source.json"
    write_json(original,dict(object_id=obj_id,sequence=record["sequence"],frames=[first,last],
               role="evaluation after recipe freeze",freeze=freeze_pin,source_reservation=file_pin(reservation),
               endpoint_image_pins=[file_pin(root/"source_first.jpg"),file_pin(root/"source_last.jpg")],
               source_poses=file_pin(root/"source_carton.npz"),geometry=assets))
    start_ik=time.perf_counter()
    normal=np.array([0.,0.,1.])
    model,wrist=control.create_scene(hand,obj[0],mesh.convex_hull,.15,.7,normal,0.,1/1200)
    fingers,tips,steps=control.solve_kinematics(model,wrist,hand,hand[:,control.TIPS])
    mjb=root/"kinematics.mjb";mujoco.mj_saveModel(model,str(mjb),None)
    trace=root/"direct_ik.npz"
    np.savez_compressed(trace,target_wrist=wrist,target_finger_q=fingers,ik_tip_targets=hand[:,control.TIPS],
                        ik_tips=tips,ik_solver_evaluations=steps)
    ik_time=time.perf_counter()-start_ik
    body=model.body("carton").id
    ik_receipt=root/"direct_ik.json"
    write_json(ik_receipt,dict(source_report=file_pin(original),mjb=file_pin(mjb),trace=file_pin(trace),
               table=dict(unit_normal=normal.tolist(),z_intercept_m=0.),gravity_m_s2=model.opt.gravity.tolist(),
               object_hypotheses=dict(mass_kg=float(model.body_mass[body]),hand_object_friction=.7,
               table_object_friction=.6,measured=False,inertia_from_uniform_convex_solid=True),
               native_integration_steps=0,wall_s=ik_time))
    motion=prepare(original,ik_receipt,output)
    manifest=motion/"export.json"
    export=json.loads(manifest.read_bytes())
    export.update(phase_scope="frozen_heldout_evaluation",frozen_recipe=freeze_pin,
                  policy_inputs="Direct kinematic robot reference, object trajectory and scan geometry. No corrected or native rollout from this motion.",
                  source_reservation=file_pin(reservation))
    write_json(manifest,export)
    write_json(output/"stage_timing.json",dict(role=export["phase_scope"],frozen_recipe=freeze_pin,
               source_prepare_wall_s=time.perf_counter()-start,direct_ik_wall_s=ik_time,
               released_reference_frames=len(frames),reward_or_policy_updates=0))
    print(motion)


def verify_eval(output):
    from materialize_hocap_flashchord import materialize
    import warp as wp
    wp.set_device("cpu")
    files=list(output.glob("assets/human_motion_data/hocap/processed/*/*/export.json"))
    if len(files)!=1:
        raise ValueError("Expected an isolated evaluation export")
    export=json.loads(files[0].read_bytes())
    if export["phase_scope"]!="frozen_heldout_evaluation" or not export["frozen_recipe"]["sha256"]:
        raise ValueError("Missing evaluation freeze provenance")
    start=time.perf_counter()
    receipt=materialize(files[0])
    receipt.update(role=export["phase_scope"],frozen_recipe_sha256=export["frozen_recipe"]["sha256"],
                   materialize_and_verify_s=time.perf_counter()-start)
    (output/"evaluation_reference.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest="mode",required=True)
    ingest=subs.add_parser("prepare")
    ingest.add_argument("reservation",type=Path)
    ingest.add_argument("frozen_recipe",type=Path)
    ingest.add_argument("output",type=Path)
    verify=subs.add_parser("verify")
    verify.add_argument("output",type=Path)
    args=parser.parse_args()
    if args.mode=="prepare":
        prepare_eval(args.reservation.resolve(),args.frozen_recipe.resolve(),args.output.resolve())
    else:
        verify_eval(args.output.resolve())
