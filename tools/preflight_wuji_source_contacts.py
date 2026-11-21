"""Inspect the unmodified first reference frame on CPU using a resolved scene JSON and prepared HO-Cap parquets."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import mujoco
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
import warp as wp

from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.evaluation.harness import apply_evaluation_composition
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from run_wuji_matched_tranche import local_path


def read(path):
    return json.loads(path.read_bytes())


def pin(path):
    with path.open("rb") as stream:
        return dict(path=str(path.resolve()), sha256=hashlib.file_digest(stream,"sha256").hexdigest())


def scene(config_path, manifest):
    config = OmegaConf.create(read(config_path))
    for assignment in manifest["overrides"]:
        field, text = assignment.split("=",1)
        names = [field, "env."+field] if field.startswith("sim.") else [field]
        for name in names:
            actual = OmegaConf.select(config,name)
            expected = json.loads(text) if not isinstance(actual,str) else text
            if actual != expected:
                raise ValueError(f"Scene differs from declared physics: {name} = {actual!r}, expected {text}")
    if config.task.motion_speed != 1 or not config.scene.support:
        raise ValueError("Expected a nominal-speed, supported tabletop reference")
    cfg = OmegaConf.to_container(config,resolve=True)
    # Preserve the physical termination config just as the source evaluator does.
    termination = deepcopy(cfg["env"]["termination"])
    apply_evaluation_composition(cfg,world_count=1,reset_mode="explicit")
    cfg["env"]["termination"] = termination
    return OmegaConf.create(cfg)


def measure(parquet, cfg, protocol):
    export_path = parquet.parent/"export.json"
    export = read(export_path)
    for asset in [export["support"],*export["asset_pins"]]:
        if pin(local_path(asset["path"]))["sha256"] != asset["sha256"]:
            raise ValueError(f"Reference asset changed: {asset['path']}")
    env_config = instantiate_typed(cfg.env,RLEnvConfig)
    setup = setup_scene(parquet=parquet,control_fps=env_config.sim.fps,motion_speed=1.,
        source_frame_playback=True,world_count=1,embodiment=instantiate_typed(cfg.embodiment,Embodiment),
        collision=instantiate_typed(cfg.collision,CollisionPolicy),include_support=cfg.scene.support,
        decompose_objects=cfg.scene.decompose_objects,object_scale_min=cfg.scene.object_scale_min,
        object_scale_max=cfg.scene.object_scale_max,contact_friction=cfg.scene.contact_friction,
        object_free_joint_damping=cfg.scene.object_free_joint_damping)
    env = RLEnv(setup.scene,setup.reference,env_config)
    env.reset(frame_id=0)
    obj, = env.scene.objects
    pose = env.state_0.body_q.numpy()[obj.bodies[0].body_id]
    np.testing.assert_allclose(pose[:3],setup.reference.object_body_pos_w()[0,0],rtol=0,atol=5e-7)
    ref = setup.reference.object_body_quat_w()[0,0]
    np.testing.assert_allclose(Rotation.from_quat(pose[3:]).as_matrix(),
                               Rotation.from_quat(ref,scalar_first=True).as_matrix(),rtol=0,atol=5e-7)
    solver = env.solver
    model = solver.mj_model
    data = mujoco.MjData(model)
    solver._update_mjc_data(data,env.model,env.state_0)
    mujoco.mj_forward(model,data)
    groups = dict(box=obj.shapes.ids(), table=env.scene.support.shapes.ids(),
                  hand=[i for link in env.scene.layout.hand("right").link_geometry for i in link.shape_ids])
    owner = {int(i): name for name, indices in groups.items() for i in indices}
    geom_shape = solver.mjc_geom_to_newton_shape.numpy()[0]
    categories = {("box","table"):("box_table","maximum_carton_penetration_mm"),
                  ("box","hand"):("hand_box","maximum_carton_penetration_mm"),
                  ("hand","table"):("hand_table",None), ("hand","hand"):("hand_self","maximum_self_penetration_mm")}
    result = {}
    for label,key in categories.values():
        result[label] = dict(threshold_field=key,threshold_mm=protocol[key] if key else None,contacts=[])
    for contact in data.contact:
        g1,g2 = int(contact.geom1),int(contact.geom2)
        s1,s2 = int(geom_shape[g1]),int(geom_shape[g2])
        if s1 not in owner or s2 not in owner:
            raise ValueError(f"Unexpected contacting shapes {s1},{s2}; check the declared tabletop scene")
        pair = tuple(sorted((owner[s1],owner[s2])))
        if pair not in categories:
            continue
        label,_ = categories[pair]
        result[label]["contacts"].append(dict(geoms=[g1,g2],shapes=[s1,s2],names=[model.geom(g1).name,model.geom(g2).name],
                                             signed_distance_mm=1000*float(contact.dist)))
    for item in result.values():
        ds = [v["signed_distance_mm"] for v in item["contacts"]]
        minimum = min(ds,default=None)
        penetration = max(0.,-minimum) if minimum is not None else 0.
        item.update(minimum_reported_distance_mm=minimum,maximum_penetration_mm=penetration,
                    exceeds_initial_threshold=(penetration > item["threshold_mm"]) if item["threshold_mm"] is not None else None)
    exceeded = [k for k,v in result.items() if v["exceeds_initial_threshold"]]
    return dict(sequence=export["sequence"],object=export["object_id"],first_source_frame=export["frame_range"][0],
                reference=pin(parquet),export=pin(export_path),support=export["support"],object_assets=export["asset_pins"],
                reset_object_xyz_xyzw=pose.tolist(),contacts_by_pair=result,initial_exceedances=exceeded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,required=True,nargs="+")
    parser.add_argument("--scene-config",type=Path,required=True,help="Resolved FlashCHORD JSON (e.g. environment.json); only configuration is used")
    parser.add_argument("--physics",type=Path,default=REPO/"configs/flashchord_wuji_physics_v2.json")
    parser.add_argument("--thresholds",type=Path,default=REPO/"configs/hocap_carton_evaluation.json")
    parser.add_argument("--warp-cache",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.warp_cache:
        wp.config.kernel_cache_dir = str(args.warp_cache)
    wp.init()
    wp.set_device("cpu")
    if wp.get_cuda_devices():
        raise RuntimeError("Expected CPU-only process")
    cfg = scene(args.scene_config,read(args.physics))
    report = dict(scope="CPU static reset geometry only. Signed distances are MuJoCo contact distances; absent contacts have null distance. "
                  "Box-table and hand-box share the frozen carton penetration threshold. Hand-table is descriptive. "
                  "A within-limit initial frame gives no evidence for later frames.",
                  device="cpu",target_training_steps=0,actions=0,integration_steps=0,tool=pin(Path(__file__)),
                  scene_config=pin(args.scene_config),physics=pin(args.physics),thresholds=pin(args.thresholds),
                  references=[measure(p.resolve(),cfg,read(args.thresholds)) for p in args.reference])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,allow_nan=False,indent=2)+"\n",encoding="utf8")
    print(args.output)


if __name__=="__main__":
    main()
