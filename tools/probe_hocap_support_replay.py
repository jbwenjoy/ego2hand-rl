"""CPU-only paired support-height sensitivity of the two full-length cold tea-carton recordings."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
import trimesh
import warp as wp

from evaluate_flash_wuji_quality import score, source_truth
from flash_chord.configuration import instantiate_typed
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.flash_sac.config import TrainingConfig
from probe_flash_wuji_replay import rebuilt
from supermimic_research.runtime.flash_quality import COMMANDS, STATES, repeat_control_trajectory
from supermimic_research.ego.source_stage_contract import score_frozen_source_stages


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf8")


def project(study, out):
    stage = read(study / "registered_stages.json")["subjects"]["subject_5"]
    source = Path(stage["source"]["path"])
    meta = read(source)
    if digest(source) != stage["source"]["sha256"] or meta["object_id"] != "G15_2":
        raise ValueError("Unexpected source")
    mesh = source.parent / "carton_collision.obj"
    if digest(mesh) != meta["geometry"]["hull"]["sha256"]:
        raise ValueError("Collision mesh differs from the frozen source")
    parq = Path(stage["parquet"]["path"])
    truth, human, frames = source_truth(parq)
    verts = trimesh.load_mesh(mesh, process=False).vertices
    bottom = float(np.min((verts @ truth[0,:3,:3].T + truth[0,:3,3])[:,2]))
    root = next(p for p in parq.parents if p.name == "assets")
    target = out / "source_aligned" / "assets"
    if not target.exists():
        shutil.copytree(root, target)
    support, = root.glob("human_motion_data/*/reconstructed_stage/*_support.usda")
    old = support.read_bytes()
    token = b"xformOp:translate = (0, 0, -0.72)"
    if old.count(token) != 1:
        raise ValueError("Expected the original 1.44 m table at z=0")
    changed = old.replace(token, f"xformOp:translate = (0, 0, {bottom-0.72:.15f})".encode())
    copied_support = target / support.relative_to(root)
    if copied_support.read_bytes() not in [old, changed]:
        raise ValueError("Unexpected copied table")
    copied_support.write_bytes(changed)
    variant = target / parq.relative_to(root)
    if digest(variant) != stage["parquet"]["sha256"]:
        raise ValueError("Reference changed")
    contract_path = Path(stage["contract"]["path"])
    if digest(contract_path) != stage["contract"]["sha256"]:
        raise ValueError("Wrong source stages")
    contract = read(contract_path)
    masks = contract_path.with_name(contract["intervals_file"])
    if digest(masks) != contract["intervals_sha256"]:
        raise ValueError("Wrong source interval masks")
    with np.load(masks, allow_pickle=False) as z:
        source_arrays = dict(z)
    return parq, variant, truth, human, frames, bottom, source_arrays, contract


def run(args):
    args.out.mkdir(parents=True,exist_ok=True)
    wp.config.kernel_cache_dir = str(args.warp_cache)
    wp.init()
    wp.set_device("cpu")
    parq, variant, truth, human, frames, bottom, source_arrays, contract = project(args.study, args.out)
    cases = []
    for budget in [104640, 209280, 313920, 418560]:
        attempts = [read(p) for p in (args.study/"jobs"/f"subject_5_cold_single_eval_{budget}").glob("attempt_*.json")]
        job, = [j for j in attempts if j.get("success")]
        p = Path(job["receipt"]["path"])
        if digest(p) != job["receipt"]["sha256"]:
            raise ValueError("Wrong score receipt")
        raw = read(p)
        actor, = (row for row in raw["realizations"] if row["id"]=="actor")
        if actor["source_intervals"] == len(frames)-1:
            cases.append((budget, job, p, raw, actor))
    plan = dict(selection="All full-length closed-loop actor recordings among four frozen cold S5 checks.",
                scope="Post-hoc paired physical replay of identical recorded low-level commands, not retraining or a closed-loop policy test.",
                steps=[c[0] for c in cases], target_source_sha256=digest(parq), support_heights_m=[0.,bottom],
                fixed="Source first-frame convex-hull lower z; only the horizontal support is translated.",
                actor_receipts=[c[1]["receipt"] for c in cases], device="cpu")
    plan_path = args.out / "comparison_inputs.json"
    if plan_path.exists() and read(plan_path) != plan:
        raise ValueError("Changed diagnostic inputs")
    save(plan_path, plan)
    protocol = read(Path(__file__).resolve().parents[1] / "configs/hocap_carton_evaluation.json")
    summary = []
    for budget, job, path, raw, actor in cases:
        original_path = path.parent/"raw/trace.npz"
        if digest(original_path) != actor["trace_sha256"]:
            raise ValueError("Wrong original command recording")
        with np.load(original_path, allow_pickle=False) as z:
            original = dict(z)
        ckpt = Path(job["command"][job["command"].index("--checkpoint")+1])
        if digest(ckpt) != raw["checkpoint_sha256"]:
            raise ValueError("Changed original checkpoint")
        training_meta = read_checkpoint_metadata(ckpt)
        train = instantiate_typed(OmegaConf.create(checkpoint_config(training_meta)["training"]), TrainingConfig)
        for name, parquet, height in (("original",parq,0.), ("aligned",variant,bottom)):
            finished = args.out / f"{budget}_{name}.json"
            if finished.exists():
                result = read(finished)
                if result["plan_sha256"] != digest(plan_path):
                    raise ValueError("Changed diagnostic plan")
                summary.append(result)
                continue
            start = time.perf_counter()
            env = rebuilt(parquet, ckpt)
            if str(env.device) != "cpu" or not getattr(env.solver, "wuji_conditioning", None):
                raise ValueError("Wrong engine or device")
            if train.curriculum is not None:
                bridge = WarpRLEnv(env)
                stage = train.curriculum.stages[int(training_meta["curriculum_stage"])]
                bridge.set_curriculum(replace(stage, voc_scale=0.))
                env.reset(frame_id=0)
            body = env.scene.objects[0].bodies[0].body_id
            shapes = list(env.scene.support.shapes.ids())
            sbody, = set(env.model.shape_body.numpy()[shapes].tolist())
            if sbody < 0:
                raise ValueError("Missing kinematic table body")
            state = {k:getattr(env.state_0,k).numpy().copy() for k in STATES}
            robot_bodies = np.arange(len(state["body_q"])) != sbody
            np.testing.assert_allclose(state["body_q"][robot_bodies], original["body_q"][0,robot_bodies], rtol=0, atol=5e-6)
            np.testing.assert_allclose(state["body_qd"], original["body_qd"][0], rtol=0, atol=5e-6)
            # The Newton importer can encode the USD shift on the body or its shape.
            shape = shapes[0]
            local_center = env.model.shape_transform.numpy()[shape,:3]
            support_state = state["body_q"][sbody]
            center_z = (support_state[:3] + Rotation.from_quat(support_state[3:]).apply(local_center))[2]
            top = float(center_z + env.model.shape_scale.numpy()[shape,2])
            if abs(top-height) > 5e-6:
                raise ValueError(f"Compiled table top {top} differs from requested {height}")
            # repeat_control_trajectory compares reset witnesses byte for byte. Supply this CPU
            # witness after independently verifying the moving bodies against the archived GPU reset.
            replay_input = {"command_"+k:original["command_"+k] for k in COMMANDS}
            replay_input.update({k:v[None] for k,v in state.items()})
            env, rec, trajectory, stop = repeat_control_trajectory(lambda:env, replay_input, actor["terminal"], len(frames))
            metrics, _, gates = score(trajectory, truth, human, frames, env, rec, protocol, raw["identity"]["sequence"], stop, 0.)
            poses = np.repeat(np.eye(4)[None],len(trajectory["body_q"]),axis=0)
            observed = trajectory["body_q"][:,body]
            poses[:,:3,3] = observed[:,:3]
            poses[:,:3,:3] = Rotation.from_quat(observed[:,3:]).as_matrix()
            loads = trajectory["substep_contacts"].reshape(-1,env.config.sim.substeps,6)
            weight = raw["identity"]["mass_kg"]*9.81
            stages, stage_metrics = score_frozen_source_stages(source_arrays, contract, contract["config"], poses,
                    loads[:,:,0].mean(axis=1)/weight, loads[:,:,1].mean(axis=1)/weight, loads[:,:,2].max(axis=1))
            result = dict(plan_sha256=digest(plan_path), budget=budget, condition=name, device=str(env.device),
                          table_top_m=top, source_intervals=len(poses)-1, first_substep_box_penetration_mm=1000*loads[0,0,3],
                          native_stop=stop, identical_low_level_commands=True, physical_seconds=time.perf_counter()-start,
                          metrics={k:v for k,v in metrics.items() if "independent_command" not in k},
                          base_gates={k:v for k,v in gates.items() if k!="recorded_command_replay"},
                          source_stages=stages, stage_metrics=stage_metrics)
            save(finished, result)
            summary.append(result)
    save(args.out/"comparison.json", dict(plan=plan, runs=summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--warp-cache",type=Path,required=True)
    run(parser.parse_args())
