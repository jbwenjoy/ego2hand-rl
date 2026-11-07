"""SOURCE DRAFT: explicit --run only, shared CPU model/initial-state functions.

No physical backend is imported by the metadata/default route. Source review is
required before Root's coordinated GPU invocation. No hand or old model weights.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CPU_SOURCE = HERE / "run_phone_cpu_bench_v1.py"
KERNEL_SOURCE = HERE / "warp_phone_kernels_v1.py"
TESTS = ("drop", "settle", "push_release")
TEST_PROTOCOL_NAMES = {"drop": "drop", "settle": "table_settling", "push_release": "force_push_release"}
NCONMAX_PER_WORLD = 64
NJMAX_PER_WORLD = 256
BLOCK_STEPS = 20


def pin(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def write(path, obj):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def plan(protocol_path, asset_path):
    p = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if p["schema"] != "phone_physics_stage1_protocol:1":
        raise ValueError("Wrong protocol schema")
    cases = p["cases"]
    masses, mus = cases["cartesian_grid"]["mass_kg"], cases["cartesian_grid"]["sliding_friction"]
    dts = cases["timesteps_s"]
    if (masses, mus, dts) != ([.1, .2, .3], [.3, .6, .9], [.0025, .00125]):
        raise ValueError("Unexpected frozen grid; no automatic adaptation")
    grid = [(.2, .6)] + [x for x in itertools.product(masses, mus) if x != (.2, .6)]
    groups = [{"mass_kg": mass, "mu": mu, "dt_s": dt, "worlds": list(TESTS),
               "steps_per_world": round(3/dt), "push_start_tick": round(1/dt),
               "push_end_tick": round(1.2/dt)} for mass, mu in grid for dt in dts]
    versions = {}
    for pkg in ("mujoco", "mujoco-warp", "warp-lang", "numpy"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    sdk_sources = {}
    for pkg, relative_files in {
        "mujoco-warp": ["mujoco_warp/_src/forward.py", "mujoco_warp/_src/io.py", "mujoco_warp/_src/types.py",
                        "mujoco_warp/_src/smooth.py", "mujoco_warp/_src/solver.py", "mujoco_warp/_src/collision_driver.py",
                        "mujoco_warp/_src/support.py"],
        "warp-lang": ["warp/_src/context.py"],
    }.items():
        distribution = importlib.metadata.distribution(pkg)
        for relative in relative_files:
            sdk_sources[relative] = pin(distribution.locate_file(relative))
    asset_pins = {"asset_XML": pin(asset_path),
                  "original_GLB": pin(ROOT/"hot3d/object_models/obj_000024.glb"),
                  "baked_visual_mesh": pin(Path(asset_path).parent/"meshes/phone24_visual_m.obj"),
                  "asset_geometry_manifest": pin(Path(asset_path).parent/"geometry_report.json"),
                  "geometry_approximation_audit": pin(Path(asset_path).parent/"aabb_geometry_audit_v1/report.json")}
    return {"schema": "phone_warp_bench_plan:1", "status": "METADATA_ONLY_NOT_EXECUTED",
            "protocol": pin(protocol_path), "asset": pin(asset_path),
            "asset_pins": asset_pins,
            "sources": {"runner": pin(__file__), "kernels": pin(KERNEL_SOURCE), "CPU_shared_model": pin(CPU_SOURCE)},
            "versions_metadata_only": versions, "groups": groups, "case_count": 54,
            "relevant_backend_sources": sdk_sources,
            "batch_steps_planned": sum(g["steps_per_world"] for g in groups),
            "world_steps_planned": sum(3*g["steps_per_world"] for g in groups),
            "additional_final_forward_calls_planned": len(groups),
            "xfrc_input_source_contract": {
                "order": "[world_Fx,world_Fy,world_Fz,world_COM_tau_x,world_COM_tau_y,world_COM_tau_z]",
                "io.py": "put_data lines2118-2123 and _create_array lines65-93 copy native MjData field without six-component reordering",
                "support.py": "_apply_ft lines265-307 swaps cdof into [linear,angular] before dot(ft), and transform_force lines470-478 reads spatial_top(frc) as force, spatial_bottom as torque",
                "smooth.py": "lines805-812 freejoint translation has world basis; rotations child/body basis",
                "forward.py": "line1314 xfrc_accumulate maps d.xfrc_applied into qfrc_smooth",
                "evidence": "pinned local installed SDK source, not an assumption about generic spatial_vector order"},
            "contact_phase": "each step's native forward contacts belong to tick i; post state is tick i+1; final extra forward captures tick N contacts",
            "extra_forward_is_not_physics_step": True, "GPU_initialization_executed": False}


def post_descriptors(np, cpu, qpos, qvel, geometry, mass):
    # Exact rigid-body algebra on saved POST freejoint state, not FK/collision.
    q = qpos[:, 3:7].astype(np.float64)
    norms = np.linalg.norm(q, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Nonfinite/zero quaternion")
    # Match shared CPU helper's raw saved quaternion arithmetic; never repair it.
    w, x, y, z = q.T
    R = np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                  2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                  2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], axis=1).reshape(-1, 3, 3)
    center = np.asarray(geometry["center_m"], dtype=np.float64)
    com, vcom, omega_world = cpu.kinematics_from_state(qpos.astype(np.float64), qvel.astype(np.float64), center)
    omega_body = qvel[:, 3:6].astype(np.float64)
    bottom = com[:, 2] - (np.abs(R[:, 2, :])*np.asarray(geometry["half_m"])).sum(axis=1)
    U = mass*9.81*com[:, 2]
    T = .5*mass*np.sum(vcom*vcom, axis=1) + .5*np.sum(omega_body**2*np.asarray(geometry["inertia_kg_m2"]), axis=1)
    return {"com_world_m": com, "com_velocity_world_m_s": vcom, "omega_world_rad_s": omega_world,
            "box_bottom_world_z_m": bottom, "quaternion_norm": norms,
            "energy_potential_kinetic_J": np.stack([U, T], axis=1)}


def run_group(np, wp, mjw, cpu, kernels, asset, group, output, device, partial):
    mass, mu, dt = group["mass_kg"], group["mu"], group["dt_s"]
    n = group["steps_per_world"]
    model, geometry = cpu.make_model(asset, mass, mu, dt)
    options = cpu.compiled_options(model)
    xml_path = output/f"warp_model_mass{mass:g}_mu{mu:g}_dt{dt:g}.xml"
    with xml_path.open("x", encoding="utf-8") as stream:
        stream.write(geometry["xml"])
    bid = model.body(cpu.BODY).id
    initial = [cpu.initial_state(model, geometry, test) for test in TESTS]
    q0 = np.stack([a.qpos.copy() for a in initial]).astype(np.float32)
    v0 = np.stack([a.qvel.copy() for a in initial]).astype(np.float32)
    d = mjw.put_data(model, initial[0], nworld=3, nconmax=NCONMAX_PER_WORLD, njmax=NJMAX_PER_WORLD)
    m = mjw.put_model(model)
    wp.copy(d.qpos, wp.array(q0, dtype=wp.float32, device=device))
    wp.copy(d.qvel, wp.array(v0, dtype=wp.float32, device=device))
    wp.synchronize()
    if not np.array_equal(d.qpos.numpy().view(np.uint8), q0.view(np.uint8)) or not np.array_equal(d.qvel.numpy().view(np.uint8), v0.view(np.uint8)):
        raise ValueError("Installed GPU initial state differs from shared CPU state cast to F32")
    # Initial/drop-derived caches copied by put_data are not sampling authority.
    # First step recomputes every world's forward fields from the installed qpos.
    cap = d.naconmax
    rows = n+1
    cursor = wp.zeros(1, dtype=int, device=device)
    def f32(shape):
        return wp.full(shape, float("nan"), dtype=float, device=device)
    oq, ov, ot, of = f32((rows, 3, 7)), f32((rows, 3, 6)), f32((rows, 3)), f32((rows, 3))
    oo, oe = wp.full((rows, 3), -1, dtype=int, device=device), wp.full((rows, 3), -1, dtype=int, device=device)
    count = wp.full(rows, -1, dtype=int, device=device)
    cd = f32((rows, cap))
    cp = wp.zeros((rows, cap), dtype=wp.vec3, device=device)
    cf = wp.zeros((rows, cap), dtype=wp.mat33, device=device)
    cg = wp.zeros((rows, cap), dtype=wp.vec2i, device=device)
    cw, ct, cdim = [wp.full((rows, cap), -1, dtype=int, device=device) for _ in range(3)]
    partial.clear()
    partial.update({"group": group, "cursor": cursor,
                    "arrays": {"qpos": oq, "qvel": ov, "time_s": ot, "applied_force": of,
                               "overflow": oo, "nefc": oe, "nacon": count, "contact_dist": cd,
                               "contact_pos": cp, "contact_frame": cf, "contact_geom": cg,
                               "contact_worldid": cw, "contact_type": ct, "contact_dim": cdim}})
    contact_args = [d.nacon, d.contact.dist, d.contact.pos, d.contact.frame, d.contact.geom,
                    d.contact.worldid, d.contact.type, d.contact.dim, cursor, cap,
                    count, cd, cp, cf, cg, cw, ct, cdim]
    def step_and_record():
        mjw.step(m, d)
        wp.launch(kernels.record_pre_contacts, dim=cap, inputs=contact_args, device=device)
        wp.launch(kernels.record_post_state, dim=3,
                  inputs=[d.qpos, d.qvel, d.time, d.xfrc_applied, d.overflow, d.nefc, cursor, bid,
                          oq, ov, ot, of, oo, oe], device=device)
        wp.launch(kernels.advance_cursor, dim=1, inputs=[cursor], device=device)
    def set_force(force):
        raw = np.zeros((3, model.nbody, 6), dtype=np.float32)
        raw[2, bid, 0] = force
        wp.copy(d.xfrc_applied, wp.array(raw, dtype=wp.spatial_vector, device=device))
        wp.synchronize()
    # Captures enqueue no launched integration. Their calls are separately logged.
    segments = [(0, group["push_start_tick"], 0.),
                (group["push_start_tick"], group["push_end_tick"], 2.),
                (group["push_end_tick"], n, 0.)]
    launches, graph_steps = [], 0
    for phase, (begin, end, force) in enumerate(segments):
        if int(cursor.numpy()[0]) != begin:
            raise ValueError("Force phase cursor mismatch")
        set_force(force)
        with wp.ScopedCapture(device=device) as capture:
            for _ in range(BLOCK_STEPS):
                step_and_record()
        remaining = end-begin
        for _ in range(remaining//BLOCK_STEPS):
            wp.capture_launch(capture.graph)
            graph_steps += BLOCK_STEPS
        # All current dt event ticks are divisible by20; reject unexpected tails.
        if remaining % BLOCK_STEPS:
            raise ValueError("Phase graph would cross input boundary")
        wp.synchronize()
        completed = int(cursor.numpy()[0])
        launches.append({"phase": phase, "first_tick": begin, "end_tick_exclusive": end,
                         "force_push_world_x_N": force, "graph_block_steps": BLOCK_STEPS,
                         "graph_launches": remaining//BLOCK_STEPS, "returned_batch_steps": completed-begin})
        if completed != end:
            raise ValueError("Graph launch step-count mismatch")
        if np.any(d.overflow.numpy() != 0):
            raise ValueError("Warp capacity overflow; no retry")
    mjw.forward(m, d)  # t=N contact sampling only, never counted as integration.
    wp.launch(kernels.record_pre_contacts, dim=cap, inputs=contact_args, device=device)
    wp.synchronize()
    final_overflow = d.overflow.numpy()
    final_nefc = d.nefc.numpy()
    q, v, times, forces, overflow, nefc = [a.numpy() for a in (oq, ov, ot, of, oo, oe)]
    q[0], v[0], times[0], forces[0], overflow[0], nefc[0] = q0, v0, 0., 0., 0, 0
    raw = {"count": count.numpy(), "dist": cd.numpy(), "pos": cp.numpy(), "frame": cf.numpy(),
           "geom": cg.numpy(), "worldid": cw.numpy(), "type": ct.numpy(), "dim": cdim.numpy()}
    if np.any(raw["count"] < 0) or np.any(raw["count"] > cap):
        raise ValueError("Missing/truncated contact row")
    valid = np.arange(cap)[None, :] < raw["count"][:, None]
    tick, slot = np.nonzero(valid)
    contacts = {"state_tick": tick, "original_Warp_contact_slot": slot,
                **{k: a[valid] for k, a in raw.items() if k != "count"}}
    if np.any(contacts["worldid"] < 0) or np.any(contacts["worldid"] >= 3) or np.any(contacts["geom"] < 0) or np.any(contacts["geom"] >= model.ngeom):
        raise ValueError("Invalid generated contact ownership/geom ID")
    if not all(np.isfinite(a).all() for a in (q, v, times, forces)):
        raise ValueError("Nonfinite recorded state")
    if np.any(overflow) or np.any(final_overflow) or np.any(nefc > NJMAX_PER_WORLD) or np.any(final_nefc > NJMAX_PER_WORLD):
        raise ValueError("Recorded capacity overflow")
    effective_dt = float(m.opt.timestep.numpy()[0])
    case_rows = []
    for world, test in enumerate(TESTS):
        case = f"{test}_mass{mass:g}_mu{mu:g}_dt{dt:g}"
        derived = post_descriptors(np, cpu, q[:, world], v[:, world], geometry, mass)
        mask = contacts["worldid"] == world
        c = {k: a[mask] for k, a in contacts.items()}
        if not np.isfinite(c["dist"]).all() or not np.isfinite(c["pos"]).all() or not np.isfinite(c["frame"]).all():
            raise ValueError("Nonfinite generated contacts")
        ncon, depth = np.bincount(c["state_tick"], minlength=rows), np.zeros(rows)
        minimum = np.full(rows, np.nan)
        for step in np.unique(c["state_tick"]):
            ds = c["dist"][c["state_tick"] == step]
            minimum[step] = ds.min()
            depth[step] = max(0., -float(ds.min()))
        trace = {"time_s": times[:, world], "integer_tick": np.arange(rows),
                 "tick_derived_requested_time_s": np.arange(rows)*dt,
                 "tick_derived_effective_time_s": np.arange(rows)*effective_dt,
                 "qpos": q[:, world], "qvel": v[:, world], **derived,
                 "applied_com_force_world_x_N": forces[:, world], "contact_count": ncon,
                 "penetration_m": depth, "minimum_generated_contact_distance_m": minimum,
                 "empty_generated_contact_set": ncon == 0, "overflow_flags": overflow[:, world],
                 "pre_solver_nefc_after_step": nefc[:, world]}
        tp, cp_path = output/(case+".npz"), output/(case+"_contacts.npz")
        np.savez_compressed(tp, **trace)
        np.savez_compressed(cp_path, **c)
        rest = np.arange(rows)*dt >= 2.5-dt/4
        com = derived["com_world_m"]
        e = derived["energy_potential_kinetic_J"].sum(1)
        work = float(np.sum(forces[1:, world]*np.diff(com[:, 0])))
        case_rows.append({"case": case, "test": test, "protocol_test_name": TEST_PROTOCOL_NAMES[test],
            "mass_kg": mass, "sliding_friction": mu, "dt_s": dt, "dt_effective_s": effective_dt,
            "metrics": {"completed_steps": n, "expected_steps": n, "final_time_s": float(times[-1, world]),
                        "clock_error_vs_N_effective_dt_s": float(times[-1, world]-n*effective_dt),
                        "finite": True, "max_penetration_m": float(depth.max()),
                        "rest_linear_speed_rms_m_s": float(np.sqrt(np.mean(np.sum(derived["com_velocity_world_m_s"][rest]**2, axis=1)))),
                        "rest_angular_speed_rms_rad_s": float(np.sqrt(np.mean(np.sum(derived["omega_world_rad_s"][rest]**2, axis=1)))),
                        "final_root_qpos": q[-1, world].tolist(), "final_com_world_m": com[-1].tolist(),
                        "com_displacement_x_m": float(com[-1, 0]-com[0, 0]),
                        "force_impulse_N_s": float(np.sum(forces[1:, world], dtype=np.float64)*dt),
                        "initial_energy_J": float(e[0]), "final_energy_J": float(e[-1]), "max_energy_J": float(e.max()),
                        "known_external_work_J": work, "energy_gain_minus_known_external_work_J": float(e[-1]-e[0]-work),
                        "max_contacts": int(ncon.max()), "contact_observed": bool(ncon.max() > 0)},
            "trajectory": pin(tp), "contact_table": pin(cp_path),
            "model_xml_sha256": geometry["model_xml_sha256"],
            "compiled_model_XML": pin(xml_path),
            "uniform_box_inertia_kg_m2": geometry["inertia_kg_m2"].tolist()})
    return case_rows, {"group": group, "graph_launches": launches, "batch_steps_returned": graph_steps,
                       "world_steps_returned": graph_steps*3, "additional_forward_calls": 1,
                       "final_cursor": int(cursor.numpy()[0]), "capacity_naconmax": cap,
                       "CPU_compiled_model_options": options,
                       "GPU_effective_options": {"timestep_F32": effective_dt, "tolerance_F32": float(m.opt.tolerance.numpy()[0]),
                            "ls_tolerance_F32": float(m.opt.ls_tolerance.numpy()[0]), "solver": int(m.opt.solver),
                            "integrator": int(m.opt.integrator), "cone": int(m.opt.cone), "iterations": int(m.opt.iterations),
                            "ls_iterations": int(m.opt.ls_iterations), "enableflags": int(m.opt.enableflags),
                            "disableflags": int(m.opt.disableflags)},
                       "capacity_njmax_per_world": NJMAX_PER_WORLD, "model_xml_sha256": geometry["model_xml_sha256"],
                       "final_overflow_flags": final_overflow.tolist(), "final_nefc": final_nefc.tolist()}


def run(args, metadata):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    write(args.output/"metadata.json", metadata)
    started = time.perf_counter()
    cases, groups = [], []
    partial = {}
    try:
        for item in metadata["sources"].values():
            if pin(item["path"])["sha256"] != item["sha256"]:
                raise ValueError("Shared/source changed after metadata binding")
        import numpy as np
        import warp as wp
        import mujoco_warp as mjw
        cpu = import_file("phone_shared_cpu_bench_v1", CPU_SOURCE)
        kernels = import_file("warp_phone_kernels_v1", KERNEL_SOURCE)
        wp.init()
        device = wp.get_device(args.device)
        if not device.is_cuda:
            raise ValueError("Warp bench requires Root-coordinated CUDA device; no fallback")
        with wp.ScopedDevice(device):
            for group in metadata["groups"]:
                c, g = run_group(np, wp, mjw, cpu, kernels, args.asset, group, args.output, device, partial)
                cases.extend(c)
                groups.append(g)
                write(args.output/f"warp_group_{len(groups):02d}.json", g)
                print(json.dumps({"status": "WARP_GROUP_RECORDED", "groups": len(groups), "cases": len(cases)}), flush=True)
        result = {"schema": "phone_free_object_warp_bench:1", "status": "RECORDED_NOT_YET_ASSESSED",
            "created_UTC": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "metadata": pin(args.output/"metadata.json"), "protocol": metadata["protocol"],
            "asset_pins": metadata["asset_pins"],
            "sources": metadata["sources"], "cases": cases, "groups": groups, "case_count": len(cases),
            "relevant_backend_sources": metadata["relevant_backend_sources"],
            "world_steps_returned": sum(g["world_steps_returned"] for g in groups),
            "batch_steps_returned": sum(g["batch_steps_returned"] for g in groups),
            "extra_forward_sampling_calls": len(groups), "elapsed_seconds": time.perf_counter()-started,
            "API_counts": {"CPU_model_compilations": len(groups), "CPU_initial_state_forward_calls": 3*len(groups),
                           "GPU_put_model_calls": len(groups), "GPU_put_data_calls": len(groups),
                           "GPU_integrating_batch_steps": sum(g["batch_steps_returned"] for g in groups),
                           "GPU_additional_nonintegrating_forward_calls": len(groups),
                           "warmup_integrating_steps": 0},
            "contact_sampling_phase": "native PRE contacts saved at tick i; final forward supplies tickN. State/rigid descriptors POST at matching same ticks",
            "contact_force_phase_scope": "PRE tick i contacts are computed with force over interval [i,i+1), while saved post-state force row i is the just-ended interval [i-1,i). No native contact forces exported; do not reinterpret boundary interval force as a contact-force label.",
            "sampling_forward_scope": "CPU runs post-forward every step; this runner adds only the final forward per group. Counts are intentionally different. Local state/geometry parity is measured, not a whole-hidden-state equivalence claim.",
            "energy_scope": "F64 rigid T + gravitational U on saved states; engine energy not exported; no soft-contact energy claim",
            "native_contact_force_exported": False, "thresholds_assessed": False,
            "real_parameters_calibrated": False, "ego_hand_contact_qualified": False,
            "Hand2_model_weights_training_executed": False}
        write(args.output/"result.json", result)
        print(json.dumps({"status": result["status"], "result": pin(args.output/"result.json"),
                          "cases": len(cases), "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    except Exception as error:
        partial_receipt = {"partial_gpu_readback_attempted": bool(partial)}
        if partial:
            try:
                wp.synchronize()
                completed = int(partial["cursor"].numpy()[0])
                partial_path = args.output/"warp_failed_group_partial.npz"
                np.savez_compressed(partial_path, **{k: a.numpy() for k, a in partial["arrays"].items()})
                partial_receipt.update(group=partial["group"], completed_batch_steps=completed,
                                       raw_nan_initialized_capture=pin(partial_path))
            except Exception as readback_error:
                partial_receipt["partial_readback_error"] = repr(readback_error)
        write(args.output/"failure.json", {"status": "FAILED_PARTIAL_NO_RETRY", "error": repr(error),
              "completed_groups": groups, "completed_case_count": len(cases), "partial": partial_receipt,
              "elapsed_seconds": time.perf_counter()-started})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT/"protocols/phone_physics_stage1_v1.json")
    parser.add_argument("--asset", type=Path, default=ROOT/"asset_stage1/phone_asset.xml")
    parser.add_argument("--output", type=Path, default=HERE/"warp_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    metadata = plan(args.protocol, args.asset)
    if not args.run:
        heavy = sorted(set(("numpy", "mujoco", "mujoco_warp", "warp", "torch")) & set(sys.modules))
        if heavy:
            raise RuntimeError(f"Metadata route imported forbidden backends: {heavy}")
        print(json.dumps({"status": metadata["status"], "case_count": metadata["case_count"],
                          "batch_steps": metadata["batch_steps_planned"], "world_steps": metadata["world_steps_planned"],
                          "heavy_backends_imported": heavy,
                          "source_pins": metadata["sources"]}), flush=True)
        return
    run(args, metadata)


if __name__ == "__main__":
    main()
