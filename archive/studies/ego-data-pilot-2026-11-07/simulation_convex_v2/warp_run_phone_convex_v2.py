"""SOURCE DRAFT: convex-v2 explicit --run only, shared CPU model/initial-state functions.

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
import math
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
COMMON_SOURCE = HERE / "phone_convex_common_v2.py"
KERNEL_SOURCE = HERE / "warp_phone_convex_kernels_v2.py"
TESTS = ("drop", "table_settling", "force_push_release")
TEST_PROTOCOL_NAMES = {name: name for name in TESTS}
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


def checked(item):
    actual = pin(item["path"])
    if actual["sha256"] != item["sha256"]:
        raise ValueError("Immutable metadata pin mismatch: "+item["path"])
    return actual


def same_pin(a, b):
    checked(a); checked(b)
    return Path(a["path"]).samefile(b["path"]) and a["sha256"] == b["sha256"]


def plan(protocol_path, asset_path, selection_path=None, cpu_result_path=None, cpu_assessment_path=None):
    p = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if p["schema"] != "phone_convex_contact_protocol:2":
        raise ValueError("Wrong protocol schema")
    if pin(protocol_path)["sha256"] != "6745a6e58ce4507c56ca6fe053a4d16fc2e4e2195cfdae6c74ea1797e6bd5ace":
        raise ValueError("Not the frozen v2 protocol")
    if p["shared_physics"]["tolerance"] != 1e-6 or p["shared_physics"]["ls_tolerance"] != .01:
        raise ValueError("Common CPU/GPU tolerances are fixed before data")
    if p["confirmation"]["test_names"] != list(TESTS):
        raise ValueError("Wrong test order")
    confirmation = p["confirmation"]
    masses = confirmation["cartesian_grid"]["mass_kg"]
    mus = confirmation["cartesian_grid"]["sliding_friction"]
    dts = confirmation["timesteps_s"]
    if (masses, mus, dts) != ([.1, .2, .3], [.3, .6, .9], [.00125, .000625]):
        raise ValueError("Unexpected frozen grid; no automatic adaptation")
    selection = None
    selected = None
    if selection_path is not None:
        selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
        if selection["schema"] != "phone_convex_contact_selection:2" or selection["audit_integrity_qualified"] is not True or selection["selection_complete"] is not True or selection["screening_complete"] is not True:
            raise ValueError("Complete independently saved CPU selection receipt required")
        sp = selection["protocol_pin"]
        if not Path(sp["path"]).samefile(protocol_path) or pin(sp["path"])["sha256"] != pin(protocol_path)["sha256"] or sp["sha256"] != pin(protocol_path)["sha256"]:
            raise ValueError("Selection protocol provenance mismatch")
        # Metadata only. Actual screened physics and selection remain delegated to
        # the pinned independent assessment; no parent arrays are decoded here.
        if not isinstance(selection["trajectory_pins"], dict) or len(selection["trajectory_pins"]) != 18:
            raise ValueError("Exactly18 screened trajectories required")
        for item in [selection["screening_result_pin"], *selection["trajectory_pins"].values()]:
            checked(item)
        eligible = selection["eligible_timeconst_s"]
        selected = selection["selected_timeconst_s"]
        if selected is not None:
            if type(selected) not in (int, float) or not math.isfinite(selected) or selected <= 0 or not eligible or selected != max(eligible) or selected not in p["screening"]["timeconst_s"] or any(t not in p["screening"]["timeconst_s"] for t in eligible):
                raise ValueError("Maximum eligible predeclared time constant required")
        elif eligible:
            raise ValueError("Null selection conflicts with eligible values")
    cpu_join = None
    if cpu_result_path is not None or cpu_assessment_path is not None:
        if cpu_result_path is None or cpu_assessment_path is None or selection is None:
            raise ValueError("CPU result+assessment+selection must be supplied together")
        result, review = (json.loads(Path(q).read_text(encoding="utf-8")) for q in (cpu_result_path, cpu_assessment_path))
        if result["schema"] != "phone_convex_contact_CPU_confirmation:2" or review["schema"] != "phone_convex_contact_CPU_confirmation_assessment:2" or review["assessment_complete"] is not True or review["CPU_case_count"] != 54:
            raise ValueError("Complete fresh CPU54 confirmation saved readback required")
        if not same_pin(review["producer_result_pin"], pin(cpu_result_path)) or not same_pin(review["protocol_pin"], pin(protocol_path)) or review["selected_timeconst_s"] != selected:
            raise ValueError("CPU confirmation readback provenance mismatch")
        reviewed = review["trajectory_pins"]
        if not isinstance(reviewed, dict) or len(reviewed) != 54:
            raise ValueError("Exactly54 reviewed CPU trajectories required")
        for item in reviewed.values():
            checked(item)
        if result["case_count"] != 54 or len(result["cases"]) != 54 or {r["case"] for r in result["cases"]} != set(reviewed):
            raise ValueError("CPU54 result/readback case map mismatch")
        for row in result["cases"]:
            if not same_pin(row["trajectory"], reviewed[row["case"]]) or row["timeconst_s"] != selected:
                raise ValueError("CPU54 trajectory/selected-tau join mismatch")
        cpu_join = {"producer_result": pin(cpu_result_path), "saved_assessment": pin(cpu_assessment_path),
                    "trajectory_pins": reviewed, "CPU_bench_qualified": review["CPU_bench_qualified"]}
    grid = [(.2, .6)] + [x for x in itertools.product(masses, mus) if x != (.2, .6)]
    groups = [{"mass_kg": mass, "mu": mu, "dt_s": dt, "worlds": list(TESTS),
               "timeconst_s": selected,
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
                  "collision_mesh": pin(Path(asset_path).parent/"meshes/phone24_collision_convex_m.obj"),
                  "asset_geometry_manifest": pin(Path(asset_path).parent/"report.json"),
                  "native_compile_frame_receipt": pin(ROOT/"outputs/convex_asset_native_compile_v1.json")}
    status = ("BLOCKED_NO_ELIGIBLE_CPU_TIME_CONSTANT" if selection is not None and selected is None
              else "METADATA_ONLY_NOT_EXECUTED" if selected is not None and cpu_join is not None
              else "METADATA_PENDING_ELIGIBLE_CPU_SELECTION_AND_CPU54_NO_RUN")
    return {"schema": "phone_convex_Warp_plan:2", "status": status,
            "protocol": pin(protocol_path), "asset": pin(asset_path),
            "asset_pins": asset_pins,
            "sources": {"runner": pin(__file__), "kernels": pin(KERNEL_SOURCE), "shared_common": pin(COMMON_SOURCE),
                        "legacy_state_helpers": pin(ROOT/"simulation_stage1/run_phone_cpu_bench_v1.py")},
            "diagnostics_source_review": pin(HERE/"warp_diagnostics_source_review_v1.json"),
            "selection_receipt": pin(selection_path) if selection_path is not None else None,
            "selected_timeconst_s": selected,
            "CPU_confirmation_join": cpu_join,
            "algorithm_contract": {"tolerance": 1e-6, "ls_tolerance": .01, "integrator": "implicitfast", "solver": "Newton", "cone": "pyramidal", "iterations": 100, "ls_iterations": 20},
            "actual_GPU_options_verified": False,
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


def validate_actual_model(np, cpu, model, m, geometry, group):
    """Read actual device model before the first integration, never in default."""
    if float(model.opt.tolerance) != 1e-6 or float(model.opt.ls_tolerance) != .01:
        raise ValueError("CPU compiled tolerances differ from the fixed common contract")
    observed = {"discrete": {}, "float32": {}, "array_guards": {}}
    for name in ("integrator", "solver", "iterations", "ls_iterations", "ccd_iterations",
                 "cone", "disableflags", "enableflags", "sdf_initpoints", "sdf_iterations"):
        a, b = int(getattr(m.opt, name)), int(getattr(model.opt, name))
        if a != b:
            raise ValueError("GPU discrete option mismatch: "+name)
        observed["discrete"][name] = a
    for name in ("timestep", "tolerance", "ls_tolerance", "ccd_tolerance", "sleep_tolerance",
                 "gravity", "wind", "magnetic", "density", "viscosity"):
        actual = np.asarray(getattr(m.opt, name).numpy())
        expected = np.asarray(getattr(model.opt, name), dtype=np.float32)
        if actual.dtype != np.float32 or not np.array_equal(actual.ravel(), expected.ravel()):
            raise ValueError("GPU FP32 option mismatch: "+name)
        observed["float32"][name] = {"dtype": str(actual.dtype), "value": actual.tolist(),
                                    "CPU_value": np.asarray(getattr(model.opt, name)).tolist()}
    for name in ("geom_type", "geom_dataid", "geom_condim", "geom_priority", "mesh_vertnum",
                 "geom_pos", "geom_quat", "geom_friction", "geom_solref", "geom_solimp", "geom_solmix", "geom_margin", "geom_gap",
                 "body_mass", "body_ipos", "body_iquat", "body_inertia", "body_gravcomp", "mesh_vert", "mesh_face", "mesh_graph"):
        actual = np.asarray(getattr(m, name).numpy())
        expected = np.asarray(getattr(model, name), dtype=actual.dtype)
        if not np.array_equal(actual.ravel(), expected.ravel()):
            raise ValueError("GPU model array mapping mismatch: "+name)
        observed["array_guards"][name] = {"shape": list(actual.shape), "dtype": str(actual.dtype), "equal_CPU_cast": True}
    gid = model.geom("phone24_collision_convex").id
    mid = int(model.geom_dataid[gid])
    if int(model.geom_type[gid]) != int(cpu.mujoco.mjtGeom.mjGEOM_MESH) or mid < 0:
        raise ValueError("Compiled collision geom must remain a mesh")
    if (int(model.mesh_vertnum[mid]), int(model.mesh_facenum[mid])) != (352, 700):
        raise ValueError("Unexpected compiled collision hull topology")
    if np.asarray(geometry["local_collision_vertices_m"]).shape != (352, 3):
        raise ValueError("Shared compiled body-local hull vertices missing")
    refsafe = not bool(int(m.opt.disableflags) & int(cpu.mujoco.mjtDisableBit.mjDSBL_REFSAFE))
    if not refsafe or int(model.opt.noslip_iterations) != 0:
        raise ValueError("Refsafety must stay enabled and unsupported noslip must stay off")
    if m.opt.warn_overflow is not True:
        raise ValueError("Supported GPU diagnostic printing must not be suppressed")
    ids = [gid, model.geom("bench_support").id]
    refs = m.geom_solref.numpy().reshape(model.ngeom, 2)[ids]
    if not np.array_equal(refs, np.tile(np.array([group["timeconst_s"], 1.], dtype=np.float32), (2, 1))):
        raise ValueError("Authored contact reference values changed")
    dt = np.float32(m.opt.timestep.numpy()[0])
    observed.update({"compiled_collision_geom_id": gid, "compiled_collision_mesh_id": mid,
                     "collision_vertices": 352, "collision_triangles": 700, "refsafe_enabled": refsafe,
                     "authored_contact_solref_F32": refs.tolist(),
                     "effective_refsafety_timeconst_F32": [float(np.maximum(t, np.float32(2)*dt)) for t in refs[:, 0]],
                     "timeconst_over_effective_dt": [float(t/dt) for t in refs[:, 0]],
                     "Warp_extra_options": {"broadphase": int(m.opt.broadphase), "broadphase_filter": int(m.opt.broadphase_filter),
                                             "graph_conditional": bool(m.opt.graph_conditional), "warn_overflow": bool(m.opt.warn_overflow)},
                     "native_CPU_style_warning_table": "NOT_AVAILABLE_BY_PINNED_WARP_DATA_API"})
    return observed


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
    # Support function of the actual compiled body-local convex hull, not its AABB.
    vertices = np.asarray(geometry["local_collision_vertices_m"], dtype=np.float64)
    bottom = qpos[:, 2].astype(np.float64)+(R[:, 2, :] @ vertices.T).min(axis=1)
    U = mass*9.81*com[:, 2]
    T = .5*mass*np.sum(vcom*vcom, axis=1) + .5*np.sum(omega_body**2*np.asarray(geometry["inertia_kg_m2"]), axis=1)
    return {"com_world_m": com, "com_velocity_world_m_s": vcom, "omega_world_rad_s": omega_world,
            "convex_hull_bottom_world_z_m": bottom, "quaternion_norm": norms,
            "energy_potential_kinetic_J": np.stack([U, T], axis=1)}


def run_group(np, wp, mjw, cpu, kernels, asset, group, output, device, partial):
    partial.clear()
    partial.update({"group": group, "phase": "PREINTEGRATION"})
    mass, mu, dt = group["mass_kg"], group["mu"], group["dt_s"]
    n = group["steps_per_world"]
    tau = group["timeconst_s"]
    if tau is None:
        raise ValueError("No eligible CPU-selected time constant: confirmation prohibited")
    model, geometry = cpu.make_model(asset, mass, mu, dt, tau)
    options = cpu.compiled_options(model)
    xml_path = output/f"warp_model_mass{mass:g}_mu{mu:g}_dt{dt:g}_tau{tau:g}.xml"
    with xml_path.open("x", encoding="utf-8") as stream:
        stream.write(geometry["xml"])
    bid = model.body(cpu.BODY).id
    initial = [cpu.initial_state(model, geometry, test) for test in TESTS]
    q0 = np.stack([a.qpos.copy() for a in initial]).astype(np.float32)
    v0 = np.stack([a.qvel.copy() for a in initial]).astype(np.float32)
    d = mjw.put_data(model, initial[0], nworld=3, nconmax=NCONMAX_PER_WORLD, njmax=NJMAX_PER_WORLD)
    m = mjw.put_model(model)
    runtime_contract = validate_actual_model(np, cpu, model, m, geometry, group)
    wp.copy(d.qpos, wp.array(q0, dtype=wp.float32, device=device))
    wp.copy(d.qvel, wp.array(v0, dtype=wp.float32, device=device))
    wp.synchronize()
    if not np.array_equal(d.qpos.numpy().view(np.uint8), q0.view(np.uint8)) or not np.array_equal(d.qvel.numpy().view(np.uint8), v0.view(np.uint8)):
        raise ValueError("Installed GPU initial state differs from shared CPU state cast to F32")
    initial_descriptors = []
    for world, test in enumerate(TESTS):
        descriptor = post_descriptors(np, cpu, q0[world:world+1], v0[world:world+1], geometry, mass)
        clearance = .1 if test == "drop" else .0005
        actual_height = float(descriptor["convex_hull_bottom_world_z_m"][0])
        allowance = 32*np.finfo(np.float32).eps*max(1., float(np.max(np.abs(geometry["local_collision_vertices_m"]))))
        if abs(actual_height-clearance) > allowance:
            raise ValueError("FP32 initial hull support placement differs from requested clearance")
        initial_descriptors.append({"test": test, "requested_clearance_m": clearance,
                                    "GPU_qpos_F32": q0[world].tolist(), "GPU_qvel_F32": v0[world].tolist(),
                                    "minimum_hull_height_m": actual_height, "fixed_FP32_arithmetic_allowance_m": float(allowance)})
    write(output/f"warp_group_preintegration_mass{mass:g}_mu{mu:g}_dt{dt:g}_tau{tau:g}.json",
          {"runtime_contract": runtime_contract, "CPU_compiled_options": options,
           "local_collision_vertices_m": np.asarray(geometry["local_collision_vertices_m"]).tolist(),
           "COM_body_m": np.asarray(geometry["center_m"]).tolist(), "inertia_kg_m2": np.asarray(geometry["inertia_kg_m2"]).tolist(),
           "initial_support": initial_descriptors, "integration_steps_so_far": 0})
    # Initial/drop-derived caches copied by put_data are not sampling authority.
    # First step recomputes every world's forward fields from the installed qpos.
    cap = d.naconmax
    rows = n+1
    cursor = wp.zeros(1, dtype=int, device=device)
    def f32(shape):
        return wp.full(shape, float("nan"), dtype=float, device=device)
    oq, ov, ot, of = f32((rows, 3, 7)), f32((rows, 3, 6)), f32((rows, 3)), f32((rows, 3))
    oo, oe = wp.full((rows, 3), -1, dtype=int, device=device), wp.full((rows, 3), -1, dtype=int, device=device)
    qa = f32((rows, 3, 6))
    native_energy = wp.array(np.full((rows, 3, 2), np.nan, dtype=np.float32), dtype=wp.vec2, device=device)
    pne, pni = [wp.full((rows, 3), -1, dtype=int, device=device) for _ in range(2)]
    nc = wp.full(rows, -1, dtype=int, device=device)
    count = wp.full(rows, -1, dtype=int, device=device)
    cd = f32((rows, cap))
    cp = wp.zeros((rows, cap), dtype=wp.vec3, device=device)
    cf = wp.zeros((rows, cap), dtype=wp.mat33, device=device)
    cg = wp.zeros((rows, cap), dtype=wp.vec2i, device=device)
    cw, ct, cdim = [wp.full((rows, cap), -1, dtype=int, device=device) for _ in range(3)]
    partial.update({"group": group, "cursor": cursor,
                    "arrays": {"qpos": oq, "qvel": ov, "time_s": ot, "applied_force": of,
                               "overflow": oo, "nefc": oe, "nacon": count, "contact_dist": cd,
                               "contact_pos": cp, "contact_frame": cf, "contact_geom": cg,
                               "contact_worldid": cw, "contact_type": ct, "contact_dim": cdim,
                               "pre_qacc": qa, "pre_nefc": pne, "pre_solver_niter": pni, "pre_ncollision": nc,
                               "pre_engine_energy": native_energy}})
    contact_args = [d.nacon, d.contact.dist, d.contact.pos, d.contact.frame, d.contact.geom,
                    d.contact.worldid, d.contact.type, d.contact.dim, cursor, cap,
                    count, cd, cp, cf, cg, cw, ct, cdim]
    def step_and_record():
        mjw.step(m, d)
        wp.launch(kernels.record_pre_contacts, dim=cap, inputs=contact_args, device=device)
        wp.launch(kernels.record_pre_diagnostics, dim=3,
                  inputs=[d.qacc, d.nefc, d.solver_niter, d.ncollision, d.energy, cursor, qa, pne, pni, nc, native_energy], device=device)
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
        partial["phase"] = phase
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
    wp.launch(kernels.record_pre_diagnostics, dim=3,
              inputs=[d.qacc, d.nefc, d.solver_niter, d.ncollision, d.energy, cursor, qa, pne, pni, nc, native_energy], device=device)
    wp.synchronize()
    final_overflow = d.overflow.numpy()
    final_nefc = d.nefc.numpy()
    qa_np, pne_np, pni_np, nc_np = [a.numpy() for a in (qa, pne, pni, nc)]
    native_energy_np = native_energy.numpy()
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
    if not all(np.isfinite(a).all() for a in (q, v, times, forces, qa_np, native_energy_np)):
        raise ValueError("Nonfinite recorded state")
    if np.any(overflow) or np.any(final_overflow) or np.any(nefc > NJMAX_PER_WORLD) or np.any(final_nefc > NJMAX_PER_WORLD) or np.any(nc_np > cap):
        raise ValueError("Recorded capacity overflow")
    if np.any(pne_np < 0) or np.any(pni_np < 0) or np.any(nc_np < 0):
        raise ValueError("Missing per-tick supported diagnostic")
    effective_dt = float(m.opt.timestep.numpy()[0])
    case_rows = []
    for world, test in enumerate(TESTS):
        case = f"{test}_mass{mass:g}_mu{mu:g}_tau{tau:g}_dt{dt:g}"
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
                 "pre_solver_nefc_after_step": nefc[:, world],
                 "solver_qacc_at_state_tick": qa_np[:, world], "solver_nefc_at_state_tick": pne_np[:, world],
                 "solver_niter_at_state_tick": pni_np[:, world], "global_ncollision_at_state_tick": nc_np,
                 "native_engine_energy_at_solver_state_tick_J": native_energy_np[:, world],
                 "supported_finite_state_qacc_flags": np.isfinite(q[:, world]).all(1) & np.isfinite(v[:, world]).all(1) & np.isfinite(qa_np[:, world]).all(1)}
        tp, cp_path = output/(case+".npz"), output/(case+"_contacts.npz")
        np.savez_compressed(tp, **trace)
        np.savez_compressed(cp_path, **c)
        rest = np.arange(rows)*dt >= 2.5-dt/4
        com = derived["com_world_m"]
        e = derived["energy_potential_kinetic_J"].sum(1)
        work = float(np.sum(forces[1:, world]*np.diff(com[:, 0])))
        case_rows.append({"case": case, "test": test, "protocol_test_name": TEST_PROTOCOL_NAMES[test],
            "backend": "MuJoCo_Warp", "mass_kg": mass, "sliding_friction": mu, "timeconst_s": tau, "dt_s": dt, "dt_effective_s": effective_dt,
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
            "trajectory": pin(tp), "trajectory_pin": pin(tp), "contact_table": pin(cp_path),
            "completed_steps": n, "checks": {"finite_state_qacc_and_engine_energy": True, "overflow_flags_clear": True},
            "diagnostic_coverage": {"native_CPU_style_warning_array": "NOT_AVAILABLE_BY_PINNED_SDK", "per_tick_overflow_9bits": True,
                                     "qacc_phase": "step PRE-state solve/integrator-produced diagnostic, not a POST-state derivative; final native forward supplies tickN",
                                     "native_energy_phase": "from native forward at solver-state tick, separately captured from rigid T+mgz",
                                     "native_contact_force": "NOT_EXPORTED", "explicit_reset_data_calls": 0},
            "model_xml_sha256": geometry["model_xml_sha256"],
            "compiled_model_XML": pin(xml_path),
            "uniform_box_inertia_kg_m2": geometry["inertia_kg_m2"].tolist()})
    return case_rows, {"group": group, "graph_launches": launches, "batch_steps_returned": graph_steps,
                       "world_steps_returned": graph_steps*3, "additional_forward_calls": 1,
                       "final_cursor": int(cursor.numpy()[0]), "capacity_naconmax": cap,
                       "CPU_compiled_model_options": options,
                       "preintegration_runtime_contract": runtime_contract,
                       "initial_support": initial_descriptors,
                       "GPU_effective_options": {"timestep_F32": effective_dt, "tolerance_F32": float(m.opt.tolerance.numpy()[0]),
                            "ls_tolerance_F32": float(m.opt.ls_tolerance.numpy()[0]), "solver": int(m.opt.solver),
                            "integrator": int(m.opt.integrator), "cone": int(m.opt.cone), "iterations": int(m.opt.iterations),
                            "ls_iterations": int(m.opt.ls_iterations), "enableflags": int(m.opt.enableflags),
                            "disableflags": int(m.opt.disableflags)},
                       "capacity_njmax_per_world": NJMAX_PER_WORLD, "model_xml_sha256": geometry["model_xml_sha256"],
                       "final_overflow_flags": final_overflow.tolist(), "final_nefc": final_nefc.tolist()}


def run(args, metadata):
    if metadata["selected_timeconst_s"] is None or metadata["CPU_confirmation_join"] is None:
        raise ValueError("Eligible independent CPU18 selection and complete CPU54 saved readback required before --run")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    write(args.output/"metadata.json", metadata)
    started = time.perf_counter()
    cases, groups = [], []
    partial = {}
    python_warnings = []
    warning_log = args.output/"warp_python_warnings.jsonl"
    warning_log.open("x", encoding="utf-8").close()
    original_showwarning = warnings.showwarning
    old_filters = warnings.filters[:]
    def showwarning(message, category, filename, lineno, file=None, line=None):
        record = {"message": str(message), "category": category.__name__, "filename": filename, "line": lineno,
                  "elapsed_seconds": time.perf_counter()-started,
                  "group": partial.get("group"), "phase": partial.get("phase")}
        python_warnings.append(record)
        with warning_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False)+"\n")
            stream.flush()
        original_showwarning(message, category, filename, lineno, file=file, line=line)
    warnings.showwarning = showwarning
    warnings.simplefilter("always")
    try:
        for item in metadata["sources"].values():
            if pin(item["path"])["sha256"] != item["sha256"]:
                raise ValueError("Shared/source changed after metadata binding")
        import numpy as np
        import warp as wp
        import mujoco_warp as mjw
        cpu = import_file("phone_convex_common_v2", COMMON_SOURCE)
        kernels = import_file("warp_phone_convex_kernels_v2", KERNEL_SOURCE)
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
        result = {"schema": "phone_convex_contact_Warp_confirmation:2", "status": "RECORDED_NOT_YET_ASSESSED",
            "created_UTC": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "metadata": pin(args.output/"metadata.json"), "protocol": metadata["protocol"],
            "asset_pins": metadata["asset_pins"],
            "selection_receipt": metadata["selection_receipt"], "selected_timeconst_s": metadata["selected_timeconst_s"],
            "CPU_confirmation_join": metadata["CPU_confirmation_join"],
            "python_warnings": pin(warning_log), "python_warning_count": len(python_warnings),
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
            "energy_scope": "F64 rigid T + gravitational U on POST saved states; native engine energy separately saved at matched solver tick; no soft-contact energy claim",
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
    finally:
        warnings.showwarning = original_showwarning
        warnings.filters[:] = old_filters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT/"protocols/phone_convex_contact_v2.json")
    parser.add_argument("--asset", type=Path, default=ROOT/"asset_stage2_convex/completed_v2/phone_asset.xml")
    parser.add_argument("--output", type=Path, default=HERE/"warp_confirmation_v2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--cpu-result", type=Path)
    parser.add_argument("--cpu-assessment", type=Path)
    parser.add_argument("--process-log-dir", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    metadata = plan(args.protocol, args.asset, args.selection, args.cpu_result, args.cpu_assessment)
    if not args.run:
        heavy = sorted(set(("numpy", "mujoco", "mujoco_warp", "warp", "torch")) & set(sys.modules))
        if heavy:
            raise RuntimeError(f"Metadata route imported forbidden backends: {heavy}")
        print(json.dumps({"status": metadata["status"], "case_count": metadata["case_count"],
                          "batch_steps": metadata["batch_steps_planned"], "world_steps": metadata["world_steps_planned"],
                          "heavy_backends_imported": heavy,
                          "source_pins": metadata["sources"]}), flush=True)
        return
    if args.process_log_dir is None or not args.process_log_dir.is_dir():
        raise ValueError("Use coordinated warp launcher with retained OS stdout/stderr")
    metadata["process_log_directory"] = str(args.process_log_dir.resolve())
    metadata["actual_process_log_pins_authority"] = "Parent launcher manifest after child exit; files are mutable while worker is alive"
    run(args, metadata)


if __name__ == "__main__":
    main()
