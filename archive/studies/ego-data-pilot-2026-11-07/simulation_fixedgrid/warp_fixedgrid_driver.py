"""Fixed-grid metadata entry; explicit --run requires a positive matched CPU audit.

No selection, alternate dt, retry or fallback. Physics and capture code are shared
in warp_fixedgrid_core_v1.py. Importing this file imports only the standard library.
"""
import argparse
import importlib.metadata
import importlib.util
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT/"simulation_warp_shared/warp_fixedgrid_core_v1.py"
KERNELS = ROOT/"simulation_warp_shared/warp_fixedgrid_kernels_v1.py"
COMMON = ROOT/"simulation_convex_v2/phone_convex_common_v2.py"
PROTOCOL_SHA = "7a84cdda18952e24e45cfd96ba0bdcb846ff656794512112911e64602b1ed14f"
COMMON_SHA = "4460ccba501e46a033e304460796103afb7b2ef4ef822e6d8653f1a496228ae9"
TESTS = ("drop", "table_settling", "force_push_release")


def core_module():
    spec = importlib.util.spec_from_file_location("phone_warp_fixedgrid_core_v1", CORE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def plan(protocol_path, asset_path, cpu_result_path=None, cpu_assessment_path=None):
    """Fixed JSON and file hashes only: never decode arrays or import backends."""
    c = core_module()
    pp = c.pin(protocol_path)
    p = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if p["schema"] != "phone_convex_fixedgrid_protocol:4" or pp["sha256"] != PROTOCOL_SHA:
        raise ValueError("Wrong frozen fixed-grid protocol")
    for item in p["source_pins"].values():
        c.checked(item)
    if not c.same_pin(p["source_pins"]["common"], c.pin(COMMON)) or c.pin(COMMON)["sha256"] != COMMON_SHA:
        raise ValueError("Frozen physical common changed")
    if not c.same_pin(p["source_pins"]["convex_asset"], c.pin(asset_path)):
        raise ValueError("Wrong convex asset")
    v2 = json.loads(Path(p["source_pins"]["v2_protocol"]["path"]).read_text(encoding="utf-8"))
    if (v2["shared_physics"]["tolerance"], v2["shared_physics"]["ls_tolerance"]) != (1e-6, .01):
        raise ValueError("Fixed shared solver changed")
    f = p["fixed_grid"]
    expected = ({"mass_kg": [.1,.2,.3], "sliding_friction": [.3,.6,.9]}, .0025,
                [.0003125,.00015625], list(TESTS), 3., 54, 777600)
    if tuple(f[k] for k in ("cartesian_grid","timeconst_s","timesteps_s","test_names",
                            "duration_s_each","case_count","integration_steps")) != expected:
        raise ValueError("Registered fixed grid/budget changed")
    grid = [(.2,.6)]+[v for v in itertools.product(f["cartesian_grid"]["mass_kg"],
                f["cartesian_grid"]["sliding_friction"]) if v != (.2,.6)]
    groups = [{"mass_kg": mass, "mu": mu, "dt_s": dt, "timeconst_s": f["timeconst_s"],
               "worlds": list(TESTS), "steps_per_world": round(3/dt),
               "push_start_tick": round(1/dt), "push_end_tick": round(1.2/dt)}
              for mass,mu in grid for dt in f["timesteps_s"]]
    if any(v % c.BLOCK_STEPS for g in groups for v in
           (g["push_start_tick"],g["push_end_tick"]-g["push_start_tick"],g["steps_per_world"]-g["push_end_tick"])):
        raise ValueError("Graph blocks would cross a force boundary")
    expected_cases = [(f"{test}_mass{g['mass_kg']:g}_mu{g['mu']:g}_tau{g['timeconst_s']:g}_dt{g['dt_s']:g}",g,test)
                      for g in groups for test in TESTS]
    cpu_join = None
    if cpu_result_path is not None or cpu_assessment_path is not None:
        if cpu_result_path is None or cpu_assessment_path is None:
            raise ValueError("Provide both CPU result and complete saved assessment")
        result, review = (json.loads(Path(q).read_text(encoding="utf-8")) for q in (cpu_result_path,cpu_assessment_path))
        if result["schema"] != "phone_convex_fixedgrid_CPU:4" or review["schema"] != "phone_convex_fixedgrid_CPU_assessment:4":
            raise ValueError("Wrong fixed-grid CPU namespace")
        if review["assessment_complete"] is not True or review["audit_integrity_qualified"] is not True or review["CPU_case_count"] != 54:
            raise ValueError("Complete independent CPU54 saved readback required")
        if type(review["CPU_bench_qualified"]) is not bool:
            raise ValueError("CPU science qualification must be a boolean")
        if not c.same_pin(review["producer_result_pin"],c.pin(cpu_result_path)) or not c.same_pin(review["protocol_pin"],pp) or not c.same_pin(result["protocol"],pp):
            raise ValueError("CPU producer/assessment protocol join mismatch")
        for doc in (result,review):
            if doc["timeconst_s"] != f["timeconst_s"] or doc["timesteps_s"] != f["timesteps_s"]:
                raise ValueError("CPU grid values differ from this protocol")
        reviewed = review["trajectory_pins"]
        if not isinstance(reviewed,dict) or set(reviewed) != {k for k,_,_ in expected_cases} or result["case_count"] != 54 or len(result["cases"]) != 54 or result["integration_steps"] != 777600:
            raise ValueError("All54 fresh CPU trajectories and original budget required")
        for row,(case,g,test) in zip(result["cases"],expected_cases):
            if (row["case"],row["test"],row["mass_kg"],row["sliding_friction"],row["timeconst_s"],row["dt_s"]) != (case,test,g["mass_kg"],g["mu"],g["timeconst_s"],g["dt_s"]):
                raise ValueError("CPU case identity/order mismatch")
            if not c.same_pin(row["trajectory"],reviewed[case]):
                raise ValueError("CPU raw/independent trajectory pin mismatch")
        if len(review["case_results"]) != 54 or len(review["timestep_comparisons"]) != 27:
            raise ValueError("Full saved assessment case and dt tables required")
        c.checked(result["binding"])
        binding = json.loads(Path(result["binding"]["path"]).read_text(encoding="utf-8"))
        if binding["schema"] != "phone_convex_fixedgrid_CPU_binding:4" or not c.same_pin(binding["protocol"],pp) or not c.same_pin(binding["common"],c.pin(COMMON)) or not c.same_pin(binding["asset"],c.pin(asset_path)) or not c.same_pin(binding["capture_source"],p["source_pins"]["CPU_run_case_source"]):
            raise ValueError("CPU execution binding/source mismatch")
        cpu_join = {"producer_result": c.pin(cpu_result_path), "saved_assessment": c.pin(cpu_assessment_path),
                    "trajectory_pins": reviewed, "CPU_bench_qualified": review["CPU_bench_qualified"],
                    "binding": result["binding"]}
    sdk_sources = {}
    for pkg,files in {"mujoco-warp": ["mujoco_warp/_src/forward.py","mujoco_warp/_src/io.py","mujoco_warp/_src/types.py",
            "mujoco_warp/_src/smooth.py","mujoco_warp/_src/solver.py","mujoco_warp/_src/collision_driver.py",
            "mujoco_warp/_src/collision_primitive.py","mujoco_warp/_src/support.py"],"warp-lang": ["warp/_src/context.py"]}.items():
        d = importlib.metadata.distribution(pkg)
        for name in files:
            sdk_sources[name] = c.pin(d.locate_file(name))
    asset = Path(asset_path)
    assets = {"asset_XML": c.pin(asset), "original_GLB": c.pin(ROOT/"hot3d/object_models/obj_000024.glb"),
              "models_info": c.pin(ROOT/"hot3d/object_models/models_info.json"),
              "baked_visual_mesh": c.pin(asset.parent/"meshes/phone24_visual_m.obj"),
              "collision_mesh": c.pin(asset.parent/"meshes/phone24_collision_convex_m.obj"),
              "asset_geometry_manifest": c.pin(asset.parent/"report.json"),
              "native_compile_frame_receipt": c.pin(ROOT/"outputs/convex_asset_native_compile_v1.json")}
    return {"schema": "phone_Warp_fixedgrid_plan:1", "status":
            "METADATA_PENDING_CPU54_NO_RUN" if cpu_join is None else
            "METADATA_ONLY_NOT_EXECUTED" if cpu_join["CPU_bench_qualified"] else "BLOCKED_CPU_FIXEDGRID_NOT_QUALIFIED",
            "result_schema": "phone_convex_fixedgrid_Warp:4", "protocol": pp, "protocol_source_pins": p["source_pins"],
            "asset": c.pin(asset), "asset_pins": assets,
            "sources": {"driver": c.pin(__file__), "core": c.pin(CORE), "kernels": c.pin(KERNELS),
                        "shared_common": c.pin(COMMON), "legacy_state_helpers": c.pin(ROOT/"simulation_stage1/run_phone_cpu_bench_v1.py")},
            "diagnostics_source_review": c.pin(ROOT/"simulation_convex_v2/warp_diagnostics_source_review_v1.json"),
            "selection_receipt": None, "selection_or_fallback_permitted": False,
            "selected_timeconst_s": f["timeconst_s"], "selected_timesteps_s": f["timesteps_s"],
            "CPU_confirmation_join": cpu_join, "groups": groups, "case_count": 54,
            "batch_steps_planned": sum(g["steps_per_world"] for g in groups),
            "world_steps_planned": sum(3*g["steps_per_world"] for g in groups),
            "additional_final_forward_calls_planned": len(groups),
            "algorithm_contract": {"tolerance": 1e-6, "ls_tolerance": .01, "integrator": "implicitfast", "solver": "Newton",
                                   "cone": "pyramidal", "iterations": 100, "ls_iterations": 20},
            "relevant_backend_sources": sdk_sources,
            "versions_metadata_only": {k:importlib.metadata.version(k) for k in ("mujoco","mujoco-warp","warp-lang","numpy")},
            "actual_GPU_options_verified": False, "GPU_initialization_executed": False,
            "contact_phase": "native PRE ticki; POST state ticki+1; final forward provides tickN",
            "qacc_phase": "native step PRE-state solve/integrator-produced diagnostic, not POST derivative",
            "GPU_native_CPU_warning_table": "NOT_AVAILABLE_BY_PINNED_SDK"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--asset", type=Path, default=ROOT/"asset_stage2_convex/completed_v2/phone_asset.xml")
    parser.add_argument("--cpu-result", type=Path)
    parser.add_argument("--cpu-assessment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--process-log-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    metadata = plan(args.protocol,args.asset,args.cpu_result,args.cpu_assessment)
    heavy = sorted(set(("numpy","mujoco","mujoco_warp","warp","torch")) & set(sys.modules))
    if heavy:
        raise RuntimeError("Metadata route imported forbidden backend: "+str(heavy))
    c = core_module()
    if args.metadata_output:
        c.write(args.metadata_output,{**metadata,"heavy_backends_imported":heavy})
    if not args.run:
        print(json.dumps({"status":metadata["status"],"case_count":54,
                          "batch_steps":metadata["batch_steps_planned"],"world_steps":metadata["world_steps_planned"],
                          "extra_forward":metadata["additional_final_forward_calls_planned"],"heavy_backends_imported":heavy}),flush=True)
        return
    if metadata["status"] != "METADATA_ONLY_NOT_EXECUTED":
        raise ValueError("Fixed-grid enrollment rejected before backend import: "+metadata["status"])
    if args.process_log_dir is None or not args.process_log_dir.is_dir():
        raise ValueError("Retained OS stdout/stderr directory required")
    args.output = args.output.resolve()
    args.asset = args.asset.resolve()
    metadata["process_log_directory"] = str(args.process_log_dir.resolve())
    metadata["actual_process_log_pins_authority"] = "Parent launcher manifest after child exit; files mutable while worker alive"
    c.run(args,metadata)


if __name__ == "__main__":
    main()
