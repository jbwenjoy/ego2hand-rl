"""Root: freeze one CPU-only consumer of completed full05; no model imports."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def pin(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    physical = ROOT / "outputs/full151105_gradient_empirical_gradient05_v2"
    watch = physical.with_name(physical.name + "_watchdog")
    producer_config = ROOT / "configs/full151105-gradient-empirical_gradient05-v2.json"
    base = read(producer_config)
    assert pin(producer_config)["sha256"] == "034e81fecb3e49a26136740054bb284d600f379bdb7ae1db2c9a6368228af077"
    ready_path = ROOT / "outputs/full151105_analysis_preparation_CPU_v1/READY.json"
    assert pin(ready_path)["sha256"] == "25bfbea1611deb7a517966b63d3d4722a77d346aecce1b01667ae243d8857ed5"
    ready = read(ready_path)
    saved_path = ROOT / "outputs/full151105_full_serial_saved_review_v1/report.json"
    assert pin(saved_path)["sha256"] == "f5fd0703639308706b4f8bf5deee834e6d207486c63fded8920a6a86fd2a62af"
    saved = read(saved_path)
    assert saved["passed"] == saved["total"] == 1762 and all(v is True for v in saved["checks"].values())
    proposal = read(base["proposal"]["path"])
    artifacts = {k: pin(physical / f) for k, f in {
        "producer_report": "physical_report.json", "physical_arrays": "physical_arrays.npz",
        "branch_index": "branch_index.json", "snapshot_records": "snapshot_records.json",
        "donor_records": "donor_records.json", "baseline_gates": "baseline_gates.json",
        "capacity_ledgers": "capacity_ledgers.json", "physical_samples": "physical_samples.jsonl",
        "reward_snapshots": "reward_snapshots.jsonl", "runtime": "runtime.json",
        "source_inventory": "source/inventory.json", "source_zip": "source/execution_sources.zip",
    }.items()}
    artifacts.update({k: pin(watch / f) for k, f in {
        "watch_state": "state.json", "engineering_identity": "engineering_identity.json",
        "post_terminal_CIM": "post_terminal_CIM.json",
    }.items()})
    artifacts.update(producer_config=pin(producer_config), saved_review=pin(saved_path))
    sources = {"physical__" + k: v for k, v in base["source_pins"].items()}
    sources.update({"analysis__" + k: v for k, v in ready["source_pins"].items()})
    sources.update(config_builder=pin(__file__), CPU_preparation_READY=pin(ready_path),
                   CPU_preparation_report=ready["cpu_report"], full_saved_auditor=saved["auditor"],
                   full_saved_pure_helper=saved["pure_saved_helper"])
    health = proposal["WM_health"]
    auditor = read(health["path"])["auditor"]
    loader = proposal["source_pins"]["WM_loader105"]
    sources.update(WM_loader=loader, WM_health_auditor=auditor)
    for value in sources.values():
        assert pin(value["path"])["sha256"] == value["sha256"] and Path(value["path"]).samefile(pin(value["path"])["path"]), value["path"]
    output = ROOT / "outputs/full151105_A2_WM_physical_gradient_CPU_v1"
    config_path = ROOT / "configs/full151105-A2-WM-physical-gradient-CPU-v1.json"
    assert not output.exists() and not config_path.exists()
    config = {
        "schema": "full151105_physical_gradient_CPU_analysis_run:1",
        "method": "A2", "iteration": 2000, "input_bundle": base["input_bundle"],
        "proposal": base["proposal"], "checkpoint": proposal["WM_checkpoint"],
        "health": health, "health_auditor": auditor, "loader": loader,
        "training_run": str(Path(proposal["WM_checkpoint"]["path"]).parent.parent),
        "physical_run": str(physical), "output": str(output),
        "artifact_pins": artifacts, "source_pins": sources,
        "CPU_run_authorized": True, "GPU_authorized": False,
        "training_authorized": False, "Env_or_physics_authorized": False,
        "cpu_threads": 4,
        "scope": "Root-authorized one strict A2 CPU WM load, one batch8 forward and one uAD after full05 saved1762. All64 epsilon rows/32 dependent groups retained. No hidden-state, unique physical derivative, HO-force, ego or hardware certificate.",
    }
    with config_path.open("x", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(json.dumps({"config": pin(config_path), "source_pins": len(sources),
                      "artifact_pins": len(artifacts), "model_executed": False}, indent=2))


if __name__ == "__main__":
    main()
