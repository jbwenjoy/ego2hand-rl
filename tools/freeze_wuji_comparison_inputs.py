"""Pin the actual source, physical assets and code consumed by the comparison process."""

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path


def read(path):
    return Path(path).read_bytes()


def pin(path):
    path = path.resolve(strict=True)
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha)


def producer_path(value):
    if os.name != "nt" and len(value) > 2 and value[1:3] in (":\\", ":/"):
        return Path("/mnt") / value[0].lower() / value[3:].replace("\\", "/")
    return Path(value)


def checked(source):
    path = producer_path(source["path"])
    actual = pin(path)
    if actual["bytes"] != source["bytes"] or actual["sha256"] != source["sha256"]:
        raise ValueError(f"Input has changed since export: {path}")
    return path


def freeze(repo, upstream, parquets, output):
    groups = {}
    def include(group, paths):
        group_set = groups.setdefault(group,set())
        for path in paths:
            group_set.add(path.resolve(strict=True))

    tools = ["time_wuji_flashsac","benchmark_wuji_parallel","run_wuji_matched_tranche",
             "run_wuji_multiref_flashsac","run_wuji_flashsac_smoke","evaluate_flash_wuji_quality"]
    include("experiment", [repo/"tools"/(name+".py") for name in tools])
    include("experiment",[Path(__file__)])
    runtime = repo/"src/supermimic_research/runtime"
    include("runtime",[runtime/(n+".py") for n in ("flash_solver","flash_multiref","flash_multiref_checkpoint",
                                                   "flash_wuji","flash_wuji_objective","flash_quality")])
    include("runtime",[repo/"src/supermimic_research/learning"/(n+".py") for n in ("flash_imagination","flash_imagination_hook")])
    include("runtime",[repo/"src/supermimic_research"/(n+".py") for n in ("artifacts","strict_json")])
    include("evaluation",[repo/"configs"/n for n in ("hocap_carton_evaluation.json","hocap_carton_physical_repeat.json")])
    include("solver_calibration",[repo/"configs/flashchord_wuji_physics_v2.json"])
    robot = repo/"runs/flashchord_wuji_asset_v1"/"adapter.json"
    adapter = json.loads(read(robot))
    include("robot",[robot,checked(adapter["model_pin"])])
    include("robot",[checked(asset) for asset in adapter["mesh_pins"]])
    for parquet in parquets:
        data = parquet.parent/"export.json"
        ref = json.loads(read(data))
        group = "reference/"+ref["sequence"]
        raw_metadata = checked(ref["raw_source"])
        original = json.loads(read(raw_metadata))
        # The running strict evaluator reads the original pose tensor through its receipt.
        include(group, [parquet,data,raw_metadata,checked(original["source_poses"]),checked(ref["support"]),
                        checked(ref["arrays"]),checked(ref["direct_ik"])])
        include(group,[checked(asset) for asset in ref["asset_pins"]])
    flash = upstream if (upstream/"src/flash_chord").is_dir() else upstream/"robotic_grounding/flash_chord"
    include("flashchord",[flash/"uv.lock",flash/"pyproject.toml",flash/"scripts/train_flash_sac.py",flash/"scripts/evaluate_policy.py"])
    include("flashchord",[f for f in (flash/"src/flash_chord").rglob("*") if f.suffix in (".py",".yaml",".yml")])
    versions = {pkg:metadata.version(pkg) for pkg in ("newton","warp-lang","mujoco","mujoco-warp","jax","jaxlib","flax","optax")}
    for pkg,suffix in (("mujoco-warp","mujoco_warp/_src/solver.py"),("newton","_src/solvers/mujoco/solver_mujoco.py")):
        dist=metadata.distribution(pkg)
        files=[dist.locate_file(f) for f in dist.files or () if str(f).endswith(suffix)]
        if len(files)!=1:
            raise ValueError(f"Cannot identify {pkg} solver source: {suffix}")
        include("installed_solver",files)
    files = [dict(group=group,**pin(p)) for group,paths in sorted(groups.items()) for p in sorted(paths)]
    result=dict(versions=versions,files=files,
                scope="Inputs used by this host's native training and scoring entry points. "
                      "The replay and evaluator are versioned separately from any future re-score.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x",encoding="utf8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(len(files),sum(f["bytes"] for f in files),pin(output)["sha256"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--upstream",type=Path,required=True)
    p.add_argument("--parquet",type=Path,required=True,action="append")
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    freeze(a.repo.resolve(),a.upstream.resolve(),a.parquet,a.out)
