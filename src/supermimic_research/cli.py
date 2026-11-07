"""Repository inspection commands import no learning or simulator backend."""
from __future__ import annotations
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

from .artifacts import ArtifactRegistry


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    root = repository_root()
    parser = argparse.ArgumentParser(prog="supermimic-research")
    parser.add_argument("--artifacts", type=Path, default=root / "configs" / "artifacts.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    artifact_parser = commands.add_parser("artifacts", help="List explicit external data roles and locations")
    artifact_parser.add_argument("--verify", action="store_true")
    commands.add_parser("doctor", help="Inspect module origins and installed versions without initializing CUDA")
    train_parser = commands.add_parser("train", help="One shared offline driver; inspect is the default")
    train_parser.add_argument("--config", type=Path, default=root / "configs" / "phone_offline.toml")
    train_parser.add_argument("--mode", choices=["inspect", "smoke", "run"], default="inspect")
    online_parser = commands.add_parser("online", help="Matched native collection/update experiment")
    online_parser.add_argument("--config", type=Path, default=root / "configs" / "phone_online_gpu.toml")
    online_parser.add_argument("--mode", choices=["inspect", "run"], default="inspect")
    collection_parser = commands.add_parser("collect-online", help="Isolated CPU collection worker")
    collection_parser.add_argument("--protocol", type=Path, required=True)
    collection_parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "online":
        if args.mode == "inspect":
            from .learning.online_contract import inspect
            result = inspect(args.config)
        else:
            from .learning.online_runner import train
            result = train(args.config)
        print(json.dumps(result, indent=2)); return 0
    if args.command == "collect-online":
        from .learning import collection_worker
        result = collection_worker.run(args.protocol) if args.run else collection_worker.inspect(args.protocol)
        print(json.dumps(result, indent=2)); return 0
    if args.command == "train":
        from .learning import runner
        action = {"inspect": runner.inspect, "smoke": runner.smoke, "run": runner.train}[args.mode]
        result = action(args.config)
        print(json.dumps(result, indent=2))
        return 0
    registry = ArtifactRegistry(args.artifacts, local=root / "paths.local.toml")
    if args.command == "artifacts":
        rows = registry.inspect(verify=args.verify)
        print(json.dumps(rows, indent=2))
        return 0 if all(row["exists"] or row["optional"] for row in rows) else 1
    versions = {}
    for package in ["numpy", "torch", "mujoco", "mujoco-warp", "warp-lang", "mjlab"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    origins = {}
    for package in ["supermimic_research", "dex_supermimic", "chord_study"]:
        spec = importlib.util.find_spec(package)
        origins[package] = str(spec.origin) if spec else None
    report = {"repository": str(root), "python": sys.executable, "versions": versions, "module_origins": origins,
              "Torch_imported": "torch" in sys.modules, "MuJoCo_imported": "mujoco" in sys.modules,
              "required_artifacts_missing": [row["name"] for row in registry.inspect() if not row["exists"] and not row["optional"]]}
    print(json.dumps(report, indent=2))
    return 0 if not report["required_artifacts_missing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
