"""Explicit artifact locations and byte identity, independent of study imports."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath
import tomllib
from typing import Any


def file_pin(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(target), "bytes": target.stat().st_size, "sha256": digest.hexdigest()}


def verify_file_pin(pin: dict[str, Any]) -> Path:
    target = Path(pin["path"]).resolve(strict=True)
    actual = file_pin(target)
    if actual["bytes"] != pin["bytes"] or actual["sha256"] != pin["sha256"]:
        raise ValueError(f"Artifact bytes changed: {target}")
    return target


@dataclass(frozen=True)
class Artifact:
    name: str
    path: Path
    role: str
    sha256: str | None = None
    optional: bool = False

    def resolve(self, *, verify: bool = True) -> Path:
        path = self.path.resolve(strict=True)
        if verify and self.sha256 and file_pin(path)["sha256"] != self.sha256:
            raise ValueError(f"Artifact checksum mismatch: {self.name}")
        return path


class ArtifactRegistry:
    """Artifact paths are configuration, not Python import locations.

    Relocation preserves the recorded digest and size while explicitly mapping
    a historical absolute prefix to a configured data root. New run provenance
    should record both the original pin and the resolved location. Legacy
    frozen programs retain their stricter original-path contract.
    """

    def __init__(self, manifest: str | Path, *, local: str | Path | None = None):
        self.manifest = Path(manifest).resolve(strict=True)
        document = tomllib.loads(self.manifest.read_text("utf-8"))
        if document.get("schema") != 1:
            raise ValueError("Unsupported artifact manifest schema")
        root_values = dict(document.get("roots", {}))
        relocations = dict(document.get("relocations", {}))
        if local is not None and Path(local).exists():
            overrides = tomllib.loads(Path(local).read_text("utf-8"))
            root_values.update(overrides.get("roots", {}))
            relocations.update(overrides.get("relocations", {}))
        self.roots = {name: self._absolute(value) for name, value in root_values.items()}
        self.relocations = relocations
        self.entries = {}
        for name, row in document.get("artifacts", {}).items():
            root = self.roots[row["root"]]
            path = (root / row["path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Artifact escapes declared root: {name}")
            self.entries[name] = Artifact(name, path, row["role"], row.get("sha256"), row.get("optional", False))

    def _absolute(self, path: str) -> Path:
        target = Path(path)
        return (target if target.is_absolute() else self.manifest.parent / target).resolve()

    def resolve(self, name: str, *, verify: bool = True) -> Path:
        return self.entries[name].resolve(verify=verify)

    def pin(self, name: str) -> dict[str, Any]:
        return file_pin(self.resolve(name))

    def resolve_pin(self, recorded: dict[str, Any]) -> Path:
        """Resolve a nested historical data pin without modifying its JSON."""
        recorded_path = str(recorded["path"])
        path_type = PureWindowsPath if PureWindowsPath(recorded_path).drive else PurePosixPath
        source = path_type(recorded_path)
        if not source.is_absolute():
            raise ValueError("Recorded artifact pins require an absolute path")
        candidates = []
        for prefix, root_name in sorted(self.relocations.items(), key=lambda item: -len(item[0])):
            prefix_path = path_type(prefix)
            try:
                relative = source.relative_to(prefix_path)
            except ValueError:
                continue
            target = self.roots[root_name].joinpath(*relative.parts).resolve()
            if not target.is_relative_to(self.roots[root_name]):
                raise ValueError("Relocated artifact escapes root")
            candidates.append(target)
            break
        if not candidates:
            candidates.append(Path(recorded_path))
        target = candidates[0].resolve(strict=True)
        actual = file_pin(target)
        if actual["bytes"] != recorded["bytes"] or actual["sha256"] != recorded["sha256"]:
            raise ValueError(f"Recorded artifact checksum mismatch: {source}")
        return target

    def inspect(self, *, verify: bool = False) -> list[dict[str, Any]]:
        result = []
        for name, artifact in sorted(self.entries.items()):
            row: dict[str, Any] = {"name": name, "role": artifact.role, "path": str(artifact.path), "optional": artifact.optional, "exists": artifact.path.is_file()}
            if verify and row["exists"]:
                artifact.resolve(verify=True)
                row["verified"] = bool(artifact.sha256)
            result.append(row)
        return result
