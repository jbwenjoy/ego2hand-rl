"""Copy historical research sources without moving data or environments.

Default is a read-only dry run. --apply creates byte-preserving archive files
and provenance inside this repository only. It never follows reparse points,
changes an origin, overwrites an archive conflict, or copies a nested .git.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from urllib.parse import urlsplit, urlunsplit

REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY.parent
ROOT_NAMES = (
    "dex-supermimic-research", "ego-data-pilot-2026-11-07",
    "supermimic-c2-020-portfolio-study", "supermimic-c2-confirmation-eval-study",
    "supermimic-c2-portfolio-study", "supermimic-c2-training-study",
    "supermimic-chord-penalty-only-study", "supermimic-chord-study",
    "supermimic-control-feasibility-draft", "supermimic-full151-observation-study",
    "supermimic-manipulation-research", "supermimic-observation-constrained-prefix-study",
    "supermimic-observation-contact-intent-study", "supermimic-observation-dynamic-wrench-study",
    "supermimic-observation-grounding-study", "supermimic-observation-prefix-study",
    "supermimic-observation-restoration-prefix-study", "supermimic-observation-retarget-study",
    "supermimic-observation-warmseed-grounding-study", "supermimic-rough-prefix-learning-study",
    "supermimic-shared-reference-study", "supermimic-target-inset-study",
    "supermimic-windows", "video-to-data-research",
)
CODE = {".py", ".ps1", ".sh", ".bat", ".cmd", ".command", ".js", ".jsx", ".ts", ".tsx",
        ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".cu", ".cuh", ".rs", ".ipynb", ".patch", ".diff"}
DOCS = {".md", ".rst", ".tex", ".bib"}
CONFIG = {".toml", ".ini", ".cfg", ".yaml", ".yml", ".xml", ".urdf"}
DEPENDENCY_DIRS = {".git", ".venv", "venv", "site-packages", "node_modules", "__pycache__",
                   ".cache", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".nox"}
BULK_DIRS = {"checkpoints", "assets", "media", "data", "raw", "runs", "results", "videos", "meshes"}
JSON_DIRS = {"configs", "config", "protocols", "design", "designs", "contracts", "schemas"}
SPECIAL = {".gitignore", ".gitattributes", ".gitmodules", ".editorconfig", ".python-version",
           "uv.lock", "poetry.lock", "pdm.lock", "requirements.txt", "requirements-dev.txt",
           "cmakelists.txt", "makefile", "dockerfile", "manifest.in"}
CHECKPOINT = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".pkl", ".pickle"}
ARRAY_ARCHIVE = {".npz", ".npy", ".csv", ".tsv", ".jsonl", ".parquet", ".h5", ".hdf5", ".db",
                 ".sqlite", ".zip", ".tar", ".gz", ".zst", ".7z", ".bag", ".vrs", ".bin"}
ASSET = {".obj", ".stl", ".ply", ".glb", ".gltf", ".fbx", ".blend", ".mjb", ".mp4", ".avi",
         ".mov", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg", ".pdf", ".docx", ".pptx",
         ".wav", ".flac", ".mp3", ".usd", ".usda", ".usdc", ".dae", ".mtl", ".bmp", ".ico"}
JSON_LIMIT = 1024 * 1024
JSON_ALLOW = re.compile(r"(^|[_\-.])(protocol|recipe|config|configuration|design|prereg|schema|spec)([_\-.]|$)")
JSON_SPECIAL = ("task_bundle", "source_contract", "source_interface", "family_plan", "new_final_family_plan")
JSON_DATA = re.compile(r"(^|[_\-.])(report|failure|terminal|selection|branches|catalog|constants|samples|readbacks)([_\-.]|$)")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dependency(parts: tuple[str, ...]) -> str | None:
    for part in parts:
        p = part.lower()
        if p == ".git":
            return "git_metadata"
        if p in DEPENDENCY_DIRS or p.startswith(".uv"):
            return "environment_or_cache"
    return None


def select(relative: Path, size: int) -> tuple[bool, str]:
    parts = tuple(p.lower() for p in relative.parts)
    if category := dependency(parts):
        return False, category
    name, extension = relative.name.lower(), relative.suffix.lower()
    if name.startswith(("license", "notice", "copying", "copyright")):
        return True, "license_notice"
    # Source factories and historical snapshots are retained even when a path
    # contains assets/data/outputs; the binary assets in those paths are not.
    if extension in CODE:
        return True, "source_code"
    if extension in DOCS:
        return True, "documentation"
    if name in SPECIAL or name.startswith("requirements") and extension == ".txt":
        return True, "build_environment_configuration"
    if extension == ".txt" and (name.startswith("readme") or "docs" in parts):
        return True, "documentation"
    if extension in CONFIG:
        if any(p in BULK_DIRS for p in parts[:-1]):
            return False, "asset_or_data_configuration"
        return True, "configuration"
    if extension == ".json":
        if size > JSON_LIMIT:
            return False, "large_json_data_or_evidence"
        if JSON_DATA.search(name):
            return False, "json_data_or_evidence"
        if any(p in BULK_DIRS for p in parts[:-1]):
            return False, "json_data_or_evidence"
        explicit = bool(JSON_ALLOW.search(name) or name.startswith(JSON_SPECIAL)
                        or any(p in JSON_DIRS for p in parts[:-1]))
        return (True, "explicit_json_configuration") if explicit else (False, "json_data_or_evidence")
    if extension in CHECKPOINT:
        return False, "checkpoint_or_pickle"
    if extension in ARRAY_ARCHIVE:
        return False, "array_archive_or_log"
    if extension in ASSET:
        return False, "simulator_asset_or_media"
    return False, "other_external"


def git(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(["git", "--no-optional-locks", "-C", str(root), *arguments],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise RuntimeError(f"Read-only git query failed in {root}: {arguments}")
    return result.stdout


def sanitize_remote(value: str) -> str:
    if "://" in value:
        u = urlsplit(value)
        # Authentication material is not useful provenance.
        host = u.netloc.rsplit("@", 1)[-1]
        return urlunsplit((u.scheme, host, u.path, "", ""))
    return value


def git_snapshot(root: Path) -> tuple[dict | None, bytes | None]:
    if not (root / ".git").exists():
        return None, None
    remotes = []
    for name in git(root, "remote").decode("utf8").splitlines():
        for url in git(root, "remote", "get-url", "--all", name).decode("utf8").splitlines():
            remotes.append({"name": name, "url": sanitize_remote(url)})
    patch = git(root, "diff", "--binary", "HEAD", "--")
    snapshot = {
        "origin": str(root), "HEAD": git(root, "rev-parse", "HEAD").decode().strip(),
        "branch": git(root, "branch", "--show-current").decode().strip(), "remotes": remotes,
        "remote_credentials_and_query_redacted": True,
        "status_porcelain_v1": git(root, "status", "--porcelain=v1", "--untracked-files=all").decode("utf8").splitlines(),
        "working_tree_patch": {"path": f"provenance/git/{root.name}/working_tree.patch",
                               "bytes": len(patch), "sha256": sha256(patch)},
        "patch_scope": "git diff --binary HEAD: staged and unstaged tracked changes; selected untracked sources are copied separately",
        "worktrees": git(root, "worktree", "list", "--porcelain").decode("utf8"),
    }
    return snapshot, patch


def inside(path: Path, parent: Path) -> bool:
    return path.is_relative_to(parent)


def create_identical(path: Path, data: bytes, *, apply: bool) -> str:
    if not inside(path.resolve(), REPOSITORY):
        raise ValueError(f"Destination outside repository: {path}")
    if path.exists():
        if path.is_symlink() or path.read_bytes() != data:
            raise ValueError(f"Existing archive differs; never overwrite: {path}")
        return "verified_existing"
    if not apply:
        return "planned"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    if path.read_bytes() != data:
        raise RuntimeError(f"Copied-byte verification failed: {path}")
    return "copied"


def consolidate(*, apply: bool) -> dict:
    expected = [WORKSPACE / name for name in ROOT_NAMES]
    if len(expected) != 24 or any(not r.is_dir() or r.resolve() == REPOSITORY for r in expected):
        raise ValueError("Exactly 24 existing legacy roots required; new repository must be excluded")
    copies = []
    roots = []
    reparse = []
    nested_git = []
    git_records = []
    patches = []
    operations = defaultdict(int)
    for root in expected:
        snapshot, patch = git_snapshot(root)
        if snapshot is not None:
            git_records.append(snapshot)
            patches.append((REPOSITORY / snapshot["working_tree_patch"]["path"], patch))
        summary = defaultdict(lambda: {"files": 0, "bytes": 0, "largest_examples": []})
        count = size = copied_bytes = copied_count = 0
        stack = [(root, Path())]
        while stack:
            directory, relative_dir = stack.pop()
            with os.scandir(directory) as entries:
                entries = sorted(entries, key=lambda e: e.name.casefold(), reverse=True)
            for entry in entries:
                relative = relative_dir / entry.name
                origin = root / relative
                st = entry.stat(follow_symlinks=False)
                if getattr(st, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                    reparse.append({"origin": str(origin), "category": "reparse_not_followed",
                                    "target": os.readlink(origin) if os.path.islink(origin) or os.path.isjunction(origin) else None})
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if entry.name == ".git":
                        nested_git.append(str(origin))
                    stack.append((origin, relative))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                count += 1
                size += st.st_size
                selected, category = select(relative, st.st_size)
                if selected:
                    before = origin.stat()
                    data = origin.read_bytes()
                    after = origin.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or len(data) != before.st_size:
                        raise RuntimeError(f"Origin changed while reading; preserve and stop: {origin}")
                    target = REPOSITORY / "archive" / "studies" / root.name / relative
                    # Even --apply first completes the entire read-only plan
                    # and conflict check before it writes any archive file.
                    action = create_identical(target, data, apply=False)
                    operations[action] += 1
                    copies.append({"origin": str(origin), "target": target.relative_to(REPOSITORY).as_posix(),
                                   "bytes": len(data), "sha256": sha256(data), "category": category})
                    copied_count += 1
                    copied_bytes += len(data)
                else:
                    # This is an external-location summary, not a claim to have
                    # content-hashed frozen datasets, weights or environments.
                    prefix = relative.parts[0] if len(relative.parts) > 1 else "[root files]"
                    value = summary[(category, prefix)]
                    value["files"] += 1
                    value["bytes"] += st.st_size
                    value["largest_examples"].append({"origin": str(origin), "bytes": st.st_size})
                    value["largest_examples"] = sorted(value["largest_examples"], key=lambda v: (-v["bytes"], v["origin"]))[:5]
        roots.append({"origin": str(root), "files_scanned": count, "logical_bytes_scanned": size,
                      "selected_source_files": copied_count, "selected_source_bytes": copied_bytes,
                      "external_path_summaries": [dict(category=k[0],relative_prefix=k[1],**v) for k,v in sorted(summary.items())]})
        if snapshot is not None:
            after_git, after_patch = git_snapshot(root)
            if snapshot != after_git or patch != after_patch:
                raise RuntimeError(f"Git source snapshot changed during extraction: {root}")
    for path, patch in patches:
        operations[create_identical(path, patch, apply=False)] += 1
    copies.sort(key=lambda v: v["target"].casefold())
    inventory = {
        "schema": "supermimic_workspace_source_inventory:1", "repository": str(REPOSITORY),
        "legacy_root_count": len(expected), "root_allowlist": list(ROOT_NAMES),
        "mode": "applied_source_copy" if apply else "dry_run_no_writes",
        "selection_policy": {"code_extensions": sorted(CODE), "documentation_extensions": sorted(DOCS),
            "configuration_extensions": sorted(CONFIG), "dependency_directories_excluded": sorted(DEPENDENCY_DIRS),
            "dot_uv_prefix_excluded": True, "bulk_configuration_paths_excluded": sorted(BULK_DIRS),
            "json_maximum_bytes": JSON_LIMIT, "json_policy": "explicit protocol/recipe/config/design/schema/prereg/spec/task_bundle/source_contract/source_interface/family_plan; or config/protocol/design/contract/schema directory; no report/selection/branch/catalog/constants/readback/sample JSON",
            "source_code_and_text_docs_in_outputs_retained": True, "large_binary_arrays_weights_assets_not_copied": True},
        "scope": "Byte-preserving historical source archive, not relocated runtime or scientific requalification. Original roots unchanged. External path/byte summaries are not content SHA manifests and logical sizes may count hard-linked storage repeatedly. No other GMR/hamer/Isaac/user project roots included.",
        "copied_files": copies, "roots": roots, "git_repositories": git_records,
        "git_directories_not_copied": nested_git, "reparse_points_not_followed": reparse,
        "summary": {"source_files": len(copies), "source_bytes": sum(v["bytes"] for v in copies),
                    "scanned_files": sum(v["files_scanned"] for v in roots),
                    "scanned_logical_bytes": sum(v["logical_bytes_scanned"] for v in roots),
                    "git_patch_count": len(patches)},
        "source_tool": {"path": "tools/consolidate_sources.py", "sha256": sha256(Path(__file__).read_bytes())},
    }
    if apply:
        path = REPOSITORY / "provenance" / "workspace_inventory.json"
        if path.exists():
            old = json.loads(path.read_text(encoding="utf8"))
            if old["copied_files"] != copies or old["git_repositories"] != git_records:
                raise ValueError("Inventory differs from preserved extraction; use a reviewed new version, never overwrite history")
        operations.clear()
        for entry in copies:
            data = Path(entry["origin"]).read_bytes()
            if len(data) != entry["bytes"] or sha256(data) != entry["sha256"]:
                raise RuntimeError(f"Origin changed after planning; stop without overwriting: {entry['origin']}")
            operations[create_identical(REPOSITORY / entry["target"], data, apply=True)] += 1
        for patch_path, patch in patches:
            operations[create_identical(patch_path, patch, apply=True)] += 1
        for record, (_, patch) in zip(git_records, patches):
            current, current_patch = git_snapshot(Path(record["origin"]))
            if current != record or current_patch != patch:
                raise RuntimeError(f"Origin git tree changed during copy; preserve archive and stop: {record['origin']}")
        if not path.exists():
            create_identical(path, (json.dumps(inventory, ensure_ascii=False, indent=2)+"\n").encode("utf8"), apply=True)
    return {"mode": inventory["mode"], **inventory["summary"], "operations": dict(operations),
            "roots": [{k:v[k] for k in ["origin","selected_source_files","selected_source_bytes"]} for v in roots],
            "inventory": str(REPOSITORY/"provenance/workspace_inventory.json") if apply else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Copy verified source bytes and write provenance; default is dry-run")
    args = parser.parse_args()
    print(json.dumps(consolidate(apply=args.apply), indent=2))
