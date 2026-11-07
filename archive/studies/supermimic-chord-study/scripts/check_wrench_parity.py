"""Numerical parity with the fixed local FlashCHORD CPU kernels.

Run with the study's training interpreter, for example:
  D:\\workspace\\dex-supermimic-research\\.venv\\Scripts\\python.exe scripts/check_wrench_parity.py

This checks geometric support/reward formulas, frame conversion and scaling.
It does not validate a learned dynamics model or a manipulation policy.
The upstream files are imported read-only; no simulator assets are needed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ATOL = 2.0e-5
RTOL = 2.0e-5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flash-repo", type=Path, default=Path(r"D:\workspace\video-to-data-research"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "wrench-parity.json")
    args = parser.parse_args()
    report = {
        "checked_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "scope": "CPU formula/frame/scale parity; no dynamics or policy claim",
        "python": sys.executable,
        "upstream_repo": str(args.flash_repo.resolve()),
        "upstream_commit": git_head(args.flash_repo),
        "atol": ATOL,
        "rtol": RTOL,
        "checks": [],
    }
    started = time.perf_counter()
    try:
        import numpy as np
        import torch
        import warp as wp

        sys.path.insert(0, str(ROOT / "src"))
        sys.path.insert(0, str(args.flash_repo / "robotic_grounding" / "flash_chord" / "src"))
        from chord_study.contacts import sample_basis, support_reward, wrench_support
        from chord_study.geometry import quat_matrix, world_to_object

        official = importlib.import_module("flash_chord.objectives.wrench")
        official_contact = importlib.import_module("flash_chord.objectives.contact")
        report["versions"] = {"torch": torch.__version__, "numpy": np.__version__, "warp": wp.__version__}
        report["source_sha256"] = {
            "study_contacts": sha256(ROOT / "src" / "chord_study" / "contacts.py"),
            "official_wrench": sha256(Path(official.__file__)),
            "official_contact": sha256(Path(official_contact.__file__)),
        }
        wp.init()
        basis_count, seed = 512, 47
        basis_t = sample_basis(basis_count, seed=seed, dtype=torch.float32, device="cpu")
        basis_np = official.sample_wrench_basis(basis_count, rc=1.0, seed=seed)
        rng = np.random.default_rng(28931)

        def compare(name, actual, expected, *, metadata=None):
            actual = np.asarray(actual)
            expected = np.asarray(expected)
            error = float(np.abs(actual - expected).max()) if actual.size else 0.0
            passed = bool(np.isfinite(actual).all() and np.isfinite(expected).all()
                          and np.allclose(actual, expected, atol=ATOL, rtol=RTOL))
            row = {"name": name, "passed": passed, "shape": list(actual.shape), "max_abs_error": error}
            if metadata:
                row.update(metadata)
            report["checks"].append(row)
            return passed

        compare("shared_unit_basis", basis_t.numpy(), basis_np,
                metadata={"seed": seed, "basis_count": basis_count, "basis_radius_parameter": 1.0})

        def warp_support(points, normals, mask, radius, mu, edges):
            points = np.asarray(points, dtype=np.float32)
            normals = np.asarray(normals, dtype=np.float32)
            mask = np.asarray(mask, dtype=np.int32)
            if points.ndim != 2:
                raise ValueError("one body per CPU launch is expected")
            cos_t, sin_t = official.friction_cone_angles(edges)
            with wp.ScopedDevice("cpu"):
                output = wp.zeros(basis_count, dtype=wp.float32, device="cpu")
                wp.launch(
                    official.wrench_support,
                    dim=basis_count,
                    inputs=[
                        wp.array(points, dtype=wp.vec3, device="cpu"),
                        wp.array(normals, dtype=wp.vec3, device="cpu"),
                        wp.array(mask, dtype=wp.int32, device="cpu"),
                        wp.array(basis_np, dtype=wp.spatial_vector, device="cpu"),
                        wp.array(cos_t, dtype=wp.float32, device="cpu"),
                        wp.array(sin_t, dtype=wp.float32, device="cpu"),
                        float(mu),
                        wp.array(np.asarray([radius], dtype=np.float32), dtype=wp.float32, device="cpu"),
                        points.shape[0],
                        edges,
                        basis_count,
                    ],
                    outputs=[output],
                    device="cpu",
                )
                return output.numpy().copy()

        def run_support(name, points, normals, mask, radius, *, mu=0.1, edges=8):
            points, normals = np.asarray(points, dtype=np.float32), np.asarray(normals, dtype=np.float32)
            mask = np.asarray(mask, dtype=np.int32)
            actual = wrench_support(
                torch.tensor(points), torch.tensor(normals), torch.tensor(mask), basis_t, radius, mu=mu, edges=edges
            ).detach().numpy()
            expected = warp_support(points, normals, mask, radius, mu, edges)
            compare(name, actual, expected, metadata={"contacts": len(mask), "active_contacts": int(mask.sum()),
                                                     "radius_m": radius, "mu": mu, "edges": edges})
            return actual, expected

        points = rng.uniform(-0.035, 0.035, (9, 3)).astype(np.float32)
        normals = rng.normal(size=(9, 3)).astype(np.float32)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        ones = np.ones(9, dtype=np.int32)
        mixed = np.asarray([1, 0, 1, 0, 0, 1, 1, 0, 1], dtype=np.int32)
        base, _ = run_support("nonempty_random_contacts", points, normals, ones, 0.04)
        run_support("mixed_active_mask", points, normals, mixed, 0.04)
        run_support("all_inactive_mask", points, normals, np.zeros(9, dtype=np.int32), 0.04)
        run_support("zero_inactive_normals", points, np.zeros_like(normals), np.zeros(9, dtype=np.int32), 0.04)
        run_support("empty_contact_array", np.empty((0, 3)), np.empty((0, 3)), np.empty((0,), dtype=np.int32), 0.04)
        poles = np.asarray([[0, 0, 1], [0, 0, -1], [1e-7, -1e-7, 1],
                            [1e-7, 1e-7, -1], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        poles /= np.linalg.norm(poles, axis=-1, keepdims=True)
        run_support("normal_poles_and_equator", points[:6], poles, np.ones(6, dtype=np.int32), 0.04)
        run_support("zero_friction", points, normals, ones, 0.04, mu=0.0)
        run_support("sixteen_cone_edges", points, normals, mixed, 0.04, edges=16)
        # Change p and rc by the same ratio: normalized torque and support must not change.
        scaled, _ = run_support("scaled_geometry_official_parity", points * 7.5, normals, ones, 0.04 * 7.5)
        compare("uniform_object_scale_invariance", scaled, base, metadata={"scale": 7.5})

        batch_points = np.stack([points, points * 0.5, points * 2])
        batch_normals = np.stack([normals, normals, normals])
        batch_masks = np.stack([ones, mixed, np.zeros(9, dtype=np.int32)])
        radii = np.asarray([0.04, 0.02, 0.08], dtype=np.float32)
        batch_actual = wrench_support(torch.tensor(batch_points), torch.tensor(batch_normals),
                                      torch.tensor(batch_masks), basis_t, torch.tensor(radii)).numpy()
        batch_expected = np.stack([warp_support(p, n, m, r, 0.1, 8)
                                   for p, n, m, r in zip(batch_points, batch_normals, batch_masks, radii, strict=True)])
        compare("batched_radius_broadcast", batch_actual, batch_expected)

        # A known COM frame rotated/translated in world; this fixture does not assert any asset has COM at its origin.
        q = np.asarray([0.7, -0.2, 0.3, 0.5], dtype=np.float64)
        q /= np.linalg.norm(q)
        pose = np.concatenate((np.asarray([0.72, -0.31, 0.58]), q))
        rotation = quat_matrix(q)
        points_w = np.einsum("ij,kj->ki", rotation, points) + pose[:3]
        normals_w = np.einsum("ij,kj->ki", rotation, normals)
        transformed_p, transformed_n = world_to_object(points_w, normals_w, pose)
        upstream_p, upstream_n = official._to_object_frame(points_w[None], normals_w[None],
                                                          pose[None, :3], pose[None, 3:])
        compare("com_frame_transform_points", transformed_p, upstream_p[0])
        compare("com_frame_transform_normals", transformed_n, upstream_n[0])
        recovered, _ = run_support("com_transform_support_official_parity", transformed_p, transformed_n, ones, 0.04)
        compare("com_transform_support_invariance", recovered, base)

        # This is an intentional stricter input contract: upstream silently ignores tiny active normals.
        for label, tiny in (("zero", 0.0), ("threshold", 1e-3)):
            tiny_normals = torch.zeros((1, 3), dtype=torch.float32)
            tiny_normals[0, 0] = tiny
            rejected = False
            try:
                wrench_support(torch.zeros((1, 3)), tiny_normals, torch.ones(1, dtype=torch.int32), basis_t, 0.04)
            except ValueError:
                rejected = True
            report["checks"].append({"name": f"strict_contract_rejects_{label}_active_normal", "passed": rejected,
                                      "upstream_difference": "upstream skips norm <= 1e-3; sidecar rejects invalid active input"})

        # Reference/live support reward: preserve nontrivial inactive bodies/hands and multiple leading worlds.
        shape = (4, 2, 3, basis_count)
        target = rng.uniform(0.02, 1.8, shape).astype(np.float32)
        current = target * rng.uniform(0.4, 1.7, shape).astype(np.float32)
        target[0] = 0  # all reference contacts absent; official score is zero, even for absent live contacts
        current[0] = 0
        target[1, 0] = 0
        target[2, :, 2] = 0
        current[2, 1, 1, ::2] = 0  # missed commanded directions
        current[3] = target[3]  # exact matching case

        def warp_reward(cur, ref, tolerance, var):
            worlds, hands, bodies, count = ref.shape
            with wp.ScopedDevice("cpu"):
                result = wp.zeros(worlds, dtype=wp.float32, device="cpu")
                wp.launch(official_contact.contact_wrench_support_objective, dim=worlds,
                          inputs=[wp.array(ref.reshape(-1), dtype=wp.float32, device="cpu"),
                                  wp.array(cur.reshape(-1), dtype=wp.float32, device="cpu"),
                                  hands, bodies, count, tolerance, var], outputs=[result], device="cpu")
                return result.numpy().copy()

        for tolerance, var in ((0.1, 0.1), (0.0, 0.03), (0.3, 0.7)):
            actual = support_reward(torch.tensor(current), torch.tensor(target), tolerance=tolerance, var=var).numpy()
            expected = warp_reward(current, target, tolerance, var)
            compare(f"support_reward_tolerance_{tolerance}_variance_{var}", actual, expected,
                    metadata={"input_shape_world_hand_body_basis": list(shape)})

        # Probe exact threshold gating independently of random fixtures.
        boundary_ref = np.asarray([0.0, 1e-3, 1.0001e-3, 0.2, 0.2], dtype=np.float32).reshape(1, 1, 1, 5)
        boundary_cur = np.asarray([1.0, 1.0, 1e-3, 0.18, 0.3], dtype=np.float32).reshape(1, 1, 1, 5)
        compare("support_reward_activity_threshold",
                support_reward(torch.tensor(boundary_cur), torch.tensor(boundary_ref)).numpy(),
                warp_reward(boundary_cur, boundary_ref, 0.1, 0.1))

        # Reference must remain fixed during actor optimization; the live side should have finite gradients.
        live = torch.tensor(current, requires_grad=True)
        fixed = torch.tensor(target, requires_grad=True)
        support_reward(live, fixed).sum().backward()
        report["checks"].append({"name": "reward_reference_detached_live_gradient_finite",
                                  "passed": bool(fixed.grad is None and live.grad is not None
                                                 and torch.isfinite(live.grad).all()),
                                  "scope": "autograd contract only, not a physical gradient validation"})
        report["passed"] = all(check["passed"] for check in report["checks"])
    except Exception as exc:
        report["passed"] = False
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    report["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": len(report["checks"]),
                      "failed_checks": [check["name"] for check in report["checks"] if not check["passed"]],
                      "error": report.get("error", {}).get("message"),
                      "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
