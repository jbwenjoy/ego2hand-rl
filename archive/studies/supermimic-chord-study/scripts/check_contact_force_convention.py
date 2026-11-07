"""Native MuJoCo static gravity probe for the rollout contact logger's sign.

The logger is imported read-only and tested against mg, inward normals and
action/reaction. Primitives exercise object-as-geom1 and object-as-geom2; XML
declaration order alone does not determine MuJoCo's collision pair ordering.
This is a convention check, not a manipulation or learning result.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGGER = ROOT / "scripts" / "controlled_pen_rollout.py"


def model_xml(object_kind: str, support_kind: str, object_first: bool) -> str:
    # A sphere may be tilted without changing geometry; this also exercises the
    # logger's object-local normal transform rather than assuming identity pose.
    height = 0.06 if support_kind == "plane" else 0.16
    if object_kind == "sphere":
        geometry = '<geom name="object_geom" type="sphere" size="0.05" mass="0.23"/>'
        orientation = 'quat="0.9659258262890683 0.2588190451025207 0 0"'
    else:
        geometry = '<geom name="object_geom" type="box" size="0.04 0.03 0.05" mass="0.23"/>'
        orientation = 'quat="1 0 0 0"'
    obj = f'<body name="object" pos="0 0 {height}" {orientation}><freejoint/>{geometry}</body>'
    support = ('<body name="support"><geom name="support_geom" type="plane" size="1 1 0.1"/></body>'
               if support_kind == "plane" else
               '<body name="support" pos="0 0 0.05"><geom name="support_geom" type="box" size="0.3 0.3 0.05"/></body>')
    bodies = obj + support if object_first else support + obj
    return f'''<mujoco model="contact_sign_{object_kind}_{support_kind}">
  <compiler angle="radian"/>
  <option timestep="0.001" gravity="0 0 -0.7" integrator="implicitfast"
          solver="Newton" iterations="100" tolerance="1e-12" cone="elliptic"/>
  <default><geom condim="3" friction="0.8 0.01 0.001" solref="0.01 1" solimp="0.99 0.999 0.001"/></default>
  <worldbody>{bodies}</worldbody>
</mujoco>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "contact-force-convention.json")
    args = parser.parse_args()
    report = {"checked_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "python": sys.executable, "scope": "native CPU primitive-contact sign/force convention only",
              "documentation": "https://mujoco.readthedocs.io/en/stable/computation/index.html#contact",
              "logger_path": str(LOGGER), "cases": []}
    started = time.perf_counter()
    try:
        import mujoco
        import numpy as np

        spec = importlib.util.spec_from_file_location("study_contact_logger_readonly", LOGGER)
        logger = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(logger)
        report["mujoco_version"] = mujoco.__version__
        report["logger_sha256"] = hashlib.sha256(LOGGER.read_bytes()).hexdigest()
        # sphere/plane tests geom2; sphere/box tests geom1, including reversed
        # XML order; box/plane also checks summation over four support contacts.
        cases = [("sphere", "plane", False), ("sphere", "plane", True),
                 ("sphere", "box", False), ("sphere", "box", True), ("box", "plane", True)]
        observed_roles = set()
        for obj_kind, support_kind, obj_first in cases:
            xml = model_xml(obj_kind, support_kind, obj_first)
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
            for _ in range(3000):
                mujoco.mj_step(model, data)
            # Synchronize contacts and solver forces to the current sample.
            mujoco.mj_forward(model, data)
            object_body = model.body("object").id
            support_body = model.body("support").id
            object_geom = model.geom("object_geom").id
            support_geom = model.geom("support_geom").id
            captured = logger.capture_contacts(model, data, object_body, {object_geom}, {support_geom})
            reverse = logger.capture_contacts(model, data, support_body, {support_geom}, {object_geom})
            total = captured["force_w_N"].sum(axis=0)
            reaction = reverse["force_w_N"].sum(axis=0)
            mg = float(model.body_mass[object_body] * 0.7)
            rotation = np.asarray(data.xmat[object_body]).reshape(3, 3)
            support_rotation = np.asarray(data.xmat[support_body]).reshape(3, 3)
            normals_w = captured["normal_o"] @ rotation.T
            reverse_normals_w = reverse["normal_o"] @ support_rotation.T
            # Body origins coincide with COM for these symmetric primitives.
            inward_dot = np.einsum("ij,ij->i", captured["normal_o"], -captured["pos_o_m"])
            vertical_error = abs(float(total[2]) - mg)
            relative_error = vertical_error / mg
            rows = []
            for index, contact in enumerate(data.contact[:data.ncon]):
                if {int(contact.geom1), int(contact.geom2)} != {object_geom, support_geom}:
                    continue
                role = "geom2" if int(contact.geom2) == object_geom else "geom1"
                observed_roles.add(role)
                force6 = np.zeros(6)
                mujoco.mj_contactForce(model, data, index, force6)
                rows.append({"object_role": role, "geom1": int(contact.geom1), "geom2": int(contact.geom2),
                             "geom1_name": model.geom(int(contact.geom1)).name,
                             "geom2_name": model.geom(int(contact.geom2)).name,
                             "frame_first_row_w": np.asarray(contact.frame).reshape(3, 3)[0].tolist(),
                             "contact_force6": force6.tolist(), "contact_pos_w_m": np.asarray(contact.pos).tolist(),
                             "contact_distance_m": float(contact.dist)})
            checks = {
                "solved_contacts_present": bool(len(rows) > 0 and captured["active"].all()),
                "support_force_upward": bool(total[2] > 0),
                "support_sum_matches_mg": bool(relative_error < 0.01),
                "horizontal_force_small": bool(np.linalg.norm(total[:2]) < 1e-4 * mg),
                "normal_upward_into_object": bool(len(normals_w) > 0 and (normals_w[:, 2] > 0.999).all()
                                                  and (inward_dot > 0).all()),
                "action_reaction_force": bool(np.allclose(total + reaction, 0, atol=1e-12, rtol=0)),
                "reverse_normal_opposite": bool(np.allclose(normals_w + reverse_normals_w, 0, atol=1e-12, rtol=0)),
                "settled_velocity": bool(np.max(np.abs(data.qvel)) < 1e-6),
            }
            report["cases"].append({
                "name": f"{obj_kind}_on_{support_kind}_object_xml_{'first' if obj_first else 'last'}",
                "passed": all(checks.values()), "checks": checks, "model_xml": xml,
                "model_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
                "object_geom_id": object_geom, "support_geom_id": support_geom,
                "settle_time_s": float(data.time), "mass_kg": float(model.body_mass[object_body]),
                "gravity_m_s2": model.opt.gravity.tolist(), "expected_support_N": mg,
                "measured_support_w_N": total.tolist(), "measured_reaction_w_N": reaction.tolist(),
                "support_vertical_absolute_error_N": vertical_error, "support_vertical_relative_error": relative_error,
                "maximum_abs_qvel": float(np.max(np.abs(data.qvel))),
                "normal_w": normals_w.tolist(), "inward_dot_m": inward_dot.tolist(), "raw_contacts": rows,
            })
        report["observed_object_roles"] = sorted(observed_roles)
        report["passed"] = all(case["passed"] for case in report["cases"]) and observed_roles == {"geom1", "geom2"}
    except Exception as exc:
        report["passed"] = False
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    report["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "cases": len(report["cases"]),
                      "failed_cases": [row["name"] for row in report["cases"] if not row["passed"]],
                      "roles": report.get("observed_object_roles"),
                      "max_support_relative_error": max((row["support_vertical_relative_error"] for row in report["cases"]), default=None),
                      "error": report.get("error", {}).get("message"), "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
