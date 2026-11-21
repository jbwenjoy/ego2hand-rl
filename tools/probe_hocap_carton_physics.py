"""Replay a single frozen hand controller as assumed carton mass and pad friction vary."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.ego import hocap_carton_control as control
from supermimic_research.strict_json import write_json
from evaluate_hocap_carton import errors, rms


def probe(parent, output):
    report = json.loads(parent.read_bytes())
    mjb = verify_file_pin(report["mjb"])
    with np.load(verify_file_pin(report["trace"]), allow_pickle=False) as z:
        wrist, fingers, times = z["target_wrist"], z["target_finger_q"], z["source_nominal_time_s"]
        source, nominal_qpos = z["world_from_carton_source"], z["sim_qpos"]
    outcomes = []
    for mass in (.1, .15, .2, .3, .48):
        for mu in (.5, .7, .9):
            model = mujoco.MjModel.from_binary_path(str(mjb))
            obj = model.body("carton").id
            geom = model.geom("carton_collision").id
            table = model.geom("table").id
            other = np.where(model.pair_geom1 == geom, model.pair_geom2, model.pair_geom1)
            pairs = ((model.pair_geom1 == geom) | (model.pair_geom2 == geom)) & (other != table)
            if not pairs.any():
                raise ValueError("Missing explicit hand-carton contacts")
            original_mu = model.pair_friction[pairs,:2]
            if model.body_mass[obj] != mass or not np.allclose(original_mu, mu, rtol=0, atol=1e-12):
                model.body_inertia[obj] *= mass / model.body_mass[obj]
                model.body_mass[obj] = mass
                model.pair_friction[pairs,:2] = mu
                mujoco.mj_setConst(model, mujoco.MjData(model))
            trace = control.replay(model, wrist, fingers, times, report["wrist"])
            obj_sim = trace["world_from_carton_sim"]
            pos, ang = errors(source, obj_sim)
            lift = (obj_sim[:, 2, 3] - obj_sim[0, 2, 3]) * 1000
            outcomes.append(dict(mass_kg=mass, hand_mu=mu, peak_lift_mm=float(lift.max()),
                                 position_rms_mm=rms(pos), rotation_rms_deg=rms(ang), max_rotation_deg=float(ang.max()),
                                 endpoint_mm=float(pos[-1]), peak_penetration_mm=1000*float(trace["interval_max_carton_penetration_m"].max())))
            if mass == .15 and mu == .7:
                np.testing.assert_allclose(trace["sim_qpos"], nominal_qpos, rtol=0, atol=1e-10)
    write_json(output, dict(parent=file_pin(parent), tool=file_pin(__file__), cases=outcomes,
                            budget_native_steps=len(outcomes)*int(round(times[-1]/model.opt.timestep)),
                            method="Frozen identical wrist reference and finger commands, same wrist feedback gains. "
                                   "Inertia scales with assumed mass. Table friction fixed."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    probe(args.parent, args.output)
