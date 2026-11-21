"""Package and independently execute a validated native hand-control trajectory."""

import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import numpy as np

from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.strict_json import write_json


def package(run, evaluation, out):
    result = json.loads(evaluation.read_bytes())
    if not result["passed_numeric_development_gates"] or verify_file_pin(result["run"]) != run.resolve():
        raise ValueError("Expected a passing evaluation of this exact run")
    source = json.loads(verify_file_pin(result["source"]).read_bytes())
    report = json.loads(run.read_bytes())
    scene = verify_file_pin(report["mjb"])
    with np.load(verify_file_pin(report["trace"]), allow_pickle=False) as t:
        payload = io.BytesIO()
        np.savez_compressed(
            payload, initial_qpos=t["sim_qpos"][0], initial_qvel=t["sim_qvel"][0],
            initial_ctrl=t["finger_ctrl"][0], frame_end_substep=t["frame_end_substep"],
            finger_target=t["substep_finger_ctrl"], wrist_wrench=t["substep_wrist_generalized_wrench"],
            source_carton=t["world_from_carton_source"], expected_carton=t["world_from_carton_sim"],
            expected_qpos=t["sim_qpos"], expected_qvel=t["sim_qvel"],
        )
    out.mkdir(parents=True, exist_ok=False)
    archive = out / "Wuji-HOCap-carton-replay.zip"
    readme = (
        f"HO-Cap {source['sequence']}, {source['object_id']}, frames {source['frames']}.\n"
        "Extract the archive and run python replay_exported_hand_controls.py with MuJoCo 3.11.0 and NumPy.\n"
        "The compiled model includes the floating right Wuji Hand 2, free carton and tabletop.\n"
        "The stored commands apply to the 20 finger actuators and the wrist. Wrist translational forces "
        "use world axes and torques use local axes. The source object pose is used for scoring.\n"
        "The physical parameters are simulation assumptions. The archive is a development-clip replay.\n"
        "Human motion and object scan: Wang et al., HO-Cap (CC BY 4.0), https://irvlutd.github.io/HOCap/\n"
    )
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(scene, "scene.mjb")
        z.write(Path(__file__).with_name("replay_exported_hand_controls.py"), "replay_exported_hand_controls.py")
        z.writestr("controls.npz", payload.getvalue())
        z.writestr("README.txt", readme)
        z.write(evaluation, "development_evaluation.json")
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                raise ValueError("Archive verification failed")
            z.extractall(tmp)
        replay = subprocess.run([sys.executable, "replay_exported_hand_controls.py"], cwd=tmp,
                                capture_output=True, check=True, text=True)
    write_json(out / "report.json", dict(archive=file_pin(archive), original=file_pin(run),
                                         evaluation=file_pin(evaluation), verification=replay.stdout.strip()))
    print(replay.stdout.strip())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("evaluation", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    package(a.run, a.evaluation, a.out)
