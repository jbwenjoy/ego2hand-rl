"""Run from the extracted replay folder with MuJoCo 3.11 and NumPy."""

import argparse
from pathlib import Path

import mujoco
import numpy as np


def replay(folder):
    model = mujoco.MjModel.from_binary_path(str(folder / "scene.mjb"))
    assert model.nmocap == model.neq == 0 and model.nu == 20
    assert model.jnt_dofadr[model.joint("wrist_free").id] == 0
    obj_joint = model.joint("carton_free").id
    assert model.jnt_type[obj_joint] == mujoco.mjtJoint.mjJNT_FREE
    assert model.jnt_dofadr[obj_joint] >= 6 and not np.any(model.actuator_trnid[:, 0] == obj_joint)
    with np.load(folder / "controls.npz", allow_pickle=False) as archive:
        clip = dict(archive)
    data = mujoco.MjData(model)
    data.qpos[:] = clip["initial_qpos"]
    data.qvel[:] = clip["initial_qvel"]
    data.ctrl[:] = clip["initial_ctrl"]
    mujoco.mj_forward(model, data)
    mujoco.mj_forward(model, data)
    body = model.body("carton").id
    frames = clip["frame_end_substep"]
    locations, rotations = [], []
    for i, end in enumerate(frames):
        if i:
            for step in range(frames[i-1], end):
                data.ctrl[:] = clip["finger_target"][step]
                data.qfrc_applied[:6] = clip["wrist_wrench"][step]
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.qpos, clip["expected_qpos"][i], rtol=0, atol=1e-9)
        np.testing.assert_allclose(data.qvel, clip["expected_qvel"][i], rtol=0, atol=1e-9)
        locations.append(data.xpos[body].copy())
        rotations.append(data.xmat[body].reshape(3, 3).copy())
    np.testing.assert_allclose(locations, clip["expected_carton"][:, :3, 3], rtol=0, atol=1e-9)
    np.testing.assert_allclose(rotations, clip["expected_carton"][:, :3, :3], rtol=0, atol=1e-9)
    delta = np.asarray(locations) - clip["source_carton"][:, :3, 3]
    rms = np.sqrt(np.mean(np.sum(delta * delta, axis=1)))
    print(f"{len(frames)} frames, {frames[-1]} physics steps, position RMS {rms*1000:.3f} mm")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path, nargs="?", default=Path(__file__).resolve().parent)
    replay(p.parse_args().folder)
