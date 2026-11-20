"""Export a fresh W1 CPU audit model including validated collision bounds."""
from pathlib import Path
import mujoco
import numpy as np
from .pen_owned_probe_recorder import save_physics
from ..mechanics.pen_body_bvh import copy_refit_pen_bvh
from ..artifacts import file_pin
from ..strict_json import write_json


def export_pen_cpu_model(runtime,destination):
    """Return pins and geometry proof; keep the old numerical snapshot for provenance."""
    destination=Path(destination)
    destination.mkdir(exist_ok=False)
    source=destination/'device_parameters'
    source.mkdir()
    physics=save_physics(runtime,source)
    m=mujoco.MjModel.from_binary_path(str(source/'model.mjb'))
    repaired,proof=copy_refit_pen_bvh(m,runtime.features.body)
    path=destination/'model.mjb'
    mujoco.mj_saveModel(repaired,str(path))
    loaded=mujoco.MjModel.from_binary_path(str(path))
    np.testing.assert_array_equal(loaded.bvh_aabb,repaired.bvh_aabb)
    record=dict(physics=physics,physical_context=file_pin(source/'physical_context.npz'),
                legacy_export=file_pin(source/'model.mjb'),model=file_pin(path),refit=proof)
    write_json(destination/'report.json',record)
    return record
