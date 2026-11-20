"""Private-data native mug audit using the exact DLL already used by Python MuJoCo."""
import ctypes
from pathlib import Path
import mujoco
import numpy as np


class MugWindow:
    def __init__(self,library,model,binding):
        self.model=model
        self.dll=ctypes.CDLL(str(Path(library).resolve(strict=True)))
        self.mj=ctypes.CDLL(str(Path(mujoco.__file__).parent/'mujoco.dll'))
        if self.dll.mug_header_version()!=mujoco.mj_version() or self.mj.mj_version()!=mujoco.mj_version():
            raise ValueError('MuJoCo header and runtime version mismatch')
        if self.dll.mug_scalar_bytes()!=8 or model.geom_size.dtype!=np.float64:
            raise ValueError('Expected double precision MuJoCo')
        self.hand=np.isin(np.arange(model.ngeom),[model.geom(n).id for n in binding['hand_collision_names']]).astype(np.uint8)
        self.mug=(model.geom_bodyid==binding['mug_body_id']).astype(np.uint8)
        self.table=model.geom('mug_task_table').id
        if not np.isclose(model.opt.timestep,.0003125) or model.nu!=20 or model.nq!=27 or model.nv!=26:
            raise ValueError('Unexpected mug asset or physics rate')
        self.window=self.dll.mug_window
        ptr=ctypes.c_void_p
        self.window.argtypes=[ptr,ptr,ptr,ptr,ptr,ptr,ctypes.c_int,ctypes.c_int,ptr]
        self.window.restype=ctypes.c_int
        self.step_ptr=ctypes.cast(self.mj.mj_step,ptr);self.force_ptr=ctypes.cast(self.mj.mj_contactForce,ptr)

    def step(self,data,command):
        if data.model._address!=self.model._address:
            raise ValueError('MjData must belong to the supplied MjModel')
        if data.qfrc_applied.any() or data.xfrc_applied.any():
            raise ValueError('Unexpected externally applied force')
        data.ctrl[:]=command
        peaks=np.empty(4,np.float64)
        err=self.window(self.model._address,data._address,self.step_ptr,self.force_ptr,self.hand.ctypes.data,
                        self.mug.ctypes.data,self.table,64,peaks.ctypes.data)
        if err:raise ValueError(f'Native mug window failed with status {err}')
        return peaks
