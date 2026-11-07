"""Read-only CPU/GPU inference wrapper for the pinned Dex pen world model."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from .geometry import quat_matrix


class FrozenPenWorldModel(torch.nn.Module):
    """Takes state58 and absolute executed target20, not raw policy action.

    Controller EMA/phase remains the caller's responsibility. Fixed simulator
    context comes from the actual training run. Returned contact logits are
    the original self-collision head, never CHORD hand-object contacts.
    """
    def __init__(self,checkpoint_path,physical_context_path,*,device="cpu"):
        super().__init__()
        from dex_supermimic.rl.dex_supermimic.networks import WorldModel
        self.checkpoint_path=Path(checkpoint_path)
        saved=torch.load(self.checkpoint_path,map_location=device,weights_only=False)
        if saved["format_version"]!=3 or saved["config"]["task"]!="hand2-pen-spin":raise ValueError("expected pen WM checkpoint")
        state=saved["modules"]["wm"]
        dimension=state["net.0.weight"].shape[-1]-20
        with np.load(physical_context_path,allow_pickle=False) as data:context=data["reader_context"].copy()
        if "context_names" not in saved:raise ValueError("checkpoint lacks declared simulator context layout")
        expected_context=sum(int(size) for _,size in saved["context_names"])
        if context.ndim!=2 or len(context)<1 or context.shape[1]!=expected_context or not np.isfinite(context).all() or not np.allclose(context,context[0],atol=1e-6):
            raise ValueError("this wrapper requires a verified fixed physical context")
        self.register_buffer("simulator_context",torch.as_tensor(context[0],dtype=torch.float32,device=device))
        self.disturbance_dim=dimension-58-context.shape[-1]
        if self.disturbance_dim!=174:raise ValueError("checkpoint/context dimension mismatch for the pinned Hand2 pen disturbance layout")
        self.model=WorldModel(dimension,saved["contact_dim"],tuple(saved["config"]["wm_hidden"]),amp=False).to(device)
        self.model.load_state_dict(state);self.model.eval();self.model.requires_grad_(False)
        self.iteration=saved["iteration"]
        self.joint_names=saved["joint_names"]
        self.reference_signature=saved["reference_signature"]
        self.metadata={"checkpoint_sha256":hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest(),
                       "physical_context_sha256":hashlib.sha256(Path(physical_context_path).read_bytes()).hexdigest(),
                       "iteration":self.iteration,"dynamics_dim":dimension,"simulator_context_dim":context.shape[-1],
                       "disturbance_dim":self.disturbance_dim,"exogenous_disturbance":"zero, matching the controlled run",
                       "contact_semantics":"self collision only"}

    def forward(self,state,executed_target):
        if state.shape[-1]!=58 or executed_target.shape!=state.shape[:-1]+(20,):raise ValueError("expected matching state58/target20")
        context=self.simulator_context.expand(state.shape[:-1]+self.simulator_context.shape)
        disturbance=state.new_zeros(state.shape[:-1]+(self.disturbance_dim,))
        dynamics=torch.cat((state,context,disturbance),-1)
        return self.model(state,executed_target,dynamics)


def rollout_palm_states(rollout):
    """Reconstruct exactly the SimReader frame from world-oriented log fields."""
    for key in ("object_lin_vel_w_m_s","object_ang_vel_w_rad_s","joint_vel_rad_s"):
        if key not in rollout:raise ValueError(f"missing state evidence {key}")
    wrist=rollout["wrist_pose_w"];pose=rollout["object_pose_w"]
    inv=np.swapaxes(quat_matrix(wrist[:,3:]),-1,-2)
    object_rotation=inv@quat_matrix(pose[:,3:])
    rotate=lambda v:np.einsum("tij,tj->ti",inv,v)
    return np.concatenate((rollout["joint_pos_rad"],rollout["joint_vel_rad_s"],rotate(pose[:,:3]-wrist[:,:3]),
                           object_rotation.reshape(-1,9),rotate(rollout["object_lin_vel_w_m_s"]),
                           rotate(rollout["object_ang_vel_w_rad_s"])),axis=-1).astype(np.float32)
