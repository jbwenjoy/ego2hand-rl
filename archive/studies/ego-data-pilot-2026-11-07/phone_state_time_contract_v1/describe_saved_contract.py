"""Read only the four already-saved zero-velocity native interface probes."""
from pathlib import Path
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np
import state_contract as c

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent


def pin(path):
    p = Path(path).resolve()
    b = p.read_bytes()
    return {"path": str(p), "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def main():
    out = HERE / "saved_zero_velocity_readback.json"
    if out.exists():
        raise FileExistsError(out)
    parent_path = PILOT / "native_equilibrium_v6/four_forward_contract_v1/report.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    asset_path = PILOT / "asset_stage2_convex/completed_v2/phone_asset.xml"
    body = ET.parse(asset_path).getroot().find(".//body[@name='hot3d_phone24']")
    ipos = np.array([float(x) for x in body.find("inertial").get("pos").split()])
    rows = []
    for name, oldpin in parent["outputs"].items():
        if not name.endswith("_raw.npz"):
            continue
        assert pin(oldpin["path"]) == oldpin
        with np.load(oldpin["path"],allow_pickle=False) as f:
            raw = {k:f[k].copy() for k in ("qpos27","qvel26","phone_root_world_m",
                "phone_body_R_world","phone_COM_world_m","COM_Jp3x26","COM_Jr3x26")}
        assert np.all(raw["qvel26"] == 0)
        s = c.to_com_world_state(raw["qpos27"][:20],raw["qvel26"][:20],
            raw["phone_root_world_m"],raw["phone_body_R_world"],raw["qvel26"][20:23],
            raw["qvel26"][23:26],ipos)
        jp_v = raw["COM_Jp3x26"] @ raw["qvel26"]
        jr_v = raw["COM_Jr3x26"] @ raw["qvel26"]
        decoded = c.from_com_world_state(s,ipos)
        rows.append({"raw":oldpin,"COM_position_max_abs_m":float(np.max(np.abs(s[40:43]-raw["phone_COM_world_m"]))),
            "COM_velocity_vs_saved_Jp_v_max_abs_m_s":float(np.max(np.abs(s[52:55]-jp_v))),
            "world_omega_vs_saved_Jr_v_max_abs_rad_s":float(np.max(np.abs(s[55:58]-jr_v))),
            "inverse_root_position_max_abs_m":float(np.max(np.abs(decoded["p_root_world"]-raw["phone_root_world_m"]))),
            "all_zero_velocity":True})
    assert len(rows) == 4
    assert not any(x in sys.modules for x in ("mujoco","torch","warp","mujoco_warp"))
    result={"schema":"phone_state_time_contract_saved_zero_velocity_readback:1",
        "status":"COMPLETE_SAVED_ZERO_VELOCITY_DESCRIPTOR_NOT_DYNAMIC_QUALIFICATION",
        "parent_report":pin(parent_path),"declared_body_ipos_source_asset":pin(asset_path),
        "body_ipos_body_m":ipos.tolist(),"source":pin(__file__),"module":pin(HERE/"state_contract.py"),
        "saved_NPZ_file_reads":4,"new_native_model_FK_forward_step_NN_GPU_calls":0,"rows":rows,
        "scope":"Declared asset COM offset agrees with these saved native COM points. Zero velocities only: Jp/Jr comparisons cannot validate nonzero angular velocity; independent analytic fixtures do. No new model/body_ipos native readback or clock qualification."}
    out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"report":str(out),"saved_NPZ_reads":4,"max_COM_position_error_m":max(x["COM_position_max_abs_m"] for x in rows),
        "max_COM_velocity_error_m_s":max(x["COM_velocity_vs_saved_Jp_v_max_abs_m_s"] for x in rows)}))


if __name__=="__main__":
    main()
