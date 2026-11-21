"""Compare observed load-bearing contacts during acquisition without rerunning a policy."""

import argparse
import json
from pathlib import Path
import re

import numpy as np
from supermimic_research.artifacts import file_pin, verify_file_pin


FINGERS = ("thumb","index","middle","ring","little")


def part(name):
    finger = re.search(r"right_finger([1-5])_",name)
    return FINGERS[int(finger[1])-1] if finger else "palm"


def window_rows(trace, roles):
    """Convert the evaluator's force-on-shape0 convention to forces acting on the carton."""
    rows = trace["contact_rows"]
    offsets = trace["contact_offsets"]
    hids = set(trace["hand_shape_ids"].tolist())
    if hids != set(roles) or offsets[0]!=0 or offsets[-1]!=len(rows):
        raise ValueError("Unrecognized shape bindings or contact row boundaries")
    obj = set(trace["object_shape_ids"].tolist())
    table = set(trace["support_shape_ids"].tolist())
    for lo,hi in zip(offsets,offsets[1:]):
        sample = []
        for a,b,_,nx,ny,nz,fx,fy,fz,load in rows[lo:hi]:
            a,b=int(a),int(b)
            if (a in obj) == (b in obj):
                continue
            counterpart = b if a in obj else a
            kind = "table" if counterpart in table else roles[counterpart]
            sign = 1 if a in obj else -1
            sample.append((kind,-sign*np.array([nx,ny,nz]),sign*np.array([fx,fy,fz]),float(load)))
        yield sample


def native_rows(model,trace):
    import mujoco
    data=mujoco.MjData(model)
    data.qpos[:]=trace["sim_qpos"][0]
    data.qvel[:]=trace["sim_qvel"][0]
    data.ctrl[:]=trace["finger_ctrl"][0]
    mujoco.mj_forward(model,data)
    mujoco.mj_forward(model,data)
    obj,table=model.geom("carton_collision").id,model.geom("table").id
    finger=trace["substep_finger_ctrl"]
    wrench=trace["substep_wrist_generalized_wrench"]
    ends=trace["frame_end_substep"]
    expected=trace["sim_qpos"]
    contact_force=np.zeros(6)
    for frame,(lo,hi) in enumerate(zip(ends,ends[1:]),1):
        for step in range(lo,hi):
            data.ctrl[:]=finger[step]
            data.qfrc_applied[:6]=wrench[step]
            mujoco.mj_step(model,data)
            sample=[]
            for idx,c in enumerate(data.contact):
                if obj not in (c.geom1,c.geom2):
                    continue
                counterpart=c.geom1 if c.geom2==obj else c.geom2
                kind="table" if counterpart==table else part(model.geom(counterpart).name)
                sign=1 if c.geom2==obj else -1
                axes=c.frame.reshape(3,3)
                mujoco.mj_contactForce(model,data,idx,contact_force)
                sample.append((kind,sign*axes[0].copy(),sign*(axes.T@contact_force[:3]),float(contact_force[0])))
            yield sample
        mujoco.mj_forward(model,data)
        np.testing.assert_allclose(data.qpos,expected[frame],atol=1e-9,rtol=0)


def summarize(rows, bounds, substeps, weight):
    # Each row corresponds to the interval ending at integer source_index = step//substeps + 1.
    periods={}
    for name,(first,last) in bounds.items():
        periods[name]=dict(first_interval_end=first,last_requested_interval_end=last,observed_substeps=0,
                           opposed_digits=0,thumb_opposed=0,with_any_hand=0,hand_z=0.,table_z=0.,
                           active={f:0 for f in (*FINGERS,"palm")},total_z_per_part={f:0. for f in (*FINGERS,"palm")})
    for i,contacts in enumerate(rows):
        frame=i//substeps+1
        relevant=[x for x in periods.values() if x["first_interval_end"]<=frame<=x["last_requested_interval_end"]]
        if not relevant: continue
        engaged=[(f,n) for f,n,_,load in contacts if f!="table" and load>.01]
        active={f for f,_ in engaged}
        opposed=[(f,g) for j,(f,n) in enumerate(engaged) for g,m in engaged[j+1:]
                 if f!=g and f!="palm" and g!="palm" and n@m<-.5]
        for x in relevant:
            x["observed_substeps"]+=1
            x["with_any_hand"]+=bool(engaged)
            x["opposed_digits"]+=bool(opposed)
            x["thumb_opposed"]+=any("thumb" in pair for pair in opposed)
            for f in active: x["active"][f]+=1
            for f,_,force,_ in contacts:
                if f=="table":
                    x["table_z"]+=force[2]
                else:
                    x["hand_z"]+=force[2]
                    x["total_z_per_part"][f]+=force[2]
    for x in periods.values():
        n=x["observed_substeps"]
        for key in ("opposed_digits","thumb_opposed","with_any_hand"):
            x[key+"_fraction"]=x[key]/n if n else None
        for key in ("hand_z","table_z"):
            total=x.pop(key)
            x[key+"_mean_bodyweights"]=total/n/weight if n else None
        totals=x.pop("total_z_per_part")
        x["vertical_bodyweights_per_hand_part"]={f:v/n/weight if n else None for f,v in totals.items()}
    return periods


def main(args):
    # Fixed ranges for subject_1, from the source-backed acquisition and lift check.
    bounds={"pre_lift_26_35":(26,35),"early_lift_36_46":(36,46),"late_lift_47_54":(47,54),"hold_55_80":(55,80)}
    evidence=[]
    if args.native:
        import mujoco
        report=json.loads(args.native.read_bytes())
        model=mujoco.MjModel.from_binary_path(str(verify_file_pin(report["mjb"])))
        evidence=[file_pin(args.native),report["mjb"],report["trace"]]
        with np.load(verify_file_pin(report["trace"]),allow_pickle=False) as z:
            measured=summarize(native_rows(model,z),bounds,40,float(model.body_mass[model.body("carton").id]*abs(model.opt.gravity[2])))
        result={"native_source_aligned_search":measured}
    else:
        import newton
        import warp as wp
        from supermimic_research.runtime.flash_wuji import WujiHand2
        wp.set_device("cpu")
        builder=newton.ModelBuilder()
        WujiHand2(args.asset_dir).build(builder)
        roles={i:part(str(label).split("/")[-1]) for i,label in enumerate(builder.shape_label)}
        result={}
        for row in json.loads(args.ledger.read_bytes()):
            for check in row["checks"]:
                if check["sequence"]!="subject1_G04_2":continue
                report=json.loads(Path(check["report"]).read_bytes())
                path=Path(check["report"]).with_name("trace.npz")
                evidence.extend((file_pin(Path(check["report"])),file_pin(path)))
                with np.load(path,allow_pickle=False) as z:
                    if z["source_frames"][0]!=0:raise ValueError("Expected subject1 from absolute source frame 0")
                    result[f'{row["stream"]}_r{row["round"]}']=summarize(window_rows(z,roles),bounds,report["substeps_per_interval"],.15*9.81)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(vertical_axis="source world +Z",source="subject1_G04_2",
        active_normal_force_N=.01,opposing_normal_angle_deg=120,inputs=evidence,tool=file_pin(__file__),
        results=result),indent=2,allow_nan=False),encoding="utf8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--native",type=Path)
    g.add_argument("--ledger",type=Path)
    p.add_argument("--asset-dir",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.ledger and not a.asset_dir:p.error("--ledger requires --asset-dir")
    main(a)
