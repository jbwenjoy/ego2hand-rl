"""Additive pad15/pad28/other saved labels and a controlled-static reference.

No old result changes, control optimization, native calls, neural model or data
filter. The static reference is a target hypothesis, not human force truth or a
necessary condition for moving transitions/holding.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import numpy as np

H=Path(__file__).resolve().parent
PARENT=H.parent
PARENT_SOURCE_SHA="e15f8ee0948abbbe7cb0eeec70719ebd2f9cbac472ce81c7b0f312e41c7c5c26"
SCHEMA="phone_two_designated_pad_static_reference_protocol:2"
PADS=(15,28)


def require(ok,msg):
    if not ok:raise ValueError(msg)


def pin(path):
    p=Path(path).resolve(strict=True);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def checked(r):
    p=Path(r["path"]);a=pin(p)
    require(p.is_absolute() and a["sha256"]==r["sha256"] and a["bytes"]==r["bytes"],"Frozen saved input changed")
    return p.resolve(strict=True)


def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))


def write(p,v):
    with Path(p).open("x",encoding="utf-8") as f:json.dump(v,f,indent=2,allow_nan=False)


def helper():
    p=PARENT/"build_saved_labels.py";require(pin(p)["sha256"]==PARENT_SOURCE_SHA,"Frozen wrench v1 helper")
    spec=importlib.util.spec_from_file_location("frozen_wrench_label_v1",p)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m


def grouped_world(contact,pre_tick,COM,m):
    phone=(contact[:,1]==71)^(contact[:,2]==71)
    other=np.where(contact[:,1]==71,contact[:,2],contact[:,1])
    out={}
    for name,mask in (("pad15",phone&(other==15)),("pad28",phone&(other==28)),("other_phone_boundary",phone&~np.isin(other,PADS))):
        out[name],_=m.contact_wrenches(contact[mask],pre_tick,COM)
    out["pad_only"]=out["pad15"]+out["pad28"]
    return out


def analytic(m):
    # Two symmetric designated-pad forces support gravity exactly. A separate
    # other-link force must not count as designated support.
    c=np.zeros((3,24));c[:,0]=0;c[:,1:3]=[[15,71],[28,71],[7,71]];c[:,3]=3
    c[:2,6:9]=[[1,0,0],[-1,0,0]]
    frame=np.array([[0.,0,1],[1,0,0],[0,1,0]])
    c[:2,9:18]=frame.ravel();c[2,9:18]=np.eye(3).ravel();c[:,18]=[.981,.981,1]
    g=grouped_world(c,np.array([0]),np.zeros((1,3)),m)
    req=np.array([[0.,0,1.962,0,0,0]])
    require(np.allclose(g["pad_only"],req,atol=1e-15,rtol=0) and np.array_equal(g["other_phone_boundary"],[[1.,0,0,0,0,0]]),"Designated-pad sum vs other force")
    Ry=np.array([[0.,0,1],[0,1,0],[-1,0,0]])
    targetbody=m.rotate_wrench_axes(req,Ry[None])
    require(np.allclose(targetbody,[[-1.962,0,0,0,0,0]],atol=1e-15,rtol=0),"Static gravity target in actual BODY axes")
    return dict(schema="phone_pad_static_target_analytic_fixture:2",passed=True,cases=2,native_NN_physics=0)


def preflight(path):
    p=read(path);require(p["schema"]==SCHEMA and p["designated_pad_geom_ids"]==[15,28]
        and p["new_model_native_NN_physics"] is False,"Fixed two-pad saved-only namespace")
    for r in [*p["inputs"].values(),*p["sources"].values()]:checked(r)
    require(checked(p["sources"]["exporter"]).samefile(__file__),"Current source")
    parent=read(checked(p["inputs"]["parent_label_report"]))
    require(parent["status"]=="EXPORTED_SIMULATOR_DECODED_WRENCHES_AND_OBSERVED_MOMENTUM_DEFECTS"
        and parent["rows"]==960 and parent["outputs"]["all_PRE_tick_wrenches.npz"]==p["inputs"]["parent_PRE_labels"],"Complete frozen wrench parent")
    out=Path(p["output"]);require(out.is_absolute() and out.parent.resolve()==H.resolve() and not out.exists(),"Fresh additive output")
    return p,out


def describe(v,mask):
    if not np.any(mask):return None
    x=v[mask]
    return dict(force_vector_RMS_N=float(np.sqrt(np.mean(np.sum(x[:,:3]**2,-1)))),
        torque_vector_RMS_Nm=float(np.sqrt(np.mean(np.sum(x[:,3:]**2,-1)))),
        signed_mean_force_N=np.mean(x[:,:3],0).tolist(),signed_mean_torque_Nm=np.mean(x[:,3:],0).tolist())


def run(path):
    p,out=preflight(path);out.mkdir();write(out/"protocol.json",p);m=helper();write(out/"analytic_fixture.json",analytic(m))
    params=read(checked(p["inputs"]["physical_parameters"]));ids=read(checked(p["inputs"]["row_identities"]))
    traces=read(checked(p["inputs"]["trace_pins"]))
    original_rows=read(checked(p["inputs"]["original_sample_identities"]))
    original_by_id={s["sample_id"]:s for s in original_rows}
    require(len(original_by_id)==960,"Original acquisition960 identities")
    req=np.r_[-params["mass_kg"]*np.asarray(params["gravity_world_m_s2"]),np.zeros(3)]
    with np.load(checked(p["inputs"]["parent_PRE_labels"]),allow_pickle=False) as z:
        ticks={k:z[k].copy() for k in ["row_id","pre_tick","native_h_s","COM_pre_world_m","Rbody_pre_world9","world_wrench6"]}
    rows=[];tickout=[];maxsum=0.;maxtable=0.;counts=dict(parent_derived_NPZ_reads=1,raw_interval_NPZ_reads=0,native_NN_newphysics=0)
    try:
        # Parent rows/ticks are sorted by original episode/time and row id.
        for s in ids:
            rid=s["sample_id"];where=np.flatnonzero(ticks["row_id"]==rid);n=len(where)
            require(n==original_by_id[rid]["executed_substeps"] and n>0,"All original actual prefix ticks")
            pre=ticks["pre_tick"][where];COM=ticks["COM_pre_world_m"][where];R=ticks["Rbody_pre_world9"][where].reshape(n,3,3)
            with np.load(checked(traces[str(rid)]),allow_pickle=False) as z:
                contact=z["contact_rows24"].copy();table=z["pre_table_normal_N"].copy()
            counts["raw_interval_NPZ_reads"]+=1;maxtable=max(maxtable,float(np.max(abs(table))))
            world=grouped_world(contact,pre,COM,m)
            maxsum=max(maxsum,float(np.max(abs(world["pad_only"]+world["other_phone_boundary"]-ticks["world_wrench6"][where]))))
            body={k:m.rotate_wrench_axes(v,R) for k,v in world.items()}
            targetW=np.broadcast_to(req,(n,6));targetB=m.rotate_wrench_axes(targetW,R)
            h=float(s["dt_s"]);row=dict(row_id=rid,duration_s=n*h)
            tout=dict(row_id=np.full(n,rid,dtype=np.int64),pre_tick=pre,static_reference_world6=targetW,
                static_reference_BODY6=targetB)
            for name in world:
                row[name+"_world_mean6"]=world[name].mean(0);row[name+"_BODY_mean6"]=body[name].mean(0)
                row[name+"_world_impulse6"]=world[name].sum(0)*h;row[name+"_BODY_integral6"]=body[name].sum(0)*h
                tout[name+"_world6"]=world[name];tout[name+"_BODY6"]=body[name]
            row.update(static_reference_world_mean6=req.copy(),static_reference_BODY_mean6=targetB.mean(0),
                pad_only_minus_static_reference_world6=(world["pad_only"]-targetW).mean(0),
                pad_only_minus_static_reference_BODY6=(body["pad_only"]-targetB).mean(0))
            rows.append(row);tickout.append(tout)
        require(len(rows)==960,"All912complete+48partial retained")
        arrays={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
        arrays.update(complete_H_bool=np.array([s["complete_H20ms"] for s in ids]),
            training_eligible_bool=np.array([s["training_eligible"] for s in ids]),
            family_id=np.array([s["family_id"] for s in ids],dtype=np.int64),episode_id=np.array([s["episode_id"] for s in ids],dtype=np.int64))
        np.savez_compressed(out/"all960_pad_only_other_and_static_reference.npz",**arrays)
        np.savez_compressed(out/"all_PRE_grouped_pad_other_wrenches.npz",**{k:np.concatenate([v[k] for v in tickout]) for k in tickout[0]})
        stats=[]
        for complete in (True,False):
            for split in ("ALL","train","validation","test"):
                mask=np.array([s["complete_H20ms"]==complete and (split=="ALL" or s["split"]==split) for s in ids])
                stats.append(dict(duration_class="complete_H" if complete else "actual_partial_prefix",split=split,rows=int(mask.sum()),
                    pad_world=describe(arrays["pad_only_world_mean6"],mask),other_world=describe(arrays["other_phone_boundary_world_mean6"],mask),
                    pad_minus_static_reference_world=describe(arrays["pad_only_minus_static_reference_world6"],mask),
                    pad_BODY=describe(arrays["pad_only_BODY_mean6"],mask),other_BODY=describe(arrays["other_phone_boundary_BODY_mean6"],mask),
                    pad_minus_static_reference_BODY=describe(arrays["pad_only_minus_static_reference_BODY6"],mask)))
        write(out/"static_reference_error_statistics.json",stats);write(out/"all960_row_identities.json",ids)
        outputs={f.name:pin(f) for f in out.iterdir() if f.is_file()}
        report=dict(schema="phone_two_designated_pad_static_reference_labels:2",status="EXPORTED_PAD_ONLY_AND_OTHER_WITH_CONTROLLED_STATIC_REFERENCE_NOT_HOLDING_GATE",
            protocol=pin(path),outputs=outputs,designated_pad_geom_ids=[15,28],phone_geom_id=71,rows=960,complete_rows=912,partial_rows=48,
            native_PRE_ticks=sum(len(v["pre_tick"]) for v in tickout),counts=counts,static_reference_world6=req.tolist(),
            max_group_sum_vs_parent_world_abs=maxsum,source_captured_table_normal_max_abs_N=maxtable,complete_ALL=stats[0],
            scopes=["Groups are actual raw contact other-geom IDs:15/28/all remaining phone-boundary. Other includes table if any; it is not silently excluded or called legal support.",
                "Static target is [-m*g,zero torque about current COM] for the assumed native model, rotated by every actual PRE Rbody before BODY averaging.",
                "Pad-only target error is a controlled-static-reference descriptor, not force truth, a moving-state necessary condition, a holding gate, data filter or new controller.",
                "Means/integrals preserve N/Nm and Ns/Nms separately; no mixed-unit6D norm is formed. BODY means use instantaneous body axes, not source-fixed axes.",
                "No three-finger reinterpretation: a future designated set requires new schema/design. No old data/label/result changes, NN/native/physics/training."])
        write(out/"report.json",report);print(json.dumps(dict(report=pin(out/"report.json"),complete_ALL=stats[0],max_group_sum_abs=maxsum)),flush=True)
    except BaseException as e:
        write(out/"failure.json",dict(error=repr(e),completed_rows=len(rows),counts=counts,automatic_retry=False));raise


def main():
    a=argparse.ArgumentParser();a.add_argument("--protocol",type=Path,required=True);a.add_argument("--run",action="store_true");p=a.parse_args()
    if p.run:run(p.protocol.resolve(strict=True))
    else:
        _,out=preflight(p.protocol.resolve(strict=True));print(json.dumps(dict(status="METADATA_ONLY",output=str(out))))

if __name__=="__main__":main()
