"""Saved-only post-hoc diagnosis of the failed v2 push dt-pair; no physics."""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent


def pin(path):
    p = Path(path).resolve(); b = p.read_bytes()
    return {"path": str(p), "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def checked(item):
    assert pin(item["path"])["sha256"] == item["sha256"]
    return Path(item["path"])


def rotations(q):
    norms = np.linalg.norm(q, axis=1)
    assert np.isfinite(q).all() and np.all(norms > 0)
    q = q / norms[:, None]
    w, x, y, z = q.T
    R = np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                  2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                  2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=1).reshape(-1,3,3)
    return q, norms, R


def rotation_difference(qc, qf):
    # qf * inverse(qc), WXYZ, rotation expressed in world coordinates.
    w = qf[:,0]*qc[:,0]+np.sum(qf[:,1:]*qc[:,1:], axis=1)
    vec = qc[:,0,None]*qf[:,1:]-qf[:,0,None]*qc[:,1:]-np.cross(qf[:,1:],qc[:,1:])
    flip = w < 0
    w[flip] *= -1; vec[flip] *= -1
    length = np.linalg.norm(vec, axis=1)
    angle = 2*np.arctan2(length,w)
    defined = length > 0
    axis = np.zeros_like(vec)
    axis[defined] = vec[defined]/length[defined,None]
    return np.degrees(angle), axis, np.degrees(angle[:,None]*axis), defined


def describe(trace, dt):
    q, norms, R = rotations(trace["qpos"][:,3:].copy())
    normal = R[:,:,1]  # Original thin local Y, initial Rx90 maps it toward world+Z.
    longaxis = R[:,:,2]
    horizontal_length = np.linalg.norm(longaxis[:,:2], axis=1)
    assert np.all(horizontal_length > 0), "Projected heading undefined; preserve rather than repair"
    heading = np.unwrap(np.arctan2(longaxis[:,1], longaxis[:,0]))
    tilt = np.arccos(np.clip(normal[:,2],-1,1))
    counts = trace["contact_count"]
    changes = np.flatnonzero(counts[1:] != counts[:-1])+1
    return {"q":q,"norms":norms,"R":R,"heading_deg":np.degrees(heading-heading[0]),
            "tilt_deg":np.degrees(tilt),"normal_world":normal,"longaxis_world":longaxis,
            "contact_count_changes":[{"tick":int(i),"tick_time_s":float(i*dt),
                 "from":int(counts[i-1]),"to":int(counts[i])} for i in changes],
            "contact_count_histogram":{str(int(k)):int(v) for k,v in zip(*np.unique(counts,return_counts=True))}}


def main():
    output = BASE/"push_orientation_diagnostic_v1.json"
    if output.exists():raise FileExistsError(output)
    selection = json.loads((BASE/"cpu_screen_assessment_v2/selection.json").read_text())
    assert selection["selected_timeconst_s"] is None and selection["eligible_timeconst_s"] == []
    result = json.loads(checked(selection["screening_result_pin"]).read_text())
    rows = sorted([r for r in result["cases"] if r["test"] == "force_push_release" and r["timeconst_s"] == .0025],key=lambda r:r["dt_s"],reverse=True)
    assert len(rows) == 2 and [r["dt_s"] for r in rows] == [.00125,.000625]
    traces, desc = [], []
    for row in rows:
        path = checked(row["trajectory"])
        assert path.samefile(checked(selection["trajectory_pins"][row["case"]]))
        with np.load(path,allow_pickle=False) as z:trace={k:z[k] for k in z.files}
        n=round(3/row["dt_s"])
        assert np.array_equal(trace["tick"],np.arange(n+1))
        # Source-native generated candidates, without force filtering.
        longtable=trace["contact_long_table"]
        assert np.array_equal(np.bincount(longtable[:,0].astype(np.int64),minlength=n+1),trace["contact_count"])
        traces.append(trace);desc.append(describe(trace,row["dt_s"]))
    coarse,fine=traces;dc,df=desc
    dt=rows[0]["dt_s"]; ticks=coarse["tick"];t=ticks*dt
    angle,axis,rotvec,axis_defined=rotation_difference(dc["q"],df["q"][::2])
    normal_angle=np.degrees(np.arccos(np.clip(np.sum(dc["normal_world"]*df["normal_world"][::2],axis=1),-1,1)))
    heading_delta=(df["heading_deg"][::2]-dc["heading_deg"]+180)%360-180
    contact_diff=coarse["contact_count"]!=fine["contact_count"][::2]
    count_difference_events=[{"coarse_tick":int(i),"fine_tick":int(2*i),"time_s":float(i*dt),
                               "coarse_count":int(coarse["contact_count"][i]),"fine_count":int(fine["contact_count"][2*i])}
                              for i in np.flatnonzero(contact_diff)]
    above=np.flatnonzero(angle>2.) # Existing G7 final-orientation limit, used descriptively over time.
    nonzero=np.flatnonzero(angle>0.)
    milestones=[]
    for sec in [0.,.5,1.,1.2,1.5,2.,2.5,3.]:
        i=round(sec/dt)
        milestones.append({"tick":i,"time_s":sec,"relative_angle_deg":float(angle[i]),
                           "relative_world_axis":axis[i].tolist() if axis_defined[i] else None,
                           "relative_world_rotvec_deg":rotvec[i].tolist(),
                           "coarse_heading_deg":float(dc["heading_deg"][i]),"fine_heading_deg":float(df["heading_deg"][2*i]),
                           "coarse_thin_axis_tilt_deg":float(dc["tilt_deg"][i]),"fine_thin_axis_tilt_deg":float(df["tilt_deg"][2*i]),
                           "coarse_generated_contacts":int(coarse["contact_count"][i]),"fine_generated_contacts":int(fine["contact_count"][2*i])})
    curves_path=BASE/"push_orientation_diagnostic_v1_arrays.npz"
    np.savez_compressed(curves_path,shared_tick=ticks,shared_time_s=t,coarse_raw_clock_s=coarse["time_s"],
        fine_tick=fine["tick"],fine_raw_clock_s=fine["time_s"],relative_angle_deg=angle,
        relative_world_axis=axis,relative_axis_defined=axis_defined,relative_world_rotvec_deg=rotvec,
        coarse_heading_deg=dc["heading_deg"],fine_heading_deg=df["heading_deg"],
        shared_heading_difference_deg=heading_delta,coarse_thin_axis_tilt_deg=dc["tilt_deg"],fine_thin_axis_tilt_deg=df["tilt_deg"],
        shared_thin_axis_normal_difference_deg=normal_angle,coarse_contact_count=coarse["contact_count"],fine_contact_count=fine["contact_count"],
        shared_contact_count_difference=contact_diff,coarse_penetration_m=coarse["penetration_m"],fine_penetration_m=fine["penetration_m"],
        coarse_COM_world_m=coarse["com_world_m"],fine_COM_world_m=fine["com_world_m"],
        coarse_quaternion_norm=dc["norms"],fine_quaternion_norm=df["norms"])
    plt.rcParams.update({"font.size":9,"svg.fonttype":"none"})
    fig,ax=plt.subplots(2,3,figsize=(13,7.5),constrained_layout=True)
    a=ax.ravel(); colors=("#1f77b4","#d95f02")
    a[0].plot(t,angle,color="black",label="Relative angle")
    for j,label in enumerate(("world X","world Y","world Z")):a[0].plot(t,rotvec[:,j],label=label,lw=1)
    a[0].axhline(2,color="grey",ls="--",lw=.8,label="Existing 2° final gate")
    a[0].set(title="Relative rotation: fine × coarse inverse",ylabel="Angle / rotation-vector component (deg)")
    for i,(trace,d) in enumerate(zip(traces,desc)):
        tt=trace["tick"]*rows[i]["dt_s"]; label=f"dt={rows[i]['dt_s']*1000:g} ms"
        a[1].plot(tt,d["heading_deg"],color=colors[i],label=label)
        a[2].plot(tt,d["tilt_deg"],color=colors[i],label=label)
        a[3].plot(tt,(trace["com_world_m"][:,0]-trace["com_world_m"][0,0])*1000,color=colors[i],label=label)
        a[4].step(tt,trace["contact_count"],where="post",color=colors[i],alpha=.8,label=label)
        a[5].plot(tt,trace["penetration_m"]*1000,color=colors[i],label=label)
    a[1].set(title="Projected body-long-axis heading",ylabel="Heading relative to initial (deg)")
    a[2].set(title="Thin-axis departure from world +Z",ylabel="Tilt (deg)")
    a[3].set(title="World COM travel",ylabel="Δ COM X (mm)")
    a[4].set(title="All generated phone–plane contacts",ylabel="Candidate count",yticks=range(5))
    a[5].set(title="Observed-candidate penetration",ylabel="Depth (mm)")
    for x in a:
        x.axvspan(1,1.2,color="#999999",alpha=.18)
        x.axvline(1,color="grey",lw=.6);x.axvline(1.2,color="grey",lw=.6)
        x.set(xlim=(0,3),xlabel="Integer-tick time (s)")
        x.grid(alpha=.2);x.legend(fontsize=7,loc="best")
    fig.suptitle("Saved v2 push pair · τ=2.5 ms, m=0.2 kg, μ=0.6 · all samples retained\nPost-hoc geometry diagnosis; no new physics, force inference or causal attribution",fontsize=12)
    png,svg=BASE/"push_orientation_diagnostic_v1.png",BASE/"push_orientation_diagnostic_v1.svg"
    fig.savefig(png,dpi=180);fig.savefig(svg);plt.close(fig)
    peak=int(np.argmax(angle))
    report={"schema":"phone_convex_saved_push_orientation_diagnostic:1","status":"COMPLETED_POSTHOC_SAVED_ONLY_NOT_NEW_PHYSICS",
        "source":pin(__file__),"selection":pin(BASE/"cpu_screen_assessment_v2/selection.json"),"screening_result":selection["screening_result_pin"],
        "input_trajectories":[r["trajectory"] for r in rows],"identity":{"test":"force_push_release","mass_kg":.2,"sliding_friction":.6,"timeconst_s":.0025,"dt_s":[.00125,.000625]},
        "all_samples":{"coarse":len(coarse["tick"]),"fine":len(fine["tick"]),"shared":len(ticks)},
        "frame_contract":{"quaternion":"WXYZ normalized only for descriptive orientation; raw input files unchanged",
            "relative":"R_fine@R_coarse.T; axis and rotation-vector components in world coordinates",
            "tilt":"angle between actual body thin local+Y and world+Z; no Euler decomposition",
            "heading":"continuous atan2 of projected body long local+Z in worldXY, relative to its initial heading",
            "contacts":"All generated source-native candidates at CPU explicitly sampled POST state, not selected by force; phase differs from solver-PRE impulses"},
        "summary":{"final_relative_angle_deg":float(angle[-1]),"final_relative_world_axis":axis[-1].tolist(),
            "final_abs_axis_world_Z_fraction":float(abs(axis[-1,2])),"final_relative_world_rotvec_deg":rotvec[-1].tolist(),
            "final_projected_heading_difference_deg":float(heading_delta[-1]),"final_thin_axis_normal_difference_deg":float(normal_angle[-1]),
            "coarse_max_thin_axis_tilt_deg":float(dc["tilt_deg"].max()),"fine_max_thin_axis_tilt_deg":float(df["tilt_deg"].max()),
            "peak_relative_angle_deg":float(angle[peak]),"peak_time_s":float(t[peak]),
            "first_exact_nonzero_orientation_difference_time_s":float(t[nonzero[0]]) if len(nonzero) else None,
            "first_above_existing_2deg_reference_time_s":float(t[above[0]]) if len(above) else None,
            "shared_ticks_with_different_generated_contact_counts":int(contact_diff.sum()),
            "first_contact_count_difference_time_s":count_difference_events[0]["time_s"] if count_difference_events else None},
        "milestones":milestones,"contact_history":{rows[i]["case"]:{"count_histogram":d["contact_count_histogram"],"all_count_change_events":d["contact_count_changes"]} for i,d in enumerate(desc)},
        "all_shared_contact_count_difference_events":count_difference_events,
        "outputs":{"full_curves":pin(curves_path),"figure_PNG":pin(png),"figure_SVG":pin(svg)},
        "limitations":["Saved after seeing the v2 failure, not a blind test or new qualification.",
            "Mostly world-Z relative rotation can be called yaw-like; this alone does not identify a solver/contact cause.",
            "Candidate-count switching and orientation changes may co-occur but neither demonstrates causality or full force-waveform parity.",
            "Exact nonzero onset is floating-arithmetic sensitive; the existing2deg final gate is only a descriptive time reference here, not a new all-time acceptance gate.",
            "No new FK, collision query, simulator integration, model, training or GPU. Original v2 selection remains null and failure is not repaired."]}
    with output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({"report":pin(output),"summary":report["summary"],"figures":[str(png),str(svg)]}))


if __name__=="__main__":main()
