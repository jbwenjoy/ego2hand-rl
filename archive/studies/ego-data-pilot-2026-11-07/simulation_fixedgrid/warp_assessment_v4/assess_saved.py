"""Saved-array G1-G8 assessment; no MuJoCo, Warp, model or physics imports.

The assessor also authored the Warp recorder. This is separate saved arithmetic,
not an independent experiment author. All cases and original gates are retained.
"""
import argparse
import collections
import datetime
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PURE = ROOT/"simulation_stage1/warp_saved_assessment_v1/assess_saved.py"
PURE_SHA = "d50d98aa3b21f55d0d1b2a9d8c63a254ba733ed2d1096c11cab4e193a87f0ddd"
PROTOCOL_SHA = "7a84cdda18952e24e45cfd96ba0bdcb846ff656794512112911e64602b1ed14f"
RESULT_SHA = "2c2a437cde4337b68f00379dd946ee4d3ee7e7b94ac3cfddd6e329e19ce9b8f5"
CPU_REVIEW_SHA = "58cfa0cff76ec5f18799a3edf019c9026848df6a1d2945a99507836967fc48be"
assert hashlib.sha256(PURE.read_bytes()).hexdigest() == PURE_SHA
spec = importlib.util.spec_from_file_location("phone_saved_arithmetic_v1",PURE)
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
TESTS = ("drop","table_settling","force_push_release")
DTS = (.0003125,.00015625)
BITS = {"NEFC":1,"NJMAX_NNZ":2,"BROADPHASE":4,"NARROWPHASE":8,"CCD":16,
        "HFIELD":32,"CONTACT_MATCH":64,"NVMAX":128,"EPA_HORIZON":256}


def same(a,b):
    A.verify(a); A.verify(b)
    return Path(a["path"]).samefile(b["path"]) and a["sha256"] == b["sha256"]


def process_and_archive(w,run):
    terminal_path = run.parent/"warp_v4_process_terminal.json"
    t = A.read(terminal_path)
    assert t["schema"] == "phone_Warp_fixedgrid_parent_terminal:4"
    assert t["exit_code"] == 0 and t["process_was_terminal_before_log_hashing"] is True
    pins = {Path(v["path"]).name:v for v in t["artifact_pins"]}
    assert same(pins["result.json"],A.pin(run/"result.json"))
    logs, hits, records = {}, [], []
    pattern = re.compile(r"\b(warning|error|overflow|nan|infinity|unstable|instability|reset|truncat\w*|unsupported)\b",re.I)
    for name in ("stdout.log","stderr.log"):
        A.verify(pins[name])
        text = Path(pins[name]["path"]).read_text(encoding="utf-8-sig")
        logs[name] = pins[name]
        hits.extend({"file":name,"line":i,"text":s} for i,s in enumerate(text.splitlines(),1) if pattern.search(s))
        if name == "stdout.log":
            for line in text.splitlines():
                if line.startswith("{"):
                    records.append(json.loads(line))
    assert [(v["groups"],v["cases"]) for v in records if v["status"]=="WARP_GROUP_RECORDED"] == [(i,3*i) for i in range(1,19)]
    assert records[-1]["status"]=="RECORDED_NOT_YET_ASSESSED" and same(records[-1]["result"],A.pin(run/"result.json"))
    A.verify(w["python_warnings"])
    warnings = [json.loads(s) for s in Path(w["python_warnings"]["path"]).read_text(encoding="utf-8").splitlines() if s]
    assert len(warnings) == w["python_warning_count"]
    archive = w["source_archive"]
    A.verify(archive["archive"]); A.verify(archive["inventory"])
    inv = A.read(archive["inventory"]["path"])
    members = set()
    with zipfile.ZipFile(archive["archive"]["path"]) as z:
        for item in inv["archived"].values():
            A.verify(item)
            data = z.read(item["archive_member"])
            assert hashlib.sha256(data).hexdigest()==item["sha256"] and len(data)==item["bytes"]
            members.add(item["archive_member"])
        assert set(z.namelist())==members
    assert len(members)==archive["unique_actual_files"]==inv["archive_unique_actual_file_count"]
    assert set(inv["CPU_raw_trajectory_pins_not_copied"])==set(w["CPU_confirmation_join"]["trajectory_pins"])
    for k,item in inv["CPU_raw_trajectory_pins_not_copied"].items():
        assert same(item,w["CPU_confirmation_join"]["trajectory_pins"][k])
    return {"terminal":A.pin(terminal_path),"logs":logs,"supported_diagnostic_text_matches":hits,
            "python_warning_count":len(warnings),"python_warning_records":warnings,
            "all18_group_completion_lines":True,"source_archive_verified_unique_files":len(members),
            "passed":not hits and not warnings,
            "native_CPU_style_warning_array":"NOT_AVAILABLE_BY_PINNED_WARP_API",
            "scope":"Retained OS streams plus supported sticky bits, qacc/finite/iteration/capacity evidence; not an unlimited CUDA printf-buffer or hidden-state certificate."}


def option_readback(g,cr,run):
    o = g["CPU_compiled_model_options"]
    rt = g["preintegration_runtime_contract"]
    assert o==cr["compiled_options"]
    assert o["option"]["tolerance"]==1e-6 and o["option"]["ls_tolerance"]==.01
    for k,v in rt["discrete"].items():
        if k in o["option"]: assert v==o["option"][k]
    for k,item in rt["float32"].items():
        assert item["dtype"]=="float32"
        x = np.asarray(item["value"],dtype=np.float64)
        y = np.asarray(item["CPU_value"],dtype=np.float32).astype(np.float64)
        assert np.array_equal(x.ravel(),y.ravel())
        if k in o["option"]: assert np.array_equal(np.asarray(item["CPU_value"]),np.asarray(o["option"][k]))
    assert all(x["equal_CPU_cast"] is True for x in rt["array_guards"].values())
    assert rt["collision_vertices"]==352 and rt["collision_triangles"]==700 and rt["refsafe_enabled"] is True
    assert rt["Warp_extra_options"]["warn_overflow"] is True
    for k in ("timestep","tolerance","ls_tolerance"):
        assert g["GPU_effective_options"][k+"_F32"]==np.float32(o["option"][k])
    for k in ("integrator","solver","iterations","ls_iterations","cone","disableflags","enableflags"):
        assert g["GPU_effective_options"][k]==o["option"][k]
    group = g["group"]
    path = run/f"warp_group_preintegration_mass{group['mass_kg']:g}_mu{group['mu']:g}_dt{group['dt_s']:g}_tau{group['timeconst_s']:g}.json"
    pre = A.read(path)
    assert pre["runtime_contract"]==rt and pre["CPU_compiled_options"]==o and pre["integration_steps_so_far"]==0
    assert np.asarray(pre["local_collision_vertices_m"]).shape==(352,3)
    assert g["initial_support"]==pre["initial_support"]
    return pre,A.pin(path)


def compare(a,b,dt,gates):
    if a["first_contact_tick"] is None or b["first_contact_tick"] is None:
        return {"metrics":{},"checks":{"first_generated_contact_present_both":False},"passed":False}
    out = A.compare_states(a,b,dt,gates)
    out["final_signed_COM_difference_m"]=(a["com"][-1]-b["com"][-1]).tolist()
    out["final_signed_COM_velocity_difference_m_s"]=(a["vcom"][-1]-b["vcom"][-1]).tolist()
    out["final_signed_world_omega_difference_rad_s"]=(a["omega"][-1]-b["omega"][-1]).tolist()
    return out


def assess(run,output):
    output.mkdir(parents=True,exist_ok=False)
    protocol = ROOT/"protocols/phone_convex_fixedgrid_v4.json"
    assert A.pin(protocol)["sha256"]==PROTOCOL_SHA and A.pin(run/"result.json")["sha256"]==RESULT_SHA
    p,w = A.read(protocol),A.read(run/"result.json")
    for item in p["source_pins"].values(): A.verify(item)
    p2 = A.read(p["source_pins"]["v2_protocol"]["path"])
    limits = p2["engineering_gates"]
    assert same(w["protocol"],A.pin(protocol)) and w["schema"]=="phone_convex_fixedgrid_Warp:4"
    for item in [*w["sources"].values(),*w["asset_pins"].values(),*w["relevant_backend_sources"].values(),w["metadata"]]: A.verify(item)
    A.verify(w["CPU_confirmation_join"]["producer_result"]);A.verify(w["CPU_confirmation_join"]["saved_assessment"])
    assert w["CPU_confirmation_join"]["saved_assessment"]["sha256"]==CPU_REVIEW_SHA
    cpu = A.read(w["CPU_confirmation_join"]["producer_result"]["path"])
    review = A.read(w["CPU_confirmation_join"]["saved_assessment"]["path"])
    assert review["CPU_bench_qualified"] is True and review["assessment_complete"] is True and review["audit_integrity_qualified"] is True
    assert same(review["producer_result_pin"],w["CPU_confirmation_join"]["producer_result"]) and same(review["protocol_pin"],A.pin(protocol))
    assert len(w["cases"])==w["case_count"]==len(cpu["cases"])==54 and len(w["groups"])==18
    assert w["world_steps_returned"]==777600 and w["batch_steps_returned"]==259200 and w["extra_forward_sampling_calls"]==18
    assert w["API_counts"]=={"CPU_model_compilations":18,"CPU_initial_state_forward_calls":54,"GPU_put_model_calls":18,"GPU_put_data_calls":18,"GPU_integrating_batch_steps":259200,"GPU_additional_nonintegrating_forward_calls":18,"warmup_integrating_steps":0}
    process = process_and_archive(w,run)
    groups={(g["group"]["mass_kg"],g["group"]["mu"],g["group"]["dt_s"]):g for g in w["groups"]}
    cm={A.case_key(r):r for r in cpu["cases"]};wm={A.case_key(r,True):r for r in w["cases"]}
    expected=set(itertools.product([.1,.2,.3],[.3,.6,.9],DTS,TESTS))
    assert set(cm)==set(wm)==expected and len(groups)==18
    bykey,rows,raw_counts,group_slots = {},[],collections.Counter(),{}
    for key,row in wm.items():
        mass,mu,dt,test=key;cr=cm[key];g=groups[key[:3]];n=round(3/dt);ticks=np.arange(n+1)
        assert same(cr["trajectory"],review["trajectory_pins"][cr["case"]])
        a,ct,ca=A.load(row["trajectory"]),A.load(row["contact_table"]),A.load(cr["trajectory"])
        raw_counts.update(Warp_trajectory=1,Warp_contact_table=1,CPU_trajectory=1)
        pre,prepin=option_readback(g,cr,run)
        failures=[]
        def require(v,message):
            if not bool(v): failures.append(message)
        require(row["test"]==row["protocol_test_name"]==test,"test identity")
        require(a["qpos"].shape==(n+1,7) and a["qvel"].shape==(n+1,6),"state shape")
        require(a["qpos"].dtype==a["qvel"].dtype==a["time_s"].dtype==np.float32,"native state FP32 dtype")
        require(a["solver_qacc_at_state_tick"].shape==(n+1,6) and a["native_engine_energy_at_solver_state_tick_J"].shape==(n+1,2),"alltick native diagnostics shape")
        require(np.array_equal(a["integer_tick"],ticks) and np.array_equal(ca["tick"],ticks),"both full integer ticks")
        effective=float(np.float32(dt));require(row["dt_effective_s"]==effective,"effective dt")
        require(np.array_equal(a["tick_derived_requested_time_s"],ticks*dt) and np.array_equal(a["tick_derived_effective_time_s"],ticks*effective),"tick-derived requested/effective time")
        clock=np.zeros(n+1,dtype=np.float32)
        for i in range(n):clock[i+1]=np.float32(clock[i]+np.float32(dt))
        clockok=np.array_equal(clock.view(np.uint32),a["time_s"].view(np.uint32));require(clockok,"FP32 clock recurrence")
        require(np.array_equal(a["qpos"][0].view(np.uint32),ca["qpos"][0].astype(np.float32).view(np.uint32)) and np.array_equal(a["qvel"][0].view(np.uint32),ca["qvel"][0].astype(np.float32).view(np.uint32)),"CPU-cast initial state")
        require(g["final_cursor"]==g["batch_steps_returned"]==n and g["world_steps_returned"]==3*n,"counts/cursor")
        require(sum(x["returned_batch_steps"] for x in g["graph_launches"])==n,"all three force phases")
        for xp in (row["compiled_model_XML"],cr["model_xml"]):A.verify(xp)
        require(Path(row["compiled_model_XML"]["path"]).read_text()==Path(cr["model_xml"]["path"]).read_text(),"same actual model XML")
        body=ET.parse(row["compiled_model_XML"]["path"]).find("./worldbody/body[@name='hot3d_phone24']")
        inertial=body.find("inertial");center=np.fromstring(inertial.get("pos"),sep=" ");inertia=np.fromstring(inertial.get("diaginertia"),sep=" ")
        require(float(inertial.get("mass"))==mass,"actual inertial mass")
        com,vc,omega,energy=A.state_geometry(a["qpos"],a["qvel"],center,inertia,mass)
        ccom,cvc,cw,ce=A.state_geometry(ca["qpos"],ca["qvel"],center,inertia,mass)
        wi=TESTS.index(test);names=g["CPU_compiled_model_options"]["geom_names"]
        allowed={names.index("phone24_collision_convex"),names.index("bench_support")}
        require(np.all(ct["worldid"]==wi),"contact world")
        require(ct["state_tick"].dtype.kind in 'iu' and np.all((ct["state_tick"]>=0)&(ct["state_tick"]<=n)),"contact tick range")
        require(all(set(pair)==allowed for pair in ct["geom"]),"complete generated phone-plane identities")
        require(np.all((ct["original_Warp_contact_slot"]>=0)&(ct["original_Warp_contact_slot"]<g["capacity_naconmax"])),"original raw slot range")
        slots=set(zip(ct["state_tick"].tolist(),ct["original_Warp_contact_slot"].tolist()))
        require(len(slots)==len(ct["state_tick"]),"duplicate raw contact slot")
        prior=group_slots.setdefault(key[:3],set());require(not (prior & slots),"raw slot duplicate across worlds");prior.update(slots)
        count=np.bincount(ct["state_tick"],minlength=n+1);empty=count==0;mins=np.full(n+1,np.inf)
        np.minimum.at(mins,ct["state_tick"],ct["dist"]);depth=np.where(empty,0.,np.maximum(0.,-mins))
        require(np.array_equal(count,a["contact_count"]) and np.array_equal(empty,a["empty_generated_contact_set"]),"complete contact counts/empty")
        require(np.isnan(a["minimum_generated_contact_distance_m"][empty]).all() and np.array_equal(mins[~empty],a["minimum_generated_contact_distance_m"][~empty]),"empty NaN/nonempty minima")
        require(np.array_equal(depth,a["penetration_m"]),"penetration from all generated contacts")
        finite=all(np.isfinite(a[k]).all() for k in ("qpos","qvel","time_s","com_world_m","com_velocity_world_m_s","omega_world_rad_s","energy_potential_kinetic_J","applied_com_force_world_x_N","solver_qacc_at_state_tick","native_engine_energy_at_solver_state_tick_J"))
        finite &= all(np.isfinite(ct[k]).all() for k in ("dist","pos","frame"))
        qnorm=A.norm(a["qpos"][:,3:]);flags=a["overflow_flags"]
        capacity=not np.any(flags) and not np.any(g["final_overflow_flags"]) and np.all((a["solver_nefc_at_state_tick"]>=0)&(a["solver_nefc_at_state_tick"]<=g["capacity_njmax_per_world"])) and np.all((a["pre_solver_nefc_after_step"]>=0)&(a["pre_solver_nefc_after_step"]<=g["capacity_njmax_per_world"])) and np.all(count<=64)
        capacity &= np.all((a["global_ncollision_at_state_tick"]>=0)&(a["global_ncollision_at_state_tick"]<=g["capacity_naconmax"])) and all(0<=x<=256 for x in g["final_nefc"])
        diagnostics=np.all(a["supported_finite_state_qacc_flags"]) and np.all((a["solver_niter_at_state_tick"]>=0)&(a["solver_niter_at_state_tick"]<=100)) and np.all((flags[1:]&flags[:-1])==flags[:-1])
        require(np.array_equal(a["pre_solver_nefc_after_step"][1:],a["solver_nefc_at_state_tick"][:-1]),"PRE nefc vs returned-step diagnostic phase")
        numeric=bool(finite and capacity and diagnostics and np.all(qnorm>0) and np.all(energy[:,1]>=0) and np.all(a["native_engine_energy_at_solver_state_tick_J"][:,1]>=0))
        force=np.zeros(n+1,dtype=np.float32)
        if test=="force_push_release":force[1:]=np.where((np.arange(n)>=round(1/dt))&(np.arange(n)<round(1.2/dt)),2.,0.)
        schedule=np.array_equal(force.view(np.uint32),a["applied_com_force_world_x_N"].view(np.uint32))
        impulse=math.fsum(float(v) for v in a["applied_com_force_world_x_N"][1:])*dt
        expected_impulse=.4 if test=="force_push_release" else 0.
        rest=ticks*dt>=2.5-dt/4;vs,ws=A.norm(vc),A.norm(omega);e=energy.sum(1)
        windows=[]
        for start,stop in ([(0.,1.),(1.2,3.)] if test=="force_push_release" else [(0.,3.)]):
            i,j=round(start/dt),round(stop/dt);gain=float(np.max(e[i:j+1]-e[i]));bound=max(.0005,.10*max(abs(float(e[i])),.005))
            windows.append({"start_tick":i,"end_tick_inclusive":j,"start_energy_J":float(e[i]),"max_positive_gain_J":gain,"limit_J":bound,"passed":gain<=bound})
        first=int(np.flatnonzero(count>0)[0]) if np.any(count>0) else None
        cfirst=int(np.flatnonzero(ca["contact_count"]>0)[0]) if np.any(ca["contact_count"]>0) else None
        metrics={"completed_steps":n,"state_rows":n+1,"empty_generated_rows":int(empty.sum()),
            "max_penetration_m":float(depth.max()),"rest_max_penetration_m":float(depth[rest].max()),
            "rest_COM_speed_RMS_m_s":A.rms(vs[rest]),"rest_COM_speed_max_m_s":float(vs[rest].max()),
            "rest_angular_speed_RMS_rad_s":A.rms(ws[rest]),"rest_angular_speed_max_rad_s":float(ws[rest].max()),
            "first_generated_contact_tick":first,"first_generated_contact_tick_time_s":first*dt if first is not None else None,
            "requested_dt_s":dt,"effective_dt_F32_s":effective,"final_accumulated_time_F32_s":float(a["time_s"][-1]),
            "max_clock_drift_vs_N_effective_dt_s":float(np.max(np.abs(a["time_s"].astype(float)-ticks*effective))),"clock_recurrence_bit_exact":bool(clockok),
            "force_tick_schedule_exact":bool(schedule),"requested_grid_force_impulse_N_s":impulse,
            "effective_grid_force_impulse_N_s":math.fsum(float(v) for v in a["applied_com_force_world_x_N"][1:])*effective,
            "known_external_work_J":math.fsum(float(f*dx) for f,dx in zip(force[1:],np.diff(com[:,0]))),
            "passive_energy_windows":windows,"quaternion_norm_range":[float(qnorm.min()),float(qnorm.max())],
            "native_qacc_maxabs":float(np.max(np.abs(a["solver_qacc_at_state_tick"]))),"native_solver_niter_max":int(a["solver_niter_at_state_tick"].max()),
            "overflow_bit_nonzero_tick_count":{k:int(np.count_nonzero(flags&bit)) for k,bit in BITS.items()},
            "saved_COM_maxabs_difference_m":float(np.max(np.abs(com-a["com_world_m"]))),
            "saved_COM_v_maxabs_difference_m_s":float(np.max(np.abs(vc-a["com_velocity_world_m_s"]))),
            "saved_world_omega_maxabs_difference_rad_s":float(np.max(np.abs(omega-a["omega_world_rad_s"]))),
            "saved_rigid_energy_maxabs_difference_J":float(np.max(np.abs(energy-a["energy_potential_kinetic_J"]))),
            "native_energy_PRE_U_T_min_J":a["native_engine_energy_at_solver_state_tick_J"].min(0).tolist(),
            "native_energy_PRE_U_T_max_J":a["native_engine_energy_at_solver_state_tick_J"].max(0).tolist(),
            "native_energy_PRE_U_T_final_J":a["native_engine_energy_at_solver_state_tick_J"][-1].tolist(),
            "CPU_native_energy_POST_U_T_final_J":ca["native_energy_potential_kinetic_J"][-1].tolist(),
            "rigid_energy_U_T_initial_J":energy[0].tolist(),"rigid_energy_U_T_final_J":energy[-1].tolist()}
        gr=limits["G4_rest_motion"];gp=limits["G3_penetration"]
        gates={"G1_identity_and_coverage":not failures,"G2_finite_and_supported_solver":numeric and process["passed"],
            "G3_penetration":metrics["max_penetration_m"]<=gp["max_observed_candidate_penetration_all_times_m"] and metrics["rest_max_penetration_m"]<=gp["max_observed_candidate_penetration_rest_window_m"],
            "G4_rest_motion":metrics["rest_COM_speed_RMS_m_s"]<=gr["RMS_COM_speed_m_s"] and metrics["rest_COM_speed_max_m_s"]<=gr["max_COM_speed_m_s"] and metrics["rest_angular_speed_RMS_rad_s"]<=gr["RMS_angular_speed_rad_s"] and metrics["rest_angular_speed_max_rad_s"]<=gr["max_angular_speed_rad_s"],
            "G5_energy":all(v["passed"] for v in windows),
            "G6_contacts_and_input":bool(np.any(count[rest]>0) and schedule and abs(impulse-expected_impulse)<=64*np.finfo(float).eps*max(1.,abs(expected_impulse)))}
        bykey[key]={"qpos":a["qpos"],"com":com,"vcom":vc,"omega":omega,"first_contact_tick":first}
        cb={"qpos":ca["qpos"],"com":ccom,"vcom":cvc,"omega":cw,"first_contact_tick":cfirst}
        parity=compare(bykey[key],cb,dt,limits["G8_CPU_Warp_parity"]);gates["G8_CPU_Warp_parity"]=parity["passed"]
        rows.append({"case":row["case"],"test":test,"mass_kg":mass,"sliding_friction":mu,"timeconst_s":.0025,"dt_s":dt,
                     "trajectory":row["trajectory"],"contact_table":row["contact_table"],"CPU_trajectory":cr["trajectory"],
                     "preintegration_snapshot_current_pin":prepin,"structural_failures":failures,"metrics":metrics,
                     "gates":{k:"PASS" if v else "FAIL" for k,v in gates.items()},"G8_parity":parity})
    timestep=[]
    for mass,mu,test in itertools.product([.1,.2,.3],[.3,.6,.9],TESTS):
        coarse=bykey[(mass,mu,DTS[0],test)];fine=bykey[(mass,mu,DTS[1],test)]
        sampled={k:(v[::2] if isinstance(v,np.ndarray) else v/2 if v is not None else None) for k,v in fine.items()}
        result=compare(coarse,sampled,DTS[0],limits["G7_timestep_pair"])
        timestep.append({"mass_kg":mass,"sliding_friction":mu,"test":test,"timesteps_s":list(DTS),**result})
        for row in rows:
            if (row["mass_kg"],row["sliding_friction"],row["test"])==(mass,mu,test):row["gates"]["G7_timestep_pair"]="PASS" if result["passed"] else "FAIL"
    counts={k:dict(collections.Counter(r["gates"][k] for r in rows)) for k in rows[0]["gates"]}
    geometry=A.read(w["asset_pins"]["asset_geometry_manifest"]["path"])
    geom_ok=geometry["maximum_original_vertex_outside_hull_m"]<=.0005 and geometry["full_sampled_hull_to_scan"]["p95_absolute_m"]<=.002 and geometry["full_sampled_hull_to_scan"]["maximum_m"]<=.004
    qualified=geom_ok and review["CPU_bench_qualified"] and all(v=="PASS" for r in rows for v in r["gates"].values())
    worst={k:max(({"case":r["case"],"value":r["G8_parity"]["metrics"][k]} for r in rows),key=lambda v:v["value"]) for k in rows[0]["G8_parity"]["metrics"]}
    report={"schema":"phone_convex_fixedgrid_Warp_saved_assessment:4","status":"COMPLETED_ALL54_SAVED_ASSESSMENT_WITH_ORIGINAL_GATES",
        "created_UTC":datetime.datetime.now(datetime.timezone.utc).isoformat(),"assessor":A.pin(__file__),"saved_math_source":A.pin(PURE),
        "protocol_pin":A.pin(protocol),"producer_result_pin":A.pin(run/"result.json"),"CPU_saved_assessment_pin":w["CPU_confirmation_join"]["saved_assessment"],
        "assessment_complete":True,"audit_integrity_qualified":not any(r["structural_failures"] for r in rows),"Warp_case_count":54,
        "world_steps":777600,"batch_steps":259200,"Warp_final_sampling_forward_calls":18,"CPU_POST_sampling_forward_calls":777654,
        "gate_case_counts":counts,"G7_passed":sum(v["passed"] for v in timestep),"G7_total":27,
        "G8_passed":sum(r["G8_parity"]["passed"] for r in rows),"G8_total":54,"geometry_qualified":bool(geom_ok),
        "Warp_bench_qualified":bool(qualified),"stage4_engineering_qualified":bool(qualified),"process_diagnostics":process,
        "max_penetration":max(({"case":r["case"],"m":r["metrics"]["max_penetration_m"]} for r in rows),key=lambda v:v["m"]),
        "worst_G8_parity":worst,"NPZ_reads":dict(raw_counts),"case_results":rows,"timestep_comparisons":timestep,
        "warning_table_scope":"Native CPU-style GPU warning table is NOT_AVAILABLE; G2 uses every supported predeclared flag/finite/capacity/log diagnostic, not a fabricated native warning zero vector.",
        "phase_scope":"GPU qacc/native energy/contact geometry indexed solver PRE tick i; one final forward supplies tickN. qpos/qvel/rigid energy indexed POST matching ticks. CPU explicit POST forward count differs. No contact-force waveform or whole-hidden-state parity claim.",
        "actual_device_options_scope":"Saved actual GPU scalar readbacks independently compared to CPU values after FP32 conversion; actual array equality guards/source pinned and ZIP checked. Full device model arrays were not independently re-exported/reloaded.",
        "authorship_scope":"Saved-only arithmetic and IO replay without importing producer/backend; assessor also authored recorder and is not an independent experimental author.",
        "real_parameter_calibrated":False,"hand_task_qualified":False,"ego_retargeting_qualified":False,"human_force_truth_qualified":False,
        "new_physics_or_GPU_calls":0}
    A.save_json(output/"all54_case_results.json",rows);A.save_json(output/"all27_timestep_pairs.json",timestep)
    A.write_csv(output/"all54_case_results.csv",rows)
    report["outputs"]={name:A.pin(output/name) for name in ("all54_case_results.json","all27_timestep_pairs.json","all54_case_results.csv")}
    assert not (set(sys.modules)&{"mujoco","mujoco_warp","warp","torch"})
    A.save_json(output/"report.json",report)
    print(json.dumps({"report":A.pin(output/"report.json"),"gate_case_counts":counts,"G7_passed":report["G7_passed"],"G8_passed":report["G8_passed"],"max_penetration":report["max_penetration"],"Warp_bench_qualified":qualified},allow_nan=False))


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--run-dir",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args();assess(args.run_dir.resolve(),args.output.resolve())
