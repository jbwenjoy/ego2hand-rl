"""Independent saved-contact object wrench screen of both warmseed candidates.

Canonicalbaseline is fixed independently of these results. No collisionquery,
physics, PD solve, measuredforce creation, sourceq label, or GPU.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import mujoco
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parents[1]
INTENT=Path("D:/workspace/supermimic-observation-contact-intent-study")
LP=ROOT/"scripts/diagnose_rest_wrench_feasibility_cpu_v2.py"
LP_SHA="ba3f0f934ff7cc2ce6d674c92f3a755487bcc0994f04736664c5ceb87f31fdca"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,default=ROOT/"outputs/warmseed_object_wrench_screen_v1")
    a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    assert sha(LP)==LP_SHA
    spec=importlib.util.spec_from_file_location("pinned_warmseed_native_lp",LP)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    cases={}
    for seed in ("baseline","midpoint"):
        directory=INTENT/f"outputs/frame0_h23_seeded_{seed}_v3"
        recipefile=INTENT/f"configs/frame0-h23-seeded-{seed}-v3.json"
        recipe,original=read(recipefile),read(directory/"report.json")
        assert sha(recipefile)==original["recipe_sha256"]
        assert sha(directory/"initialization_candidate.npz")==original["output_npz_sha256"]
        pins={k:sha(v["path"])==v["sha256"] for k,v in recipe["pins"].items()}
        assert all(pins.values())
        model=mujoco.MjModel.from_binary_path(recipe["pins"]["model"]["path"])
        obj=model.body("object/cube").id;og=set(np.flatnonzero(model.geom_bodyid==obj))
        assert model.npair==0 and not model.flg_adhesion and not np.any(model.geom_adhesion)
        assert model.opt.cone==mujoco.mjtCone.mjCONE_PYRAMIDAL
        with np.load(recipe["pins"]["context"]["path"],allow_pickle=False) as ctx:
            mass=float(ctx["body_mass"][0,obj]);frictions=ctx["geom_friction"][0].astype(float)
            assert np.array_equal(ctx["body_mass"],np.broadcast_to(ctx["body_mass"][:1],ctx["body_mass"].shape))
            assert np.array_equal(ctx["geom_friction"],np.broadcast_to(ctx["geom_friction"][:1],ctx["geom_friction"].shape))
        with np.load(directory/"initialization_candidate.npz",allow_pickle=False) as z:
            pose=z["mapped_object_pose_w"][0].copy();q=z["joint_pos_rad"][0].copy()
            selfmask=(z["geometry_pair_kind"]=="hand_self")&z["generated_contact_mask"][0]
            selfdepth=np.maximum(-z["selected_native_contact_distance_m"][0,selfmask],0)
        assert len(selfdepth)==0
        com=pose[:3]+helper.quat_matrix(pose[3:])@model.body_ipos[obj]
        target=np.r_[-mass*model.opt.gravity.astype(np.float32).astype(float),np.zeros(3)]
        records=read(directory/"native_contacts.json");contacts=[];materials=[]
        for i,c in enumerate(records):
            assert c["phase"]=="native_geometry_query_not_solved_force"
            g1,g2=c["geom"];assert (g1 in og)!=(g2 in og)
            p1,p2=model.geom_priority[[g1,g2]]
            if p1==p2:
                condim=int(max(model.geom_condim[g1],model.geom_condim[g2]));f3=np.maximum(frictions[g1],frictions[g2])
            else:
                selected=g1 if p1>p2 else g2;condim=int(model.geom_condim[selected]);f3=frictions[selected]
            f5=np.array([f3[0],f3[0],f3[1],f3[2],f3[2]])
            assert condim==c["effective_condim"] and np.array_equal(f5,c["friction5"])
            frame=(1. if g2 in og else -1.)*np.array(c["frame_rows_geom0_to_geom1_w"])
            assert np.max(np.abs(frame@frame.T-np.eye(3)))<1e-12
            contacts.append(dict(id=f"saved_geometry_{i}",frame=frame,point=np.array(c["position_w"]),friction=f5,condim=condim))
            materials.append({"geom":c["geom"],"names":c["names"],"condim":condim,"friction5":f5.tolist(),"normal_on_object_world":frame[0].tolist(),"point_world_m":c["position_w"]})
        full,W=helper.solve(contacts,com,target)
        force=linprog(np.ones(W.shape[1]),A_eq=W[:3],b_eq=target[:3],bounds=(0,None),method="highs",options={"primal_feasibility_tolerance":1e-9,"dual_feasibility_tolerance":1e-9})
        # A numerical separatingwitness is checked against every exact native
        # edge. Row scaling converts torque to N-equivalent for conditioning;
        # no contactcap or primalphysics assumption is changed.
        length=.01;scale=np.r_[np.ones(3),np.ones(3)/length]
        Ws=scale[:,None]*W;bs=scale*target;requested_margin=1e-7
        dual=linprog(bs,A_ub=-Ws.T,b_ub=-np.ones(W.shape[1])*requested_margin,
                     bounds=[(-1,1)]*6,method="highs",options={"primal_feasibility_tolerance":1e-9,"dual_feasibility_tolerance":1e-9})
        assert dual.success
        ray_dots=dual.x@Ws;target_dot=float(dual.x@bs)
        assert ray_dots.min()>0 and target_dot<0
        assert force.success and not full["feasible_in_native_object_wrench_screen"]
        # Quantify residualmoment for gravitybalanced forces, without claiming
        # that the forceonly allocation or this minimizer will be realized.
        nc=W.shape[1]
        moment=linprog(np.r_[np.zeros(nc),1.],A_ub=np.vstack((np.c_[W[3:]/length,-np.ones(3)],np.c_[-W[3:]/length,-np.ones(3)])),b_ub=np.zeros(6),A_eq=np.c_[W[:3],np.zeros(3)],b_eq=target[:3],bounds=(0,None),method="highs",options={"primal_feasibility_tolerance":1e-9,"dual_feasibility_tolerance":1e-9})
        assert moment.success
        case={"status":"SAVED_GEOMETRY_NATIVE_OBJECT_WRENCH_INFEASIBLE",
              "canonicalbaseline_fixed_before_screen":True,"candidate_selected_by_this_LP":False,
              "input_directory":str(directory),"recipe_sha256":sha(recipefile),"source_pin_checks":pins,
              "input_sha256":{n:sha(directory/n) for n in ("initialization_candidate.npz","native_contacts.json","report.json")},
              "data_semantics":"Savedfloat64 nativecollisiongeometry, not solvedforce. No newquery; FP32contactset is a separatependingcontract.",
              "contact_count":len(contacts),"unique_geom_pairs":len({tuple(sorted(c["geom"])) for c in records}),"self_generated_pair_count":len(selfdepth),
              "materials_and_geometry":materials,"mass_kg":mass,"COM_world_m":com.tolist(),"required_world_wrench":target.tolist(),
              "force_only_LP":{"feasible":True,"status":int(force.status),"candidate_normal_force_sum_N":float(force.x.sum()),"candidate_COM_torque_Nm":(W[3:]@force.x).tolist(),"scope":"One forcebalanced allocation only, not measured or optimizedzero moment"},
              "full_wrench_LP":full,
              "minimum_inf_COM_torque_for_exact_gravity_force_Nm":float(moment.fun*length),
              "separating_certificate":{"scaled_wrench_y":dual.x.tolist(),"torque_length_scale_m":length,"min_y_dot_native_ray":float(ray_dots.min()),"y_dot_required_wrench":target_dot,"requested_positive_ray_margin":requested_margin,"proof":"Every nonnegative nativeedge combination has positiveydot unlesszero; requiredwrench hasnegativeydot. Therefore no nonnegative allocation can equalrequiredwrench.","scope":"Numericallyverified serialized-nativepyramid certificate; no claim for a different elliptic/material/contactgeometry model"},
              "q_FP32_cast_max_abs_rad":float(np.abs(q-q.astype(np.float32)).max()),
              "source_geometry_qualification":original["geometry_candidate_accepted_for_dynamics"],
              "PD_precondition":"This fixedgeometry objectwrench prerequisite fails; no PDpreload feasible declaration follows. Robotstiction/PD cannot fix objectwrench cone without geometry/contactset/state change.",
              "physical_reference_qualified":False,"measured_or_reference_force_labels_created":False}
        cases[seed]=case
        np.savez_compressed(out/f"{seed}_matrices.npz",native_world_wrench_matrix=W,required_world_wrench=target,
                            scaled_wrench_separator=dual.x,ray_dots=ray_dots,
                            force_only_edge_weights=force.x,minimax_torque_edge_weights=moment.x[:nc])
    report={"status":"COMPLETE_CPU_OBJECT_SCREEN_BOTH_WARMSEEDS",
            "source_sha256":sha(__file__),"pinned_LP_source":{"path":str(LP),"sha256":LP_SHA},
            "canonical":"baseline","canonical_not_quality_selected":True,"cases":cases,
            "normal_force_upper_cap_N":None,"ray_discretization":"Exactlyinstallednative2*(condim-1)edges/contact; effectivecondim4 here. Noextraangular discretization.",
            "limits":["Forcebalance is possible but zeroCOMmoment is not in either saved contactwrenchcone", "These are static existence screens of generated geometry, not actualsolverforces or dynamicstability evidence", "Sourceintent/normal/geometry error andphysicalqualification remainseparate; no objective/threshold/controller change or bestcandidate selection", "Any laterq32 collisionrecords must be independentlysaved/audited, not substituted for theseq64negative results", "No wholehand equilibrium inferred; no PD/stiction/Jgravity calculations in thisscript", "Failure of fixedinitial staticbalance does not prove the entire candidate motion infeasible"],
            "GPU_used":False,"new_physics_steps":0,"new_collision_queries":0}
    (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    lines=[report["status"],"Canonicalbaseline fixed, never selectedby thisscreen. Savedq64geometry only."]
    for seed,c in cases.items():lines.append(f"{seed}: {c['contact_count']}contacts/{c['unique_geom_pairs']}pairs/no self; forceLPfeasible, full6DwrenchLPinfeasible; min exactforce residualtorque_inf={1000*c['minimum_inf_COM_torque_for_exact_gravity_force_Nm']:.6f}mNm; dualrequired={c['separating_certificate']['y_dot_required_wrench']:.9f}, minraydot={c['separating_certificate']['min_y_dot_native_ray']:.3g}.")
    lines+=report["limits"]
    (out/"REPORT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({s:{"force_feasible":c["force_only_LP"]["feasible"],"wrench_feasible":c["full_wrench_LP"]["feasible_in_native_object_wrench_screen"],"min_torque_inf_Nm":c["minimum_inf_COM_torque_for_exact_gravity_force_Nm"],"dual_target":c["separating_certificate"]["y_dot_required_wrench"]} for s,c in cases.items()},indent=2))


if __name__=="__main__":main()
