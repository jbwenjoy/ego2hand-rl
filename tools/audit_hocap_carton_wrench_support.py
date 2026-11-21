"""Prepare a geometry-only CHORD support target and compare native contact sets."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
import trimesh

from chord_study.contacts import sample_basis, support_reward, wrench_support
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.assets.proximity import SurfaceQuery
from supermimic_research.strict_json import write_json


TIPS = [4, 8, 12, 16, 20]
BASE_URL = "https://github.com/nvidia-isaac/video_to_data/blob/v0.4.0/robotic_grounding/flash_chord/src/flash_chord/"


def support(points, normals, mask, basis, radius):
    tensors = [torch.as_tensor(x, dtype=torch.float32) for x in (points, normals, mask)]
    with torch.inference_mode():
        return wrench_support(*tensors, basis, radius, mu=.1, edges=8).numpy()


def native_contact_records(model, trace, center):
    # Re-integrate every stored action to recover the same solver contacts and normal
    # loads. Native geometric normals, not measured force magnitudes, form each cone.
    poses = trace["sim_qpos"]
    qvel = trace["sim_qvel"]
    frame_ends = trace["frame_end_substep"]
    controls = trace["substep_finger_ctrl"]
    wrist = trace["substep_wrist_generalized_wrench"]
    data = mujoco.MjData(model)
    data.qpos[:] = poses[0]; data.qvel[:] = qvel[0]; data.ctrl[:] = trace["finger_ctrl"][0]
    mujoco.mj_forward(model, data)
    carton_geom = model.geom("carton_collision").id
    table = model.geom("table").id
    carton_body = model.body("carton").id
    force = np.zeros(6)
    rows = []
    max_err = 0.
    for t, end in enumerate(frame_ends):
        if t:
            for s in range(frame_ends[t-1], end):
                data.ctrl[:] = controls[s]
                data.qfrc_applied[:6] = wrist[s]
                mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        max_err = max(max_err, float(np.max(abs(data.qpos-poses[t]))),
                      float(np.max(abs(data.qvel-qvel[t]))))
        rotation = data.xmat[carton_body].reshape(3,3)
        row = []
        for k, c in enumerate(data.contact):
            geoms = (int(c.geom1), int(c.geom2))
            if carton_geom not in geoms or table in geoms or c.efc_address < 0:
                continue
            mujoco.mj_contactForce(model, data, k, force)
            if force[0] <= 1e-5:
                continue
            # MuJoCo normal points geom1 -> geom2. Use the force ON the carton.
            other = geoms[0] if geoms[1] == carton_geom else geoms[1]
            sign = 1. if geoms[1] == carton_geom else -1.
            normal = sign*c.frame[:3]@rotation
            point = (c.pos-data.xpos[carton_body])@rotation-center
            name = model.geom(other).name
            digit = next((i for i in range(1,6) if f"right_finger{i}_" in name), 0)
            row.append((point, normal, name.endswith("_tip_sensor_col"), digit))
        rows.append(row)
    if max_err > 1e-9:
        raise ValueError(f"Independent native replay diverged by {max_err}")
    width = max(1, max(map(len, rows)))
    points = np.zeros((len(rows), width, 3))
    normals = np.zeros_like(points)
    active = np.zeros((len(rows), width), dtype=bool)
    pad = np.zeros_like(active)
    digit_ids = np.full(active.shape, -1, dtype=int)
    for t, row in enumerate(rows):
        for k, (p, n, is_pad, digit) in enumerate(row):
            points[t,k], normals[t,k], pad[t,k], digit_ids[t,k] = p, n, is_pad, digit
            active[t,k] = True
    return points, normals, active, pad, digit_ids, max_err


def disagreement(pred, target, sim_digit, ref_digit, phases):
    commanded = target > .001
    observed = pred > .001
    ref_active = commanded.any(axis=1)
    cur_active = observed.any(axis=1)
    with torch.inference_mode():
        frame_score = support_reward(torch.from_numpy(pred[:,None,None,:]),
                                     torch.from_numpy(target[:,None,None,:])).numpy()
    missed_basis = (commanded & ~observed).sum(axis=1) / np.maximum(1, commanded.sum(axis=1))
    stats = {}
    for name, indices in phases.items():
        mask = indices & ref_active
        neither = indices & ~ref_active
        stats[name] = dict(
            frames=int(indices.sum()), reference_candidate_frames=int(mask.sum()),
            sim_contact_frames=int((indices & cur_active).sum()),
            missed_contact_frames=int((mask & ~cur_active).sum()),
            unexpected_contact_frames=int((neither & cur_active).sum()),
            candidate_only_mean_cws=float(frame_score[mask].mean()) if mask.any() else None,
            all_frame_mean_cws=float(frame_score[indices].mean()),
            candidate_only_missed_basis_fraction=float(missed_basis[mask].mean()) if mask.any() else None,
            missing_reference_digit_slots=int((indices[:,None] & ref_digit & ~sim_digit).sum()),
            additional_simulated_digit_slots=int((indices[:,None] & sim_digit & ~ref_digit).sum()),
            unintended_inactive_object_penalty=float(np.where(~ref_active,cur_active+np.mean(pred**2,axis=1),0.)[indices].mean()),
        )
    return stats, frame_score


def run(args):
    args.out.mkdir(parents=True,exist_ok=False)
    receipt=json.loads(args.source.read_bytes())
    cfg=json.loads(args.protocol.read_bytes())
    first=json.loads(args.run[0].read_bytes())
    m=mujoco.MjModel.from_binary_path(str(verify_file_pin(first["mjb"])))
    center=m.body_ipos[m.body("carton").id].copy()
    mesh=trimesh.load_mesh(verify_file_pin(receipt["geometry"]["source"]),process=False)
    if not (mesh.is_watertight and mesh.is_winding_consistent and mesh.volume>0):
        raise ValueError("Original carton scan must have outward oriented faces")
    radius=float(np.linalg.norm(mesh.vertices-center,axis=1).max())
    with np.load(verify_file_pin(receipt["source_poses"]),allow_pickle=False) as src:
        obj=src["world_from_carton"]; hand=src["world_right_hand21"]
    local=np.einsum("fti,fij->ftj",hand[:,TIPS]-obj[:,None,:3,3],obj[:,:3,:3])
    closest, dist, ids, _ = SurfaceQuery(mesh.vertices,mesh.faces).nearest(local.reshape(-1,3),chunk=24)
    points=closest.reshape(-1,5,3)-center
    inward=-mesh.face_normals[ids].reshape(-1,5,3)
    dist_mm=dist.reshape(-1,5)*1000
    basis=sample_basis(512,0)
    torch.set_num_threads(2)
    thresholds=(5,10,15)
    target={k:support(points,inward,dist_mm<=k,basis,radius) for k in thresholds}
    # Wider proximity masks can only enlarge a support function.
    assert np.all(target[15]>=target[10]-1e-6) and np.all(target[10]>=target[5]-1e-6)
    frames=np.arange(len(hand))
    phases={"all":np.ones(len(hand),dtype=bool)}
    start=0
    for name,end in cfg["phase_end_inclusive"].items():
        phases[name]=(frames>=start)&(frames<=end)
        start=end+1
    assert start==len(hand)
    path=args.out/"source_contact_support.npz"
    np.savez_compressed(path,frame_index=frames,basis=basis.numpy(),finger_order=np.array(TIPS),
                        object_origin_to_nominal_com_m=center,radius_m=radius,
                        projected_contact_comframe_m=points,inward_normals_object=inward,
                        distance_to_scan_mm=dist_mm,nearest_triangle=ids.reshape(-1,5),
                        candidate_10mm=dist_mm<=10,ambiguous_5to15mm=(dist_mm>5)&(dist_mm<=15),
                        support_5mm=target[5],support_10mm=target[10],support_15mm=target[15])
    results=[]
    for runfile in args.run:
        report=json.loads(runfile.read_bytes())
        model=mujoco.MjModel.from_binary_path(str(verify_file_pin(report["mjb"])))
        np.testing.assert_allclose(model.body_ipos[model.body("carton").id],center,atol=1e-7,rtol=0)
        with np.load(verify_file_pin(report["trace"]),allow_pickle=False) as trace:
            np.testing.assert_array_equal(trace["frame_index"],frames)
            np.testing.assert_allclose(trace["world_from_carton_source"],obj,atol=1e-9,rtol=0)
            ps,ns,valid,pad_mask,digits,err=native_contact_records(model,trace,center)
        saved=dict(points_comframe_m=ps,normals_object=ns,valid=valid,pad=pad_mask,digit=digits)
        measures={}
        for label,mask in [("all_hand",valid),("sensor_pads",pad_mask)]:
            pred=support(ps,ns,mask,basis,radius)
            sim_digit=np.stack([((digits==f)&mask).any(axis=1) for f in range(1,6)],axis=1)
            saved[f"support_{label}"]=pred
            for k in thresholds:
                score, per_frame=disagreement(pred,target[k],sim_digit,dist_mm<=k,phases)
                measures[f"{label}_ref{k}mm"]=score
                saved[f"cws_{label}_ref{k}mm"]=per_frame
        out=args.out/f"{runfile.parent.name}.npz"
        np.savez_compressed(out,**saved)
        results.append(dict(run=file_pin(runfile),native=file_pin(out),replay_error=err,phases=measures))
    write_json(args.out/"report.json",dict(source=file_pin(args.source),tool=file_pin(__file__),protocol=file_pin(args.protocol),
               semantics="Five source fingertip landmarks projected to closest original-scan triangle. Inward face normal and unsigned proximity only. Far/empty fingertip set does not exclude other human surface contacts. No human forces or confidence probabilities.",
               native="Positive normal load >1e-5 N, all hand/carton native contacts at nominal source frame endpoints, table excluded. Also report sensor-pad-only sensitivity. Multi-point native manifold retained; not FlashCHORD link-slot reduction.",
               configuration=dict(basis=512,basis_seed=0,edges=8,normal_ray=True,descriptor_mu=.1,nominal_physics_mu=.7,
                                  radius_m=radius,radius_definition="max original-scan vertex distance from nominal uniform-hull COM",
                                  tolerance=.1,variance=.1,active_threshold=.001,force_rays_unit_norm=True),
               links=[BASE_URL+"configs/objective/tracking.yaml",BASE_URL+"objectives/wrench.py",BASE_URL+"objectives/contact.py"],
               thresholds_mm=list(thresholds),target=file_pin(path),results=results))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True);p.add_argument("--protocol",type=Path,required=True)
    p.add_argument("--run",type=Path,nargs="+",required=True);p.add_argument("--out",type=Path,required=True)
    run(p.parse_args())
