"""Draw the actual saved Wuji/box states from both solvers in the original moving head camera."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import yaml

from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.strict_json import write_json


W, H = 512, 288
BG, INK, GOLD = "#12222d", "#eef4f6", "#ffd371"


def read(path):
    return json.loads(path.read_bytes())


def camera_pixels(points, k):
    xy = points[:, :2] / np.maximum(points[:, 2:], 1e-5)
    return (xy * [k["fx"], k["fy"]] + [k["ppx"], k["ppy"]]) * [W/k["width"], H/k["height"]]


def make_geometry(model):
    """Build lightweight convex visual surfaces in actual compiled geom frames."""
    geom = []
    carton = model.body("carton").id
    for i in range(model.ngeom):
        box = model.geom_bodyid[i] == carton
        if (not box and model.geom_group[i] != 1) or model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[i]
        addr = model.mesh_vertadr[mesh]
        verts = np.array(model.mesh_vert[addr:addr+model.mesh_vertnum[mesh]], dtype=np.float64)
        hull = ConvexHull(verts)
        keep, inverse = np.unique(hull.simplices, return_inverse=True)
        faces = inverse.reshape(hull.simplices.shape)
        # Color is a plotting aid. All geometry dimensions and stored poses come from the recordings.
        base = np.array([34, 144, 214] if box else [182, 195, 196])
        geom.append((i, verts[keep], faces, hull.equations[:, :3], base))
    return geom


def draw_physical(data, geometry, camera, k, source_outline):
    frame = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(frame)
    # Use a common world-height grid as a visual reference, separately from the simulated support.
    for c in np.arange(-.4,.81,.1):
        for a,b in [([c,-.4,0],[c,.8,0]),([-.4,c,0],[.8,c,0])]:
            v = (np.array([a,b]) - camera[:3,3]) @ camera[:3,:3]
            if v[:,2].min() > .04:
                draw.line([tuple(p) for p in camera_pixels(v,k)],fill="#263945",width=1)
    triangles, depths, colors = [], [], []
    view_light = np.array([-.42, -.58, -.70])
    for geom, verts, faces, normals, base in geometry:
        rot = data.geom_xmat[geom].reshape(3,3).T @ camera[:3,:3]
        position = (data.geom_xpos[geom] - camera[:3,3]) @ camera[:3,:3]
        cv = verts @ rot + position
        n = normals @ rot
        surface = cv[faces]
        center = surface.mean(axis=1)
        visible = (np.sum(n * center,axis=1)<0) & (surface[:,:,2].min(axis=1)>.015)
        if not visible.any():
            continue
        xy = camera_pixels(cv,k)[faces[visible]]
        box_min,box_max = xy.min(axis=1),xy.max(axis=1)
        inside = (box_min[:,0]<W) & (box_min[:,1]<H) & (box_max[:,0]>=0) & (box_max[:,1]>=0)
        if not inside.any():
            continue
        light = np.maximum(0,n[visible][inside]@view_light)
        triangles.append(np.rint(xy[inside]).astype(np.int32))
        depths.append(center[visible][inside,2])
        colors.append(np.minimum(255, base[None,:]*(.55+.46*light[:,None])).astype(np.uint8))
    if triangles:
        xy,z,rgb=np.concatenate(triangles),np.concatenate(depths),np.concatenate(colors)
        for j in np.argsort(z)[::-1]:
            draw.polygon(xy[j].ravel().tolist(),fill=tuple(rgb[j]))
    draw.line(source_outline,fill=GOLD,width=2)
    return frame


def transform_body(v):
    tr = np.eye(4)
    tr[:3,:3] = Rotation.from_quat(v[3:]).as_matrix()
    tr[:3,3] = v[:3]
    return tr


def nearest_body(initial, truth):
    errors = np.linalg.norm(initial[:,:3]-truth[:3,3],axis=1)
    good = [int(i) for i in np.flatnonzero(errors < 1e-5)
            if np.allclose(transform_body(initial[i])[:3,:3],truth[:3,:3],atol=1e-5)]
    if len(good)!=1:
        raise ValueError("No unique carton body at the stored source reset")
    return good[0]


def compare(args):
    report, normal = read(args.evaluation/"report.json"), read(args.native/"report.json")
    binding, source = read(args.mapping), read(args.source/"report.json")
    if report["sequence"] != source["sequence"].split("/")[-1] and report["sequence"] != args.sequence:
        raise ValueError("Select a physical evaluation for this source sequence")
    if not read(args.native/"phase_evaluation.json")["passed_numeric_development_gates"]:
        raise ValueError("Expected an independently qualified native replay")
    if normal["best"]["target_slew_cap_rad_s"]!=30:
        raise ValueError("Unexpected offline target slew condition")
    with np.load(args.evaluation/"trace.npz",allow_pickle=False) as record:
        frames, bq, jq = record["source_frames"], record["body_q"], record["joint_q"]
        times = record["frame_time"]
        truth = record["raw_object_pose"]; human_tips = record["raw_human_tips"]
    with np.load(verify_file_pin(source["source_poses"]),allow_pickle=False) as ref:
        camera = ref["world_from_egocamera"]; np.testing.assert_allclose(truth,ref["world_from_carton"])
    with np.load(args.native/"trace.npz",allow_pickle=False) as nr:
        native_frames=nr["frame_index"]; nq=nr["sim_qpos"]; native=nr["world_from_carton_sim"]
        native_tips=nr["sim_tips"]; np.testing.assert_allclose(nr["world_from_carton_source"],truth)
    np.testing.assert_array_equal(frames,np.arange(len(bq)))
    np.testing.assert_array_equal(native_frames,np.arange(len(truth)))
    np.testing.assert_allclose(times,frames/30,atol=1e-7)
    assert len(frames)==report["physical_intervals"]+1 and len(truth)==report["requested_intervals"]+1
    inputs=read(verify_file_pin(source["input_frames"]))
    k=yaml.safe_load(verify_file_pin(next(p for p in source["source_inputs"] if "intrinsics/hololens" in p["path"].replace("\\","/"))).read_text())["color"]
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(normal["mjb"])))
    geom=make_geometry(model); d=mujoco.MjData(model)
    palm=binding["body_labels"].index("right_palm_link"); object_id=nearest_body(bq[0],truth[0])
    native_names=[(i,mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n)) for i,n in enumerate(binding["body_labels"])]
    native_names=[(i,j) for i,j in native_names if j>=0]
    fingers=[(int(model.joint(name).qposadr[0]), idx) for name,idx in binding["finger_q_indices"].items()]
    box_q=int(model.jnt_qposadr[model.body("carton").jntadr[0]])
    wrist_q=int(model.joint("wrist_free").qposadr[0])
    site_ids=[model.site(f"right_finger{i}_tip").id for i in range(1,6)]
    object_mesh=np.loadtxt(verify_file_pin(source["geometry"]["hull"]),comments=["f","#"],
                           usecols=(1,2,3))
    pred=np.array([transform_body(v) for v in bq[:,object_id]])
    p_err=np.linalg.norm(pred[:,:3,3]-truth[frames,:3,3],axis=1)*1000
    r_err=np.rad2deg(Rotation.from_matrix(truth[frames,:3,:3].transpose(0,2,1)@pred[:,:3,:3]).magnitude())
    np.testing.assert_allclose(np.sqrt(np.mean(p_err**2)),report["metrics"]["prefix_position_rms_mm"],atol=1e-3)
    np.testing.assert_allclose(np.sqrt(np.mean(r_err**2)),report["metrics"]["prefix_rotation_rms_deg"],atol=1e-3)
    height=[1000*(t[:,2,3]-t[0,2,3]) for t in (truth,pred,native)]
    args.output.mkdir(parents=True,exist_ok=False)
    video=args.output/"source_learned_and_offline.mp4"
    preview=args.output/"source_learned_and_offline.jpg"
    picks=sorted(set([0,min(34,frames[-1]),min(40,frames[-1]),frames[-1]]))
    cell_w,cell_h=290,163; left=100
    sheet=Image.new("RGB",(left+len(picks)*cell_w,3*cell_h+103),BG)
    label=ImageDraw.Draw(sheet)
    fs=ImageFont.truetype("msyh.ttc",17); big=ImageFont.truetype("msyh.ttc",22)
    for row,name in enumerate(("源视频","学习策略","离线修正")):
        label.text((9,74+cell_h*row),name,font=fs,fill=INK)
    tip_diff=[]; max_fk=0.; max_angle=0.
    with imageio.get_writer(video,fps=15,codec="libx264",quality=8,pixelformat="yuv420p",
                            macro_block_size=1,ffmpeg_params=["-threads","1","-movflags","+faststart"]) as stream:
        for i in frames:
            cv_geom=(object_mesh@truth[i,:3,:3].T+truth[i,:3,3]-camera[i,:3,3])@camera[i,:3,:3]
            poly=camera_pixels(cv_geom,k); h=ConvexHull(poly).vertices; outline=[tuple(x) for x in poly[np.r_[h,h[0]]]]
            # Rebuild robot visual frames only. Check every shared body against the saved Newton endpoints.
            d.qpos[:]=nq[i]
            for index,body_id in ((wrist_q,palm),(box_q,object_id)):
                v=bq[i,body_id]; d.qpos[index:index+7]=np.r_[v[:3],v[[6,3,4,5]]]
            for idx,j in fingers: d.qpos[idx]=jq[i,j]
            mujoco.mj_kinematics(model,d)
            for j,n in native_names:
                max_fk=max(max_fk,float(np.linalg.norm(d.xpos[n]-bq[i,j,:3])))
                rot=Rotation.from_matrix(d.xmat[n].reshape(3,3)).inv()*Rotation.from_quat(bq[i,j,3:])
                max_angle=max(max_angle,float(rot.magnitude()))
            tip_diff.extend(np.linalg.norm(d.site_xpos[site_ids]-human_tips[i],axis=1))
            learned=draw_physical(d,geom,camera[i],k,outline)
            d.qpos[:]=nq[i]; mujoco.mj_kinematics(model,d)
            np.testing.assert_allclose(d.site_xpos[site_ids],native_tips[i],atol=1e-7)
            replay=draw_physical(d,geom,camera[i],k,outline)
            if inputs[i]["frame"]!=i: raise ValueError("Source image ordering mismatch")
            with Image.open(verify_file_pin(inputs[i]["ego_rgb"])) as pic:
                rgb=pic.convert("RGB").resize((W,H),Image.Resampling.LANCZOS)
            panels=(rgb,learned,replay); screen=Image.new("RGB",(3*W,H+140),BG); pen=ImageDraw.Draw(screen)
            for col,(title,view,z) in enumerate(zip(
                    ("头戴相机原片","FlashSAC 策略 · Newton / MJWarp","离线修正指令 · 原生 MuJoCo"),panels,height,strict=True)):
                x=W*col;pen.text((x+11,8),title,font=big,fill=INK);screen.paste(view,(x,44))
                msg=f"纸盒距初始高度 {z[i]:+.0f} mm"
                if col: msg+=f"     位置误差 {np.linalg.norm((pred if col==1 else native)[i,:3,3]-truth[i,:3,3])*1000:.0f} mm"
                pen.text((x+11,H+49),msg,font=fs,fill=INK)
            status="物体跟踪触发终止" if i==frames[-1] and report["terminal"]["failed"] else "已记录"
            pen.text((11,H+91),f"源帧 {i:03d} / {len(truth)-1}     {times[i]:.2f} s     0.5 倍速",font=fs,fill=INK)
            pen.text((W+11,H+91),f"{status}  ·  学习策略记录截至第 {frames[-1]} 帧",font=fs,fill=GOLD)
            pen.text((2*W+11,H+91),"金线为源纸盒位置。关节目标限速 30 rad/s。",font=fs,fill=GOLD)
            stream.append_data(np.asarray(screen))
            if i in picks:
                x=left+cell_w*picks.index(i)
                label.text((x+8,6),f"第 {i} 帧    {times[i]:.2f} s",font=fs,fill=INK)
                for row,pic in enumerate(panels):
                    sheet.paste(pic.resize((cell_w,cell_h),Image.Resampling.LANCZOS),(x,39+row*cell_h))
                label.text((x+6,3*cell_h+45),f"高度  {height[0][i]:+.0f} / {height[1][i]:+.0f} / {height[2][i]:+.0f} mm",font=fs,fill=INK)
    np.testing.assert_allclose(1000*np.sqrt(np.mean(np.square(tip_diff))),report["metrics"]["prefix_fingertip_rms_mm"],atol=1e-3)
    if max_fk>2e-6 or max_angle>1e-5:
        raise ValueError(f"Newton-to-visual FK mismatch: {max_fk} m and {max_angle} rad")
    label.text((left+6,3*cell_h+73),f"高度按三行顺序，相对各自起点。金线为源纸盒轮廓；学习策略记录截至第 {frames[-1]} 帧。",font=fs,fill=GOLD)
    sheet.save(preview,quality=94)
    with imageio.get_reader(video) as reader:
        assert reader.count_frames()==len(frames) and reader.get_meta_data()["fps"]==15.
    write_json(args.output/"report.json",dict(
        evaluation=file_pin(args.evaluation/"report.json"), trace=file_pin(args.evaluation/"trace.npz"),
        qualified_native=file_pin(args.native/"phase_evaluation.json"), native_trace=file_pin(args.native/"trace.npz"),
        canonical_visual_model=normal["mjb"],source_report=file_pin(args.source/"report.json"),
        binding=file_pin(args.mapping),renderer=file_pin(__file__),frames=frames.tolist(),preview_frames=picks,
        source_frame_count=len(truth),termination=report["terminal"],carton_body_index=object_id,
        verified_robot_bodies=len(native_names),max_visual_fk_m=max_fk,max_visual_fk_rotation_rad=max_angle,
        learned_position_error_mm=p_err.tolist(),learned_rotation_error_deg=r_err.tolist(),
        box_lift_mm_source_learned_native=np.stack([v[frames] for v in height],axis=1).tolist(),
        display="15 fps from 30 Hz original; CPU-only convex visual surfaces using saved states; no dynamics integration. Only overlapping recorded frames are displayed.",
        video=file_pin(video),preview=file_pin(preview)))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("evaluation","native","mapping","source","output"):
        p.add_argument("--"+name,required=True,type=lambda s:Path(s).resolve())
    p.add_argument("--sequence",default="subject1_G04_2")
    compare(p.parse_args())
