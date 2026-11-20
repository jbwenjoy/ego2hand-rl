"""Trace the pen reference to its raw capture and inspect a simpler egocentric object."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

import mujoco
import numpy as np
import requests
import torch
import trimesh
from PIL import Image, ImageDraw, ImageOps
from scipy.spatial import ConvexHull

from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.assets.mesh import bake_scene
from supermimic_research.assets.proximity import SurfaceQuery
from supermimic_research.ego.clip_inventory import read_clip
from supermimic_research.ego.hand_frames import read_hand_window, reconstruct, transform_points
from supermimic_research.strict_json import write_json


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = 'https://github.com/wuji-technology/toporetarget/releases/download/v2026.10.23/toporetarget-assets.zip'
REV = '30fe9674782f32e1e5edba98476b6ff4300132c5'
ERASER = f'https://huggingface.co/datasets/bop-benchmark/hot3d/resolve/{REV}/object_models/obj_000029.glb'


def download(url, expected, path):
    response = requests.get(url, timeout=(15, 60))
    response.raise_for_status()
    if hashlib.sha256(response.content).hexdigest() != expected:
        raise ValueError(f'Source checksum mismatch for {path.name}')
    path.write_bytes(response.content)
    return file_pin(path)


def pen_lineage(folder):
    archive = download(CAPTURE, '3caa00e625f56e928b4f931b2e0e96b9dc1b59bdfd1d5626f48045a33820fe77', folder/'pen_capture.zip')
    catalog = file_pin(ROOT/'runs/pen_refit_catalog_v2/catalog/report.json')
    sources = json.loads(verify_file_pin(catalog).read_bytes())['results']
    rows = []
    with zipfile.ZipFile(verify_file_pin(archive)) as z:
        for clip in ('clip_001', 'clip_002', 'clip_003'):
            member = f'toporetarget-assets/pen_spin/rawdata/01_single_axis/rawdata/{clip}.npz'
            raw_bytes = z.read(member)
            with np.load(io.BytesIO(raw_bytes), allow_pickle=False) as raw, np.load(verify_file_pin(sources[clip]['source']), allow_pickle=False) as used:
                for k in ('mediapipe_landmarks_w', 'object_pose_w'):
                    np.testing.assert_array_equal(raw[k], used[k])
                rows.append(dict(clip=clip, member=member, raw_member_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                    raw_shapes={k:list(raw[k].shape) for k in raw.files},
                    processed_shapes={k:list(used[k].shape) for k in used.files},
                    hand_and_object_arrays_exact=True, processed=sources[clip]['source']))
    return dict(release_url=CAPTURE, archive=archive, catalog=catalog, rows=rows)


def eraser_source(folder):
    intake = file_pin(ROOT/'runs/hot3d_adjacent_mugs_v1/report.json')
    tar_pin = next(row['archive'] for row in json.loads(verify_file_pin(intake).read_bytes())['rows'] if row['clip']==1883)
    tar_path = verify_file_pin(tar_pin)
    info_pin = json.loads((ROOT/'runs/mug_mesh_v1/report.json').read_bytes())['models_info']
    info = json.loads(verify_file_pin(info_pin).read_bytes())['29']
    bounds = np.array([[info[f'min_{a}'] for a in 'xyz'], [info[f'min_{a}']+info[f'size_{a}'] for a in 'xyz']])
    asset = download(ERASER, 'f5dbb2634e430b8e7b963b5f82d1b9e6057bcdefd442185ce9ec5c59b76bc312', folder/'obj_000029.glb')
    v,f,bake = bake_scene(trimesh.load_scene(verify_file_pin(asset), process=False), bounds)
    hull = trimesh.Trimesh(vertices=v, faces=f, process=False).convex_hull
    proxy = folder/'eraser_convex.obj'
    hull.export(proxy)
    xml = folder/'eraser_geometry.xml'
    xml.write_text('<mujoco model="HOT3D_BOP29_geometry"><compiler fusestatic="false"/><asset>'
                   '<mesh name="source_hull" file="eraser_convex.obj"/></asset><worldbody>'
                   '<body name="eraser_BODY"><geom name="eraser_collision" type="mesh" mesh="source_hull"/>'
                   '</body></worldbody></mujoco>\n',encoding='utf8')
    model = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(model)
    mujoco.mj_forward(model,d)
    gid = model.geom('eraser_collision').id
    mid = model.geom_dataid[gid]
    addr = model.mesh_vertadr[mid]
    vv = model.mesh_vert[addr:addr+model.mesh_vertnum[mid]]@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid]
    np.testing.assert_allclose(np.stack([vv.min(0),vv.max(0)]),bounds,rtol=0,atol=1e-6)
    planes = ConvexHull(v).equations
    gaps = np.concatenate([-np.max(v[i:i+512]@planes[:,:3].T+planes[:,3],axis=1) for i in range(0,len(v),512)])
    # Annotated surface proximity is a geometric clue only. A missing hand stays unknown.
    uid = '258906041248094'
    all_frames = list(range(31))
    inventory = read_clip(tar_path,[29])
    query = SurfaceQuery(v,f)
    validity = {}
    arrays = {}
    for side in ('right','left'):
        ids = [i for i in all_frames if inventory[i]['hands'][side]['pose'] is not None]
        if not ids:
            validity[side] = dict(frames_annotated=0, near_3mm_frames=None)
            continue
        source = read_hand_window(tar_path,ids,side,29,uid)
        hand = reconstruct(source['profile'],source['angles22'],source['wrist'],side)
        counts=[]
        for pose,points in zip(source['objects'],hand['vertices'],strict=True):
            local = transform_points(np.linalg.inv(pose),points)
            near = query.center_tree.query(local)[0] <= query.maximum_radius+.003+1e-10
            dist = query.nearest(local[near])[1] if near.any() else np.empty(0)
            counts.append(int((dist<=.003).sum()))
        validity[side]=dict(frames_annotated=len(ids), near_3mm_frames=int(np.count_nonzero(counts)),
                            near_3mm_frame_ids=[i for i,n in zip(ids,counts,strict=True) if n])
        arrays[f'{side}_frames']=ids
        arrays[f'{side}_wrist']=source['wrist']
        arrays[f'{side}_hand_landmarks21']=hand['landmarks21']
        arrays[f'{side}_surface_point_counts_3mm']=counts
        if side=='right':
            assert ids==all_frames
            arrays.update(object_pose=source['objects'], timestamps_ns=source['timestamps_ns'])
    archive = folder/'source_eraser_001883.npz'
    np.savez_compressed(archive,**arrays)
    points = arrays['object_pose'][:,:3,3]
    wrist = arrays['right_wrist'][:,:3,3]
    rgb = folder/'source_eraser_rgb.png'
    image = Image.new('RGB',(825,265),'#14202a')
    with tarfile.open(tar_path) as t:
        for col,frame in enumerate((0,15,30)):
            raw = t.extractfile(f'{frame:06d}.image_214-1.jpg').read()
            view = Image.open(io.BytesIO(raw)).convert('RGB').transpose(Image.Transpose.ROTATE_270)
            view = ImageOps.contain(view,(275,230))
            image.paste(view,(col*275+(275-view.width)//2,25))
            ImageDraw.Draw(image).text((col*275+9,6),f'HOT3D 001883  |  frame {frame}',fill='white')
    image.save(rgb)
    return dict(clip=1883,frame_range=[0,30],tar=tar_pin,intake=intake,asset_url=ERASER,asset=asset,metadata=info_pin,bake=bake,
        name=info['name'],object_id=29,uid=uid,size_m=np.ptp(v,axis=0).tolist(),proxy=file_pin(proxy), geometry_only_mjcf=file_pin(xml),
        vertices=len(v), faces=len(f),hull_vertices=len(hull.vertices),hull_faces=len(hull.faces),
        hull_inset_at_source_vertices_mm_p50_p95_max=(1000*np.quantile(gaps,[.5,.95,1])).tolist(),
        start_to_max_object_displacement_mm=1000*np.linalg.norm(points-points[0],axis=1).max(),
        start_to_max_wrist_displacement_mm=1000*np.linalg.norm(wrist-wrist[0],axis=1).max(),
        duration_s=(arrays['timestamps_ns'][-1]-arrays['timestamps_ns'][0])*1e-9,side_checks=validity,
        source_poses=file_pin(archive),preview=file_pin(rgb),geometry_only=True,dynamics_calibrated=False)


def main(out):
    torch.set_num_threads(2)
    out.mkdir(exist_ok=False,parents=True)
    pen=pen_lineage(out)
    write_json(out/'pen_lineage.json',pen)
    eraser=eraser_source(out)
    write_json(out/'eraser_intake.json',eraser)
    write_json(out/'report.json',dict(tool=file_pin(__file__),pen=file_pin(out/'pen_lineage.json'),eraser=file_pin(out/'eraser_intake.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    main(parser.parse_args().out.resolve())
