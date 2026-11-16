"""Original RGB/camera/known-mesh alignment, without simulation or fitting."""
import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import time
import tomllib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.ndimage import binary_erosion
from PIL import Image,ImageDraw,ImageFont
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.ego.hand_frames import pose_matrix,transform_points
from supermimic_research.ego.camera_projection import fisheye624,decode_hot3d_mask,rasterize,compare_masks
from supermimic_research.ego.vendor_camera import camera


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def byte_pin(name,data):return dict(member=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def main(a):
    start=time.perf_counter();cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());pilot=Path(cfg['pilot_root']);clip=pilot/cfg['clip']
    if (cfg['schema'],cfg['stream'],cfg['object_bop_id'],cfg['object_uid'],cfg['first_frame'],cfg['last_frame'],cfg['display_frames'])!=(1,'214-1','9','223371871635142',47,107,[47,77,107]):raise ValueError('Fixed original61-frame mug RGB audit required')
    p={k:(cp.parent/cfg[k]).resolve() for k in ['mesh_report','asset_contract','upstream_references','protocol','registration','output']}
    download=read(pilot/'download_manifest_v1.json');expected=next(v for v in download['clips'] if Path(v['path']).resolve()==clip.resolve());verify_file_pin(expected)
    mesh=read(p['mesh_report']);asset=read(p['asset_contract']);validation=read(verify_file_pin(asset['validation']));intent=read(verify_file_pin(validation['intent']));decomp=read(verify_file_pin(intent['decomposition']))
    if validation['passed']!=validation['total'] or asset['collision_parts']!=len(decomp['components']) or len(decomp['components'])!=128 or intent['mesh_report']!=file_pin(p['mesh_report']):raise ValueError('Original validated128-piece asset required')
    refs=read(p['upstream_references']);vendor=ROOT/'src/supermimic_research/ego/vendor_camera';vendor_manifest=read(vendor/'SOURCES.json')
    for pin in refs['records']:verify_file_pin({k:pin[k] for k in ['path','bytes','sha256']})
    for pin in vendor_manifest['records']:
        verify_file_pin(dict(path=str(vendor/pin['name']),bytes=pin['bytes'],sha256=pin['sha256']))
    inputs=dict(clip=file_pin(clip),download_manifest=file_pin(pilot/'download_manifest_v1.json'),mesh_report=file_pin(p['mesh_report']),source_mesh=mesh['raw'],asset_contract=file_pin(p['asset_contract']),
        validation=asset['validation'],intent=validation['intent'],decomposition=intent['decomposition'],components=decomp['raw'],native_asset=asset['asset_model'],upstream=file_pin(p['upstream_references']),vendor=file_pin(vendor/'SOURCES.json'),protocol=file_pin(p['protocol']))
    for pin in inputs.values():verify_file_pin(pin)
    sources=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,p['protocol'],vendor/'LICENSE',vendor/'SOURCES.json']
    def freeze(out):
        receipt=freeze_sources(sources,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt);return receipt
    if a.register:
        out=p['registration'].parent
        if out.exists():raise ValueError('Fresh registration required')
        out.mkdir();freeze(out);write(p['registration'],dict(schema='mug_RGB_projection_registration:1',config=file_pin(cp),inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),future_output=str(p['output']),
            scope='61 source frames,predeclared3 displays,diagnostic metrics only,zero fitting/physics.'))
        print(json.dumps(dict(registration=file_pin(p['registration']))),flush=True);return
    reg=read(p['registration']);verify_source_snapshot(read(verify_file_pin(reg['source_receipt'])),live=True)
    if reg['inputs']!=inputs or reg['config']!=file_pin(cp):raise ValueError('Registered source inputs changed')
    out=p['output']
    if out.exists():raise ValueError('Fresh audit output required')
    out.mkdir();receipt=freeze(out);phase='compiled_geometry'
    try:
        import mujoco,trimesh
        from unittest.mock import patch
        model=mujoco.MjModel.from_binary_path(str(verify_file_pin(asset['asset_model'])));body=model.body('hot3d_mug9').id
        source=np.load(verify_file_pin(mesh['raw']));vertices=source['vertices_model_m'];faces=source['faces'];parts=np.load(verify_file_pin(decomp['raw']));cv=[];cf=[];offset=0;part_errors=[]
        for component in decomp['components']:
            i=component['index'];geom=model.geom(f'mug9_collision_{i:03d}').id;mid=int(model.geom_dataid[geom]);v0=int(model.mesh_vertadr[mid]);n=int(model.mesh_vertnum[mid]);f0=int(model.mesh_faceadr[mid]);nf=int(model.mesh_facenum[mid])
            if model.geom_bodyid[geom]!=body:raise ValueError('All colliders must share original BODY')
            R=Rotation.from_quat(model.geom_quat[geom,[1,2,3,0]]).as_matrix();v=model.mesh_vert[v0:v0+n].astype(np.float64)@R.T+model.geom_pos[geom];f=model.mesh_face[f0:f0+nf].copy()
            expected=parts[f'part{i:03d}_vertices'];error=max(float(cKDTree(expected).query(v)[0].max()),float(cKDTree(v).query(expected)[0].max()));part_errors.append(error)
            cv.append(v);cf.append(f+offset);offset+=n
        cv,cf=np.concatenate(cv),np.concatenate(cf)
        sv,sf=trimesh.remesh.subdivide_to_size(vertices,faces,cfg['max_raster_triangle_edge_m'],max_iter=10)
        cv,cf=trimesh.remesh.subdivide_to_size(cv,cf,cfg['max_raster_triangle_edge_m'],max_iter=10)
        np.savez_compressed(out/'projection_geometry.npz',source_vertices=sv,source_faces=sf,compiled_BODY_vertices=cv,compiled_faces=cf)
        rows=[];samples={};boxes=[];member_pins=[]
        with patch.object(mujoco,'mj_step',side_effect=AssertionError('No integration')) as steps,patch.object(mujoco,'mj_forward',side_effect=AssertionError('No forward')) as forwards,tarfile.open(clip) as tar:
            for frame in range(47,108):
                phase=f'frame{frame:06d}';raw={k:tar.extractfile(f'{frame:06d}.{k}.json').read() for k in ['info','cameras','objects']};info,cameras,objects=[json.loads(raw[k]) for k in ['info','cameras','objects']]
                member_pins += [byte_pin(f'{frame:06d}.{k}.json',v) for k,v in raw.items()]
                ob=next(v for v in objects['9'] if str(v['object_uid'])==cfg['object_uid']);cal=cameras[cfg['stream']];intr=cal['calibration'];shape=(intr['image_height'],intr['image_width'])
                image_bytes=tar.extractfile(f'{frame:06d}.image_{cfg["stream"]}.jpg').read();member_pins.append(byte_pin(f'{frame:06d}.image_{cfg["stream"]}.jpg',image_bytes));image=Image.open(io.BytesIO(image_bytes)).convert('RGB')
                if image.size!=(shape[1],shape[0]):raise ValueError('Original image/calibration dimensions differ')
                T=pose_matrix(ob['T_world_from_object']);C=pose_matrix(cal['T_world_from_camera']);world=transform_points(T,sv);cw=transform_points(T,cv);eye=(world-C[:3,3])@C[:3,:3];ceye=(cw-C[:3,3])@C[:3,:3]
                uv=fisheye624(eye,intr['projection_params']);cuv=fisheye624(ceye,intr['projection_params']);oracle=camera.from_json(cal)
                same_error=float(abs(uv-oracle.eye_to_window(eye)).max());pipeline_error=float(abs(uv-oracle.world_to_window(world)).max())
                rendered=rasterize(uv,sf,shape);compound=rasterize(cuv,cf,shape);amodal=decode_hot3d_mask(ob['masks_amodal'][cfg['stream']]);modal=decode_hot3d_mask(ob['masks_modal'][cfg['stream']])
                def roundtrip(mask,encoded):
                    pixels=np.r_[False,mask.ravel(),False];runs=np.flatnonzero(pixels[1:]!=pixels[:-1])+1;runs[1::2]-=runs[::2];return runs.tolist()==encoded['rle']
                np.savez_compressed(out/f'masks_{frame:06d}.npz',source=np.packbits(rendered),compound=np.packbits(compound),provided_amodal=np.packbits(amodal),provided_modal=np.packbits(modal),shape=np.array(shape))
                row=dict(frame=frame,ref_timestamp_ns=info['ref_timestamp_ns'],RGB_timestamp_ns=info['image_timestamps_ns'][cfg['stream']],image_shape=list(shape),same_eye_max_error_px=same_error,FP32_oracle_pipeline_max_error_px=pipeline_error,
                    all_source_front_facing=bool(np.all(eye[:,2]>0)),max_source_angle_rad=float(np.arctan2(np.linalg.norm(eye[:,:2],axis=1),eye[:,2]).max()),calibration_max_solid_angle=intr['max_solid_angle'],
                    RLE_roundtrip=roundtrip(amodal,ob['masks_amodal'][cfg['stream']]) and roundtrip(modal,ob['masks_modal'][cfg['stream']]),
                    source_vs_provided_amodal=compare_masks(amodal,rendered),source_vs_provided_modal=compare_masks(modal,rendered),compound_vs_source=compare_masks(rendered,compound),compound_vs_provided_amodal=compare_masks(amodal,compound),
                    masks=file_pin(out/f'masks_{frame:06d}.npz'))
                rows.append(row);boxes.append(ob['boxes_amodal'][cfg['stream']])
                if frame in cfg['display_frames']:
                    (out/f'RGB_{frame:06d}.jpg').write_bytes(image_bytes);samples[frame]=(image,amodal,modal,rendered,compound)
                print(json.dumps(dict(frame=frame,amodal_IoU=row['source_vs_provided_amodal']['IoU'],compound_IoU=row['compound_vs_source']['IoU'])),flush=True)
        checks=dict(all61=len(rows)==61,compiled128=len(part_errors)==128,compiled_geometry=max(part_errors)<=cfg['compiled_mesh_tolerance_m'],
            same_eye_projection=max(r['same_eye_max_error_px'] for r in rows)<=cfg['same_eye_projection_tolerance_px'],oracle_serialization_precision=max(r['FP32_oracle_pipeline_max_error_px'] for r in rows)<=cfg['full_pipeline_precision_tolerance_px'],
            exact_RGB_clock=all(r['ref_timestamp_ns']==r['RGB_timestamp_ns'] for r in rows),strict_timestamps=all(b['ref_timestamp_ns']>a['ref_timestamp_ns'] for a,b in zip(rows,rows[1:])),
            RLE_roundtrip=all(r['RLE_roundtrip'] for r in rows),images1408=all(r['image_shape']==[1408,1408] for r in rows),no_native_steps=steps.call_count==0,no_native_forward=forwards.call_count==0)
        boxes=np.array(boxes);crop=(max(0,int(np.floor(boxes[:,0].min()))-30),max(0,int(np.floor(boxes[:,1].min()))-30),min(1408,int(np.ceil(boxes[:,2].max()))+30),min(1408,int(np.ceil(boxes[:,3].max()))+30))
        font_path=Path('C:/Windows/Fonts/arial.ttf');font=ImageFont.truetype(str(font_path),18);sheet=Image.new('RGB',(1536,3*550),(24,27,33));draw=ImageDraw.Draw(sheet)
        for ri,frame in enumerate(cfg['display_frames']):
            image,amodal,modal,rendered,compound=samples[frame]
            provided=np.asarray(image).copy();geometry=provided.copy()
            for mask,color,target in [(amodal,(255,205,40),provided),(modal,(30,240,240),provided),(rendered,(40,165,255),geometry),(compound,(255,80,80),geometry)]:target[mask&~binary_erosion(mask)]=color
            for ci,(label,picture) in enumerate([('Original RGB',image),('Dataset amodal yellow / modal cyan',Image.fromarray(provided)),('Source blue / compiled128 red',Image.fromarray(geometry))]):
                tile=picture.crop(crop);tile.thumbnail((500,500));x=ci*512+(512-tile.width)//2;y=ri*550+45;sheet.paste(tile,(x,y));draw.text((ci*512+8,ri*550+8),f'Frame {frame} | {label}',font=font,fill=(245,245,245))
        sheet.save(out/'RGB_projection_contact_sheet.png');write(out/'member_receipts.json',dict(clip=file_pin(clip),records=member_pins));verify_source_snapshot(receipt,live=True)
        report=dict(schema='real_ego_RGB_projection_audit:1',config=file_pin(cp),registration=file_pin(p['registration']),inputs=inputs,source_receipt=file_pin(out/'source_receipt.json'),rows=rows,checks=checks,passed=sum(checks.values()),total=len(checks),
            compiled_part_max_error_m=max(part_errors),projection_geometry=file_pin(out/'projection_geometry.npz'),member_receipts=file_pin(out/'member_receipts.json'),contact_sheet=file_pin(out/'RGB_projection_contact_sheet.png'),font=file_pin(font_path),display_frames=cfg['display_frames'],fixed_crop_xyxy=list(crop),
            aggregate={name:{'min_IoU':min(r[name]['IoU'] for r in rows),'median_IoU':float(np.median([r[name]['IoU'] for r in rows])),'max_boundary_px':max(r[name]['symmetric_boundary_max_px'] for r in rows)} for name in ['source_vs_provided_amodal','source_vs_provided_modal','compound_vs_source','compound_vs_provided_amodal']},
            wall_s=time.perf_counter()-start,actual_native_integrations=0,optimizer_steps=0,GPU_kernels=0,scope='Known mesh/annotated poses/provided masks/original RGB consistency only. Dataset masks are rendered or SAM2-derived,not independent manual truth. No gravity/dynamics/contact/robot/ego-generalization qualification.')
        write(out/'report.json',report)
        if not all(checks.values()):raise ValueError('Numeric/input checks failed;keep full report: '+repr(checks))
        print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[report['passed'],report['total']],aggregate=report['aggregate'])),flush=True)
    except Exception as error:write(out/'failure.json',dict(type=type(error).__name__,message=str(error),phase=phase,actual_native_integrations=0,optimizer_steps=0));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--register',action='store_true');main(p.parse_args())
