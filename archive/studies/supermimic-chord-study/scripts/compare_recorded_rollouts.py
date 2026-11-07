"""Compose two verified recorded-state renders; no physics or GPU work."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from chord_study.geometry import quat_matrix


def load(folder,video):
    meta=json.loads((folder/"rollout.json").read_text(encoding="utf-8-sig"))
    render=json.loads((folder/"render_metadata.json").read_text(encoding="utf-8-sig"))
    with np.load(folder/"rollout.npz",allow_pickle=False) as d:run={k:d[k] for k in d.files}
    with np.load(folder/"reference.npz",allow_pickle=False) as d:ref={k:d[k] for k in d.files}
    pos=1000*np.linalg.norm(run["object_pose_w"][:,:3]-ref["object_pose_w"][:,:3],axis=-1)
    a=quat_matrix(run["object_pose_w"][:,3:])[:,:,0];b=quat_matrix(ref["object_pose_w"][:,3:])[:,:,0]
    angle=np.rad2deg(np.arccos(np.clip((a*b).sum(-1),-1,1)))
    return meta,render,run,pos,angle,imageio.get_reader(folder/video)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a0",type=Path,required=True);p.add_argument("--a1",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    m0,v0,r0,p0,a0,reader0=load(args.a0,"rollout_a0.mp4")
    m1,v1,r1,p1,a1,reader1=load(args.a1,"rollout_a1_bounded.mp4")
    if m0["model_sha256"]!=m1["model_sha256"] or m0["reference_bundle_sha256"]!=m1["reference_bundle_sha256"] or v0["cameras"]!=v1["cameras"]:
        raise ValueError("comparison requires same model/reference/fixed camera")
    if not np.array_equal(r0["time_s"],r1["time_s"]):raise ValueError("sample clocks differ")
    font=ImageFont.truetype("C:/Windows/Fonts/consola.ttf",19);small=ImageFont.truetype("C:/Windows/Fonts/consola.ttf",16)
    movie=args.output/"a0_vs_a1_bounded.mp4"
    writer=imageio.get_writer(movie,fps=50,codec="libx264",quality=8,pixelformat="yuv420p",macro_block_size=16)
    stills=[]
    try:
        for k,t in enumerate(r0["time_s"]):
            f0=reader0.get_data(k);f1=reader1.get_data(k)
            canvas=Image.new("RGB",(1280,720),(14,20,29))
            canvas.paste(Image.fromarray(f0[48:688,640:1280]),(0,48));canvas.paste(Image.fromarray(f1[48:688,640:1280]),(640,48))
            draw=ImageDraw.Draw(canvas)
            draw.text((16,6),"A0 / direct reference",font=font,fill=(255,205,160))
            draw.text((656,6),"A1 / SuperMimic + bounded targets",font=font,fill=(164,238,204))
            draw.text((16,28),f"t={t:.2f}s  position={p0[k]:.1f}mm  shaft={a0[k]:.1f}deg",font=small,fill=(220,227,235))
            draw.text((656,28),f"t={t:.2f}s  position={p1[k]:.1f}mm  shaft={a1[k]:.1f}deg",font=small,fill=(220,227,235))
            draw.text((16,695),"Same reference / initial state. Recorded physics states; drawing only.",font=small,fill=(201,214,229))
            draw.text((880,695),"No CHORD shaping in A1.",font=small,fill=(201,214,229))
            writer.append_data(np.asarray(canvas))
            if k in (0,25,50,75,100,150):stills.append(canvas.copy())
    finally:
        writer.close();reader0.close();reader1.close()
    sheet=Image.new("RGB",(1920,720),(14,20,29))
    for k,frame in enumerate(stills):
        frame.thumbnail((640,360));sheet.paste(frame,((k%3)*640,(k//3)*360))
    sheet.save(args.output/"comparison_contact_sheet.png")
    decoded=imageio.get_reader(movie)
    count=decoded.count_frames();decoded.close()
    if count!=len(r0["time_s"]):raise ValueError("comparison frame count mismatch")
    report={"scope":"composition of verified physical state renders; no physics, new policy or state interpolation",
            "frames":count,"model_sha256":m0["model_sha256"],"reference_sha256":m0["reference_bundle_sha256"],
            "a0_rollout_sha256":hashlib.sha256((args.a0/"rollout.npz").read_bytes()).hexdigest(),
            "a1_rollout_sha256":hashlib.sha256((args.a1/"rollout.npz").read_bytes()).hexdigest(),
            "a1_limit_note":"same trained A1 weights, explicit real/imagined target Limit added after original training; not a contact-enhanced run"}
    (args.output/"comparison.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"video":str(movie),"frames":count}))


if __name__=="__main__":main()
