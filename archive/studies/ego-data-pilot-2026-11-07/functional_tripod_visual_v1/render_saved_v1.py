"""Visualize saved original/three-tip candidate poses; no forward/integration."""
from pathlib import Path
import argparse,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np

P=Path(__file__).resolve().parent.parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def checked(r):assert pin(r['path'])==r;return Path(r['path'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate-report',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    assert not a.output.exists();a.output.mkdir()
    report=read(a.candidate_report);modelpin=report['model_mjb'];entries=report['candidates'];poses=[];pins={}
    for branch in ['exact_source_original_hand','source1mm_opposed_side_original_hand']:
        path=P/f'surface_assembly_v11/{branch}_completed_v1/seed1_rank0.json';c=read(path)
        # Source v11 final saved native raw qpos is authoritative, not a fresh IK.
        q=np.asarray(c['candidate']['kinematics']['qpos27']) if 'candidate' in c and 'kinematics' in c['candidate'] else None
        if q is None:
            continue  # Do not guess a new pose if a source schema is different.
        poses.append((branch+' (old 2-pad source)',q));pins[branch]=pin(path)
    for e in entries:
        c=read(checked(e['candidate']));q=np.array(c['qpos27_F64'])
        poses.append((c['candidate_id']+' | geometry '+str(c['geometry']['passed']),q));pins[c['candidate_id']]=e['candidate']
    assert poses,'No saved poses to render'
    m=mujoco.MjModel.from_binary_path(str(checked(modelpin)));d=mujoco.MjData(m)
    m.vis.headlight.ambient[:]=.3;m.vis.headlight.diffuse[:]=.65
    opt=mujoco.MjvOption();opt.geomgroup[3]=0
    renderer=mujoco.Renderer(m,width=640,height=480);camera=mujoco.MjvCamera()
    fig,axes=plt.subplots(len(poses),2,figsize=(12,4*len(poses)),squeeze=False,constrained_layout=True)
    try:
        for row,(name,q) in enumerate(poses):
            assert q.shape==(27,) and np.isfinite(q).all()
            d.qpos[:]=q;d.qvel[:]=0
            mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_camlight(m,d)
            camera.lookat[:]=(d.xpos[m.body('right_palm_link').id]+d.xpos[m.body('hot3d_phone24').id])/2
            camera.distance=.36;camera.elevation=-20
            for col,azimuth in enumerate((135,-45)):
                camera.azimuth=azimuth;renderer.update_scene(d,camera=camera,scene_option=opt)
                axes[row,col].imshow(renderer.render().copy());axes[row,col].axis('off');axes[row,col].set_title(name+'\nview '+str(azimuth)+' deg',fontsize=10)
        fig.suptitle('Fixed-wrist Hand2 + HOT3D phone: saved poses\nVisual comparison only; not a contact/load or dynamic-holding certificate',fontsize=13)
        fig.savefig(a.output/'saved_pose_comparison.png',dpi=130)
    finally:renderer.close();plt.close(fig)
    with (a.output/'report.json').open('x',encoding='utf8') as f:
        json.dump(dict(schema='saved_tripod_pose_visual_comparison:1',candidate_report=pin(a.candidate_report),model=modelpin,saved_poses=pins,
            source=pin(__file__),figure=pin(a.output/'saved_pose_comparison.png'),
            counts=dict(MJB_reads=1,MjData=1,mj_kinematics=len(poses),mj_comPos=len(poses),mj_camlight=len(poses),render=2*len(poses),mj_forward=0,integration_steps=0,NN=0),
            visual_lighting_only_model_changes=True,contact_or_holding_or_ego_qualified=False),f,indent=2,allow_nan=False)
    print(json.dumps({'report':pin(a.output/'report.json')}))

if __name__=='__main__':main()
