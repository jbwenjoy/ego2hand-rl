"""Scientific annotation/reference figures; no image reconstruction or fitting."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin


def main(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import mujoco
    stage,proximity,out=args.reference.resolve(),args.proximity.resolve(),args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified runs output')
    report=json.loads((stage/'report.json').read_text());dist_report=json.loads((proximity/'report.json').read_text())
    for pin in report['outputs'].values():verify_file_pin(pin)
    verify_file_pin(dist_report['output'])
    z=np.load(stage/'reference61.npz');dist=np.load(proximity/'source_proximity61.npz');mesh=np.load(ROOT/'runs/mug_mesh_v1/source_mesh.npz')
    v,f=mesh['vertices_model_m'],mesh['faces'];groups=dist['dominant_skinning_group']
    colors=['#64748b','#d97706','#dc2626','#16a34a','#2563eb','#9333ea'];names=['wrist','thumb','index','middle','ring','little']
    paths=[[5,6,7,0],[5,8,9,10,1],[5,11,12,13,2],[5,14,15,16,3],[5,17,18,19,4]]
    fig=plt.figure(figsize=(15,5),layout='constrained')
    axes=[fig.add_subplot(1,4,i+1,projection='3d') for i in range(4)]
    for ax in axes:
        # Every original source triangle, no collision proxy replacing the handle.
        ax.add_collection3d(Poly3DCollection(v[f]*1000,facecolor='#a8a29e',edgecolor='none',alpha=.07,rasterized=True))
        ax.set(xlim=(-105,80),ylim=(-65,135),zlim=(-85,100),xlabel='BODY x (mm)',ylabel='y (mm)',zlabel='z (mm)')
        ax.set_box_aspect((185,200,185));ax.view_init(22,-65);ax.tick_params(labelsize=7)
    for ax,row in zip(axes[:3],[0,30,60]):
        points=z['hand_vertices_object_BODY_F64'][row]*1000
        landmarks=z['hand_landmarks21_object_BODY_F64'][row]*1000
        for group in range(1,6):
            mask=groups==group;ax.scatter(*points[mask].T,c=colors[group],s=2,alpha=.4,depthshade=False)
            ax.plot(*landmarks[paths[group-1]].T,color=colors[group],lw=1.6)
            ax.scatter(*landmarks[group-1],c=colors[group],s=18)
        ax.set_title(f"Human right annotation FK\nframe {z['frame_ids'][row]}, t={z['elapsed_s'][row]:.3f}s",fontsize=10)
    model=mujoco.MjModel.from_binary_path(str(stage/'hand2_mug.mjb'));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    pose=z['T_robot_world_from_object_BODY'][0];R=pose[:3,:3];p=pose[:3,3]
    def to_object(points):return (np.asarray(points)-p)@R*1000
    for digit in range(1,6):
        anchors=[data.xanchor[model.joint(f'right_finger{digit}_joint{j}').id].copy() for j in range(1,5)]
        anchors.append(data.site_xpos[model.site(f'right_finger{digit}_tip').id].copy())
        a=to_object(anchors);axes[3].plot(*a.T,color=colors[digit],lw=2);axes[3].scatter(*a[-1],c=colors[digit],s=20)
    axes[3].set_title('Hand2 at q=0, mapped mug\nNO fitted grasp; 0 native contacts',fontsize=10)
    fig.suptitle('Same mug BODY frame | annotation proximity is not contact truth or robot feasibility',fontsize=12)
    out.mkdir(parents=True);fig.savefig(out/'right_mug_geometry.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,3.5),layout='constrained');t=z['elapsed_s']
    for key,label in [('T_source_world_from_object_BODY_F64','source world'),('T_robot_world_from_object_BODY','after removing wrist')]:
        p=z[key][:,:3,3];ax[0].plot(t,np.linalg.norm(p-p[0],axis=1)*1000,label=label)
    ax[0].set(xlabel='Original elapsed time (s)',ylabel='Origin displacement from first (mm)',title='Fixed wrist changes the task');ax[0].legend()
    for group in range(1,6):
        minimum=dist['distances_m'][:,groups==group].min(1)*1000
        ax[1].plot(t,minimum,label=names[group],color=colors[group])
    ax[1].axhline(3,color='#475569',ls='--',lw=.8);ax[1].set(xlabel='Original elapsed time (s)',ylabel='Minimum sampled unsigned gap (mm)',title='All788 vertices to original mug triangles',yscale='log');ax[1].legend(ncol=2,fontsize=8)
    for a in ax:a.grid(alpha=.2)
    fig.savefig(out/'right_mug_motion_proximity.png',dpi=160);plt.close(fig)
    result=dict(source=file_pin(__file__),reference_report=file_pin(stage/'report.json'),proximity_report=file_pin(proximity/'report.json'),
        files=[file_pin(out/n) for n in ['right_mug_geometry.png','right_mug_motion_proximity.png']],
        native_forward_calls=1,integration_steps=0,neural_updates=0)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--proximity',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
