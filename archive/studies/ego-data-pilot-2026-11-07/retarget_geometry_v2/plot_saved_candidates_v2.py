"""Saved-pose visualization only: one passive model read, no MjData/FK."""
from pathlib import Path
import hashlib,importlib.util,json,os
os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['MPLBACKEND']='Agg'
ROOT=Path(__file__).resolve().parent


def pin(p):
    p=Path(p).resolve();return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    report=json.loads((ROOT/'report.json').read_text(encoding='utf8'));protocol=json.loads((ROOT/'protocol_v2.json').read_text(encoding='utf8'))
    prep=json.loads((ROOT/'fixed_anchors_and_targets.json').read_text(encoding='utf8'))['preparation']
    spec=importlib.util.spec_from_file_location('saved_contact_first_loader',protocol['inputs']['loader']['path']);loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader);model=loader.load_model()
    target=json.loads(Path(protocol['inputs']['targets']['path']).read_text(encoding='utf8'))
    hand=[g for g in range(model.ngeom) if (model.geom_contype[g] or model.geom_conaffinity[g]) and model.geom(g).name!='stage1_single_table']
    phone=model.geom('phone24_collision_convex').id
    def mesh(g):
        kind=int(model.geom_type[g])
        if kind==7:
            mid=int(model.geom_dataid[g]);start=int(model.mesh_vertadr[mid]);n=int(model.mesh_vertnum[mid]);v=model.mesh_vert[start:start+n].astype(np.float64)
        elif kind==3:
            radius,half=map(float,model.geom_size[g,:2]);theta=np.arange(12)*2*np.pi/12;points=[]
            for sign in [-1,1]:
                for phi in np.linspace(0,np.pi/2,5):
                    points.extend(np.c_[radius*np.sin(phi)*np.cos(theta),radius*np.sin(phi)*np.sin(theta),np.full(12,sign*(half+radius*np.cos(phi)))])
            v=np.unique(points,axis=0)
        else:return None,None
        return v,ConvexHull(v).simplices
    local={g:mesh(g) for g in hand};fig=plt.figure(figsize=(15,8),layout='constrained');plot_records=[]
    for column,result in enumerate(report['seed_results']):
        ax=fig.add_subplot(2,3,column+1,projection='3d');bottom=fig.add_subplot(2,3,column+4)
        if 'selected_state' not in result:ax.text2D(.1,.5,result['status'],transform=ax.transAxes);bottom.axis('off');continue
        with np.load(result['selected_state']['path'],allow_pickle=False) as a:gp=a['geom_xpos'];gr=a['geom_xmat'].reshape(-1,3,3);R=a['phone_R_world'];p=a['phone_p_world'];tips=a['site_xpos'][[model.site(f'right_finger{i}_tip').id for i in range(1,6)]]
        world=[];phoneframe=[]
        for g in hand:
            v,f=local[g]
            if v is None:continue
            w=v@gr[g].T+gp[g];o=(w-p)@R;world.append(w);phoneframe.append(o)
            color='#777777' if g==phone else '#439bb0';alpha=.35 if g==phone else .27
            ax.add_collection3d(Poly3DCollection(w[f]*1000,facecolor=color,edgecolor='none',alpha=alpha))
            bottom.scatter(o[:,0]*1000,o[:,2]*1000,s=2,c=color,alpha=.25)
        actual=[];desired=[]
        for i,entry in enumerate(prep):
            g=model.geom(entry['geom']['name']).id;point=np.asarray(entry['anchor_geom_local_m'])@gr[g].T+gp[g];want=np.asarray(entry['target_projected_phone_convex_object_m'])@R.T+p;actual.append(point);desired.append(want)
            ax.plot(*np.stack([point,want]).T*1000,c='#d22c2c',lw=1.4)
        actual=np.asarray(actual);desired=np.asarray(desired);ax.scatter(*actual.T*1000,c='#147a2d',s=23);ax.scatter(*desired.T*1000,c='#d22c2c',marker='x',s=30)
        cloud=np.concatenate(world)*1000;lo=cloud.min(0);hi=cloud.max(0);ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]));ax.set_box_aspect(hi-lo);ax.view_init(20,-62);ax.set_axis_off()
        gaps=[q['gap_m']*1000 for q in result['verification']['all_phone_pairs'] if q['is_candidate_pad']]
        ax.set_title(f"Seed{column}: geometric qualification {result['geometry_candidate_qualified']}\npad gaps(mm) {', '.join(f'{v:.4f}' for v in gaps)}",fontsize=10)
        ao=(actual-p)@R;de=(desired-p)@R;bottom.scatter(ao[:,0]*1000,ao[:,2]*1000,c='#147a2d',s=35,label='Fixed robot anchors');bottom.scatter(de[:,0]*1000,de[:,2]*1000,c='#d22c2c',marker='x',s=40,label='Projected reference targets')
        tipobject=(tips-p)@R;reference=np.array([x['reference_tip_object_m'] for x in target['mapping']]);bottom.scatter(tipobject[:,0]*1000,tipobject[:,2]*1000,c='#176e9f',s=10);bottom.scatter(reference[:,0]*1000,reference[:,2]*1000,c='#d9a72d',marker='+',s=25)
        bottom.set_aspect('equal');bottom.set(xlabel='Phone X (mm)',ylabel='Phone Z (mm)');bottom.grid(alpha=.15);bottom.legend(fontsize=7)
        plot_records.append({'seed':column,'selected_state':result['selected_state'],'geometry_candidate_qualified':result['geometry_candidate_qualified'],'pad_gaps_mm':gaps})
    fig.suptitle('Contact-first static geometry / all three original starts — no preload, dynamics or holding claim')
    fig.savefig(ROOT/'all3_contact_first_candidates.png',dpi=160,bbox_inches='tight');plt.close(fig)
    out={'schema':'contact_first_saved_visualization:1','report':pin(ROOT/'report.json'),'source':pin(__file__),'figure':pin(ROOT/'all3_contact_first_candidates.png'),'all3':plot_records,'counts':{'passive_MJB_reads':1,'MjData_FK_distance_optimization_physics_training_GPU':0},'figure_scope':'Ideal compiled colliders only. Capsules tessellated solely for display; source mesh convex hulls solely for display. These approximations never feed optimization or pass/fail queries.'}
    with (ROOT/'visualization_receipt_v2.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({'figure':out['figure']}))


if __name__=='__main__':main()
