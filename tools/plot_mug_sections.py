"""Source-only scientific sections; no inferred contact or collision repair."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin


def section_segments(vertices,faces,axis,value):
    t=vertices[faces];d=t[:,:,axis]-value;hits=np.full((len(t),3,3),np.nan);valid=np.zeros((len(t),3),dtype=bool)
    for k,(a,b) in enumerate(((0,1),(1,2),(2,0))):
        mask=d[:,a]*d[:,b]<0;alpha=d[mask,a]/(d[mask,a]-d[mask,b])
        hits[mask,k]=t[mask,a]+alpha[:,None]*(t[mask,b]-t[mask,a]);valid[:,k]=mask
    keep=valid.sum(1)==2;indices=np.argsort(~valid[keep],axis=1)[:,:2]
    return np.take_along_axis(hits[keep],indices[:,:,None],axis=1)


def main(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    meta=json.loads(args.report.read_text());raw=meta['raw'];out=args.output.resolve()
    if file_pin(raw['path'])!=raw or out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Pinned source and fresh plot')
    with np.load(raw['path'],allow_pickle=False) as z:v=z['vertices_model_m'];f=z['faces']
    fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    for ax,(axis,value,i,j) in zip(axes,[(2,.0025,0,1),(1,.050,0,2),(0,0.,2,1)]):
        if args.components:
            from scipy.spatial import ConvexHull
            with np.load(args.components,allow_pickle=False) as parts:
                names=sorted(name[:-9] for name in parts.files if name.endswith('_vertices'))
                for name in names:
                    lines=section_segments(parts[name+'_vertices'],parts[name+'_faces'],axis,value)
                    if not len(lines):continue
                    points=np.unique(np.round(lines[:,:,[i,j]].reshape(-1,2),12),axis=0)
                    if len(points)<3:continue
                    hull=ConvexHull(points);polygon=points[hull.vertices]*1000
                    ax.fill(polygon[:,0],polygon[:,1],color='#2f80b9',alpha=.2,edgecolor='#2f80b9',linewidth=.4)
        segments=section_segments(v,f,axis,value);ax.add_collection(LineCollection(segments[:,:,[i,j]]*1000,colors='#233a55',linewidths=.8))
        ax.autoscale();ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_xlabel('XYZ'[i]+' (mm)');ax.set_ylabel('XYZ'[j]+' (mm)');ax.set_title(f'Source section {"XYZ"[axis]}={value*1000:g} mm')
    axes[0].scatter([-52,0],[50,65],marker='+',s=90,c=['#c54128','#278757']);axes[0].text(-52,50,' handle gauge',fontsize=8);axes[0].text(0,65,' cavity gauge',fontsize=8)
    fig.suptitle('HOT3D mug9: original BOP model-frame triangle sections\n'+('Black: source; blue: convex-piece sections (internal cuts shown); gauges are geometry tests only' if args.components else 'No recentering, extra scale, hole filling or convexification; gauges are geometry tests only'))
    out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=150);plt.close(fig)
    print(json.dumps(dict(image=file_pin(out),source=raw)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--components',type=Path);main(p.parse_args())
