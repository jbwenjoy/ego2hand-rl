"""One bounded CPU CoACD call in a child process; source surface stays immutable."""
from pathlib import Path
import argparse,datetime,importlib.metadata,json,os,subprocess,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin

PARAMETERS=dict(threshold=.001,real_metric=True,max_convex_hull=64,preprocess_mode='auto',preprocess_resolution=120,resolution=1000,
    mcts_nodes=10,mcts_iterations=50,mcts_max_depth=3,pca=False,merge=True,decimate=False,max_ch_vertex=128,extrude=False,apx_mode='ch',seed=20261108)
FAST_PARAMETERS={**PARAMETERS,'preprocess_resolution':100,'resolution':500,'mcts_nodes':5,'mcts_iterations':10,'mcts_max_depth':2}
FINE_PARAMETERS={**FAST_PARAMETERS,'max_convex_hull':128}


def write(path,value):
    with Path(path).open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def worker(args):
    import coacd,trimesh
    from scipy.spatial import ConvexHull
    intent=json.loads(args.intent.read_text());source=json.loads(Path(intent['mesh_report']['path']).read_text());out=args.intent.parent
    if importlib.metadata.version('coacd')!='1.0.14' or source['raw']!=file_pin(source['raw']['path']):raise ValueError('Pinned backend/input')
    with np.load(source['raw']['path'],allow_pickle=False) as z:v=z['working_vertices_model_m'].copy();f=z['working_faces'].copy()
    tick=time.perf_counter();parts=coacd.run_coacd(coacd.Mesh(v,f),**intent['parameters']);arrays={};records=[]
    if not 1<=len(parts)<=intent['parameters']['max_convex_hull']:raise ValueError('Declared convex piece budget')
    for i,(vertices,unused_faces) in enumerate(parts):
        vertices=np.array(vertices,dtype=float);h=ConvexHull(vertices);faces=h.simplices.copy()
        normals=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])
        flip=np.einsum('ij,ij->i',normals,h.equations[:,:3])<0;faces[flip]=faces[flip][:,[0,2,1]]
        if not np.isfinite(vertices).all() or h.volume<=0:raise ValueError('Finite full-dimensional convex pieces')
        mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
        if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume<=0:raise ValueError('Canonical outward convex component')
        name=f'part{i:03d}';arrays[name+'_vertices']=vertices;arrays[name+'_faces']=faces;arrays[name+'_halfspaces']=h.equations.copy()
        path=out/(name+'.obj');path.write_text(trimesh.exchange.obj.export_obj(mesh,include_normals=False,include_color=False,include_texture=False,digits=17),encoding='utf8')
        records.append(dict(index=i,vertices=len(vertices),faces=len(faces),volume_m3=float(h.volume),obj=file_pin(path)))
    with (out/'components.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
    write(out/'report.json',dict(schema='supermimic_research.compound_decomposition:1',intent=file_pin(args.intent),mesh_report=intent['mesh_report'],components=records,raw=file_pin(out/'components.npz'),
        parameters=intent['parameters'],backend_version=importlib.metadata.version('coacd'),elapsed_s=time.perf_counter()-tick,CoACD_calls=1,native_steps=0,learning_updates=0,
        scope='CoACD auto preprocessing of exact-weld source, then Qhull canonicalization of convex vertex sets. Original surface untouched; preprocessing surface is not exposed. Output error/void preservation must be independently checked, not guaranteed by threshold. No collision qualification yet.'))


def run(args):
    import coacd
    out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    if importlib.metadata.version('coacd')!='1.0.14':raise ValueError('Pinned isolated asset environment')
    out.mkdir(parents=True);module=Path(coacd.__file__).parent
    backend={p.name:file_pin(p) for p in module.iterdir() if p.is_file() and p.suffix in ('.py','.pyd','.dll')}
    intent=dict(schema='supermimic_research.decomposition_intent:1',tool=file_pin(__file__),mesh_report=file_pin(args.mesh_report),parameters={'balanced':PARAMETERS,'fast':FAST_PARAMETERS,'fine':FINE_PARAMETERS}[args.recipe],
        isolated_interpreter=sys.executable,backend=backend,maximum_child_wall_seconds=240,requested_thread_limits=dict(OMP_NUM_THREADS='4',OMP_THREAD_LIMIT='4'),
        primary_docs=['https://github.com/SarahWeiii/CoACD','https://pypi.org/project/coacd/1.0.14/'],automatic_retry=False,role='ASSET_DEVELOPMENT_NO_FITTING')
    write(out/'intent.json',intent);env=dict(os.environ,OMP_NUM_THREADS='4',OMP_THREAD_LIMIT='4',CUDA_VISIBLE_DEVICES='-1');tick=time.perf_counter();timed_out=False
    with (out/'worker.log').open('x',encoding='utf8') as log:
        process=subprocess.Popen([sys.executable,__file__,'--worker','--intent',str(out/'intent.json')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:code=process.wait(timeout=240)
        except subprocess.TimeoutExpired:timed_out=True;process.kill();code=process.wait()
    write(out/'terminal.json',dict(pid=process.pid,returncode=code,timed_out=timed_out,elapsed_s=time.perf_counter()-tick,automatic_retry=False,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    if code or timed_out:raise RuntimeError('Bounded decomposition failed; inspect retained worker.log and terminal')
    print(json.dumps(dict(report=file_pin(out/'report.json'),terminal=file_pin(out/'terminal.json'))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--intent',type=Path);p.add_argument('--mesh-report',type=Path);p.add_argument('--output',type=Path);p.add_argument('--recipe',choices=('balanced','fast','fine'),default='balanced');args=p.parse_args()
    if args.worker:worker(args)
    else:run(args)
