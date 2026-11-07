"""Source-derived gauge protocol before looking at decomposition QA results."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin
from supermimic_research.assets.validation import gauges,nearest_surface,corridor_clearance
from supermimic_research.assets.mesh import winding_numbers


def run(args):
    import trimesh
    report=json.loads(args.mesh_report.read_text());out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or file_pin(report['raw']['path'])!=report['raw']:raise ValueError('Fresh gauge protocol and pinned source')
    with np.load(report['raw']['path'],allow_pickle=False) as z:v=z['vertices_model_m'];f=z['faces']
    mesh=trimesh.Trimesh(vertices=v,faces=f,process=False);paths,guesses=gauges();surface,_=nearest_surface(mesh,guesses);records={};arrays=dict(source_surface_controls=surface)
    for name,points in paths.items():
        _,distance=nearest_surface(mesh,points);w=winding_numbers(points,v,f)
        arrays[name+'_points']=points;arrays[name+'_source_distance']=distance;arrays[name+'_source_winding_descriptor']=w
        records[name]=dict(source_minimum_surface_distance_m=float(distance.min()),maximum_absolute_winding_descriptor=float(abs(w).max()),
            source_corridors={str(r):corridor_clearance(distance,points,r) for r in (.001,.003,.005)})
    out.mkdir(parents=True)
    with (out/'gauges.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
    intent=dict(schema='supermimic_research.mug_gauge_protocol:1',mesh_report=file_pin(args.mesh_report),source_only_selection=True,radii_m=[.001,.003,.005],paths=records,raw=file_pin(out/'gauges.npz'),
        required_primary_radius_m=.001,sampled_error_limits_m=dict(source_outside_union_max=.0015,exposed_union_surface_to_source_max=.003),
        counts=dict(handle_points=29,cavity_points=22,source_surface_controls=3),
        sources={name:file_pin(ROOT/name) for name in ('tools/prepare_mug_gauges.py','src/supermimic_research/assets/mesh.py','src/supermimic_research/assets/validation.py')},
        scope='Gauges chosen from original BOP sections. Source distance/winding and finite clear tubes are numerical descriptors, not true material/contact or finger-fit guarantees. Future decomposition/native tests must preserve primary1mm corridors and3mm source-surface controls. No fitting or dynamics.')
    (out/'protocol.json').write_text(json.dumps(intent,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(dict(protocol=file_pin(out/'protocol.json'),source_corridors=records)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mesh-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
