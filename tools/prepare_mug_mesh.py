"""Pin, bake and inspect the existing BOP9 source; no repair or simulation."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin
from supermimic_research.assets.mesh import bake_scene,exact_weld,topology,surface_mass_prior


def main(args):
    import trimesh
    pilot=args.pilot_root.resolve();out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified runs output')
    manifest=json.loads((pilot/'selected_models_manifest_v1.json').read_text());source=pilot/'hot3d/object_models/obj_000009.glb';info_path=pilot/'hot3d/object_models/models_info.json'
    for path in (source,info_path):
        actual=file_pin(path);expected=next(x for x in manifest['assets'] if Path(x['path']).resolve()==path)
        if (actual['bytes'],actual['sha256'])!=(expected['bytes'],expected['sha256']):raise ValueError('Changed source asset')
    info=json.loads(info_path.read_text())['9'];bounds=[[info['min_'+a] for a in 'xyz'],[info['min_'+a]+info['size_'+a] for a in 'xyz']]
    v,f,bake=bake_scene(trimesh.load_scene(source,process=False),bounds);wv,wf=exact_weld(v,f);prior=surface_mass_prior(v,f,.25)
    out.mkdir(parents=True)
    with (out/'source_mesh.npz').open('xb') as stream:np.savez_compressed(stream,vertices_model_m=v,faces=f,working_vertices_model_m=wv,working_faces=wf)
    for name,vertices,faces in [('visual_model_m.obj',v,f),('working_exact_weld_m.obj',wv,wf)]:
        text=trimesh.exchange.obj.export_obj(trimesh.Trimesh(vertices=vertices,faces=faces,process=False),include_normals=False,include_color=False,include_texture=False,digits=17)
        (out/name).write_text(text,encoding='utf8')
    report=dict(schema='supermimic_research.mug_mesh_intake:1',object_bop_id=9,revision=manifest['revision'],source=file_pin(source),models_info=file_pin(info_path),bake=bake,
        source_topology=topology(v,f),exact_weld_topology=topology(wv,wf),exact_weld_surface_changed=False,
        physical_prior=prior,unmeasured_ranges=dict(mass_kg=[.15,.5],hand_sliding_friction=[.3,1.0]),
        raw=file_pin(out/'source_mesh.npz'),visual=file_pin(out/'visual_model_m.obj'),working=file_pin(out/'working_exact_weld_m.obj'),
        sources=dict(tool=file_pin(__file__),mesh_module=file_pin(ROOT/'src/supermimic_research/assets/mesh.py')),
        scope='Source geometry in original BOP model frame; no hole filling/repair/recentering, collider/native solve/NN/learning. Inertial and friction values are declared exploratory assumptions, not data-derived calibration.')
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(dict(report=file_pin(out/'report.json'),topology=report['exact_weld_topology'],prior=prior)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
