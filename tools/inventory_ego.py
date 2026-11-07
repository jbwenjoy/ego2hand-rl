"""Verify local pilot inputs and screen all already downloaded known-mesh objects."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin
from supermimic_research.ego.clip_inventory import read_clip,inventory


def run(args):
    pilot=args.pilot_root.resolve();out=args.output.resolve()
    if not out.is_relative_to(ROOT/'runs') or out.exists():raise ValueError('Fresh unified runs output')
    download=json.loads((pilot/'download_manifest_v1.json').read_text());models=json.loads((pilot/'selected_models_manifest_v1.json').read_text())
    if download['revision']!=models['revision']:raise ValueError('Same fixed dataset revision required')
    ids=sorted(map(int,models['selection']));pins=[]
    for item in download['clips']+models['assets']:
        p=Path(item['path']).resolve()
        if not p.is_relative_to(pilot):raise ValueError('Explicit local pilot input root')
        actual=file_pin(p)
        if actual['bytes']!=item['bytes'] or actual['sha256']!=item['sha256']:raise ValueError('Changed downloaded input')
        pins.append(actual)
    out.mkdir(parents=True);reports=[]
    for item in download['clips']:
        clip=Path(item['path']).stem;rows=read_clip(item['path'],ids);result=inventory(rows)
        result['clip']=clip;path=out/(clip+'.json');path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
        reports.append(dict(clip=clip,report=file_pin(path),best_per_object=result['best_per_object'],candidate_windows=len(result['candidates'])))
    summary=dict(schema='supermimic_research.known_mesh_ego_inventory:1',revision=download['revision'],input_pins=pins,
        sources=dict(tool=file_pin(__file__),reader=file_pin(ROOT/'src/supermimic_research/ego/clip_inventory.py')),
        heuristic=dict(window_frames=61,min_RGB_co_visible_frames=31,min_object_world_translation_m=.01,max_median_origin_wrist_distance_m=.3),reports=reports,
        role='INTAKE_ONLY_NO_FITTING_NO_CONTACT_QUALIFICATION',simulation_steps=0,learning_updates=0,
        scope='Existing three raw ego clips and local meshes only; no new download, images, FK, contact classification or physical feasibility claim.')
    path=out/'report.json';path.write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(dict(report=file_pin(path),clips=reports)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
