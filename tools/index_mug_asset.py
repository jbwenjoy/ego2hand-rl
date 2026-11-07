"""Expose the checked object asset through the existing explicit registry."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import ArtifactRegistry,file_pin


def run(args):
    qa=json.loads(args.report.read_text());meta=json.loads(args.mesh_report.read_text());decomp=json.loads(args.decomposition.read_text());out=args.report.resolve().parent
    if not qa['qualified_numerical_asset_prototype'] or qa['passed']!=qa['total']:raise ValueError('Checked numerical prototype required')
    for name in ('physical_prior.json','asset_contract.json','artifacts.toml'):
        if (out/name).exists():raise ValueError('Fresh index, never overwrite')
    prior=dict(schema='supermimic_research.object_physical_prior:1',object_bop_id=9,**meta['physical_prior'],measured=False,
        unmeasured_ranges=meta['unmeasured_ranges'],nominal_object_geom_sliding_mu=.7,nominal_explicit_table_pair_sliding_mu=.6,condim=3,
        contact_time_prior_s=qa['contact_time_prior_s'],contact_damping_ratio=1.,
        source_mesh_report=file_pin(args.mesh_report),scope='Exploratory surface-lamina inertia/mass/friction/compliance prior, not identified physical parameters. Composed Hand2 pairs must declare their actual mixing/override rules.')
    (out/'physical_prior.json').write_text(json.dumps(prior,indent=2),encoding='utf8')
    contract=dict(schema='supermimic_research.compound_object_asset_contract:1',object_bop_id=9,body_name='hot3d_mug9',model_units='metres',
        BODY_frame='Original BOP model frame after baking GLB node transforms exactly once',COM_frame='Explicit source-lamina prior offset in BODY; not the mesh root',
        standalone_native_layout=dict(nq=7,nv=6,nu=0,Hand2_present=False),collision_parts=len(decomp['components']),
        visual_collidable=False,asset_model=file_pin(qa['asset_mjb']['path']),validation=file_pin(args.report),
        calibrated=False,robot_task=False,old_phone_Actor_or_WM_compatible=False,
        remaining='Right-hand/wrist-palm mapping, actual handle intent/finger fit, source gravity, new Hand2 scene/reference/observations/labels and no-aid holding remain unverified.')
    (out/'asset_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf8')
    ego=args.pilot_root.resolve();entries=[
        ('mug9_model',Path(qa['asset_mjb']['path']),'research','numerical_object_only_asset_not_robot_task'),
        ('mug9_contract',out/'asset_contract.json','research','explicit_frame_layout_contract'),
        ('mug9_physical_prior',out/'physical_prior.json','research','unmeasured_nominal_prior_not_calibration'),
        ('mug9_components',Path(decomp['raw']['path']),'research','model_frame_convex_union_geometry'),
        ('mug9_mesh_report',args.mesh_report.resolve(),'research','source_geometry_and_frames'),
        ('mug9_validation',args.report.resolve(),'research','scoped_numerical_asset_checks'),
        ('mug9_drop_scene',out/'drop_scene.xml','research','isolated_object_table_fixture_not_holding'),
        ('mug9_source_glb',Path(meta['source']['path']),'ego','original_known_object_mesh'),
    ]
    roots=dict(research=ROOT,ego=ego);lines=['schema = 1','','[roots]']
    for key,path in roots.items():lines.append(f'{key} = '+json.dumps(Path(os.path.relpath(path,out)).as_posix()))
    lines+=['','[relocations]',json.dumps(ROOT.as_posix())+' = "research"',json.dumps(ego.as_posix())+' = "ego"']
    for name,path,root,role in entries:
        pin=file_pin(path);relative=path.resolve().relative_to(roots[root]).as_posix()
        lines+=['',f'[artifacts.{name}]','root = '+json.dumps(root),'path = '+json.dumps(relative),'role = '+json.dumps(role),'sha256 = '+json.dumps(pin['sha256'])]
    manifest=out/'artifacts.toml';manifest.write_text('\n'.join(lines)+'\n',encoding='utf8');registry=ArtifactRegistry(manifest);inspection=registry.inspect(verify=True)
    if not all(row.get('verified') and row['exists'] for row in inspection):raise ValueError('All asset index entries verify')
    print(json.dumps(dict(manifest=file_pin(manifest),entries=len(inspection),contract=file_pin(out/'asset_contract.json'))))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for flag in ('report','mesh-report','decomposition','pilot-root'):p.add_argument('--'+flag,type=Path,required=True)
    run(p.parse_args())
