"""Conditional gravity mapping from pinned official device-axis convention."""
from pathlib import Path
import argparse,ast,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin


def main(args):
    sources=args.sources.resolve();reference=args.reference.resolve();out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh unified output')
    manifest=json.loads((sources/'manifest.json').read_text())
    for item in manifest['files']:verify_file_pin(item)
    viewer=next(x for x in manifest['files'] if x['path'].endswith('Hot3DVisualizer.py'))
    tree=ast.parse(Path(viewer['path']).read_text())
    matches=[x for x in ast.walk(tree) if isinstance(x,ast.If) and
             ast.unparse(x.test)=='self._hot3d_data_provider.get_device_type() is Headset.Aria' and
             any('RIGHT_HAND_Z_UP' in ast.unparse(y) for y in x.body)]
    if len(matches)!=1:raise ValueError('Explicit official Aria Z-up branch required')
    z=np.load(reference/'reference61.npz');P=z['T_source_world_from_rest_palm'][:,:3,:3];R=z['T_robot_world_from_robot_palm'][:3,:3]
    assumed_source=np.array([0.,0.,-9.81]);g=np.einsum('ij,nkj,k->ni',R,P,assumed_source)
    angles=np.rad2deg(np.arccos(np.clip(-g[:,2]/9.81,-1,1)))
    if not np.allclose(np.linalg.norm(g,axis=1),9.81,rtol=0,atol=1e-12):raise ValueError('Rigid mapping gravity magnitude')
    out.mkdir(parents=True);np.savez_compressed(out/'conditional_gravity61.npz',frame_ids=z['frame_ids'],timestamps_ns=z['timestamps_ns'],
        gravity_source_world_assumed_m_s2=assumed_source,gravity_mapped_into_robot_world_m_s2=g,angle_to_simulator_gravity_deg=angles)
    result=dict(schema='mug_gravity_mapping_review:1',source=file_pin(__file__),official_sources=file_pin(sources/'manifest.json'),
        official_viewer=viewer,viewer_revision=manifest['revision'],reference=file_pin(reference/'reference61.npz'),
        official_Aria_world_up='Z_UP',device='Aria',source_gravity_direction_from_viewer_convention='negative_Z',
        source_gravity_measured_for_this_sequence=False,gravity_magnitude_assumed_m_s2=9.81,
        viewer_convention_is_not_per_sequence_calibration=True,
        mapped_first_gravity_m_s2=g[0].tolist(),angle_to_simulator_gravity_deg_min_max=[float(angles.min()),float(angles.max())],
        fixed_palm_mapping_preserves_gravity=False,source_contact_force_labels_transferable=False,
        output=file_pin(out/'conditional_gravity61.npz'),neural_updates=0,integration_steps=0,
        scope='Official viewer device-up convention plus conditional9.81 magnitude. Not an observed per-sequence gravity vector or proof that clips preserve full-dataset world alignment. Current mapping changes gravity-relative task; retain the old UNKNOWN binding rather than silently rewriting it.')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
