"""Register the new task layout explicitly, without changing frozen phone pins."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin,ArtifactRegistry


def main(args):
    stage,proximity,review=args.reference.resolve(),args.proximity.resolve(),args.review.resolve()
    path=stage/'artifacts.toml'
    if path.exists() or not stage.is_relative_to(ROOT/'runs'):raise ValueError('New unified task index only')
    report=json.loads((stage/'report.json').read_text());check=json.loads((review/'report.json').read_text())
    if report['passed']!=report['total'] or check['passed']!=check['total']:raise ValueError('Require numeric and saved review')
    for pin in report['outputs'].values():verify_file_pin(pin)
    records=[('hand2_mug_model',stage/'hand2_mug.mjb','COMPILED_SCENE_NOT_GRASP_QUALIFIED'),
        ('right_mug_reference61',stage/'reference61.npz','ANNOTATED_KINEMATIC_REFERENCE_NOT_ROBOT_JOINT_TARGETS'),
        ('right_mug_contract',stage/'report.json','FRAME_CONTRACT_NOT_POLICY_BENEFIT'),
        ('right_mug_binding',stage/'binding.json','COMPOUND_BODY_COM_AND_CONTACT_PAIR_BINDING'),
        ('right_mug_proximity',proximity/'source_proximity61.npz','UNSIGNED_SOURCE_GEOMETRY_INTAKE_NO_FITTING'),
        ('right_mug_proximity_report',proximity/'report.json','PROXIMITY_NOT_CONTACT_TRUTH'),
        ('right_mug_saved_review',review/'report.json','SAVED_CONTRACT_REVIEW_NOT_HOLDING'),
        ('mug9_physical_prior',ROOT/'runs/mug_asset_validation_v2/physical_prior.json','UNMEASURED_DYNAMICS_PRIOR')]
    lines=['schema = 1','', '[roots]', 'research = '+json.dumps(Path(os.path.relpath(ROOT,stage)).as_posix()),'']
    for name,p,role in records:
        pin=file_pin(p)
        lines += ['[artifacts.'+name+']','root = "research"','path = '+json.dumps(p.relative_to(ROOT).as_posix()),
            'role = '+json.dumps(role),'sha256 = '+json.dumps(pin['sha256']),'']
    path.write_text('\n'.join(lines),encoding='utf8')
    registry=ArtifactRegistry(path)
    verified={name:file_pin(registry.resolve(name)) for name,_,_ in records}
    result=dict(source=file_pin(__file__),manifest=file_pin(path),verified=verified,entries=len(records),
        frozen_phone_artifact_manifest_changed=False,compatible_with_phone_checkpoints=False,neural_updates=0,integration_steps=0)
    (stage/'index_report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--proximity',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True);main(p.parse_args())
