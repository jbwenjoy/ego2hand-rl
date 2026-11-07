"""Matched geometry/mass/friction drop comparison of two contact-time priors."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin
from supermimic_research.assets.compound import scene_xml
from validate_mug_asset import assert_prior,drop_test


def run(args):
    import mujoco
    out=args.output.resolve();meta=json.loads(args.mesh_report.read_text());decomp=json.loads(args.decomposition.read_text())
    if out.exists() or not out.is_relative_to(ROOT/'runs') or decomp['mesh_report']!=file_pin(args.mesh_report):raise ValueError('Fresh matched contact-time test')
    out.mkdir(parents=True);intent=dict(mesh_report=file_pin(args.mesh_report),decomposition=file_pin(args.decomposition),contact_times_s=[.02,.002],dt_s=[.0003125,.00015625],
        maximum_steps=38400,unchanged='same geometry/inertial prior/gravity/friction/initial pose/no aid',role='ISOLATED_ASSET_CONTACT_PRIOR_DIAGNOSTIC',automatic_retry=False,
        sources={name:file_pin(ROOT/name) for name in ('tools/check_asset_contact_time.py','tools/validate_mug_asset.py','src/supermimic_research/assets/compound.py')})
    (out/'intent.json').write_text(json.dumps(intent,indent=2),encoding='utf8');records=[]
    for tau in intent['contact_times_s']:
        folder=out/f'tau{tau}';folder.mkdir();path=folder/'drop.xml';path.write_text(scene_xml(meta,decomp,folder,drop=True,contact_time=tau),encoding='utf8')
        model=mujoco.MjModel.from_xml_path(str(path));assert_prior(model,meta,mujoco)
        for dt in intent['dt_s']:records.append(dict(contact_time_s=tau,drop=drop_test(model,dt,folder,mujoco)))
    report=dict(schema='supermimic_research.asset_contact_time_comparison:1',intent=file_pin(out/'intent.json'),records=records,native_steps=sum(x['drop']['steps'] for x in records),learning_updates=0,
        scope='Controlled isolated100mm drop under declared unmeasured priors. Contact-time changes solver compliance, not source geometry/mass/friction. Numerical calibration only, not material identification or robot holding.')
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(dict(report=file_pin(out/'report.json'),cases=[dict(tau=x['contact_time_s'],dt=x['drop']['dt_s'],maximum_penetration_m=x['drop']['metrics']['maximum_native_penetration_m'],passed=x['drop']['passed']) for x in records])))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for flag in ('mesh-report','decomposition','output'):p.add_argument('--'+flag,type=Path,required=True)
    run(p.parse_args())
