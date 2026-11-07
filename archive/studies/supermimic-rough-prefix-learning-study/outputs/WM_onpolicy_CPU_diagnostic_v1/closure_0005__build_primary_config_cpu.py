"""Prepare a from-scratch primary seed0 configuration, always GPU-disabled.

Revision4 and all fixed engineering prerequisites must exist first. This
builder never imports a trainer/device, starts a process, or authorizes a run.
"""
from __future__ import annotations
import argparse,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rough_prefix.io import read,write,pin,checked,canonical


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--method',choices=['A1L','P_ONLY','A2'],required=True)
    ap.add_argument('--version',required=True)
    ap.add_argument('--protocol',type=Path,default=ROOT/'protocols/rough-prefix-v4.json')
    ap.add_argument('--evidence',type=Path,action='append',default=[])
    a=ap.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_]+',a.version):raise ValueError('A new simple version label is required')
    from train_rough_prefix_pen_v5 import PRIMARY_TRAINING,verify_primary_prerequisites
    from rough_prefix.probe import GATES
    protocol=a.protocol.resolve(strict=True);p=read(protocol);bundle=ROOT/'outputs/input_bundle_v1/bundle.json';data=read(bundle)
    if p.get('schema')!='rough_prefix_learning_protocol:1' or type(p.get('protocol_revision')) is not int or p['protocol_revision']!=4 or p.get('allowed_roles')!=['primary_matched'] or canonical(p.get('allowed_seeds'))!=canonical([0]) or canonical(p.get('primary_matched_training'))!=canonical(PRIMARY_TRAINING):raise ValueError('Only the separately frozen exact revision4 primary budget is accepted')
    if p['input_bundle']!=pin(bundle):raise ValueError('Primary protocol input differs')
    pins={'protocol':pin(protocol),'input_bundle':pin(bundle),'config_builder':pin(__file__),'reviewed_v3_identity':pin('D:/workspace/supermimic-observation-restoration-prefix-study/scripts/run_track_prefix_v3.py'),'reviewed_lifecycle':pin('D:/workspace/supermimic-control-feasibility-draft/scripts/run_braking_microprobe.py'),'executed_external_C2_projection':pin('D:/workspace/supermimic-c2-training-study/src/target_inset.py')}
    def add(key,value):
        checked(value)
        if key in pins:
            if pins[key]!=value:raise ValueError('Source alias conflict '+key)
            return
        # Preserve every prior path+SHA while avoiding redundant archived copies
        # of the same model/source shared by six prerequisite configurations.
        if value not in pins.values():pins[key]=value
    paths=[ROOT/'scripts/train_rough_prefix_pen_v5.py',ROOT/'scripts/run_rough_prefix_v6.py',Path(__file__)]
    paths += [ROOT/'src/rough_prefix'/name for name in ('__init__.py','io.py','metadata.py','bundle.py','bundle_v2.py','native_v2.py','training.py','capacity.py','probe.py','runtime_v2.py','controller.py')]
    if a.method in ('P_ONLY','A2'):paths += [ROOT/'src/rough_prefix/contact_binding.py']
    paths += sorted((ROOT/'vendor').rglob('*.py'))
    for path in paths:add('study__'+path.relative_to(ROOT).as_posix(),pin(path))
    dex=Path('D:/workspace/dex-supermimic-research')
    for path in sorted((dex/'src/dex_supermimic').rglob('*.py')):add('dex__'+path.relative_to(dex).as_posix(),pin(path))
    lib=dex/'.venv/Lib/site-packages/mjlab'
    for suffix in ('envs/manager_based_rl_env.py','sim/sim.py','entity/entity.py','managers/action_manager.py','managers/termination_manager.py','managers/observation_manager.py','utils/buffers/circular_buffer.py','utils/buffers/delay_buffer.py'):
        add('mjlab__'+suffix,pin(lib/suffix))
    for name in ('READY_controller_v1.json','READY_runtime_v2.json','READY_contact_binding_v1.json','READY_training_capacity_v1.json','READY_capacity_v1.json'):
        add(name,pin(ROOT/'outputs'/name))
    for method,spec in p['primary_matched_prerequisites'].items():
        for key,value in spec.items():add('prerequisite__'+method+'__'+key,value)
        for kind in ('smoke_config','evaluation_config'):
            old=read(checked(spec[kind]))
            for key,value in old['source_pins'].items():add('prerequisite__'+method+'__'+kind+'__'+key,value)
    for index,path in enumerate(a.evidence):add('independent_evidence__'+str(index),pin(path))
    if a.method in ('P_ONLY','A2'):
        contact=pin(ROOT/'outputs/contact_intent_v1/objective_spec.json');spec=read(checked(contact));binding=read(checked(spec['binding_manifest']));add('contact_spec',contact);add('contact_binding_manifest',spec['binding_manifest'])
        for key,value in binding.get('source_pins',{}).items():add('contact_source__'+key,value)
        for key,name in {'manifest_sha256':'manifest.json','target_npz_sha256':'contact_targets.npz','rules_sha256':'rules.json','raw_geometry_sha256':'raw_geometry.json','time_windows_sha256':'time_windows.json'}.items():
            value=pin(Path(spec['bundle_dir'])/name)
            if value['sha256']!=spec['expected_hashes'][key]:raise ValueError('Frozen prior artifact drift: '+name)
            add('contact_artifact__'+key,value)
    else:contact=None
    for key,value in data['source_pins'].items():add('input_source__'+key,value)
    config={'schema':'rough_prefix_native_run:1','role':'primary_matched','method':a.method,'seed':0,'input_bundle':pin(bundle),'output':str(ROOT/'outputs'/('primary_'+a.method.lower()+'_seed0_'+a.version)),'training':p['primary_matched_training'],'contact_binding':contact,'source_pins':pins,'protocol':pin(protocol),'GPU_run_authorized':False,'training_authorized':True,'hard_process_seconds':7200,'soft_worker_seconds':7100,'automatic_retry':False,'compile_fallback':False,'controller':{'kind':'ROUGH_PREFIX_BASELINE_EMA_C2','format_version':104,'margin_rad':.020,'resume':'from_scratch_only'},'evaluation_gates':GATES}
    verify_primary_prerequisites(p,config)
    path=ROOT/'configs'/(a.method.lower()+'-primary_matched-'+a.version+'.json')
    if path.exists():raise FileExistsError('Never overwrite a frozen primary config')
    write(path,config);print(pin(path))


if __name__=='__main__':main()
