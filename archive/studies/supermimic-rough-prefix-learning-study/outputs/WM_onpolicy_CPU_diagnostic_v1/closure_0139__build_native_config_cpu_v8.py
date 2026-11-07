"""Freeze a new finite A0/smoke config; never launches a device or trainer."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rough_prefix.io import read,write,pin,checked
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--role',choices=['native_a0','engineering_smoke'],required=True);ap.add_argument('--method',choices=['A0','A1L','P_ONLY','A2'],required=True);ap.add_argument('--version',required=True);ap.add_argument('--authorize',action='store_true');ap.add_argument('--evidence',type=Path,action='append',default=[]);a=ap.parse_args()
    from rough_prefix.probe import GATES
    protocol=ROOT/'protocols/rough-prefix-v3.json';p=read(protocol);bundle=ROOT/'outputs/input_bundle_v1/bundle.json';data=read(bundle)
    if (a.role=='native_a0')!=(a.method=='A0'):raise ValueError('A0 is a nontraining role; three arms use smoke')
    pins={'protocol':pin(protocol),'input_bundle':pin(bundle),'config_builder':pin(__file__),'reviewed_v3_identity':pin('D:/workspace/supermimic-observation-restoration-prefix-study/scripts/run_track_prefix_v3.py'),'reviewed_lifecycle':pin('D:/workspace/supermimic-control-feasibility-draft/scripts/run_braking_microprobe.py'),'executed_external_C2_projection':pin('D:/workspace/supermimic-c2-training-study/src/target_inset.py')}
    # Pin the execution closure. Independent checker/proposal files may be
    # authored concurrently and are evidence pins, not worker dependencies.
    paths=[ROOT/'scripts/train_rough_prefix_pen_v4.py',ROOT/'scripts/run_rough_prefix_v5.py',Path(__file__)]
    paths += [ROOT/'src/rough_prefix'/name for name in ('__init__.py','io.py','metadata.py','bundle.py','bundle_v2.py','native_v2.py','training.py','capacity.py','probe.py','runtime_v2.py','controller.py')]
    if a.method in ('P_ONLY','A2'):paths += [ROOT/'src/rough_prefix/contact_binding.py']
    paths += sorted((ROOT/'vendor').rglob('*.py'))
    for path in paths:pins['study__'+path.relative_to(ROOT).as_posix()]=pin(path)
    dex=Path('D:/workspace/dex-supermimic-research')
    for path in sorted((dex/'src/dex_supermimic').rglob('*.py')):pins['dex__'+path.relative_to(dex).as_posix()]=pin(path)
    lib=dex/'.venv/Lib/site-packages/mjlab'
    for suffix in ('envs/manager_based_rl_env.py','sim/sim.py','entity/entity.py','managers/action_manager.py','managers/termination_manager.py','managers/observation_manager.py','utils/buffers/circular_buffer.py','utils/buffers/delay_buffer.py'):
        path=lib/suffix;pins['mjlab__'+suffix]=pin(path)
    for name in ('READY_controller_v1.json','READY_runtime_v2.json','READY_contact_binding_v1.json','READY_training_capacity_v1.json','READY_capacity_v1.json'):pins[name]=pin(ROOT/'outputs'/name)
    for index,path in enumerate(a.evidence):pins['independent_evidence__'+str(index)]=pin(path)
    if a.role=='engineering_smoke':
        a0=ROOT/'outputs/native_a0_v2';pins['native_A0_evaluation']=pin(a0/'evaluation.json');pins['native_A0_identity']=pin(a0.with_name(a0.name+'_watchdog')/'engineering_identity.json')
    if a.method in ('P_ONLY','A2'):
        contact=pin(ROOT/'outputs/contact_intent_v1/objective_spec.json');spec=read(contact['path']);binding=read(spec['binding_manifest']['path']);pins['contact_spec']=contact;pins['contact_binding_manifest']=spec['binding_manifest']
        for key,value in binding.get('source_pins',{}).items():pins['contact_source__'+key]=value
        for key,name in {'manifest_sha256':'manifest.json','target_npz_sha256':'contact_targets.npz','rules_sha256':'rules.json','raw_geometry_sha256':'raw_geometry.json','time_windows_sha256':'time_windows.json'}.items():
            value=pin(Path(spec['bundle_dir'])/name)
            if value['sha256']!=spec['expected_hashes'][key]:raise ValueError('Frozen prior artifact drift: '+name)
            pins['contact_artifact__'+key]=value
    else:contact=None
    pins.update({'input_source__'+k:v for k,v in data['source_pins'].items()})
    for value in pins.values():checked(value)
    config={'schema':'rough_prefix_native_run:1','role':a.role,'method':a.method,'seed':0,'input_bundle':pin(bundle),'output':str(ROOT/'outputs'/('native_a0_'+a.version if a.role=='native_a0' else 'smoke_'+a.method.lower()+'_'+a.version)),'training':{'num_envs':1} if a.role=='native_a0' else p['engineering_smoke_training'],'contact_binding':contact,'source_pins':pins,'protocol':pin(protocol),'GPU_run_authorized':a.authorize,'training_authorized':a.role!='native_a0','hard_process_seconds':120 if a.role=='native_a0' else 300,'soft_worker_seconds':110 if a.role=='native_a0' else 280,'automatic_retry':False,'compile_fallback':False,'controller':{'kind':'ROUGH_PREFIX_BASELINE_EMA_C2','format_version':104,'margin_rad':.020,'resume':'from_scratch_only'},'evaluation_gates':GATES}
    path=ROOT/'configs'/((a.method.lower())+'-'+a.role+'-'+a.version+'.json')
    if path.exists():raise FileExistsError('Never overwrite a frozen config')
    write(path,config);print(pin(path))
if __name__=='__main__':main()
