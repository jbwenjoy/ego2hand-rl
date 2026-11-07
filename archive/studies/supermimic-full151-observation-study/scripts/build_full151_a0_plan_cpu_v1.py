"""Freeze one concrete full151 native A0 plan; no environment or device."""
from pathlib import Path
import argparse,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_native.io import checked,pin,read,write
from full151_native.probe import GATES,CLOCK_CONTRACT


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--authorize-gpu',action='store_true');a=parser.parse_args()
    if any(x in sys.modules for x in ('torch','warp','mujoco_warp')):raise AssertionError('Plan creation imported device frameworks')
    cfgdir=ROOT/'configs';cfgdir.mkdir(exist_ok=True)
    protocol_path=ROOT/'protocols/full151-native-a0-v1.json';config_path=cfgdir/'full151-a0-authorized-v1.json'
    if protocol_path.exists() or config_path.exists():raise FileExistsError('Freeze once, never overwrite')
    data=ROOT/'outputs/full151_native_input_v1';bundle=pin(data/'bundle.json')
    readiness={
      'input_CPU16':pin(data/'receipt.json'),
      'input_independent67':pin(ROOT/'outputs/full151_native_input_saved_independent_v2/report.json'),
      'core_CPU61_READY':pin(ROOT/'outputs/full151_native_core_READY_v1/READY.json'),
      'native_A0_CPU44_READY':pin(ROOT/'outputs/READY_full151_native_A0_v1.json'),
      'consumers_CPU65_READY':pin(ROOT/'outputs/READY_checkpoint_consumers_v1.json'),
      'A0_front_static27':pin(ROOT/'outputs/full151_A0_front_static_review_v1/report.json')}
    protocol={'schema':'full151_knownrough_native_A0_protocol:1','allowed_roles':['native_a0'],'allowed_seeds':[0],
      'input_bundle':bundle,'reference_states':151,'control_intervals':150,'source_frame_range_half_open':[200,351],
      'physical_reference_qualified':False,'training_authorized':False,'known_rough_reference':True,'geometric_candidate_qualified':False,
      'environment':{'num_envs':1,'nconmax':192,'njmax':4096,'control_hz':50,'physics_dt_s':.005,'substeps_per_interval':4,'quality_early_stop':False},
      'episode_counts':{'real_intervals':150,'solver_steps':600,'raw_states':601,'native_postinterval_forwards':150,'last_real_phase':149,'terminal_observation':150,'u150_permitted':False},
      'clock_contract':CLOCK_CONTRACT,'evaluation_gates':GATES,'readiness':readiness,
      'runtime_scope':'One fromscratch-zeroresidual A0; noActor/WM/Runner instance or optimizer. Actual8copy-setconst0restore-doubleread then explicitanalyticphase0reset. Fullqualityfailuresrecordedwithoutresetorassist; hardunstable/nonfinite/clock/APIerror partialuncensoredmask.',
      'capacity_scope':'Frozenstickyobserver beginsafterEnvconstructor, beforeactual8/explicitreset, readsbeforeeachobservedresetandoncefinal. Constructor-before-observer work separatelyUNOBSERVED, not promotedtofullprocesscapacityproof.',
      'sampling':'Rawpoststep qpos/qvel/clock600+startup. Solverderived contact/force preintegration before original150Managerforwards. Rewardcapture/state58/self15/PENmemory/terms samecadence and phase explicitlysaved.',
      'attempt_policy':'One Root-authorized run, soft180s/hard240s watchdog, noautomaticretry/fallback, nohistoricalqualityselection. Partialexceptions/metadata/identity failuresretained.',
      'source_history':'Parentfull151SCPFAIL preserved; accepting an explicitlyknownrough engineeringinput doesnotchangegeometry/physicalqualification. No old104checkpoint/16framepadding orshared[0,151)identity reuse.',
      'checkpoint_scope':'105 newidentityreserved; noactualcheckpointcreatedbyA0. Futuretraining/actual105health/A2independenttargets remainpending.'}
    source={};seen={}
    def include(key,value):
        path=checked(value).resolve();marker=str(path).casefold()
        if marker in seen:
            if seen[marker]!=value['sha256']:raise ValueError('Samefile different expected bytes')
            return
        seen[marker]=value['sha256'];source[key]=value
    include('input_bundle',bundle)
    for key,value in read(data/'bundle.json')['source_pins'].items():include('input__'+key,value)
    for key,value in readiness.items():include('readiness__'+key,value)
    # Real application/backend Python source closure. Binary library versions
    # are recorded separately; this is not an OS/container freeze claim.
    roots=[Path('D:/workspace/dex-supermimic-research/src/dex_supermimic'),Path('D:/workspace/dex-supermimic-research/.venv/Lib/site-packages/mjlab'),Path('D:/workspace/dex-supermimic-research/.venv/Lib/site-packages/mujoco_warp')]
    listed=subprocess.run(['rg','--files',*(str(x) for x in roots)],capture_output=True,text=True,check=True).stdout.splitlines()
    for i,path in enumerate(sorted(x for x in listed if x.endswith('.py'))):include(f'native_python_{i:04d}',pin(path))
    for path in sorted((ROOT/'src/full151_native').glob('*.py')):include('full151__'+path.name,pin(path))
    paths={'worker':ROOT/'scripts/execute_full151_a0_v1.py','supervisor':ROOT/'scripts/run_full151_a0_v1.py','plan_builder':Path(__file__),
      'qualified_identity_v3':Path('D:/workspace/supermimic-observation-restoration-prefix-study/scripts/run_track_prefix_v3.py'),
      'qualified_lifecycle':Path('D:/workspace/supermimic-control-feasibility-draft/scripts/run_braking_microprobe.py'),
      'local_atomic_writer':ROOT/'vendor/runtime_io.py','local_measured_contacts':ROOT/'vendor/measured_contacts.py',
      'actual_projection':Path('D:/workspace/supermimic-c2-training-study/src/target_inset.py'),
      'capacity_observer':Path('D:/workspace/supermimic-rough-prefix-learning-study/src/rough_prefix/capacity.py'),
      'original_full151_clock_evaluator':Path('D:/workspace/supermimic-chord-study/scripts/evaluate_pen_checkpoint.py')}
    for key,path in paths.items():include(key,pin(path))
    write(protocol_path,protocol);source['protocol']=pin(protocol_path)
    config={'schema':'full151_knownrough_native_A0_run:1','role':'native_a0','method':'A0','seed':0,'input_bundle':bundle,
      'output':str((ROOT/'outputs/full151_native_A0_v1').resolve()),'training':{'num_envs':1},'contact_binding':None,
      'source_pins':source,'protocol':pin(protocol_path),'GPU_run_authorized':bool(a.authorize_gpu),'training_authorized':False,
      'hard_process_seconds':240,'soft_worker_seconds':180,'automatic_retry':False,'compile_fallback':False,
      'controller':{'kind':'FULL151_KNOWNROUGH_BASELINE_EMA_C2','format_version':105,'margin_rad':.020,'resume':'from_scratch_only'},'evaluation_gates':GATES}
    write(config_path,config)
    print(__import__('json').dumps({'config':pin(config_path),'protocol':pin(protocol_path),'pins':len(source),'GPU_run_authorized':bool(a.authorize_gpu),'GPU_started':False,'Torch_imported':False,'Warp_imported':False}))


if __name__=='__main__':main()
