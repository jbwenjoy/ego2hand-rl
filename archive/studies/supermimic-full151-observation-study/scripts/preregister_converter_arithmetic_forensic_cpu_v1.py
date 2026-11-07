"""Freeze an arithmetic-only condition; no converter or scientific input load."""
import hashlib,json
from pathlib import Path
ROOT=Path('D:/workspace/supermimic-full151-observation-study')


def pin(p):
    p=Path(p).resolve(strict=True);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return {'path':str(p),'sha256':h.hexdigest()}


def main():
    dest=ROOT/'protocols/full151105-converter-arithmetic-forensic-cpu-v1.json'
    if dest.exists():raise ValueError('No overwrite')
    parent=ROOT/'protocols/full151105_solver_phase_penalty_observability_CPU_prereg_v2.json'
    failed=ROOT/'outputs/full151105_solver_phase_penalty_CPU_v1/failure.json'
    case=ROOT/'configs/full151105-solver-phase-penalty-CPU-v1.json'
    require=lambda x:None if x else (_ for _ in ()).throw(ValueError('Fixed failed condition mismatch'))
    require(pin(parent)['sha256']=='1ec18db963bc2931920bedce3f2bf43e345544798ee4ac1e0f856ac119cea13c')
    require(pin(case)['sha256']=='160c2b8d27012631ffab821f04cdd8ae115c84765b5051daa0289926192020a3')
    failure=json.loads(failed.read_text(encoding='utf-8-sig'))
    require(failure['completed_scored_rows']==0 and failure['counts']['native_state_conversion']['returned']==1 and failure['counts']['objective_factory']['attempted']==0)
    d={'schema':'full151105_converter_arithmetic_forensic_protocol:1','status':'FROZEN_ARITHMETIC_DIAGNOSTICS_ONLY_NOT_AUTHORIZED',
       'CPU_run_authorized':False,'GPU_run_authorized':False,'proxy_scoring_authorized':False,
       'parent_protocol':pin(parent),'failed_config':pin(case),'failed_record':pin(failed),'writer':pin(__file__),
       'known_failure':'Original all452post20 p/R/v/omega output-scaled8eps32 gate failed. CauseUNKNOWN. Original00e1/config160c/outputv1 immutable, factory/FW/scoredrows0.',
       'explicit_condition':'Same data preparation + one original unbound nativeState conversion2260. Save alloutputs before arithmetic tests. Never call factory/scoring or use diagnostic outcomes to grant eligibility.',
       'fixed_budget':{'passive_MJB_mapping_loads':1,'native_state_calls':1,'native_state_rows':2260,'pre_rows':1808,'known_post20_rows':452,
                       'leading_anchor_comparisons':160,'solver_pose_diagnostics':1808,'factory_calls':0,'objective_WM_Actor_forward':0,'weights_load_AD_optimizer_Env_nativeFK_collision_physics_GPU':0},
       'old_rule_retained':'q/qd exactFP32 bits; otherstatecomponents abs(actual-ref)<=8*eps32*max(1,abs(actual),abs(ref)). Record everyflag/maxerror; NO earlyrequire(all flags), NO replacementacceptancecriterion.',
       'required_saved_arrays':['raw_converted2260_native58','post20_reference452_native58','pre_qpos1808x27','pre_qvel1808x26','post20_qpos452x27','post20_qvel452x26',
         'actualReaderpalm_pos2260x3','actualReaderpalm_rot2260x3x3','mapping','all_endpoint_diff58_oldbounds58_flags58',
         'all160_anchor_actual_reference_diff58_bounds58_flags58','all1808_solver_pose_actual_pre_pose_diff_bounds_flags',
         'raw_object_quaternions','normalized_quaternions_CPU_FP32_and_F64','object_rotations_CPU_FP32_and_F64',
         'F64_shadow_state58_SECONDARY_NOT_ALTERNATE_CONVERTER','F64_operand_absolute_sums_and_coordinate_scales','output_cancellation_ratios_DIAGNOSTIC_ONLY'],
       'conditioning':'Record inputs/rotations and a pureF64 formula shadow of the same state definition, including bodyomega->world->palm. abs-sum/cancellation measures are explanatory arithmetic descriptors, not an error theorem or a relaxedgate. Do not assume all failures harmless.',
       'phase_analysis':'All452 post20 endpoints,160 leadingpreanchors,1808 solverPREpose comparisons preserved; no chosenrows. The fixed actualReader frames and mapping equal originalprep. Wrongframe/mapping/phase remain possible until actual diagnostics.',
       'source_guard':'Borrow only exactfrozen00e1 purehelpers; do not monkeypatch arithmetic_check, originalgate or originalrun function. New module/CLI/CPUfixtureReady/config/output with complete pins required.',
       'readonly':'Fresh CPU CUDA_VISIBLE_DEVICES=-1; noWarp/Env/modelweight loading, nofactory/binding import or reachable scoringcall. PassiveMJB constant mapping allowed only on Root separately authorized --run. Pre/post sources/data/RNG hashchecks; partials preserved.',
       'interpretation':'Forensic status is diagnostics complete, not dense penalty READY. Originalfailedgate staysFAIL even if a new CPU replay has differentflags. No proxy/physics/gradient eligibilityupgrade, no autorerun/scoring afterdiagnosis.'}
    with dest.open('x',encoding='utf-8') as f:json.dump(d,f,indent=2,allow_nan=False)
    print(json.dumps({'protocol':pin(dest)},indent=2))


if __name__=='__main__':main()
