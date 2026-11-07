"""Freeze a saved-only audit checklist without reading any scientific output."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path('D:/workspace/supermimic-full151-observation-study')
OUT=ROOT/'outputs/full151105_trained_bonus_saved_audit_prereg_v1'


def pin(path):
    p=Path(path).resolve(strict=True)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    if OUT.exists():raise ValueError('Fresh checklist directory only')
    sources={'consumer':ROOT/'src/full151_gradient/trained_bonus_analysis105_v1.py',
             'prereg':ROOT/'protocols/full151105_trained_geometric_bonus_secant_CPU_prereg_v1.json',
             'CPU66':ROOT/'outputs/full151105_trained_bonus_preparation_CPU_v1/report.json',
             'CPU_READY':ROOT/'outputs/full151105_trained_bonus_preparation_CPU_v1/READY.json',
             'Root_preflight_case':ROOT/'configs/full151105-trained-bonus-CPU-v1.json',
             'checklist_writer':Path(__file__)}
    pins={k:pin(p) for k,p in sources.items()}
    if pins['consumer']['sha256']!='9c34ea62f2aab6c1492bc544e916ef5c6d77348452e282501a148047a9505fbd':raise ValueError('Frozen consumer mismatch')
    if pins['prereg']['sha256']!='9d48ae798471638bb6bbc3e47d80112b185de68fd861e9eac5449286428413b7':raise ValueError('Frozen prereg mismatch')
    if pins['Root_preflight_case']['sha256']!='0aef37ffbe8b0d5f84dfcb2b623a5832e2dba79f325b925d4811790ebf1214d0':raise ValueError('Root case mismatch')
    checklist={
      'schema':'full151105_trained_bonus_saved_audit_checklist:1',
      'status':'FROZEN_BEFORE_SAVED_AUDIT_NO_SCIENTIFIC_OUTPUT_READ',
      'created_UTC':datetime.now(timezone.utc).isoformat(),'pins':pins,
      'own_execution_scope':{'actual_output_paths_opened':0,'NPZ_opened':0,'model_loads':0,'objective_or_WM_or_Actor_forward':0,'AD':0,'Env_physics_GPU':0},
      'trigger':'Only after Root actual terminal notice; do not read live scientific output or run a score/model again.',
      'checks_to_perform':[
        {'id':'source_and_case','rule':'Bind actual command config separately if authorized copy differs; fixed prereg9d48 and consumer9c34. Check complete source table/current files/execution ZIP order+bytes, all immutable inputs+checkpoint hashes. No model deserialization.'},
        {'id':'outcome_and_partial','rule':'Read actual report or preserved failure exactly; do not turn missing fields into false/success. Audit attempted versus returned counts and pre-failure phase. Preserve partial/NOT_RUN if not completed.'},
        {'id':'exact_runtime_budget','rule':'Complete expectation only: strictWMloader1 with torch.load2 internal, factories2, physical objective2(batch160), WMforward1(batch8), predicted objective2(batch8), uVJP4. Hooks capture4 existing proxy calls; never infer additional geometry computation from files.'},
        {'id':'module_authority','rule':'Exact trained GPU spec08d9 read unchanged at CPU; actual binding_v3/R frozen math64, P_ONLY eagerA2 base. Verify saved receipts/source paths/registered-constants equality evidence, not a lookalike implementation or512 primary.'},
        {'id':'row_and_phase_identity','rule':'Physical160 order=donor8 then actualbranch152; labels A0/A2 x [0,49,99,149]. Next20 state and PREACTION memoryk,start0,end151. k149 remains active; no u150, target150 not primary. Verify all states/drop/far rows retained.'},
        {'id':'actual_input_identity','rule':'Saved x,u equal pinned full05 state58/absoluteu20, unprojected and unreclipped. Compare physical phases/predicted phases and target64 to frozen151 table. Compare beforeAD and postAD prediction files without rerunningWM.'},
        {'id':'capture_layout_and_consistency','rule':'All4 proxy NPZ captures: finiteFP32 CPU source-backed, support64/presence6/SDF[P]/weight[P]/projectedCOMpoints[P,3]/inwardnormal[P,3]/minSDF. Compare P_ONLY andA2 saved outputs. Hook extraction only; no new Torch/objective/geometry forward in audit.'},
        {'id':'proxy_math_saved_only','rule':'Independently reconstruct weights exp(-.5*(relu(sdf)/.001)^2), minSDF, rawP=(max(relu(-sdf-.003))/.001)^2 from SAVED SDF; official64 support match from SAVED support+target, originalthreshold1e-3/tolerance.1/var.1. Arithmetic FP32/F64 checks are not native geometry/force truth. Report all captured near-surface/support rows, not aggregate RMS inference.'},
        {'id':'raw_weighted_and_dt_units','rule':'Use actualfactory saved BP andA2 primary. Independently compare rawP/rawS/CS, BP=-.1P, A2=CS+BP arithmetic residual; .02 derivatives only once. Do not rewrite actualscore via subtraction/cancellation or replace primary withF64.'},
        {'id':'predeclared_noise','rule':'Independently recompute3baseline range/donor error perstate/component,8eps32scale, rawP1e-4/rawS&CS1e-5 floors, BP=.1floorP, A2=max(floorBP+floorCS,10N_A2,8eps32scaleA2). BP representation-noise inconsistency cancels eligibility without floor tuning. dtfloor=.02unscaled.'},
        {'id':'all_secants_and_group_labels','rule':'For each5channels x32 dependentgroups x2eps, F64-promoteFP32 endpoints before subtract/(2eps); strict signal >floor, same sign/relativeeps<=.5. Retain all320 rows/160 labels; original native-object9stable not a filter; clipped orthogonal.'},
        {'id':'saved_AD_projection','rule':'Read saved4 gradient8x20 only. RawSgradient=CSgradient only after sourceweights1 verified. Recompute d_eff projections inF64 and nominalsecondary, all center offsets/clipped flags; compare componentVJP identities as diagnostics, not new AD/physical proof.'},
        {'id':'denominator_and_statistics','rule':'Recompute perchannel/eps all32 and stable/unclipped secondary statistics, max(abs(slope),abs(AD),floor/(2eps)) denominator, zeroAD signnoagreement, flat ratiosnull+reason. No mixedunit aggregate, weighted totalscorewinner,CI,pvalue or enlargedN acrossfunctions/eps.'},
        {'id':'RNG_and_parameters_evidence_scope','rule':'Input/file hashes can be independently checked. In-memory RNG/weights/normalizer/buffer pre/post booleans are SOURCE-BACKED runtime assertions only unless both raw states were saved; never claim independent remeasurement or reload/checking trainedweights.'},
        {'id':'domain_and_semantics','rule':'No q/R clipping/projection/filtering; finite but offmanifold/far/drop inputs retained. Proxy max/threshold/SDF branches are nonsmooth. Measured self15/HO force/directtorque are separate; none is a bonusgradient label or eligibility mask.'},
        {'id':'interpretation','rule':'CPU FP32shadow != actual native GPUAMP policygradient. Wholehidden/physicalJacobian/forceclosure/taskbenefit remain false. Nsource1,Nseed1,Nego0,eightdependentstates,twoactionpaths. Historical prepolicy100 training divergence prevents originaltraining causalbenefit claim.'}
      ],
      'formula_changes_after_outcome_permitted':False,
      'actual_saved_audit_status':'PENDING_ROOT_TERMINAL',
      'future_reader_privileges':'JSON/CSV/NPZ NumPy and byte hashing only; no torch.load/Actor/WM/objective construction, no geometry/FK/query/Env/GPU/physics.'}
    OUT.mkdir(parents=True)
    (OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    p=OUT/'checklist.json'
    with p.open('x',encoding='utf-8') as f:json.dump(checklist,f,indent=2,allow_nan=False)
    print(json.dumps({'checklist':pin(p),'writer':pins['checklist_writer'],'status':checklist['status']}))


if __name__=='__main__':main()
