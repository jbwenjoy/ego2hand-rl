"""Review document claims against existing JSON receipts only."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path('D:/workspace/supermimic-full151-observation-study')
OUT=Path(__file__).resolve().parent
def pin(path):
    p=Path(path).resolve(strict=True)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

docpath=ROOT/'docs/full151_input_findings_2026-11-04.md'
docpin=pin(docpath);text=docpath.read_text(encoding='utf-8-sig')
paths={
    'fit':ROOT/'outputs/full151_cpu_input_v1/receipt.json',
    'joint14':ROOT/'outputs/full151_cpu_input_v1/joint_LPs/joint14_receipt.json',
    'geometry':ROOT/'outputs/full151_failed_candidate_geometry_v1/receipt.json',
    'geometry_metadata':ROOT/'outputs/full151_failed_candidate_geometry_v1/query_metadata.json',
    'cue_READY':ROOT/'outputs/observation_intent_cues_v1/READY.json',
    'fit_raw_review':ROOT/'outputs/full151_cpu_input_saved_independent_v1/report.json',
    'lifecycle_review':ROOT/'outputs/full151_input_saved_lifecycle_independent_v1/report.json',
    'root_execution':ROOT/'outputs/full151_cpu_input_v1/root_execution_observation.json',
}
sources={k:pin(v) for k,v in paths.items()};d={k:read(v) for k,v in paths.items()}
checks={
    'explicit_candidateFAIL_noREADY_nophysical':all(s in text for s in ('没有READY','未通过原数值合同','没有新训练或物理积分')),
    'source_lineage_not_independent_ego_shared_window_distinct':all(s in text for s in ('不是新增独立ego数据','[200,351)','[0,151)')),
    'raw_and_functional_failures_not_conflated':all(s in text for s in ('joint14的fit辅助行','joint4紧化target上界行','15/20关节通过')) and d['joint14']['maximum_raw_Ax_minus_b']==d['joint14']['row_group_max_raw_Ax_minus_b']['fit_abs_negative'],
    'fitgate33of34_and5joints_correct':len(d['fit']['checks'])==34 and sum(v is True for v in d['fit']['checks'].values())==33 and [r['joint'] for r in d['fit']['joint_results'] if not r['arithmetic_1e12_gate']]==[4,12,13,14,15],
    '15rad_s_targetrate_not_hardware_qd':all(s in text for s in ('解析初速度峰值16.055520602','不是上述目标更新率的越界','目标索引仅0..149','没有u150')),
    'referenceforce_not_measured_force': '参考PD证书也不是实测关节力或接触力' in text,
    'oldprefix_qualification_not_inherited': '旧prefix的几何或物理资格不能继承' in text,
    'no_false_no_wrist_read_claim': '没有读取源wrist' not in text and 'source_cached_wrist_pose_read=false' not in text,
    'macro21_explicitly_not_nativeBODY21': '并非native BODY21参考ABI' in text,
    'unknown_gap_not_clearance_release_safety':all(s in text for s in ('gap保持NaN','不能当作正间隙','实际释放','连续几何安全')),
    'cue21_checks_and_false_target_objective_flags': d['cue_READY']['synthetic_boundary_checks']=={'passed':21,'total':21} and d['cue_READY']['target_ready'] is False and d['cue_READY']['objective_loading_permitted'] is False,
    'geometry21hand273pairs602FK601query_correct': d['geometry_metadata']['hand_geometry_enabled_count']==21 and d['geometry_metadata']['hand_self_potential_pairs']+d['geometry_metadata']['hand_object_mask_compatible_pairs']==273 and d['geometry']['actual_calls']['allowed']=={'mj_kinematics':602,'mj_comPos':602,'mj_collision':601},
    'geometry_max_and_counts_document_rounding_correct':abs(d['geometry']['summary']['maximum_generated_hand_object_depth_m']*1000-15.158818)<.000001 and abs(d['geometry']['summary']['maximum_generated_hand_self_depth_m']*1000-5.884525)<.000001 and d['geometry']['summary']['sample_counts_over_original3mm']=={'hand_object':350,'hand_self':16},
    'worst_HO_time_and_pair_correct':d['geometry']['worst_generated_overlap']['time_s']==.085 and d['geometry']['worst_generated_overlap']['geom_names']==['object/cube','robot/right_finger2_link3_col'],
    'geometry_engineering_done_not_depthpass':len(d['geometry']['checks'])==7 and all(d['geometry']['checks'].values()) and d['geometry']['summary']['sampled_depth3mm_pass'] is False,
    'fit44_and_lifecycle26_reports_are_pass_independent':d['fit_raw_review']['passed']==44 and d['fit_raw_review']['engineering_review_passed'] is True and d['lifecycle_review']['all_checks_pass'] is True,
    'OS_PID_creation_absence_explicit':d['root_execution']['PID_and_creation_time']=='NOT_SAMPLED_SHORT_FOREGROUND_CPU_COMMAND' and 'PID/Creation未采样' in text,
    'cost_scopes_nested_not_added': '各时间范围嵌套' in text and '不能相加成完整研究开销' in text,
    'geometry_failure_not_global_motion_or_CHORD_impossibility':all(s in text for s in ('不等于原运动全局不可行','SuperMimic/CHORD无效','仍未接入')),
}
result={
    'schema':'full151_findings_text_independent_review:1',
    'status':'PASS_FACT_AND_SCOPE_REVIEW_WITH_WORDING_NOTES',
    'document':docpin,'sources':sources,'checks':checks,'all_checks_pass':all(checks.values()),
    'wording_notes':[
        {'id':'planned_FP32_update_not_real_execution','text_present': '149次实际FP32目标更新' in text,'recommendation':'Use 计划控制表的149次FP32目标更新 to avoid implying physical execution; same numerical value.'},
        {'id':'local_image_rendering','relative_image_present':'![full151 CPU候选数值诊断](../' in text,'recommendation':'Use absolute D:/workspace image target for Codex rendering.'},
        {'id':'pending_geometry_raw_audit','recommendation':'Link final corrected Contact review receipt, not initial31/32 harness-failed review. Current findings does not falsely claim it passed.'},
    ],
    'limits':'151knot max/counts/self peak time/first exceed time were provided by Root and are consistent with quoted document; this review did not reload arrays to recompute them. No model/physics recalculation. Geometry raw independent audit remains separate.',
    'scope':{'JSON_doc_read_hash_only':True,'arrays_models_Reader_FK_query_solver_Env_WM_physics_GPU':False,'framework_imported':any(x in sys.modules for x in ('numpy','scipy','torch','warp','mujoco')),'old_sources_written':False},
    'review_source':pin(__file__),
}
result['document_unchanged_during_review']=pin(docpath)==docpin
target=OUT/'report.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'report':pin(target),'checks':len(checks),'all_pass':all(checks.values())}))
if not all(checks.values()):raise SystemExit(2)
