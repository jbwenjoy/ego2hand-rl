"""Finalize a read-only protocol/source review; no framework or simulation."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
    p=Path(p).resolve(strict=True)
    return {'path':p.as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


proposal_path=ROOT/'proposals/current104_samewarp_physical_fd_v1/protocol.json'
correction_path=proposal_path.parent/'CORRECTION_v1.json'
p=read(proposal_path);c=read(correction_path);d=read(OUT/'directions_review.json')
checks={}
def check(name,v):checks[name]=bool(v)
check('reviewed_exact_proposal60f3',pin(proposal_path)['sha256']=='60f390eb8b6e395046ee041f0af58331647551009ea1e872ea07ee413682eded')
check('mandatory_correction_exact38f2',pin(correction_path)['sha256']=='38f28982eb0d9a52836d9fad71a0f4b873060ae867f2997227dadec496ff9978'
      and c['applies_to']['sha256']==pin(proposal_path)['sha256'])
check('all_proposal_authority_sources_current',all(pin(v['path'])['sha256']==v['sha256'] for v in p['source_authorities'].values()))
check('all_correction_source_evidence_current',all(pin(v['path'])['sha256']==v['sha256'] for v in c['source_evidence'].values()))
check('no_execution_authorization_or_actual_new_results',p['GPU_run_authorized'] is False and p['training_authorized'] is False
      and p['automatic_retry'] is False and p['status']=='DRAFT_NOT_AUTHORIZED_NOT_IMPLEMENTED_NOT_RUN' and c['GPU_run_authorized'] is False)
check('fixed16states_560branches_2480substeps_consistent',p['selection']['trajectory_order']==['A0','A1L','P_ONLY','A2']
      and p['selection']['phase_k']==[0,4,8,12] and p['selection']['state_count']==16
      and p['counts_maximum']['all_branches']==16*(3+8*2*2)==560
      and p['counts_maximum']['total_physics_substeps']==560*4+4*60==2480
      and p['counts_maximum']['total_postinterval_forwards']==560+4*15==620)
check('independent_saved_direction40_checks_passed',d['status']=='PASS_SAVED_PREPARATION_ONLY' and d['passed']==d['total']==40)
check('all16actual_direction_ranks8_not_complete20Jacobian',len(d['rows'])==16 and all(r['direction_rank']==8 and 18<=r['free_coordinate_count']<=20 for r in d['rows']))
check('no_framework_or_production_import_by_reviewer',not any(x in sys.modules for x in ('torch','mujoco','warp','mujoco_warp','rough_prefix')))
root_native=Path('D:/workspace/dex-supermimic-research/.venv/Lib/site-packages/mujoco_warp/_src')
extra_sources={name:pin(root_native/name) for name in ('support.py','types.py','sensor.py','forward.py')}
extra_sources['mjtype.h']=pin(Path('D:/workspace/dex-supermimic-research/.venv/Lib/site-packages/mujoco/include/mujoco/mujoco/mjtype.h'))
report={
    'schema':'same_Warp_physical_FD_independent_static_review:1',
    'status':'DESIGN_ACCEPTABLE_WITH_MANDATORY_CORRECTION_AND_V2_CLARIFICATIONS_NOT_EXECUTION_READY' if all(checks.values()) else 'BINDING_FAILED_NOT_READY',
    'checks_kind':'Read-only source/proposal/byte bindings plus separate saved direction arithmetic; not a physics qualification',
    'checks':checks,'passed':sum(checks.values()),'total':len(checks),
    'proposal':pin(proposal_path),'mandatory_correction':pin(correction_path),'direction_review':pin(OUT/'directions_review.json'),
    'source_evidence_additional_installed_API':extra_sources,
    'v1_execution_blockers':[
        {'issue':'Strict104 WM loader cannot run in already-CUDA-initialized physics process','resolution':'Correction requires physicsworker termination then separate CPU-only WM process','state':'RESOLVED_IN_DESIGN_SIDECAR_IMPLEMENTATION_PENDING'},
        {'issue':'startup_forwards=4 undercounts actual native Manager+rough reset forwards','resolution':'Correction withdraws4; at least8 plus measured constructor/graphwarmups,2480/620 unchanged','state':'RESOLVED_IN_DESIGN_SIDECAR_IMPLEMENTATION_PENDING'}],
    'remaining_before_execution_freeze':[
        'Integrate mandatory corrections into future protocol/implementation; v1 alone cannot generate execution config',
        'Include HISTORY in explicit integration ABI and assert nplugin0 because installed Warp PLUGIN state unsupported',
        'Pin complete snapshot-sidecar/closure/alias inventories and independent exact restore checks; external capacity ledger must never restore',
        'Fix tiny/zero masked-norm fail/omit wording; current fixed128 artifact unaffected',
        'Predeclare primary AD dot effective_FP32_direction; retain nominal AD and midpoint drift without favorable selection',
        'Nominal self15 truth fixed to freshdonor reward capture; all3baseline label consistency/prevalence separately retained',
        'Separate pinned GPU freshdata producer and CPU WM consumer, negative tests, bounded owner process lifecycle and explicit authorization'],
    'unchanged_threshold_assessment':'Legacy replay/signal numbers are predeclared engineering rules, not transferred qualification. No source/result/tolerance change made by reviewer.',
    'actual_physics_or_WM_outcome_assessed':False,
    'reviewer_execution':{'Env_or_Reader_constructed':False,'framework_imported':False,'WM_forward_calls':0,'physics_or_geometry_queries':0,'GPU_used':False,'production_implementation_written':False},
    'documentation':pin(OUT/'README.md'),'reviewer_source':pin(__file__)}
target=OUT/'report.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'report':pin(target)},indent=2))
raise SystemExit(0 if all(checks.values()) else 2)
