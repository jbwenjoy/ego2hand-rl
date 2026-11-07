"""Small pure plan/metadata and CLI-help contracts, never native execution."""
import ast,copy,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_native.io import pin,read
from full151_native.metadata import write
from full151_gradient import serial_probe105_calibration_v1 as worker

OUT=ROOT/'outputs/full151105_calibration_response_preparation_CPU_v1'
SOURCES={'worker':Path(worker.__file__),'entry':ROOT/'scripts/execute_full151_calibration_response_v1.py','supervisor':ROOT/'scripts/run_full151_calibration_response_v1.py','checker':Path(__file__),
 'transaction105':ROOT/'src/full151_gradient/transaction105.py','io':ROOT/'src/full151_native/io.py','metadata':ROOT/'src/full151_native/metadata.py'}

def main():
    if OUT.exists():raise ValueError('Fresh source-contract preparation only')
    OUT.mkdir();sources={k:pin(p) for k,p in SOURCES.items()};write(OUT/'source_identity_before_checks.json',sources)
    for key,path in SOURCES.items():(OUT/(key+'_source.py')).write_bytes(path.read_bytes())
    checks={}
    def check(name,value):
        if not value:raise AssertionError(name)
        checks[name]=True
    def rejects(fn):
        try:fn()
        except (ValueError,worker.InventoryReviewRequired):return True
        return False
    discovery=worker.execution_plan('inventory_discovery_all');full=worker.execution_plan('serial_full')
    check('new_exact8_labels_and_list_phases',worker.PHASES==[24,74,124,148] and worker.SNAPSHOT_LABELS==tuple(f'{p}/{k}' for p in ('A0','A2') for k in worker.PHASES))
    check('discovery1200_8captures_no_scheduled_branch',discovery['episode_substeps']==1200 and discovery['snapshot_count']==8 and discovery['branch_count']==0 and len(discovery['branch_slots'])==152 and not any(r['scheduled'] for r in discovery['branch_slots']))
    rows=full['branch_slots']
    check('full1808_24baseline128direction_order',full['episode_substeps']==1808 and full['branch_count']==152 and sum(r['type']=='baseline' for r in rows)==24 and sum(r['type']=='direction' for r in rows)==128 and all(rows[s*19+b]['baseline_repeat']==b for s in range(8) for b in range(3)))
    check('same4joint2epsilon_plusminus_and_no_terminal150',[(r['joint_index'],r['epsilon_rad'],r['sign']) for r in rows[3:19]]==[(j,e,sign) for e in (.002,.001) for j in (0,4,8,12) for sign in (-1,1)] and full['next_ticks']==[100,300,500,596] and all(r['status']=='NOT_RUN' and not r['restorequalified'] for r in rows))
    proposal=read(ROOT/'protocols/full151105_fresh_calibration_response_proposal_v1.json')
    check('actual_new_proposal_fields_not_old_executionstatus_guess',proposal['schema']=='full151105_fresh_calibration_response_physical_proposal:1' and proposal['role']=='fresh_response_calibration' and proposal['status']=='FROZEN_RESPONSE_ACQUISITION_DESIGN_NOT_SELF_AUTHORIZATION' and proposal['snapshot_phases']==worker.PHASES and proposal['scalar_contract']==worker.SCALAR_CONTRACT and proposal['GPU_authorized_by_this_proposal'] is False and proposal['training_authorized'] is False)
    from full151_gradient import transaction105
    fixture={'input_bundle':{'path':'D:/synthetic/nonexistent_bundle.json','sha256':'0'*64}}
    review={'schema':'full151105_foreign_inventory_per_snapshot_source_review:2','worker':pin(worker.__file__),'transaction':pin(transaction105.__file__),'input_bundle':fixture['input_bundle'],
      'source_reviewed_closure_globals':True,'approved_probe_modes':['serial_full'],'allowed_snapshot_labels':list(worker.SNAPSHOT_LABELS),'phase0_only':False,'conditional_engineering_only':True,'whole_graph_hidden_state_qualified':False,
      'per_snapshot':{label:{'inventory':{'description':{'path':'D:/synthetic/description.json','sha256':'1'*64},'topology':{'path':'D:/synthetic/topology.json','sha256':'2'*64}},'allowed_identity_only':[{'path':'native_entry[0].value[0]','type':'synthetic.ReadOnlyHandle','reason':'Synthetic fixture only; no actual inventory evidence.'}],'source_reviewed_closure_globals':True} for label in worker.SNAPSHOT_LABELS}}
    check('new_review_header_exact8_synthetic_only',set(worker._review_header(review,fixture))==set(worker.SNAPSHOT_LABELS))
    old=copy.deepcopy(review);old['allowed_snapshot_labels']=['A0/0','A0/49','A0/99','A0/149','A2/0','A2/49','A2/99','A2/149']
    missing=copy.deepcopy(review);missing['per_snapshot'].pop('A0/24')
    wrong=copy.deepcopy(review);wrong['worker']=pin(worker.PARENT_WORKER)
    check('oldlabels_oldworker_missinglabel_rejected',all(rejects(lambda v=v:worker._review_header(v,fixture)) for v in (old,missing,wrong)))
    for alias in ('entry','supervisor'):
        path=SOURCES[alias];ast.parse(path.read_text(encoding='utf-8'))
        result=subprocess.run([sys.executable,str(path),'--help'],text=True,capture_output=True,timeout=10,env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1','PYTHONDONTWRITEBYTECODE':'1'})
        (OUT/(alias+'_help.stdout.txt')).write_text(result.stdout,encoding='utf-8');(OUT/(alias+'_help.stderr.txt')).write_text(result.stderr,encoding='utf-8')
        check(alias+'_CLI_help_without_config_or_execution',result.returncode==0 and '--config' in result.stdout and '--output' in result.stdout and '--run' in result.stdout)
    check('no_native_framework_import_in_pure_contract',not any(k in sys.modules for k in ('torch','warp','mujoco','mujoco_warp')))
    report={'schema':'full151105_calibration_response_CPU_contract:1','status':'PASS_PURE_CALIBRATION_PLAN_METADATA_NO_NATIVE','checks':checks,'passed':len(checks),'total':len(checks),'source_pins':sources,
      'actual_proposal':pin(ROOT/'protocols/full151105_fresh_calibration_response_proposal_v1.json'),'actual_new_inventory_or_response_runs':0,'native_framework_Env_model_MJB_FK_physics_GPU_calls':0,
      'qualification_scope':'Source identity and pure plans/metadata only. No fresh discovery, native restore, baseline, response, sealing or physical-gradient qualification. Draft changes invalidate these source identities.'}
    write(OUT/'report.json',report);write(OUT/'READY.json',{'schema':'full151105_calibration_response_CPU_READY:1','source_pins':sources,'cpu_report':pin(OUT/'report.json'),'cpu_checks':{'passed':len(checks),'total':len(checks)},'GPU_or_discovery_authorization':False,'source_stage':'DRAFT_IDENTITY_SNAPSHOT_NOT_SCIENTIFIC_EXECUTION_FREEZE'})
    print(json.dumps({'status':report['status'],'passed':len(checks),'total':len(checks),'READY':pin(OUT/'READY.json')}))

if __name__=='__main__':main()
