"""CPU preparation only: clone fixed physics, bind the nine audited final103s."""
from pathlib import Path
import ast
import copy
import difflib
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('D:/workspace/supermimic-c2-020-portfolio-study')
TRAIN=Path('D:/workspace/supermimic-c2-training-study')
MAIN=Path('D:/workspace/supermimic-chord-study')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
def write(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def replace(text,a,b):
    assert a in text,a
    return text.replace(a,b,1)

def main():
    for d in ('src','scripts','configs/models','outputs/preparation_v1'):(ROOT/d).mkdir(parents=True,exist_ok=True)
    shutil.copyfile(OLD/'src/target_inset.py',ROOT/'src/target_inset.py')
    shutil.copyfile(OLD/'control_log_schema_v1.json',ROOT/'control_log_schema_v1.json')
    worker=(OLD/'scripts/evaluate_c2_portfolio.py').read_text(encoding='utf-8')
    start=worker.index('    proposal=read(')
    end=worker.index('    for model in p["models"]:',start)
    worker=worker[:start]+'''    index=read(p["source_pins"]["audited_model_index"]["path"])
    schedule=read(p["source_pins"]["frozen_schedule"]["path"])
    require(index["status"]=="passed","Nine-final training audit not passed")
    require({(m["method"],m["training_seed"]) for m in p["models"]}=={(a,s) for a in ("A1L","P_ONLY","A2") for s in (4,5,6)},"Wrong prospective cohort")
    require(len(index["models"])==9 and len(schedule["evaluation_jobs"])==27,"Authority counts changed")
    byid={m["model_id"]:m for m in index["models"]}
    for model in p["models"]:
        authority=byid[model["model_id"]]
        require(model["method"]==authority["trained_method"] and model["training_seed"]==authority["training_seed"] and model["checkpoint"]==authority["checkpoint"] and model["checkpoint_sha256"]==authority["checkpoint_sha256"],"Audited final identity changed")
        require(sha(model["contract_file"]["path"])==model["contract_file"]["sha256"],"Checkpoint contract changed")
        require(read(model["contract_file"]["path"])=={k:v for k,v in model.items() if k!="contract_file"},"Contract/plan model mismatch")
        for name,evidence in model["independent_inputs"].items():
            require(sha(evidence["path"])==evidence["sha256"],"Model evidence changed: "+name)
    for job,original in zip(p["jobs"],schedule["evaluation_jobs"]):
        require(all(job[k]==v for k,v in original.items()),"Prospectively frozen evaluation order changed")
''' + worker[end:]
    worker=replace(worker,'    for job,original in zip(p["jobs"],proposal["jobs"]):\n        require(all(job[k]==v for k,v in original.items()),"Original proposal job/order changed")','    for job in p["jobs"]:')
    worker=replace(worker,'"--seed","0","--device","cuda:0","--output",job["output_dir"]]','"--seed","0","--device","cuda:0","--output",job["output_dir"],\n                  "--checkpoint-contract",model["contract_file"]["path"]]')
    worker=worker.replace('"schema":"C2_target_inset_zero_shot:1"','"schema":"C2_trained_confirmation:1","cohort":"fresh_training_seed4_5_6_common_C2_020",\n                     "training_checkpoint_format":103,"checkpoint_contract_sha256":model["contract_file"]["sha256"],\n                     "training_controller_evidence":model["training_run"]')
    worker=worker.replace('"checkpoint_sha256":model["checkpoint_sha256"]}', '"checkpoint_sha256":model["checkpoint_sha256"],\n               "training_checkpoint_format":103,"checkpoint_contract_sha256":model["contract_file"]["sha256"]}')
    worker=replace(worker,'model_id=model["model_id"],trained_method=model["method"],training_seed=model["training_seed"])',
        'model_id=model["model_id"],trained_method=model["method"],training_seed=model["training_seed"],\n            training_checkpoint_format=103,checkpoint_contract_sha256=model["contract_file"]["sha256"])')
    (ROOT/'scripts/evaluate_c2_portfolio.py').write_text(worker,encoding='utf-8')
    queue=(OLD/'scripts/run_c2_portfolio_queue.py').read_text(encoding='utf-8')
    queue=queue.replace('configs/c2-portfolio-v1.json','configs/c2-confirmation-evaluation-v1.json')
    anchor='    for record in (condition,runtime):'
    queue=replace(queue,anchor,'''    worker.require(meta["checkpoint_format_version"]==103 and meta["checkpoint_contract_sha256"]==model["contract_file"]["sha256"],"Original103 contract missing")
    worker.require(meta["checkpoint_controller"]["trained_method"]==model["method"] and meta["checkpoint_controller"]["training_seed"]==model["training_seed"],"Actual103 controller identity mismatch")
    worker.require(meta["runtime_reference_signature"]==model["expected"]["reference_signature"],"Actual runtime reference signature mismatch")
    worker.require(condition["schema"]=="C2_trained_confirmation:1" and condition["training_checkpoint_format"]==103,"Wrong confirmation condition schema")
    for record in (condition,runtime):''')
    queue=replace(queue,'    worker.require(capacity["status"]=="OK"',
        '    worker.require(capacity["training_checkpoint_format"]==103 and capacity["checkpoint_contract_sha256"]==model["contract_file"]["sha256"],"Capacity103 contract mismatch")\n    worker.require(capacity["status"]=="OK"')
    (ROOT/'scripts/run_c2_portfolio_queue.py').write_text(queue,encoding='utf-8')
    evaluator=(MAIN/'scripts/evaluate_pen_checkpoint.py').read_text(encoding='utf-8')
    evaluator=replace(evaluator,'No checkpoint means A0 zero residual. A checkpoint evaluates its deterministic\nactor (A1). Both cases use the same play=True scene, reference, initial state,',
        'This copy explicitly accepts only the frozen original format103 trained C2\ncheckpoints. Its deterministic residual actor uses the same play=True scene, reference, initial state,')
    evaluator=replace(evaluator,'STUDY_ROOT = Path(__file__).resolve().parents[1]','STUDY_ROOT = Path("D:/workspace/supermimic-chord-study")\nsys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))\nfrom checkpoint103_contract import validate_checkpoint\nsys.path.insert(0, str(STUDY_ROOT / "scripts"))')
    evaluator=replace(evaluator,'    parser.add_argument("--checkpoint", type=Path)','    parser.add_argument("--checkpoint", type=Path, required=True)\n    parser.add_argument("--checkpoint-contract", type=Path, required=True)')
    a=evaluator.index('        checkpoint = torch.load(')
    b=evaluator.index('        state = checkpoint["modules"]["actor"]',a)
    evaluator=evaluator[:a]+'''        model_contract=json.loads(args.checkpoint_contract.read_text(encoding="utf-8-sig"))
        if args.checkpoint.resolve()!=Path(model_contract["checkpoint"]).resolve() or sha256(args.checkpoint)!=model_contract["checkpoint_sha256"]:
            raise ValueError("Original103 checkpoint path/hash differs from independently frozen contract")
        checkpoint = torch.load(args.checkpoint.resolve(strict=True), map_location=args.device, weights_only=False)
        checkpoint_descriptor=validate_checkpoint(checkpoint,model_contract)
        if checkpoint.get("joint_names") != list(RIGHT_FINGER_JOINT_NAMES):
            raise ValueError("runtime native joint order differs from103 contract")
''' + evaluator[b:]
    anchor='        rows = []\n'
    evaluator=replace(evaluator,anchor,'''        # Compare immutable reference arrays with the native training signature;
        # this reads reference data, never predicted/live future physical state.
        import hashlib
        reference_hash=hashlib.sha256()
        for field in ("joint_pos","object_pose","link_pos"):
            reference_hash.update(motion._frames[field].cpu().numpy().tobytes())
        reference_hash.update(motion.group.trajectory_end.cpu().numpy().tobytes())
        runtime_reference_signature=reference_hash.hexdigest()
        if runtime_reference_signature!=checkpoint["reference_signature"]:
            raise ValueError("Runtime fixed reference differs from trained103 reference signature")
        action_term=env.action_manager.get_term("joint_pos")
        lower,upper=action_term.target_position_limits
        if cpu(lower)[0].tolist()!=checkpoint["c2_controller"]["target_lower_rad"] or cpu(upper)[0].tolist()!=checkpoint["c2_controller"]["target_upper_rad"]:
            raise ValueError("Runtime outer hard/ctrl bounds differ from trained103 controller")
        rows = []
''')
    evaluator=replace(evaluator,'"checkpoint_iteration": checkpoint.get("iteration") if checkpoint else None,','"checkpoint_iteration": checkpoint.get("iteration") if checkpoint else None,\n                "checkpoint_format_version": checkpoint["format_version"],\n                "checkpoint_contract_sha256": sha256(args.checkpoint_contract),\n                "checkpoint103_validation": checkpoint_descriptor,\n                "checkpoint_controller": checkpoint_descriptor["controller"],\n                "runtime_reference_signature": runtime_reference_signature,')
    evaluator=evaluator.replace('"A1 deterministic actor" if actor is not None else "A0 zero residual"','"C2-trained "+checkpoint_descriptor["trained_method"]+" deterministic residual actor"')
    (ROOT/'scripts/evaluate_pen_checkpoint_103.py').write_text(evaluator,encoding='utf-8')
    # Bounds are independently computed from the frozen original MJB, cast to
    # FP32 exactly as the native action contract, before the one20mrad inset.
    import mujoco
    import numpy as np
    indexpath=TRAIN/'outputs/confirmation_training_audit_v1/model_index.json'
    index=read(indexpath);schedulepath=TRAIN/'protocols/confirmatory_schedule_draft_v1.json';schedule=read(schedulepath)
    proto=TRAIN/'protocols/final_execution_protocol_v1.json'
    names=[f'right_finger{f}_joint{j}' for f in range(1,6) for j in range(1,5)]
    models=[]
    for item in index['models']:
        run=Path(item['checkpoint']).parents[2];cfg=read(item['input_config']['path']);inventorypath=run/'source/inventory.json';inventory=read(inventorypath)
        archive=inventory['c2_controller'];m=mujoco.MjModel.from_binary_path(str(run/'compiled/nominal_model.mjb'))
        jids=np.asarray([m.joint('robot/'+n).id for n in names]);aids=np.asarray([np.flatnonzero(m.actuator_trnid[:,0]==j).item() for j in jids])
        lo=np.maximum(m.jnt_range[jids,0],m.actuator_ctrlrange[aids,0]).astype(np.float32).tolist()
        hi=np.minimum(m.jnt_range[jids,1],m.actuator_ctrlrange[aids,1]).astype(np.float32).tolist()
        expectedcontroller={'kind':'C2_TARGET_INSET','margin_rad':.020,'format_version':103,'exact_resume_supported':False,
          'trained_method':item['trained_method'],'base_method':item['trained_method'],'training_seed':item['training_seed'],
          'run_role':'primary_matched_arm','config_sha256':item['input_config']['sha256'],
          'paired_protocol_sha256':inventory['paired_protocol']['sha256'],'study_protocol_sha256':sha(proto),
          'native_source_inventory_sha256':sha(inventorypath),'adapter_sha256':archive['files']['adapter']['sha256'],
          'pure_projection_sha256':archive['files']['pure_projection']['sha256'],'entry_sha256':archive['files']['entry']['sha256'],
          'objective':cfg.get('contact_objective'),'target_lower_rad':lo,'target_upper_rad':hi}
        model={'model_id':item['model_id'],'method':item['trained_method'],'training_seed':item['training_seed'],
          'training_run':str(run),'checkpoint':item['checkpoint'],'checkpoint_sha256':item['checkpoint_sha256'],
          'checkpoint_selection':'new_run_final2000_only','checkpoint_format':103,
          'independent_inputs':{'input_config':item['input_config'],'training_audit':item['independent_audit'],
             'native_config':pin(run/'compiled/effective_training_config.json'),'source_inventory':pin(inventorypath),
             'nominal_model':pin(run/'compiled/nominal_model.mjb'),'training_protocol':pin(proto)},
          'expected':{'native_config':read(run/'compiled/effective_training_config.json'),
             'reference_signature':item['reference_signature'],'joint_names':names,'controller':expectedcontroller}}
        contract=ROOT/'configs/models'/f'{item["model_id"]}.json';write(contract,model);model['contract_file']=pin(contract);models.append(model)
    oldplan=read(OLD/'configs/c2-portfolio-v1.json');p=copy.deepcopy(oldplan)
    p.update(schema='C2_trained_confirmation_evaluation_plan:1',status='CPU_prepared_GPU_not_authorized',experiment='C2_020_fresh_training_confirmation',
      output_root=str(ROOT/'outputs/confirmation_evaluation_v1'),models=models,blocks=schedule.get('evaluation_blocks',[]),
      cohort='fresh_training_seed4_5_6_common_C2_020',training_seeds=[4,5,6],separate_cohort=True,
      purpose='Prospective frozen final2000 confirmation; original103 loading, not zero-shot transfer',
      margin_selection='Frozen C2_020 training and evaluation protocol; no new dose selection',
      comparison_scope='Trainseed N3 per method, process repeats nested; no pooling with old seed1/2/3',
      scope='Evaluation only: same controller as training, no new policy/WM learning, no C1 or dynamics guarantee')
    # Keep the complete already-frozen native/helper pins, replace only new
    # entrypoints, and remove old cohort-specific proposal/analysis pins.
    remove=[k for k in p['source_pins'] if any(x in k for x in ('portfolio_proposal','analysis','cpu_contract'))]
    for k in remove:p['source_pins'].pop(k)
    for key,path in {'worker':ROOT/'scripts/evaluate_c2_portfolio.py','queue':ROOT/'scripts/run_c2_portfolio_queue.py',
      'pure_projection':ROOT/'src/target_inset.py','evaluator':ROOT/'scripts/evaluate_pen_checkpoint_103.py',
      'checkpoint103_contract':ROOT/'src/checkpoint103_contract.py','control_log_schema':ROOT/'control_log_schema_v1.json',
      'original_format3_evaluator':MAIN/'scripts/evaluate_pen_checkpoint.py','audited_model_index':indexpath,
      'frozen_schedule':schedulepath,'prospective_training_protocol':proto,'training_queue_terminal':Path(index['queue_state']['path'])}.items():p['source_pins'][key]=pin(path)
    planpath=ROOT/'configs/c2-confirmation-evaluation-v1.json';p['jobs']=[]
    for original in schedule['evaluation_jobs']:
        job=copy.deepcopy(original);model=next(m for m in models if m['model_id']==job['model_id']);out=Path(p['output_root'])/job['job_id']
        job.update(repeat_id=job['job_id'],margin_rad=.020,margin_mrad=20,condition_id='C2_TRAINED_TARGET_INSET_020mrad',
          output_dir=str(out),wrapper_metadata_dir=str(out)+'_wrapper')
        job['original_evaluator_args']=['--reference',p['reference']['path'],'--checkpoint',model['checkpoint'],'--seed','0',
          '--device','cuda:0','--output',str(out),'--checkpoint-contract',model['contract_file']['path']]
        job['command']=[p['python'],'-u',p['source_pins']['worker']['path'],'--plan',str(planpath),'--job-id',job['job_id'],'--output',str(out),'--run']
        p['jobs'].append(job)
    write(planpath,p)
    diffs={}
    for name,new in [('worker','evaluate_c2_portfolio.py'),('queue','run_c2_portfolio_queue.py')]:
        before=(OLD/'scripts'/new).read_text(encoding='utf-8');after=(ROOT/'scripts'/new).read_text(encoding='utf-8')
        (ROOT/'outputs/preparation_v1'/f'{name}.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=str(OLD/'scripts'/new),tofile=str(ROOT/'scripts'/new))),encoding='utf-8')
        diffs[name]=pin(ROOT/'scripts'/new)
    before=(MAIN/'scripts/evaluate_pen_checkpoint.py').read_text(encoding='utf-8')
    (ROOT/'outputs/preparation_v1/evaluator.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),evaluator.splitlines(True),fromfile='original_format3_evaluator',tofile='explicit103_evaluator')),encoding='utf-8')
    def factory_ast(text):
        return ast.dump(next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name=='factory'),include_attributes=False)
    checks={'factory_AST_identical_to_measured_common20':factory_ast(worker)==factory_ast((OLD/'scripts/evaluate_c2_portfolio.py').read_text()),
      'pure_bytes_identical':sha(ROOT/'src/target_inset.py')==sha(OLD/'src/target_inset.py'),
      'models9':len(models)==9,'jobs27':len(p['jobs'])==27,'seed_cohort456':sorted({m['training_seed'] for m in models})==[4,5,6]}
    write(ROOT/'outputs/preparation_v1/clone_contract.json',{'checks':checks,'passed':all(checks.values()),'plan':pin(planpath),'sources':diffs,'GPU_started':False})
    print(json.dumps({'plan':pin(planpath),'checks':checks}))

if __name__=='__main__':main()
