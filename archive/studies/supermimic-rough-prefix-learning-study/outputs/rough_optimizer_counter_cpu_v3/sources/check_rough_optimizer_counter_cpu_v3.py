"""Strict lazy-zero counter contract only; no environment or training."""
from pathlib import Path
from collections import defaultdict
import copy,difflib,hashlib,importlib.util,json,os,sys
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/rough_optimizer_counter_cpu_v3'
OLD=ROOT/'scripts/audit_rough_checkpoint_cpu_v2.py';NEW=ROOT/'scripts/audit_rough_checkpoint_cpu_v3.py'
OLD_SHA='f230ad395703412d6386fd72b04d98bb44e61e2f9c4400d6375b49f0f3e305a8'
NEW_SHA='7a6a4e21d24f5acad0eccc6908bef03fd7e1c4acaec7196edf296cc75772850f'
def pin(p):return {'path':str(Path(p).resolve()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def main():
 OUT.mkdir(exist_ok=False);snap=OUT/'sources';snap.mkdir()
 for p in (OLD,NEW,Path(__file__)):(snap/p.name).write_bytes(p.read_bytes())
 (OUT/'exact_v2_to_v3.diff').write_text(''.join(difflib.unified_diff(OLD.read_text().splitlines(True),NEW.read_text().splitlines(True),fromfile=OLD.name,tofile=NEW.name)))
 assert pin(OLD)['sha256']==OLD_SHA and pin(NEW)['sha256']==NEW_SHA
 spec=importlib.util.spec_from_file_location('frozen_rough_health_v3_CPU',NEW);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 expected={'wm':200,'fail_head':200,'actor':0,'critic':0};lazy=defaultdict(int)
 # Mirror actual native post-optimizer hook insertion, not a full-zero fixture.
 for _ in range(200):lazy['wm']+=1;lazy['fail_head']+=1
 observed=dict(lazy);frozen_observed=copy.deepcopy(observed);frozen_expected=copy.deepcopy(expected);checks={};cases=[]
 def case(name,value,wanted=True,exp=None):
  exp=expected if exp is None else exp;result=m.optimizer_counts_match(value,exp);checks[name]=result is wanted;cases.append({'name':name,'observed':value,'expected':exp,'match':result,'wanted':wanted})
 case('real_native_lazy_WM_only_missing_two_zeros',observed)
 case('explicit_four_counts',expected.copy())
 case('one_expectedzero_missing',{'wm':200,'fail_head':200,'actor':0})
 case('all_expectedzero_absent',{},exp={'wm':0,'fail_head':0,'actor':0,'critic':0})
 case('complete_positive_final',{'wm':4000,'fail_head':4000,'actor':1800,'critic':7200},exp={'wm':4000,'fail_head':4000,'actor':1800,'critic':7200})
 case('reject_boolFalse_as_zero',{'wm':200,'fail_head':200,'actor':False},False)
 case('reject_boolTrue_as_positive',{'wm':True,'fail_head':200},False)
 case('reject_float_zero',{'wm':200,'fail_head':200,'actor':0.},False)
 case('reject_integral_float_positive',{'wm':200.,'fail_head':200},False)
 case('reject_negative',{'wm':200,'fail_head':200,'actor':-1},False)
 case('reject_unknown_even_zero',{'wm':200,'fail_head':200,'ppo':0},False)
 case('reject_missing_expected_positive',{'fail_head':200},False)
 case('reject_missing_all_positive',{},False)
 case('reject_wrong_positive',{'wm':199,'fail_head':200},False)
 case('reject_nonzero_expectedzero',{'wm':200,'fail_head':200,'actor':1},False)
 case('reject_nonzero_other_expectedzero',{'wm':200,'fail_head':200,'critic':1},False)
 case('reject_none',None,False)
 case('reject_list',[],False)
 case('reject_expected_missing_contract_key',observed,False,exp={'wm':200,'fail_head':200,'actor':0})
 case('reject_expected_bool',observed,False,exp={'wm':200,'fail_head':200,'actor':False,'critic':0})
 case('reject_expected_float',observed,False,exp={'wm':200,'fail_head':200,'actor':0.,'critic':0})
 checks['inputs_are_not_mutated']=observed==frozen_observed and expected==frozen_expected and set(lazy)=={'wm','fail_head'}
 a1meta=ROOT/'outputs/smoke_a1l_v2/train/model_final.meta.json';actual=json.loads(a1meta.read_text());case('actual_existing_A1L_full_counter_payload',actual['actual_optimizer_steps'],exp={'wm':20,'fail_head':20,'actor':8,'critic':32})
 cfg={'warmup_steps':64,'wm_window':8,'utd':.4};prefix=m.expected_collect_steps(cfg,100)
 checks['unchanged_first100_budget_101calls_40192transitions']=len(prefix)==101 and sum(prefix)==314 and 128*sum(prefix)==40192
 # The sole changed old expression must be the hook dictionary comparison.
 text=NEW.read_text();start=text.index('def optimizer_counts_match(');end=text.index('def expected_collect_steps(',start)
 restored=text[:start]+text[end:]
 restored=restored.replace("optimizer_counts_match(meta.get('actual_optimizer_steps'),counts) and (not isfinal or optimizer_counts_match(result.get('actual_optimizer_steps'),counts))", "meta.get('actual_optimizer_steps')==counts and (not isfinal or counts==result.get('actual_optimizer_steps'))")
 # Insertion is after a pre-existing blank separator; tolerate only its exact
 # extra separator line, without normalizing executable text or other checks.
 checks['all_other_v2_source_preserved']=restored.replace('return canonical(saved_config)==canonical(effective)\n\n\n','return canonical(saved_config)==canonical(effective)\n\n')==OLD.read_text()
 checks['Adam_empty_and_positive_state_checks_unchanged']="steps==([float(counts[n])] if counts[n] else [])" in text
 checks['old_and_frozen_new_source_bytes_preserved']=pin(OLD)['sha256']==OLD_SHA and pin(NEW)['sha256']==NEW_SHA
 checks['CPU_no_CUDA_no_native_environment']=not m.torch.cuda.is_initialized() and not any(k.split('.')[0] in ('warp','mujoco','mujoco_warp','mjlab','dex_supermimic') for k in sys.modules)
 report={'schema':'rough_health_lazy_zero_counter_CPU_contract:3','status':'PASS_STRICT_LAZY_ZERO_COMPATIBILITY' if all(checks.values()) else 'FAIL','passed':sum(checks.values()),'total':len(checks),'checks':checks,'cases':cases,'source_pins':{'original_v2':pin(OLD),'new_v3':pin(NEW),'checker':pin(__file__),'actual_A1L_meta_readonly':pin(a1meta)},'scope':'New auditor source frozen before this small CPU helper run. Only absent counters with independently expected zero may be treated as zero. No budget/Adam state/threshold/physical/training/controller/source changes. No actual primary checkpoint exists or was graded here. Existing v2/full smoke audits retained.','CUDA_initialized':m.torch.cuda.is_initialized(),'Env_or_Runner_constructed':False,'physics_steps':0,'optimizer_steps':0}
 (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'failed':[k for k,v in checks.items() if not v],'report':pin(OUT/'report.json')}));return 0 if all(checks.values()) else 1
if __name__=='__main__':raise SystemExit(main())
