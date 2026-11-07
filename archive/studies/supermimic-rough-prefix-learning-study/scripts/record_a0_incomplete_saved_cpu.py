"""Saved-only classification of the retained zero-step native A0 v1 failure."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin=lambda p:{'path':str(Path(p).resolve()),'sha256':sha(p)}
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'outputs/native_a0_v1_incomplete_saved_review_v1');a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'review_source.py').write_bytes(Path(__file__).read_bytes());raw=ROOT/'outputs/native_a0_v1';logs=ROOT/'outputs/native_a0_v1_watchdog';p=read(raw/'config.json');runtime=read(raw/'runtime.json');failure=read(raw/'failure.json');state=read(logs/'state.json');post=read(logs/'post_terminal_CIM.json');checks=[]
    def check(name,ok):checks.append({'name':name,'passed':bool(ok)})
    check('failed_before_probe0steps',runtime['status']=='failed_engineering_partial' and runtime['physics_steps']==0 and runtime['optimizer_updates']==0 and not (raw/'raw_states.npz').exists() and not (raw/'evaluation.json').exists())
    check('actual_installed_obs_API_assertion_retained','env_ids is only meaningful with update_history=True' in failure['error'] and 'runtime.py' in failure['traceback'])
    check('actual43088_exit1_no_timeout_ownedPIDs_absent',runtime['pid']==43088 and state['launcher_pid']==19912 and state['actual_process_OS_observation']['status']=='exact_OS_identity_observed' and state['exit_code']==1 and state['timeout'] is False and post['actual_CIM_rows']==[])
    for k,q in p['source_pins'].items():check('pin__'+k,sha(q['path'])==q['sha256'])
    source=read(p['input_bundle']['path']);context=np.load(source['context']['path'],allow_pickle=False);expected=np.load(raw/'expected_context_fields.npz',allow_pickle=False);rb=np.load(raw/'actual_context_readbacks.npz',allow_pickle=False)
    for key in ('body_mass','body_inertia','body_iquat','geom_size','geom_pos','geom_friction','actuator_gainprm','actuator_biasprm'):
        check('completed_actual8__'+key,np.array_equal(expected[key],context[key][0]) and all(np.array_equal(rb[stage+'__'+key],np.broadcast_to(context[key][0],rb[stage+'__'+key].shape)) for stage in ('after_copy','after_set_const_0')))
    result={'status':'INCOMPLETE_ENGINEERING_0STEPS_QUALITY_NOT_ASSESSED','independent_saved_checks_pass':all(c['passed'] for c in checks),'checks':checks,'check_count':len(checks),'quality_pass':None,'quality_label':'NOT_ASSESSED_NOT_TASK_FAIL','native_probe_steps':0,'new_physics_steps':0,'source_v1_immutable':True,'failure_stage':'After actual8 restore/first reset state write, before probe reader/zero-residual intervals','original_actual8_stage':'PASS_SAVED_READBACKS','original_startup_state_values':'NOT_SAVED_INDEPENDENT_INIT_UNKNOWN','original_exit_code':state['exit_code'],'original_worker_wall_seconds':runtime.get('wall_seconds'),'original_watchdog_slot_seconds':state.get('elapsed_s'),'input_pins':{'config':pin(raw/'config.json'),'runtime':pin(raw/'runtime.json'),'failure':pin(raw/'failure.json'),'actual8':pin(raw/'actual_context_readbacks.npz'),'watchdog':pin(logs/'state.json'),'postterminal':pin(logs/'post_terminal_CIM.json')},'review_source':pin(__file__),'scope':'Preserved failed implementation attempt, not a dynamic quality result or a reason to repeat the same directory.'};(out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'saved_checks':result['check_count'],'all_pass':result['independent_saved_checks_pass'],'quality_pass':None},indent=2));raise SystemExit(not result['independent_saved_checks_pass'])
if __name__=='__main__':main()
