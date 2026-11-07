"""Small supplemental pinned-C2/CLI metadata proof; no candidate or exporter run."""
from pathlib import Path
import copy,hashlib,importlib.util,json,os,subprocess,sys,tempfile,time
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import prefix_control as control
PROTOCOL=Path('D:/workspace/supermimic-observation-dynamic-wrench-study/proposals/short_dynamic_tracking_v1/protocol.json')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))

def main():
    start=time.perf_counter();out=ROOT/'outputs/export_contract_cpu_v1';out.mkdir(parents=True,exist_ok=False)
    script=ROOT/'scripts/export_prefix_curves_control_cpu.py';spec=importlib.util.spec_from_file_location('prefix_export_contract_only',script);exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
    protocol=read(PROTOCOL);mp,cp=protocol['inputs']['model'],protocol['inputs']['actual_context'];p,proof=control.load_actual_pd(mp['path'],cp['path'],model_sha256=mp['sha256'],context_sha256=cp['sha256'])
    sourcepins={str(x):sha(x) for x in [Path(__file__),script,ROOT/'src/prefix_curves.py',ROOT/'src/prefix_control.py',Path(protocol['inputs']['pure_C2']['path'])]}
    for path in sourcepins:(out/Path(path).name).write_bytes(Path(path).read_bytes())
    checks=[]
    def check(name,fn):
        try:fn();checks.append({'name':name,'pass':True})
        except Exception as exc:checks.append({'name':name,'pass':False,'error':repr(exc)})
    def truth(x):
        if not bool(x):raise AssertionError('Expected true')
    def reject(fn):
        try:fn()
        except (ValueError,TypeError):return
        raise AssertionError('Expected rejection')
    purepin=protocol['inputs']['pure_C2'];spec=importlib.util.spec_from_file_location('actual_pinned_C2_supplement',purepin['path']);pure=importlib.util.module_from_spec(spec);spec.loader.exec_module(pure)
    values=np.stack([np.full(20,-10.,np.float32),np.full(20,10.,np.float32),p.target_lower,p.target_upper,(p.target_lower+p.target_upper)/np.float32(2)])
    before=values.copy();tp=control.torch_parameters(p);oracle=pure.target_inset(torch.tensor(values),torch.tensor(p.target_lower.copy()),torch.tensor(p.target_upper.copy()),.020).numpy()
    check('real_pinned2d_C2_saturated_boundary_interior_NumPy_bitexact',lambda:truth(np.array_equal(control.c2_numpy(values,p).view(np.uint32),oracle.view(np.uint32))))
    check('real_pinned2d_C2_saturated_boundary_interior_Torch_bitexact',lambda:truth(np.array_equal(control.c2_torch(torch.tensor(values),tp).numpy().view(np.uint32),oracle.view(np.uint32))))
    check('saturation_exercised_and_input_unchanged',lambda:truth(np.count_nonzero(oracle!=values)>=80 and np.array_equal(before,values)))
    with tempfile.TemporaryDirectory(prefix='prefix_metadata_only_') as td:
        fake=Path(td)/'hash_only_not_a_candidate.bin';fake.write_bytes(b'metadata-pin-fixture-not-NPZ')
        manifest={'schema':'prefix16_geometry_candidate:1','protocol':{'path':str(PROTOCOL),'sha256':exporter.PROTOCOL_SHA},'candidate':{'path':str(fake),'sha256':sha(fake)},'q0_source':protocol['inputs']['canonical_H123_q0'],'source_identity':copy.deepcopy(protocol['source_identity']),'optimizer_success':True,'hardq0_bitexact':True,'curve_controller_qualification':'PENDING','qd_or_u_exported':False}
        check('metadata_pin_positive_fixture_only_no_NPZ_read',lambda:exporter.validate_manifest(manifest,protocol))
        mutations=[('old_window',lambda x:x['source_identity'].update(primary_pose_source_frame_range_half_open=[0,16])),('wrong_dt',lambda x:x['source_identity'].update(dt_goal_s=.01)),('wrong_candidate_hash',lambda x:x['candidate'].update(sha256='0'*64)),('self_qualified_control',lambda x:x.update(curve_controller_qualification='PASS')),('geometry_exports_u',lambda x:x.update(qd_or_u_exported=True)),('integer_not_bool_optimizer',lambda x:x.update(optimizer_success=1)),('wrong_q0_pin',lambda x:x['q0_source'].update(sha256='0'*64))]
        for name,mutate in mutations:
            wrong=copy.deepcopy(manifest);mutate(wrong);check('reject_'+name,lambda wrong=wrong:reject(lambda:exporter.validate_manifest(wrong,protocol)))
    proc=subprocess.run([sys.executable,str(script),'--help'],capture_output=True,text=True,timeout=15,env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'})
    (out/'CLI_help.txt').write_text(proc.stdout,encoding='utf-8');check('CLI_help_without_candidate_or_output',lambda:truth(proc.returncode==0 and '--candidate-manifest' in proc.stdout))
    check('source_unchanged_CPU_only',lambda:truth(all(sha(path)==value for path,value in sourcepins.items()) and not torch.cuda.is_initialized() and not torch.cuda.is_available() and torch.cuda.device_count()==0 and 'warp' not in sys.modules))
    report={'status':'PASS' if all(x['pass'] for x in checks) else 'FAIL_RETAINED','checks':checks,'count':len(checks),'source_pins':sourcepins,'actual_pinned_C2':purepin,'clamped_test_components':int(np.count_nonzero(values!=oracle)),'scope':'Temporary hash-only metadata fixture is never read as NPZ/exported/registered; no real candidate exists or was read here. No original math checker rerun or source overwrite.','real_candidate_exported':False,'optimizer_runs':0,'physics_steps':0,'GPU_used':False,'Torch_CUDA_initialized':torch.cuda.is_initialized(),'wall_s_excluding_imports':time.perf_counter()-start}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'status':report['status'],'count':len(checks),'failures':[x for x in checks if not x['pass']],'report_SHA256':sha(out/'report.json')},indent=2));raise SystemExit(report['status']!='PASS')

if __name__=='__main__':main()
