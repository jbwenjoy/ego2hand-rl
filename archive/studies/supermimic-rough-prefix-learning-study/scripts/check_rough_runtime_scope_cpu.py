"""Small additional CPU rejection/finally checks; no physics or training."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse,copy,hashlib,importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rough_runtime_v1_fixture_definitions',ROOT/'scripts/check_rough_runtime_cpu.py');fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
rt=fixture.rt
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'checker_source.py').write_bytes(Path(__file__).read_bytes());b=fixture.load_bundle(ROOT/'outputs/input_bundle_v1/bundle.json');native=json.loads(Path(b['spec']['source_pins']['native_manifest']['path']).read_text());path=Path(native['native']['path']).parents[2]/'native_reader_loaded_CPU.npz'
    with np.load(path,allow_pickle=False) as z:fixture.LOADED={k:z[k].copy() for k in z.files}
    classes=(fixture.StubEnv,fixture.Motion,fixture.StubMDP,fixture.ScopeRunner);names=((classes[0],'__init__'),(classes[0],'reset'),(classes[1],'prepare_reset'),(classes[1],'_configure_fixed_wrist_reference'),(classes[2],'termination'),(classes[3],'policy_update'),(classes[3],'__init__'));original={(c,n):getattr(c,n) for c,n in names};checks=[]
    def restored():return all(getattr(c,n) is f for (c,n),f in original.items())
    def cfg():return NS(auto_reset=False,events={'reset_to_reference_state':NS(mode='reset')},terminations={'reference_end':NS(func=lambda env:None,time_out=True)})
    def rejected(name,operation):
        try:operation()
        except (ValueError,RuntimeError):checks.append({'name':name,'passed':restored()})
        else:checks.append({'name':name,'passed':False,'error':'Invalid operation accepted'})
    def scope_call(operation,bundle=None,descriptor=None):
        with rt.rough_runtime(*classes,bundle or b['arrays'],descriptor or b['descriptor']):operation()
    rejected('exception_inside_scope_restores_all_class_methods',lambda:scope_call(lambda:(_ for _ in ()).throw(RuntimeError('intentional_CPU_fixture'))))
    rejected('auto_reset_true_rejected_finally_restores',lambda:scope_call(lambda:fixture.StubEnv(NS(**{**vars(cfg()),'auto_reset':True}))))
    def geometry_reset():
        config=cfg();config.events['pen_dimensions']=NS(mode='reset');fixture.StubEnv(config)
    rejected('reset_geometry_event_rejected_not_silently_overwriting_actual8',lambda:scope_call(geometry_reset))
    def adaptive():
        env=fixture.StubEnv(cfg());env.motion.cfg.trajectory_sampling_mode='adaptive';env.reset()
    rejected('adaptive_RSI_rejected',lambda:scope_call(adaptive))
    def hold():
        env=fixture.StubEnv(cfg());env.motion.cfg.first_frame_hold_prob=.1;env.reset()
    rejected('held_first_frame_rejected',lambda:scope_call(hold))
    rejected('duplicate_reset_ids_rejected',lambda:scope_call(lambda:fixture.StubEnv(cfg()).reset(env_ids=torch.tensor([1,1]))))
    rejected('outofrange_reset_ids_rejected',lambda:scope_call(lambda:fixture.StubEnv(cfg()).reset(env_ids=torch.tensor([4]))))
    bad=copy.deepcopy(b['arrays']);bad['init']['qd_init_requested_FP32_rad_s'][0]+=.1
    rejected('analytic_init_mismatch_rejected',lambda:scope_call(lambda:None,bundle=bad))
    bad=copy.deepcopy(b['descriptor']);bad['physical_reference_qualified']=True
    rejected('false_physical_qualification_rejected',lambda:scope_call(lambda:None,descriptor=bad))
    bad=copy.deepcopy(b['arrays']);bad['init']['u_planned0_FP32_rad'][0]=1000;bad['control']['control__fp32_old_export__tracking_u_exec_15_FP32_rad'][0,0]=1000
    rejected('outofbounds_u0_rejected_without_clamp_repair',lambda:scope_call(lambda:fixture.StubEnv(cfg()).reset(),bundle=bad))
    # A pinned native method is mandatory; do not silently adapt an arbitrary
    # replacement whose source anchors happen to look similar.
    rejected('unreviewed_policy_source_rejected',lambda:rt.make_policy_update(scope_call))
    checks.append({'name':'CUDA_uninitialized_no_cuda_tensors','passed':not torch.cuda.is_initialized()})
    report={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','check_count':len(checks),'passed':sum(c['passed'] for c in checks),'checks':checks,'runtime_sha256':sha(rt.__file__),'checker_sha256':sha(__file__),'fixture_definitions_sha256':sha(ROOT/'scripts/check_rough_runtime_cpu.py'),'native_Env_constructed':False,'stub_only_Env_IO':True,'actual_optimizer_steps':0,'physics_steps':0,'GPU_used':False,'Torch_CUDA_initialized':torch.cuda.is_initialized()};(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));raise SystemExit(report['status']!='PASS')
if __name__=='__main__':main()
