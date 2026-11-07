"""JSON/AST/NumPy contract only; never constructs or runs an NN/native Reader."""
from pathlib import Path
import ast,hashlib,importlib.util,json,subprocess,sys
import numpy as np
H=Path(__file__).resolve().parent

def run():
    spec=importlib.util.spec_from_file_location('online_pure_contract',H/'collect_online_v2.py');O=importlib.util.module_from_spec(spec);spec.loader.exec_module(O)
    protocol=H/'protocol_CPUFalse_v2.json';p,b,out,pending=O.metadata_preflight(protocol)
    nodes={f.name:ast.parse(f.read_text(encoding='utf8')) for f in [H/'collect_online_v2.py',H/'prepare_protocol_v2.py',H/'bind_final_result_v1.py']}
    plan=O.profile_plan();profiles=[np.array(v['action20x20_F64'],dtype=np.float64) for v in plan['profiles']]
    assert all(a.shape==(20,20) and np.isfinite(a).all() and np.max(abs(a))<=1 for a in profiles)
    assert np.array_equal(profiles[0],-profiles[1]) and np.array_equal(profiles[2],-profiles[3])
    eps=p['rules']['exploration_mixture']
    assert all(np.all((1-eps)*corner+eps*a<=1) and np.all((1-eps)*corner+eps*a>=-1) for a in profiles for corner in [-1.,1.])
    n=p['rules']['substeps'];assert p['rules']['maximum_native_steps']==2*4*20*sum(n)==30720
    assert p['rules']['maximum_rows']==2*4*2*20==320
    root=nodes['collect_online_v2.py'];calls=[v for v in ast.walk(root) if isinstance(v,ast.Call)]
    assert not any(isinstance(c.func,ast.Attribute) and c.func.attr in ['backward','zero_grad'] for c in calls)
    # Deliberately executes only the CLI's default metadata route in a fresh process.
    cp=subprocess.run([sys.executable,str(H/'collect_online_v2.py'),'--protocol',str(protocol)],capture_output=True,text=True)
    assert cp.returncode==0,cp.stderr
    result=json.loads(cp.stdout)
    assert result['status']=='SOURCE_METADATA_ONLY_NO_TORCH_NATIVE_NPZ_VALUES' and result['pending_final_checkpoint'] is True
    assert 'torch' not in sys.modules and 'mujoco' not in sys.modules
    checks={key:True for key in ['current_source_AST_and_exact_JSON_pins','four_bounded_opposed_named_profiles','convex_action_rule_no_clipping_at_extremeActoroutputs',
        'paired_arm4profile2dt20H_exact_320H30720steps_budget','pending_final_no_checkpoint_deserialization','default_actualCLI_exit0_metadataonly','no_optimizer_or_backward_in_collector','no_Torch_MuJoCo_import_in_pure_contract']}
    report=dict(schema='phone_tripod_online_collection_source_contract:1',status='PASS_PURE_SOURCE_METADATA_NOT_NN_NATIVE_QUALIFICATION',
        checks=checks,passed=len(checks),total=len(checks),protocol=O.pin(protocol),sources={f:O.pin(H/f) for f in nodes},
        CLI=dict(argv=cp.args,returncode=cp.returncode,stdout=result,stderr=cp.stderr),
        counts=dict(NN_constructors=0,NN_FW=0,native_model_load=0,native_FK_forward_steps=0,NPZ_payload_reads=0,training_updates=0),
        actual_inference_and_native_collection_qualified=False,scope='Only implemented source/metadata/analytic action bounds; no final checkpoint binding or actual policy/native rollout was executed.')
    dest=H/'source_contract_v1.json';assert not dest.exists();dest.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(dict(report=O.pin(dest),passed=len(checks),total=len(checks))))

if __name__=='__main__':run()
