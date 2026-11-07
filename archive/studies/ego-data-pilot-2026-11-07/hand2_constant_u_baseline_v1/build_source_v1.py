"""New metadata adapter; retain the parent's entire physics main."""
import ast,difflib,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'hand2_holding_v4/run_cpu_native_equilibrium_holding_v4.py'
NEW=ROOT/'run_cpu_constant_u_baseline_v1.py'
def pin(p):
    p=p.resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
text=OLD.read_text()
text=text.replace("SCHEMA = 'hand2_phone_native_equilibrium_CPU_holding_protocol:4'","SCHEMA = 'hand2_phone_matched_constant_u_baseline_CPU_protocol:1'\nACCEL_GATE_KEYS = {'COM_acceleration_infinity_m_s2','angular_acceleration_infinity_rad_s2','joint_acceleration_infinity_rad_s2'}")
helper="""def required_native_gates(value):
    return (isinstance(value,dict) and set(value)==NATIVE_GATE_KEYS
            and all(type(v) is bool for v in value.values())
            and all(value[k] is True for k in NATIVE_GATE_KEYS-ACCEL_GATE_KEYS))


"""
text=text.replace('def metadata_preflight(protocol_path):',helper+'def metadata_preflight(protocol_path):')
text=text.replace("    protocol=read(protocol_path)\n","    protocol=read(protocol_path)\n    if protocol['baseline_selection']!={'seed':0,'rank':0} or set(protocol['acceleration_descriptors_only'])!=ACCEL_GATE_KEYS:\n        raise ValueError('Fixed baseline and exactly three descriptive acceleration metrics')\n",1)
old="            if candidate.get('candidate_pass') is not True:reasons.append('candidate_pass not True')"
new="            if (seed['seed'],candidate['rank'])!=(0,0):reasons.append('Not preregistered seed0rank0 constant-u baseline; no rank substitution')"
assert old in text;text=text.replace(old,new)
text=text.replace("all_true_exact(candidate.get('native_entry_gates'),NATIVE_GATE_KEYS)","required_native_gates(candidate.get('native_entry_gates'))")
text=text.replace("all_true_exact(candidate.get('fresh_repeat_entry_gates'),NATIVE_GATE_KEYS)","required_native_gates(candidate.get('fresh_repeat_entry_gates'))")
text=text.replace("complete primary native entry gates not allTrue","required nonacceleration primary entry gates not allTrue")
text=text.replace("complete fresh-repeat native entry gates not allTrue","required nonacceleration repeat entry gates not allTrue")
text=text.replace("'rank':candidate['rank'],'qualified_for_dynamic_diagnostic':qualified,","'rank':candidate['rank'],'enrolled_for_matched_constant_u_baseline':qualified,\n                'original_static_candidate_pass':candidate.get('candidate_pass'),\n                'acceleration_descriptors_only':{tag:{k:candidate.get(key,{}).get(k) for k in sorted(ACCEL_GATE_KEYS)} for tag,key in [('primary','native_entry_gates'),('repeat','fresh_repeat_entry_gates')]},")
text=text.replace('f"native_seed{seed[\'seed\']}_rank{candidate[\'rank\']}"',"'matched_constant_u_seed0_rank0'")
replacements={
 'Bounded native-equilibrium candidate CPU holding diagnostic; no training or object aid.':'Matched constant-u baseline CPU diagnostic; no training or object aid.',
 'hand2_phone_native_equilibrium_CPU_holding_result:4':'hand2_phone_matched_constant_u_baseline_CPU_result:1',
 'BLOCKED_NO_NATIVE_EQUILIBRIUM_CANDIDATE_NO_MODEL_OR_STEPS':'BLOCKED_FIXED_BASELINE_EVIDENCE_NO_MODEL_OR_STEPS',
 'COMPLETED_ALL_SELECTED_NATIVE_EQUILIBRIUM_SEEDS_BOTH_DT_UNASSISTED_DIAGNOSTIC':'COMPLETED_MATCHED_CONSTANT_U_BASELINE_BOTH_DT',
 'Exact finite native-equilibrium verified pose and static zero speed':'Exact finite original rank0 pose and zero speed',
 'Native equilibrium entry/repeat/global-model-audit and independent22/189 new initial pad window [-0.1,+0.2]mm; no old point LP gate':'Registered constant-u baseline: fixedrank0, all nonacceleration entry/repeat/audit/22+189 gates; three static accelerations descriptive only',
 'Each native-search seed first passing shortlist rank only, both fixed dt, unchanged original2s case gates. ':'Fixed original seed0rank0 constant-u baseline, both fixed dt, unchanged original2s case gates. ',
}
for a,b in replacements.items():text=text.replace(a,b)
ast.parse(text)
with NEW.open('x',encoding='utf8') as f:f.write(text)
def main_ast(t):return next(n for n in ast.parse(t).body if isinstance(n,ast.FunctionDef) and n.name=='main')
class Normalize(ast.NodeTransformer):
    def visit_Constant(self,n):
        if isinstance(n.value,str):
            for a,b in replacements.items():n.value=n.value.replace(b,a)
        return n
same=ast.dump(Normalize().visit(main_ast(text)),include_attributes=False)==ast.dump(main_ast(OLD.read_text()),include_attributes=False)
assert same,'Entire physics main differs only by explicit namespace/descriptive labels'
diff=ROOT/'metadata_adapter_diff_v1.txt'
with diff.open('x',encoding='utf8') as f:f.write(''.join(difflib.unified_diff(OLD.read_text().splitlines(True),text.splitlines(True),fromfile=str(OLD),tofile=str(NEW))))
with (ROOT/'source_derivation_v1.json').open('x',encoding='utf8') as f:json.dump({'schema':'matched_constant_u_source_derivation:1','parent':pin(OLD),'newsource':pin(NEW),'builder':pin(Path(__file__)),'whole_main_AST_exact_except_declared_strings':same,'declared_string_changes':replacements,'diff':pin(diff),'model_or_forward_or_steps':0},f,indent=2)
checker=(ROOT.parent/'hand2_holding_v4/check_source_preservation_v1.py').read_text()
checker=checker.replace("ROOT/'run_cpu_native_equilibrium_holding_v4.py'","ROOT/'run_cpu_constant_u_baseline_v1.py'")
checker=checker.replace('metadata_adapter_v3_to_v4.diff','metadata_adapter_v3_to_baseline.diff').replace('native_equilibrium_holding_v4_source_preservation:1','matched_constant_u_source_preservation:1')
with (ROOT/'check_source_preservation_v1.py').open('x',encoding='utf8') as f:f.write(checker)
print(json.dumps({'source':pin(NEW),'whole_main_preserved':same,'native_calls':0}))
