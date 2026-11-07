"""Build a new native-equilibrium adapter; no model/data/physics execution."""
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'hand2_holding_v3/run_cpu_support_holding_v3.py'
TARGET=ROOT/'run_cpu_native_equilibrium_holding_v4.py'
source=PARENT.read_text(encoding='utf8')
source=source.replace('Bounded joint-support-pose candidate','Bounded native-equilibrium candidate')
source=source.replace("SCHEMA = 'hand2_phone_support_CPU_holding_protocol:3'", "SCHEMA = 'hand2_phone_native_equilibrium_CPU_holding_protocol:4'")
start=source.index('def metadata_preflight(protocol_path):')
end=source.index('\n\ndef dt_pair_summary',start)
adapter='''NATIVE_GATE_KEYS = {
    'COM_acceleration_infinity_m_s2','angular_acceleration_infinity_rad_s2',
    'joint_acceleration_infinity_rad_s2','native_other_penetration_m',
    'native_other_normal_force_N','native_finite_no_warnings',
    'native_two_pads_loaded','native_pad_budget','native_pad_compression',
    'requested_ctrl_unclamped',
}
REPEAT_KEYS = {'qpos27','ctrl20','qacc26','qfrc_actuator26','qfrc_constraint26',
               'a_COM_world_m_s2','alpha_world_rad_s2','contact_rows24'}


def all_true_exact(value, keys):
    return isinstance(value,dict) and set(value)==keys and all(v is True for v in value.values())


def metadata_preflight(protocol_path):
    protocol=read(protocol_path)
    if protocol['schema']!=SCHEMA:
        raise ValueError('New native-equilibrium holding protocol namespace')
    if protocol['gates']!=GATES or protocol['dt_s']!=DTS or protocol['duration_s']!=DURATION:
        raise ValueError('Unchanged original physical budget/gates')
    if protocol['dt_pair_gates']!=PAIR_GATES:
        raise ValueError('Unchanged2mm/2degree saved endpoint dt comparison')
    if (type(protocol['CPU_run_authorized']) is not bool
            or protocol['training_or_GPU_authorized'] is not False):
        raise ValueError('Explicit CPU-only boundary')
    if set(protocol['inputs'])!={'equilibrium_protocol','equilibrium_report','model_binding','model_mjb'}:
        raise ValueError('Exact four native-equilibrium semantic inputs')
    for rec in [*protocol['inputs'].values(),*protocol['source_pins'].values()]:
        checked(rec)
    if protocol['source_pins']['runner']!=pin(__file__):
        raise ValueError('Current runner')
    if protocol['source_pins']['old_holding_runner']!=pin(ROOT.parent/'hand2_holding_v1/run_cpu_holding_v1.py'):
        raise ValueError('Original physics loop authority')
    output=Path(protocol['output'])
    if not output.is_absolute() or output.exists() or output.parent.resolve()!=ROOT.resolve():
        raise ValueError('Absolute fresh holding_v4 output')
    pose=read(checked(protocol['inputs']['equilibrium_report']))
    cp=read(checked(protocol['inputs']['equilibrium_protocol']))
    if (pose.get('schema')!='native_soft_contact_equilibrium_result:6'
            or pose.get('status')!='COMPLETED_BOUNDED_NATIVE_STATIC_SEARCH'
            or pose.get('protocol')!=protocol['inputs']['equilibrium_protocol']
            or pose.get('integration_training_GPU')!=0
            or cp.get('schema')!='native_soft_contact_equilibrium_protocol:6'
            or cp.get('training_GPU_integration_authorized') is not False):
        raise ValueError('Completed matching native-equilibrium static search only')
    # The actual producer spelling is model_audit, not globalmodel_audit.
    audit=pose.get('model_audit',{})
    if audit.get('passed') is not True or not isinstance(audit.get('model_digest'),str):
        raise ValueError('Global endpoint model audit must pass')
    for rec in [*cp['inputs'].values(),*cp['source_pins'].values()]:
        checked(rec)
    binding=read(checked(protocol['inputs']['model_binding']))
    if (binding['model']!=protocol['inputs']['model_mjb']
            or binding['source']!=protocol['source_pins']['scene']
            or protocol['inputs']['model_binding'] not in cp['inputs'].values()
            or protocol['inputs']['model_mjb'] not in cp['inputs'].values()):
        raise ValueError('Same original explicit-pair compiled model, directly bound by native protocol')
    if cp.get('pad_compression_limit_m')!=.0001:
        raise ValueError('New designated-pad initial compression assumption exactly0.1mm')
    seeds=pose['seed_results']
    if len(seeds)!=2 or sorted(v['seed'] for v in seeds)!=[0,1]:
        raise ValueError('Both predefined native search seeds retained')
    variants,screening=[],[]
    total_qualified=0
    for seed in seeds:
        shortlist=seed['shortlist']
        ranks=[c['rank'] for c in shortlist]
        if any(type(v) is not int or v<0 for v in ranks) or len(ranks)!=len(set(ranks)):
            raise ValueError('Distinct nonnegative frozen shortlist ranks')
        candidates=[];selected_rank=None
        for candidate in sorted(shortlist,key=lambda c:c['rank']):
            reasons=[]
            if candidate.get('candidate_pass') is not True:reasons.append('candidate_pass not True')
            if candidate.get('unknown')!=[]:reasons.append('candidate unknown not empty')
            if not all_true_exact(candidate.get('fresh_repeat_bit_equal'),REPEAT_KEYS):
                reasons.append('complete eight-field fresh repeat not bit-equal')
            if not all_true_exact(candidate.get('native_entry_gates'),NATIVE_GATE_KEYS):
                reasons.append('complete primary native entry gates not allTrue')
            if not all_true_exact(candidate.get('fresh_repeat_entry_gates'),NATIVE_GATE_KEYS):
                reasons.append('complete fresh-repeat native entry gates not allTrue')
            native=candidate.get('native',{});raw=native.get('raw',{})
            repeat=candidate.get('fresh_repeat_native',{})
            if native.get('unknown')!=[] or repeat.get('unknown')!=[]:
                reasons.append('native or repeated evidence unknown')
            if native.get('model_parameter_digest')!=audit['model_digest'] or repeat.get('model_parameter_digest')!=audit['model_digest']:
                reasons.append('native/endpoint model digest identity mismatch')
            geometry=candidate.get('geometry',{});state=geometry.get('state',{})
            if (geometry.get('geometry_gate_pass') is not True
                    or geometry.get('complete22_189') is not True
                    or geometry.get('new_pad_gap_hypothesis_m')!=[-.0001,.0002]
                    or geometry.get('constraints',{}).get('all_satisfied') is not True
                    or state.get('unknown')!=[]
                    or len(state.get('phone_geometry_rows',[]))!=22
                    or len(state.get('self_geometry_rows',[]))!=189):
                reasons.append('complete independent22/189/new-compression geometry not qualified')
            x=candidate.get('x46')
            if not finite_vector(x,46):reasons.append('finite x46 missing')
            if not finite_vector(raw.get('qpos27'),27):reasons.append('finite native qpos27 missing')
            if not finite_vector(raw.get('ctrl20'),20):reasons.append('finite native ctrl20 missing')
            if not finite_vector(raw.get('qvel26'),26) or any(raw.get('qvel26',[1])):
                reasons.append('native velocity must be zero26')
            if finite_vector(x,46) and raw.get('ctrl20')!=x[26:]:
                reasons.append('x46 ctrl versus native raw ctrl mismatch')
            selected=candidate.get('state')
            if not isinstance(selected,dict):reasons.append('candidate state pin missing')
            else:
                checked(selected)
                if Path(selected['path']).parent.resolve()!=Path(protocol['inputs']['equilibrium_report']['path']).parent.resolve():
                    raise ValueError('Candidate state outside matching native search output')
            qualified=not reasons
            if candidate.get('candidate_pass') is True:total_qualified+=1
            chosen=qualified and selected_rank is None
            if chosen:selected_rank=candidate['rank']
            candidates.append({'rank':candidate['rank'],'qualified_for_dynamic_diagnostic':qualified,
                'selected':chosen,'reasons':reasons,
                'selection_status':'FIRST_PASSING_RANK_SELECTED' if chosen else 'QUALIFIED_LATER_RANK_NOT_SELECTED_PREDECLARED_RULE' if qualified else 'REJECTED_EVIDENCE',
                'state':selected})
            if chosen:
                variants.append({'name':f"native_seed{seed['seed']}_rank{candidate['rank']}",
                    'seed':seed['seed'],'rank':candidate['rank'],'reachable':True,'ctrl20':x[26:],
                    'candidate_state':selected,'verified_qpos27':raw['qpos27'],
                    'verified_qvel26':raw['qvel26'],'verified_ctrl20':raw['ctrl20'],'verified_x46':x,
                    'moment_infinity_Nm':None,'strict_support_proposal_pass':None})
        screening.append({'seed':seed['seed'],'selected_rank':selected_rank,'enrolled':selected_rank is not None,
            'shortlist_screening':candidates,'empty_shortlist':not shortlist})
        if seed.get('qualified_candidate_count')!=sum(c.get('candidate_pass') is True for c in shortlist):
            raise ValueError('Full per-seed producer qualification count consistent')
    if pose.get('qualified_candidate_count')!=total_qualified:
        raise ValueError('Full global producer qualification count consistent')
    return protocol,output,binding,variants,screening
'''
source=source[:start]+adapter+source[end:]
source=source.replace('hand2_phone_support_CPU_holding_result:3','hand2_phone_native_equilibrium_CPU_holding_result:4')
source=source.replace('BLOCKED_NO_SUPPORT_POSE_SEED_ELIGIBLE_NO_MODEL_OR_STEPS','BLOCKED_NO_NATIVE_EQUILIBRIUM_CANDIDATE_NO_MODEL_OR_STEPS')
source=source.replace('COMPLETED_ALL_ENROLLED_SUPPORT_POSE_SEEDS_BOTH_DT_UNASSISTED_DIAGNOSTIC','COMPLETED_ALL_SELECTED_NATIVE_EQUILIBRIUM_SEEDS_BOTH_DT_UNASSISTED_DIAGNOSTIC')
old="""                qv0 = z['qvel26'].copy()
            if (q0.shape != (27,)"""
new="""                qv0 = z['qvel26'].copy()
                state_ctrl = z['ctrl20'].copy()
                state_x46 = z['x46'].copy()
            expected={'qpos27':variant['verified_qpos27'],'qvel26':variant['verified_qvel26'],
                      'ctrl20':variant['verified_ctrl20'],'x46':variant['verified_x46']}
            actual={'qpos27':q0,'qvel26':qv0,'ctrl20':state_ctrl,'x46':state_x46}
            for key,raw in actual.items():
                ref=np.asarray(expected[key],dtype=np.float64)
                if raw.dtype!=np.dtype('float64') or raw.shape!=ref.shape or raw.tobytes()!=ref.tobytes():
                    raise ValueError('NPZ/native/x46 exact raw binding: '+key)
            if not np.array_equal(state_x46[:20],q0[:20]) or not np.array_equal(state_x46[26:],state_ctrl):
                raise ValueError('NPZ x46 q/control ordering')
            if (q0.shape != (27,)"""
assert source.count(old)==1
source=source.replace(old,new)
source=source.replace('Exact finite support-pose verified pose and static zero speed','Exact finite native-equilibrium verified pose and static zero speed')
source=source.replace("                    predecessor_moment_infinity_Nm=variant['moment_infinity_Nm'],\n                    predecessor_strict_support_proposal_pass=variant['strict_support_proposal_pass'],\n                    enrollment='Force-hard LP/caps/reachable controller and geometry; moment residual explicitly allowed')",
    "                    native_equilibrium_rank=variant['rank'],\n                    enrollment='Native equilibrium entry/repeat/global-model-audit and independent22/189 new initial pad window [-0.1,+0.2]mm; no old point LP gate')")
source=source.replace("'scope':'All joint-support-pose force-hard/geometry-eligible seeds, both fixed dt, unchanged original2s case gates. '",
    "'scope':'Each native-search seed first passing shortlist rank only, both fixed dt, unchanged original2s case gates. '")
source=source.replace("'Point-support moment residual is permitted only for this finite-mesh free-object diagnostic. '",
    "'New designated-pad initial compression up to0.1mm is a hypothesis, not measured compliance; no old point LP or positive-gap condition. '")
with TARGET.open('x',encoding='utf8',newline='\n') as f:f.write(source)
print(TARGET)
