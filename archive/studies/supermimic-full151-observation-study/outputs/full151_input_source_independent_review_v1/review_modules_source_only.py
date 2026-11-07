"""Independent source-only review receipt; imports no studied module."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('D:/workspace/supermimic-full151-observation-study')
OUT = Path(__file__).resolve().parent
paths = {
    'object_curves': ROOT/'src/full151/object_curves.py',
    'continuous_control': ROOT/'src/full151/continuous_control.py',
    'io': ROOT/'src/full151/io.py',
    'frozen_object_math': Path('D:/workspace/supermimic-observation-prefix-study/src/prefix_curves.py'),
    'frozen_FP32_PD': Path('D:/workspace/supermimic-observation-prefix-study/src/prefix_control.py'),
}

def pin(path):
    p = Path(path).resolve(strict=True)
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

before = {name: pin(p) for name, p in paths.items()}
sources = {name: p.read_text(encoding='utf-8-sig') for name, p in paths.items()}
symbols = {}
for name, source in sources.items():
    tree = ast.parse(source)
    symbols[name] = {n.name: {'line': n.lineno, 'end_line': n.end_lineno}
                     for n in ast.walk(tree)
                     if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
checks = {
    'object_primitives_actual_sha_matches_declared': before['frozen_object_math']['sha256'] in sources['object_curves'],
    'old_PD_actual_sha_matches_declared': before['frozen_FP32_PD']['sha256'] in sources['continuous_control'],
    'modules_syntax_parse': len(symbols) == 5,
    'object_exact151_clock_and150_intervals': 'np.arange(151)*.02' in sources['object_curves'] and '149)' in sources['object_curves'],
    'rotational_endpoint_acceleration_uses_DJr': 'right_jacobian_directional(phi,d1)@d1' in sources['object_curves'],
    'object_world_omega_and_alpha_consistent_with_R_Jr': 'R@(jr@dphi)' in sources['object_curves'] and 'R@(jr@ddphi+m.right_jacobian_directional(phi,dphi)@dphi)' in sources['object_curves'],
    'source_centered_curve_retains_P0': 'delta=P[1:]-P[0]' in sources['continuous_control'],
    'LP_fixed_P0_P1_not_free_or_clipped': 'free=mat[...,2:]' in sources['continuous_control'],
    'first_LP_command_is_fixed_but_final_audit_does_not_replace_it': 'U[0]=0.;ub[0]=u0[j]' in sources['continuous_control'] and 'U=old.c2_numpy(pre,params)' in sources['continuous_control'],
    'final_FP32_u0_equality_audited': 'same_fixed_initial_u0' in sources['continuous_control'] and 'U[0].view(np.uint32)' in sources['continuous_control'],
    'only150_proposal_rows_and149_neighbor_rates': "data['q_goal_rad'][:150]" in sources['continuous_control'] and 'rates=np.diff(U.astype(float),axis=0)/H' in sources['continuous_control'],
    'terminal_diagnostic_maps_to_u149_not_new_u150': 'np.minimum(np.arange(601)//4,149)' in sources['continuous_control'],
    'postcast_Bernstein_and_two_FP32_groupings_retained': all(s in sources['continuous_control'] for s in ('postC2_real_affine_rawforce_Bernstein_Nm', 'reference_affine601_FP32_Nm', 'reference_difference601_FP32_Nm')),
    'modules_do_not_call_linprog_minimize_Env_or_WM': not any(isinstance(n, ast.Call) and ((isinstance(n.func, ast.Name) and n.func.id in ('linprog','minimize','ManagerBasedRlEnv','WorldModel')) or (isinstance(n.func, ast.Attribute) and n.func.attr in ('linprog','minimize'))) for name in ('continuous_control','object_curves','io') for n in ast.walk(ast.parse(sources[name]))),
}
report = {
    'schema': 'full151_module_source_independent_review:1',
    'status': 'MODULE_STATIC_REVIEW_COMPLETE_PRODUCER_PROTOCOL_PENDING',
    'source_pins': before,
    'source_AST_symbols': symbols,
    'checks': checks,
    'all_checks_pass': all(checks.values()),
    'mathematics_review': {
        'SO3': 'R=RiExp(phi), bodyomega=Jr(phi)*phidot, worldomega=R*bodyomega. alpha world uses R*(Jr*phiddot+DJr(phi,phidot)*phidot); omega-cross-itself term is zero. Endpoint inverse-Jr coefficients consistent.',
        'LP': 'Each joint has151 absolute free coefficients P2..152 plus151 fit and151 second-difference epigraphs. FixedP0/P1 and original hard coefficient box suffice for convex-hull range (declared arithmetic tolerance only). Linear rows use150 ZOH targets,149 rate differences and600 cubic Bernstein rawforce bounds; tracking-only surrogate, not dynamics/physical-force certificate.',
        'FP32': 'LP row0 fixedu0 is not a repair of final controls: audit recomputes frozen FP32 proposal+C2 over150 knots and separately requires exact U0 bits. Real-affine coefficients at U32 and FP32 sample arithmetic are separate gates.',
    },
    'producer_requirements_pending': [
        'Bind exact P0/P1 authority and verify q0/qd0 cast and preserved requestedinit; never optimize/clip/reconstruct after failure to hide mismatch.',
        'Object quaternion primitive unit check then normalizes within1e-6. Requested FP32 objectpose/twist must retain original bytes; sampled rotation roundoff reported separately.',
        'Pass hash-checked frozen760 module and PDParameters object to final audit; pure dict is only supported by new linear-parameter validator, not old proposal API.',
        'Fresh output and nooverwrite/partial failure preservation in entry: io.write by itself overwrites and is not atomic.',
        'All source/hash/context/nativewindow joins, authorization and all20LP schedule pending producer/protocol review. No module receipt is an execution authorization.',
    ],
    'execution_scope': {'source_JSON_hash_AST_only': True, 'modules_imported': False, 'arrays_loaded_or_computed': False, 'LP_optimizer_FK_collision_Reader_Env_WM_physics_GPU': False, 'framework_imported': any(x in sys.modules for x in ('numpy','scipy','torch','warp','mujoco')), 'source_written': False},
    'review_source': pin(__file__),
    'all_sources_unchanged': before == {name: pin(p) for name, p in paths.items()},
}
target = OUT/'modules_review.json'
if target.exists(): raise FileExistsError(target)
target.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
print(json.dumps({'report': pin(target), 'all_checks_pass': report['all_checks_pass'], 'checks': len(checks), 'scope': report['execution_scope']}))
if not all(checks.values()) or not report['all_sources_unchanged']: raise SystemExit(2)
