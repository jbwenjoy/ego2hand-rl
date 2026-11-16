"""Pinned source facts and honestly phase-labeled solve-exit metrics."""
import ast

import numpy as np

from .pen_solve_exit_epoch_v2 import physical_view


def source_facts(text):
    module=ast.parse(text); functions={n.name:n for n in module.body if isinstance(n,ast.FunctionDef)}
    def dump(node): return ast.dump(node,include_attributes=False)
    def expression(text): return dump(ast.parse(text,mode='eval').body)
    def calls(node,name):
        return sorted(n.lineno for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==name)
    def launch(node,name):
        return sorted(n.lineno for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['launch','capture_while'] and n.args and isinstance(n.args[0],ast.Name) and n.args[0].id==name)
    iteration=functions['_solver_iteration']; factory=functions['_linesearch_iterative_kernel']; solve=functions['_solve']
    positions={name:calls(iteration,name) for name in ['_linesearch','_update_constraint','_update_gradient_incremental']}
    positions['_solve_done']=launch(iteration,'_solve_done')
    ordered=all(len(v)==1 for v in positions.values()) and list(positions.values())==sorted(positions.values())
    assignments=[n for n in ast.walk(factory) if isinstance(n,ast.Assign)]
    gtol=[n.value for n in assignments if any(isinstance(t,ast.Name) and t.id=='gtol' for t in n.targets)]
    exhausted=[n.value for n in assignments if any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='ctx_ls_exhausted_out' for t in n.targets)]
    initial=[n.value for n in assignments if any(isinstance(t,ast.Name) and t.id=='initial_converged' for t in n.targets)]
    marker=expression('efcid == 0 and ctx_ls_exhausted_in[worldid]')
    forced=[n for n in ast.walk(functions['_update_constraint_efc']) if isinstance(n,ast.If) and dump(n.test)==marker]
    atomic=expression('wp.atomic_add(state_changed_count_out, worldid, 1)')
    forced_ok=len(forced)==1 and any(isinstance(n,ast.Call) and dump(n)==atomic for n in ast.walk(forced[0]))
    shifted=functions['_eval_pt_direct_shifted']; shifted_return=[n for n in shifted.body if isinstance(n,ast.Return)]
    expected_shift=expression('wp.vec3(alpha * (jvD * jaref + 0.5 * alpha_h) + offset, jvD * jaref + alpha_h, hessian)')
    recovery=launch(solve,'_qfrc_constraint_from_grad'); loop=[n.lineno for n in ast.walk(solve) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='capture_while']
    gradient=functions['_update_gradient_zero_grad_dot']
    unchanged=[n.value for n in ast.walk(gradient) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='ctx_search_unchanged_out' for t in n.targets)]
    checks=dict(
        line_then_constraint_then_gradient_then_done=ordered,
        exhausted_is_step_noise_test=len(exhausted)==1 and dump(exhausted[0])==expression('wp.abs(alpha) < noise_floor'),
        gtol_floor_1e_minus6=len(gtol)==1 and dump(gtol[0])==expression('wp.max(tolerance * ls_tolerance * snorm * scale, 1e-6)'),
        initial_accept_requires_negative_delta=len(initial)==1 and dump(initial[0])==expression('wp.abs(lo_in[1]) < gtol and lo_in[0] < 0.0'),
        exhausted_forces_state_change=forced_ok,
        unchanged_uses_state_change_count=any(dump(n)==expression('ctx_done_in[worldid] or state_changed_count_in[worldid] == 0') for n in unchanged),
        quadratic_cost_already_shifted=len(shifted_return)==1 and dump(shifted_return[0].value)==expected_shift,
        pyramidal_uses_shifted_quadratic=bool(calls(functions['_compute_efc_eval_pt_pyramidal'],'_eval_pt_direct_shifted')),
        public_constraint_recovered_after_loop=len(recovery)==1 and len(loop)==1 and recovery[0]>loop[0])
    selected=['_solver_iteration','_linesearch_iterative_kernel','_linesearch_iterative','_linesearch',
        '_update_constraint_efc','_update_gradient_zero_grad_dot','_update_gradient_incremental',
        '_eval_pt_direct_shifted','_compute_efc_eval_pt_pyramidal','_compute_efc_eval_pt_3alphas_pyramidal',
        '_qfrc_constraint_from_grad','_solve']
    return dict(checks=checks,iteration_call_lines=positions,
                source_ranges={k:[functions[k].lineno,functions[k].end_lineno] for k in selected},
                scope='Static facts for the exact pinned noncompact Newton/pyramidal implementation;not an execution trace or a numerical root-cause proof.')


def endpoint_metrics(observations,tick):
    o=physical_view(observations,26)
    grad=o['context.grad'][tick,0].astype(np.float64)*float(o['context.grad_scale'][tick,0])
    search=o['context.search'][tick,0].astype(np.float64)
    return dict(iterations=int(o['data.solver_niter'][tick,0]),alpha_from_last_linesearch=float(o['context.alpha'][tick,0]),
        improvement_from_last_linesearch=float(o['context.improvement'][tick,0]),step_noise_flag_from_last_linesearch=bool(o['context.ls_exhausted'][tick,0]),
        post_update_search_unchanged=bool(o['context.search_unchanged'][tick,0]),post_update_state_changed=int(o['context.state_changed_count'][tick,0]),
        post_update_quad_changed=int(o['context.quad_changed_count'][tick,0]),post_update_grad_scale=float(o['context.grad_scale'][tick,0]),
        post_update_gradient_norm=float(np.linalg.norm(grad)),post_update_search_norm=float(np.linalg.norm(search)),
        post_update_G_dot_stored_search=float(grad@search),post_update_newton_decrement=float(o['context.newton_decrement'][tick,0]),
        post_update_grad_dot=float(o['context.grad_dot'][tick,0]))
