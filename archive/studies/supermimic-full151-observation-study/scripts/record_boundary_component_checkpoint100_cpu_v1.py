"""Metadata-only checkpoint100 handoff; never loads models or checkpoints."""
from pathlib import Path
import datetime
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OTHER = Path('D:/workspace/supermimic-chord-study')
RUN = ROOT/'outputs/full151105_boundary_impulse_matched_training_CPU_v1'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def pin(p):
    p = Path(p).absolute()
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def write(p, value, replace=False):
    with Path(p).open('w' if replace else 'x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    health_path = ROOT/'outputs/full151105_boundary_impulse_checkpoint100_saved_independent_v1/report.json'
    health = read(health_path)
    assert health['engineering_qualified'] is True and health['iteration'] == 100
    assert health['passed'] == health['total'] == len(health['checks'])
    assert all(v is True for v in health['checks'].values())
    assert health['own_CPU_checkpoint_loads'] == 3
    eta = read(RUN/'checkpoint100_ETA.json')
    runtime = read(RUN/'runtime.json')
    assert runtime['completed_pair_iteration'] >= 100
    assert runtime['status'] != 'FAILED_PARTIAL_NO_RETRY' and not (RUN/'failure.json').exists()
    assert eta['completed_pairs'] == 100 and eta['remaining_pairs'] == 900
    terminal = (RUN/'result.json').exists()
    result = read(RUN/'result.json') if terminal else None
    if terminal:
        assert result['status'] == 'COMPLETED_FIXED1000_MATCHED_COMPONENT_TRAINING'
    protocol_path = ROOT/'protocols/full151105_boundary_impulse_matched_training_CPU_run_v1.json'
    protocol = read(protocol_path)
    assert protocol['background_continue_to_final'] is True
    with (RUN/'updates.jsonl').open('rb') as f:
        rows = [json.loads(f.readline()) for _ in range(100)]
    assert [r['iteration'] for r in rows] == list(range(1, 101))
    arm_summaries = {a: {'PRE1_losses': rows[0]['arms'][a]['losses'],
                          'PRE100_losses': rows[99]['arms'][a]['losses'],
                          'counts_after100': rows[99]['arms'][a]['counts']}
                     for a in rows[0]['arms']}
    audit_history = {f'initial_saved_v{v}': pin(ROOT/f'outputs/full151105_boundary_component_initial_parity_saved_independent_v{v}/report.json') for v in (1, 2, 3)}
    pins = {
        'initial_native_parity': pin(ROOT/'outputs/full151105_boundary_component_initial_parity_CPU_v1/report.json'),
        'initial_native_saved_review10': audit_history['initial_saved_v3'],
        'training_recipe': protocol['recipe'], 'training_preregistration': protocol['preregistration'],
        'training_protocol': pin(protocol_path),
        'training_source_review': protocol['source_pins']['training_source_review'],
        'checkpoint100_pair': pin(RUN/'paired_checkpoint_0100.json'),
        'checkpoint100_health': pin(health_path), 'checkpoint100_ETA': pin(RUN/'checkpoint100_ETA.json'),
        'hidden_worker_launch': pin(ROOT/'outputs/full151105_boundary_impulse_matched_training_launch_v1/launch.json'),
        'additional_training_health_source_review': pin(ROOT/'outputs/full151105_boundary_training_health_source_review_v1/report.json'),
    }
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    snapshot = {
        'schema': 'full151105_boundary_impulse_checkpoint100_goal_pause_handoff:1',
        'created_UTC': now, 'status': 'FIRST_PAIRED100_SAVED_HEALTH_AND_ETA_VERIFIED_GOAL_PAUSE_DUE',
        'pins': pins, 'auditor_adapter_history': audit_history,
        'health_checks': {'passed': health['passed'], 'total': health['total']},
        'saved_training_log_summaries': arm_summaries,
        'loss_scope': 'PRE100 loss is from weights after99 updates; checkpoint100 is after100. No post-checkpoint model forward or task accuracy evaluation.',
        'worker_runtime_snapshot': runtime, 'worker_terminal_at_snapshot': terminal,
        'checkpoint100_ETA': eta,
        'background_continue_to_final': True,
        'accuracy_task_CHORD_physicalgradient_ego_qualified': False,
        'next_after_user_resumes': ['Observe existing worker/terminal result once; do not restart or rerun training.', 'Independently read fixed1000 own checkpoints, then preregister saved-component point/impulse/mode/consistency and oldself15 diagnostics, reporting tradeoffs on source-dependent rows.', 'Return to physically feasible reference/contact support and matched actual initial warmup task experiments; multiple trajectories and real ego/hardware contracts remain open.'],
    }
    if terminal:
        snapshot['terminal_result'] = pin(RUN/'result.json')
        snapshot['terminal_completed_iteration'] = result['iteration']
        snapshot['remaining_seconds_at_snapshot'] = 0
    handoff_path = ROOT/'outputs/full151105_boundary_impulse_checkpoint100_pause_handoff_v1.json'
    write(handoff_path, snapshot)
    findings_path = ROOT/'docs/full151_boundary_impulse_checkpoint100_findings_2026-11-06.md'
    worker_text = '本轮1000次已在记录时完成，实际剩余0；首100次ETA保留作历史测量。' if terminal else '训练进程已获协议明确授权在后台继续到固定1000次；暂停Goal不终止该进程。'
    findings = f'''# 分源边界冲量组件：实际初始验证与首个配对检查点

2026-11-06。首个双方100次检查点已实际保存，独立健康读取 {health['passed']}/{health['total']} 通过，ETA已保存，达到用户约定的Goal暂停条件。{worker_text}

实际初始验证只读自有初始组件，1次native构造、2次raw复制、3次152批前向；原WM与两旁支的native58/self15逐位一致，14个新通道为正零，原始RNG、输入、normalizer、context及参数不变。独立保存审计v3为10/10，零Torch/模型/前向。v1 8/10和v2 9/10完整保留：前者错误假定NumPy sqrt复刻Torch位级值，后者错写非旋转差分符号；均为审计器错误，未改变producer、数据或科学容差。新审计绑定两份此前逐位审核过的native scale52原字节，使用原生target−pred非旋转差分。

训练前固定recipe：两臂同一实际WM初始、相同零14头、152条顺序、有效s52和m/I/g/DeltaT，公共native点/mode/无量纲冲量损失系数均1。两臂均计算一致性，只有一致性系数control0、method1不同。CPU4线程FP32原网络、F64冲量/求解，AdamW lr1e-4、betas(.9,.999)、eps1e-8、weight_decay0，clip1，固定1000次，仅100/1000保存。首次训练前向复用为actual初始位级验证，没有附加验证前向。v1合成训练器因可见CUDA被优化器初始化而失败已留存；v2在importTorch前隔离CUDA并完成3项Toy检查，原数学/recipe不变。

独立健康进程只加载这次自己的initial_pair与两份AFTER100字典，共3次CPU读取，检查16个参数、原始Adam状态/step100、PRE100的命名裁剪前后梯度、冻结norm/context、THIS RUN原始RNG及100条日志前缀身份；没有模型重建、前向、AD或物理。PRE100梯度/日志来自已更新99次的权重，不能当作AFTER100模型复测。

首100次测得配对均时 {eta['observed_mean_pair_seconds_after_trim']:.6f}s；按剩900对和保存/显式20s收尾估算剩余 {eta['estimated_remaining_seconds']:.1f}s，预计UTC {eta['estimated_finish_UTC']}。范围为规划启发式，不是置信区间或加速证据。日志两臂摘要及进程快照见handoff。

本轮只获得构造/优化与序列化健康证据。原151参考几何FAIL、旧三臂任务FAIL、有限响应确认FAIL、H8取舍及warmup差异均保留。152条来自单一已见机器人模拟谱系，r_saved保留observer/solver非零缺陷；不是成功人类力监督。旧self15没有公共标签损失，fit后可改变；尚未完成任务、Actor、物理梯度、force-capacity、多轨迹、ego或真机资格。

- [可复查handoff]({handoff_path.as_posix()})
- [首100独立健康]({health_path.as_posix()})
- [ETA]({(RUN/'checkpoint100_ETA.json').as_posix()})
- [冻结配方]({Path(protocol['recipe']['path']).as_posix()})
'''
    findings_path.write_text(findings, encoding='utf-8') if not findings_path.exists() else (_ for _ in ()).throw(FileExistsError(findings_path))
    plan_path = OTHER/'docs/research-plan-v0.14.md'
    with plan_path.open('x', encoding='utf-8') as f:
        f.write(f'''# 研究方案 v0.14：首个冲量一致性配对训练与暂停交接

2026-11-06。已完成实际初始native三批152前向的位级验证与独立10/10；新recipe冻结，匹配的公共损失/一致性两臂实际启动并保存首个100次检查点，独立健康{health['passed']}/{health['total']}通过，ETA已记录。按用户明确约定暂停Goal。{worker_text}

两臂同WM/零14头/context/norm/152顺序、公共点/mode/impulse损失和AdamW1000预算，唯一科学差异为跨目标一致性系数0/1。r_saved仅作监督标签，不能进入WM输入；不强迫原始观测缺陷为零。无新增物理、Actor或GPU训练。训练健康不证明方法收益。

恢复后先读取现有进程/终态，不重启。对固定1000检查点另做独立saved健康，再预注册组件预测/冲量/mode/一致性与旧self15诊断，报告各项取舍。随后回到可行参考/接触支持、真实共同warmup初态和同预算任务对照。已见缓存/fresh仅探索性使用，不重新宣称blind或跨轨迹泛化。

原参考与旧任务FAIL、有限响应确认FAIL、H8取舍、whole-hidden未知均保留。单缓存谱系、ego N=0；共享多轨迹G2–G8、heldout009/010和硬件子版本/观测契约仍待完成，完整目标尚未完成。

- [本轮证据与限制]({findings_path.as_posix()})
- [首100交接]({handoff_path.as_posix()})
- [上一版]({(OTHER/'docs/research-plan-v0.13.md').as_posix()})
''')
    ledger = read(ROOT/'outputs/full151105_matched_replay_dense_compute_ledger_v12.json')
    ledger['schema'] = 'full151105_matched_replay_dense_compute_ledger:13'
    ledger['previous_preserved'] = pin(ROOT/'outputs/full151105_matched_replay_dense_compute_ledger_v12.json')
    ledger['boundary_initial_component_actual_CPU'] = {'report': pins['initial_native_parity'], 'saved_review': pins['initial_native_saved_review10'], 'own_checkpoint_load': 1, 'native_constructor': 1, 'native_forwards': 3, 'predicted_rows': 456, 'AD_optimizer_MJB_physics_GPU': 0}
    ledger['boundary_matched_training_first100_CPU'] = {'pair': pins['checkpoint100_pair'], 'saved_health': pins['checkpoint100_health'], 'ETA': pins['checkpoint100_ETA'], 'each_arm_updates': 100, 'training_native_forwards_through100': 200, 'training_rows_through100': 30400, 'extra_validation_forward': 0, 'Actor_MJB_Env_physics_GPU': 0, 'saved_health_own_checkpoint_loads': 3, 'saved_health_model_FW_AD': 0, 'worker_may_have_progressed_after_saved100': True, 'terminal_at_handoff': terminal, 'whole_job_terminal_counts_only_in_terminal_result': terminal}
    ledger_path = ROOT/'outputs/full151105_matched_replay_dense_compute_ledger_v13.json'
    write(ledger_path, ledger)
    for state_path in [ROOT/'STUDY_STATE.json', OTHER/'STUDY_STATE.json']:
        state = read(state_path)
        state['updated_UTC'] = now
        state['goal_status'] = 'active'
        state['goal_pause_condition_met'] = True
        state['phase'] = 'ACTUAL boundary initialparity independent10; matchedCPU firstpaired100 SAVED independenthealth PASS+ETA; Goal pause due per user. Background fixed1000 authorized. Original complete task remains unfinished.'
        state['latest_research_update'] = pin(findings_path)
        state['latest_research_plan'] = pin(plan_path)
        state['latest_controlled_compute_ledger'] = pin(ledger_path)
        state['boundary_checkpoint100_pause_handoff'] = pin(handoff_path)
        state['running_jobs'] = [] if terminal else [{'kind': 'explicitly_authorized_background_fixed1000_CPU', 'run_dir': str(RUN), 'protocol': pin(protocol_path), 'launch': pins['hidden_worker_launch'], 'pid': read(pins['hidden_worker_launch']['path'])['pid'], 'last_saved_qualified_iteration': 100, 'live_status_requires_existing_PID_check_after_resume': True}]
        branch = state.setdefault('full151_native105_boundary_impulse_WM_next', {})
        branch.update(status='FIRST_PAIRED100_ACTUAL_SAVED_HEALTH_AND_ETA_PASS', training_recipe_frozen=True, training_started=True, actual_native_component_preflight_executed=True, accuracy_or_benefit_qualified=False, first_checkpoint100_saved_health_pass=True, goal_pause_due=True, background_continue_to_final=True, pins=pins)
        branch['limitations'] = 'Initial function/optimization/serialization only; no task/Actor/physicalgradient/forcecap/ego certificate. Self15 unsupervised afterfit. Original failures retained. Fixed1000 health and scientific comparisons remain pending after Goal resume.'
        write(state_path, state, True)
    with (OTHER/'docs/progress-2026-11-05.md').open('a', encoding='utf-8') as f:
        f.write(f'\n\n2026-11-06 {now}: actual boundary initialparity3x152 + independent10; matchedCPU own16 paired100 checkpoints saved, independenthealth{health["passed"]}/{health["total"]}, ETA saved. User Goal pause condition met; background1000 explicitly authorized. Training/saved health not accuracy/CHORD/ego qualification. Details: {findings_path.as_posix()}\n')
    print(json.dumps({'handoff': pin(handoff_path), 'findings': pin(findings_path), 'plan': pin(plan_path), 'ledger': pin(ledger_path), 'terminal_at_snapshot': terminal, 'ETA': eta, 'health_passed': health['passed'], 'health_total': health['total']}))


if __name__ == '__main__':
    main()
