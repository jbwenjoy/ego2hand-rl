"""Saved-array/source inspection and proposal authoring only; never import a model."""
from pathlib import Path
import ast
import hashlib
import json
import sys
import numpy as np

ROOT = Path('D:/workspace/supermimic-full151-observation-study')
OLD = Path('D:/workspace/supermimic-rough-prefix-learning-study')
NATIVE = Path('D:/workspace/dex-supermimic-research')
OUT = ROOT / 'outputs/full151105_gradient_probe_readonly_preparation_v1'
DOC = ROOT / 'docs/full151_same_warp_control_gradient_proposal_v1.md'
PROTOCOL = ROOT / 'protocols/full151_same_warp_control_gradient_proposal_v1.json'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def pin(p):
    p = Path(p).resolve(strict=True)
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def main():
    if OUT.exists() or DOC.exists() or PROTOCOL.exists():
        raise FileExistsError('Fresh proposal/report paths required')
    sources = {
        'WM_loader105': ROOT/'src/full151_native/world_model_checkpoint.py',
        'Actor_loader105': ROOT/'src/full151_native/checkpoint.py',
        'bundle_loader105': ROOT/'src/full151_native/bundle.py',
        'controller105': ROOT/'src/full151_native/controller.py',
        'runtime105': ROOT/'src/full151_native/runtime.py',
        'native105': ROOT/'src/full151_native/native.py',
        'policy_probe105': ROOT/'src/full151_native/policy_probe.py',
        'evaluation105_v2': ROOT/'src/full151_native/evaluation_v2.py',
        'native_state': NATIVE/'src/dex_supermimic/rl/dex_supermimic/state.py',
        'native_pen': NATIVE/'src/dex_supermimic/rl/dex_supermimic/pen.py',
        'native_snapshot': NATIVE/'src/dex_supermimic/rl/dex_supermimic/snapshot.py',
        'Warp_support': NATIVE/'.venv/Lib/site-packages/mujoco_warp/_src/support.py',
        'Warp_io': NATIVE/'.venv/Lib/site-packages/mujoco_warp/_src/io.py',
        'old104_physical_worker': OLD/'src/rough_prefix/physical_fd_v2.py',
        'old104_transaction': OLD/'src/rough_prefix/probe_transaction_v3.py',
        'old104_capacity': OLD/'src/rough_prefix/capacity.py',
        'old104_consumer_math': OLD/'src/rough_prefix/physical_probe_consumer_math.py',
        'measured_contacts': OLD/'vendor/measured_contacts.py',
        'previous105_WM_CPU_entry': ROOT/'scripts/diagnose_full151_105_WM_saved_cpu_v1.py',
        'this_readonly_preparation': Path(__file__),
    }
    source_pins = {k: pin(v) for k,v in sources.items()}
    symbols = {}
    for k,p in sources.items():
        tree = ast.parse(p.read_text(encoding='utf-8-sig'))
        symbols[k] = {n.name: {'line': n.lineno, 'end_line': n.end_lineno}
                      for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
    paths = {'A0':ROOT/'outputs/full151_native_A0_v1',
             'A2':ROOT/'outputs/full151_v2_policy_eval_primary2000_A2_seed0_v1'}
    checks = {}; contracts = {}; inputs = {}
    for name,d in paths.items():
        t=read(d/'native_reward_transitions.json');e=read(d/'control_events.json')
        evaluation=read(d/'evaluation.json')
        x0=np.asarray([r['x0'] for r in t],np.float32)
        x1=np.asarray([r['x1_reward_snapshot'] for r in t],np.float32)
        labels=np.asarray([r['native_selfcollision15'] for r in t],np.float32)
        u=np.asarray([r['actualu'] for r in t],np.float32)
        with np.load(d/'raw_states.npz',allow_pickle=False) as z:
            shape={k:list(z[k].shape) for k in z.files}
            checks[name+'_raw_full601']=z['full_qpos'].shape==(601,27) and z['full_qvel'].shape==(601,26)
            checks[name+'_raw_clock601']=z['raw_time_s'].shape==(601,) and bool(np.isfinite(z['raw_time_s']).all())
        checks[name+'_150_transitions_events']=len(t)==len(e)==150
        checks[name+'_phase_order']=all(r['phase']==k and e[k]['phase']==k for k,r in enumerate(t))
        checks[name+'_state58_finite']=x0.shape==x1.shape==(150,58) and bool(np.isfinite(x0).all() and np.isfinite(x1).all())
        checks[name+'_actualu20_finite']=u.shape==(150,20) and bool(np.isfinite(u).all())
        checks[name+'_self15_binary']=labels.shape==(150,15) and bool(np.isin(labels,[0,1]).all())
        checks[name+'_event_u_exact']=all(np.array_equal(np.asarray(r['actualu'],np.float32).reshape(20),u[k]) for k,r in enumerate(e))
        checks[name+'_full_horizon']=evaluation['complete'] is True and evaluation['physical_steps']==600 and evaluation['control_intervals']==150
        checks[name+'_known_rough_preserved']=evaluation['physical_reference_qualified'] is False
        if name=='A2':
            action=np.asarray([r['action'] for r in e],np.float32).reshape(150,20)
            checks['A2_action_source150']=bool(np.isfinite(action).all() and np.max(np.abs(action))<=1)
        else:
            checks['A0_action_explicitly_absent']=all('action' not in r for r in e)
        contracts[name]={'directory':str(d),'transition_keys':list(t[0]),'event_keys':list(e[0]),
                         'raw_shapes':shape,'native_task_pass':evaluation.get('native_policy_pass',evaluation.get('native_A0_pass')),
                         'source_action':'saved action[150,1,20]' if name=='A2' else 'DECLARED_ZERO_NOT_SAVED_ACTION'}
        for f in ('native_reward_transitions.json','control_events.json','raw_states.npz','evaluation.json','config.json','fixed_physics_context.npz','runtime.json'):
            inputs[name+'_'+f]=pin(d/f)
    extras={
        'bundle':ROOT/'outputs/full151_native_input_v1/bundle.json',
        'A2_final_health':ROOT/'outputs/full151_v3_primary_A2_final2000_health_v1/report.json',
        'A2_eval_saved_review':ROOT/'outputs/full151_v3_A2_primary2000_policy_saved_review_v1/report.json',
        'A2_training_config':ROOT/'configs/full151-v3-primary2000-A2-seed0-v1.json',
        'previous105_WM_CPU_protocol':ROOT/'protocols/full151105-saved-WM-CPU-diagnostic-v1.json',
        'previous105_WM_CPU_report':ROOT/'outputs/full151105_saved_WM_CPU_diagnostic_v1/report.json',
        'old104_interface_review':OLD/'outputs/current104_physical_probe_interface_review_v1/REVIEW.md',
        'old104_saved_WM_review':OLD/'outputs/current104_physical_v2_WM_CPU_saved_independent_v2/report.json',
        'prior_research_plan_README':Path('D:/workspace/supermimic-chord-study/README.md'),
        'knownrough_findings':ROOT/'docs/full151_contact_coupled_findings_2026-11-04.md',
    }
    inputs.update({k:pin(p) for k,p in extras.items()})
    health=read(extras['A2_final_health']);saved=read(extras['A2_eval_saved_review'])
    checks['A2_final_health_engineering']=health['status']=='PASS_SAVED_ENGINEERING_HEALTH' and health['final_iteration_scope'] is True and health['iteration']==2000 and all(v is True for v in health['checks'].values())
    checks['A2_eval_saved_engineering']=saved.get('engineering_qualified',saved.get('engineering_passed')) is True or all(v is True for v in saved['checks'].values())
    checks['no_framework_import']=not any(n in sys.modules for n in ('torch','warp','mujoco','mjlab'))
    checks={k:bool(v) for k,v in checks.items()}
    protocol={
      'schema':'full151105_same_Warp_control_gradient_diagnostic_proposal:1',
      'status':'proposal_only','execution_status':'NOT_EXECUTED','GPU_authorized':False,'training_authorized':False,
      'mode':'rough_observation_reference_full151_v1','retarget_variant_id':'legacy_H123_full151_SCP8_knownrough_v1',
      'physical_reference_qualified':False,'source_lineage_N':1,'independent_ego_N':0,'training_seed':0,
      'WM_methods':['A2'],'WM_checkpoint':health['checkpoint'],'WM_health':inputs['A2_final_health'],
      'donor_action_sources':['A0_declared_zero','A2_final2000_saved_action_replay'],
      'donor_semantics':'Fresh native action-source replays; not archived-state restoration or fresh closed-loop Actor rollouts.',
      'snapshot_phases':[0,49,99,149],'direction_joint_indices':[0,4,8,12],'eps_rad':[0.002,0.001],
      'baseline_repeats':3,'snapshot_count':8,'branches_per_snapshot':19,'planned_branches':152,
      'donor_substeps':1200,'branch_substeps':608,'planned_total_episode_substeps':1808,
      'branch_order':'snapshot donor-order then k; baseline0/1/2 then epsilon-order, joint-order, sign -1/+1',
      'intervention':'Native process_actions once preserves EMA; replace only cached absolute target by prevalidated uplus/uminus, no second EMA/C2, original apply holds four 5ms steps.',
      'perturbation':'uplus/minus=FP32 clamp(u_center +/- FP32(epsilon)*onehot, exact frozen FP32 C2 innerbounds); save nominal offset, actual endpoints, clipping and d_eff.',
      'primary_scalar':'FP32 unscaled native object_pos + object_rot terms (2+8), goal at preaction k; step object reward=.02*score separately; goal k+1 secondary.',
      'primary_slope':'F64(F32 score_plus)-F64(F32 score_minus), divided by 2*epsilon; no F32 subtraction',
      'primary_WM_comparison':'grad_u score(WM(x58,u_center),goal_k) dot d_eff, with d_eff=F64(uplus-uminus)/(2*epsilon)',
      'primary_units':'dimensionless object score / rad; scaled .02s step reward / rad reported separately',
      'baseline_state_tolerances':{'q_rad':2e-5,'qd_rad_s':2e-4,'p_m':2e-5,'R_element':2e-5,'v_m_s':2e-4,'omega_rad_s':2e-4,'unscaled_object_score':2e-5},
      'classification':{'signal_floor':'max(1e-5,10*max(baseline_range,baseline_donor_error))','stable_nonflat':'both absolute score differences exceed floor, same slope sign, relative disagreement<=.5','relative_disagreement':'abs(s0-s1)/max(abs(s0),abs(s1),1e-12)','labels':['NOT_RUN','CLIPPED_ASYMMETRIC','FLAT_OR_BELOW_NOISE','NONFLAT_UNSTABLE','STABLE_NONFLAT'],'contact_signature_change':'orthogonal diagnostic flag, never silently remove rows'},
      'restore_qualification':'New105 explicit inventory + integration bitexact + 3 fresh baseline repeats; old104 qualification not inherited.',
      'capacity':'External host OR/UNKNOWN ledger before every restore, after each of four substeps, branch finish, donor terminal, final; never reset flags or restore ledger.',
      'scope':'Local same-Warp finite-interval secants vs WM self derivative, not true/unique physical gradient, HO force gradient, contact feasibility or full-motion qualification.',
      'proposed_worker_budget_seconds':{'soft':180,'hard':240},'retry':False,'root_authorization_required':True,
      'implementation_status':{'worker105':'PENDING','transaction105_inventory':'PENDING','raw_sensor_capture':'PENDING','baseline_repeat_qualification':'PENDING','CPU_consumer105':'PENDING','current_schema2_A2_WM_CPU_load':'PENDING','budget_actual_engineering_qualification':'PENDING'},
      'source_pins':source_pins,'input_pins':inputs,
    }
    write(PROTOCOL,protocol)
    text = '''# Full151/105同Warp控制差分诊断草案

状态：**proposal_only / NOT_EXECUTED**。本文件不授权物理、模型前向、训练或GPU。本次只读源码和已保存JSON/NPZ，未导入Torch/Warp/MuJoCo/Env。参数为下一次实施前供Root审阅的一套有限方案，不是已冻结执行计划。

## 已有证据与缺口

实际WM文件为`src/full151_native/world_model_checkpoint.py`，不是`world_model105_loader.py`。原119f加载器可严格消费105、显式方法/最终2000健康/训练配置/同bundle与actual8；物理xi为46维，不能用critic context368代替。`forward(x58,u20)`附加xi46和zero174，返回state58与self15 logits；不含EMA或C2，冻结参数仍允许u输入自动微分。源码`validate_health_metadata`使用显式expected_method/expected_health_auditor，后续绑定primary_matched/实际config/2000，没有schema1、A1L或旧e9f194 learning硬编码；Actor105检查root五个extra和动态execution_source_pins（逐pin核hash），只固定仍沿用的controller/pure源。因此源码接口可接schema2/A2，但不会独立重审prior/capacity，须传新5950健康producer及实际final健康。当前schema2 A2的实际WM加载/前向仍PENDING，本次没有加载checkpoint。若未来实际guard拒绝，须另新隔离消费者修复，不能绕过或猴补旧loader。

旧105诊断只测历史schema1 A1L WM、A0/A1L各150条保存转移及WM内部FD。23PASS是工程/有限/不变性检查；不是预测精度门或物理梯度资格。旧104同Warp结果亦不能转移到151/150、105、解析速度及新控制签名。旧104事务`_native_profile`硬编码`_rough_*`，新上下文是`_full151_*`，必须另版本适配并审完整inventory；不能直接把旧成功当新恢复证明。

当前schema2 A2 final2000健康196/196、完整原生评估saved审188/188工程通过，任务7/13通过，位置553.991mm/笔轴150.494°/穿透28.4029mm且掉落、非手接触、速率超量。其任务失败原样保留，不因此换checkpoint、换状态或缩短窗口。当前新A1L长训练正在进行；本草案不预设其结果，也不声称已有三臂新WM公平对照。

## 单一有限设计

先用**一个已固定A2 final2000 WM**，两个fresh donor：A0零残差、A2最终评估保存的150个clamped action按原顺序重放。A0原日志没有action字段，必须显式`declared_zero`；A2有action和policy_input155。fresh donor是动作来源类别，不是旧闭环轨迹复现，不重新运行Actor，不加载WM到物理进程。旧轨迹和fresh漂移独立列报，WM只消费fresh snapshot。

每条完整150 interval donor在k=[0,49,99,149]截取4个pre-action状态：前一原生4×5ms、Manager forward、sense/command/obs已完成，当前process_actions尚未发生。共8 snapshots，每个先3次baseline，再4个onehot关节[0,4,8,12]×两eps[.002,.001]rad×正负；19分支/状态、152分支/608步，连同两donor1200步共1808集成步。无随机方向、状态挑选、重试或history择优。k149只产生obs150/timeout，不查u150、不自动reset。该最后一次物理state/scalar响应可比较，但不是有效后继policy决策或bootstrap资格；phase150必须survival0，无第151个控制决策。

捕获后的分支调用原`env.step(source_action)`，原process_actions恰一次并保持EMA残差c1/c2；只在它之后改cached absolute `_targets`。u±以**执行目标空间**构造：由fresh中心u32加减eps32，在原硬∩ctrl且内收20mrad的FP32域预验证。中心及端点均须C2 identity，真正分支不再投影、不改变残差，不额外forward或写q/qd/物体。原apply_actions每5ms硬限须identity，四步实际ctrl逐位等u±。保存clip、不对称中心偏移与实际d_eff；这不等于Actor action→EMA→C2的全链导数。

## 相同标量、相同时间和单位

主标量为原生物体两项未乘dt的FP32 score：s=2 exp[-(||p−pg||/.05)^2]+8 exp[-(e_axis/.04)^2]；e_axis为±r轴端点平均距离，r从冻结command axis_point_radius_m读取并断言实际.1m，不另猜半径。目标采用preaction k，与原native reward一致；k+1仅并列次口径。step物体reward=.02*s另列，不能混分母。

物理x1来自`PenSimReader.attach_capture`的`reward_snapshot.physics`（20ms postintegration、Manager最终forward前）。self15和torque来自末次solver的15ms preintegration时相，必须单独标时钟；x1_after_Manager只作次记录。物理和WM同输入x58、actualu20、xi46、相同goal，WM不再次EMA/C2。保存两个FP32 term和其FP32 sum；独立F64几何核验只用预声明8eps32算术界。FD为两个F32 score先转F64再相减/2eps，单位score/rad；WM AD主比较为∇u s(WM(x,u))·d_eff，nominal onehot次列。不得把F64重算换成native主标量。

## 恢复、实际context与容量资格

可复用API见source_pins及source_symbols.json：`full151_runtime`外层、`full151_controller`内层，构造Manager+PenSimReader；`install_actual_context`保持CPU nominal MJB，actual8复制后`set_const_0(restore=True)`，两次真实readback，xi1×46与bundle逐位一致；analytic非零init和defaultmocap沿现有reset。实际8字段/shape必须由原源取，不能把nominal MJB当实际随机化context。

`WarpTaskSnapshot(SimpleNamespace(env,reader)).restore()`恢复原WP model/data数组、Torch张量/manager容器与RNG，但NumPy、函数closure、foreign object及新105属性必须由新事务sidecar显式cover。snapshot需包括solver warmstart、sensor/cache/obs histories、termination、motion phase、action `_raw/_processed/_targets/_targets_valid`、controller pending/active、runtime generation、PEN memory、mocap/exogenous/model实际参数和RNG。`mjSTATE_INTEGRATION` size/signature从实际模型算，set/get bitexact只是必要条件；`get_data_into`是WP→host读回，原raw_states qpos/qvel也不含完整warmstart，不能反构完整snapshot。

overflow会随snapshot恢复，故容量observer不能挂env/reader或进入sidecar。host OR/UNKNOWN账本在每次restore前、每substep后、branch finish、donor terminal和全末读，记录实际dtype/shape/nacon/nefc与source；OR非0、UNKNOWN或读取缺失工程FAIL。不把末次0覆盖早前overflow。固定原容量，不调大容量补跑。

每snapshot三baseline须先全部工程/actualu/clock/integration/inventory合格，并对fresh donor x1按原旧门q2e−5rad、qd2e−4rad/s、p/R元素2e−5、v/ω2e−4和score2e−5比较；fresh baseline失败不得进入方向分支或WM。原始实际误差照存。不同clock来源区分canonical k*.02与全程顺序FP32时钟；不要继承旧.3秒2µs门。

## 非光滑、标签与失败解释

所有计划行必须落表：NOT_RUN、clip/asymmetric、flat/below noise、nonflat unstable、stable nonflat，不能删outlier。score差阈max(1e−5,10×三baseline噪声)，两eps同号且相对差≤.5才stable；边界不对称单列，不冒称中心局部导数。接触geom/efc/类型/normal/active-set与self15改变作为独立非光滑flag；传感阈值跳变不强行求连续真实导数。每eps指同32方向，不能当64独立样本，8状态仍同源/同seed相依。

self15为10中节+5远端(link4 OR tip)的native hand self-collision。它不是HO、力、穿透或人类接触真值。新worker应在权威reward capture同时保存raw finger_collision found和原sensordata以便独立标签重建；现旧full151日志只有self15，不能假称可从HO几何反推。测得HO contact force6/COM wrench、distance和接触signature只作原native测量诊断（必须包含direct torque/侧别与preintegration相位）；WM没有HO力head，不能拿proxy P的梯度当真实HO力梯度。实际力可很大而标量差平，二者不互替。

全151几何SCP候选仍FAIL（HO11.448mm/self11.431mm），声明surface cone302的可行见证是条件假设、无真实contact/wholehand/constitutive资格；本探针不能补这些缺口。FD是有限区间secant，不是真实唯一梯度；WM AD是模型自身FP32CPU导数，二者吻合也不证明物理Jacobian、GPU AMP、训练收益、真实ego或全运动可行。

## 实施边界与待补接口

需要新增105事务profile/新物理worker/入口及独立saved-only finalizer和CPUconsumer，不改旧冻结源。物理进程零WM/optimizer；CPUconsumer用原119f strict loader、显式schema2 A2 final健康producer和2000，保持checkpoint/权重/norm/RNG不变，actual8/xi sourcebound，先独立重算48?不：本方案**24个baseline**及全部工程门，再允许WM。consumer不得继承旧104固定16/560/903字段。

建议预先soft180/hard240秒、共同GPU锁、精确PID/PPID/CreationDate/actualcommand base samefile与终态CIM空、原159c有界metadata I/O、无retry。此为预算提案，Root须在实施前确认并单独授权；超时/未知退出/容量/restore/baseline失败完整保partial，CPU WM状态NOT_RUN。工程门不看任务quality，质量fail保留且不缩短完整horizon。没有新训练或自动后续run。

本次被动数组检查只证明现有150转移/601状态/动作字段与源ABI可接，不能证明未来105事务恢复、GPU容量/长时钟、A2 WM加载或物理分支已合格。详见同目录proposal JSON及只读准备report。
'''
    text=text.replace('48?不：本方案**24个baseline**','本方案**24个baseline**')
    DOC.parent.mkdir(parents=True,exist_ok=True)
    with DOC.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    OUT.mkdir(parents=True)
    write(OUT/'source_pins.json',source_pins)
    write(OUT/'source_symbols.json',symbols)
    report={'schema':'full151105_gradient_probe_readonly_preparation:1','status':'PASS_SAVED_ABI_REVIEW_PROPOSAL_ONLY' if all(checks.values()) else 'FAILED_READONLY_CONTRACT',
            'proposal_status':'proposal_only','executed_probe':False,'checks':checks,'passed':sum(checks.values()),'total':len(checks),
            'contracts':contracts,'source_pins':source_pins,'input_pins':inputs,'protocol':pin(PROTOCOL),'document':pin(DOC),
            'frameworks_imported':[],'physics_steps':0,'WM_forward_calls':0,'Actor_forward_calls':0,'optimizer_updates':0,
            'scope':'Source/JSON/NumPy saved ABI only. All future native restore, WM load and physical-FD qualification remain PENDING.'}
    write(OUT/'report.json',report)
    write(OUT/'READY.json',{'status':'PROPOSAL_ONLY_NOT_AUTHORIZED','report':pin(OUT/'report.json'),'protocol':pin(PROTOCOL),'document':pin(DOC),'readonly_checker':pin(__file__)})
    print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'report':pin(OUT/'report.json'),'protocol':pin(PROTOCOL),'document':pin(DOC)},ensure_ascii=False))
    return 0 if all(checks.values()) else 2

if __name__=='__main__':
    raise SystemExit(main())
