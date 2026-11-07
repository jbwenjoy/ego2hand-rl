"""Only saved source/geometry metadata -> functional-support recommendation."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent;P=H.parent
def pin(p):
    p=p.resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(p.read_text(encoding='utf8'))
sources=[]
for branch in ['exact_source_original_hand','source1mm_opposed_side_original_hand']:
    p=P/'surface_assembly_v11'/f'{branch}_completed_v1/seed1_rank0.json';r=read(p)
    row=next(x for x in r['final_geometry']['state']['phone_geometry_rows'] if x['geom_id']==41)
    sources.append(dict(branch=branch,record=pin(p),third_tip_current_gap_m=row['exact_gap_m'],reachability_qualified=False))
memo=dict(schema='functional_grasp_mode_route_memo:1',status='SOURCE_RECOMMENDATION_NOT_PROTOCOL_OR_EXECUTION',
    recommendation='建议新增具名机器人三指功能支持分支，完整保留旧二指结果与门，不把新分支通过写成旧2pad通过。',
    goal_basis=dict(source=pin(P/'outputs/goal_extension_authorization_v1.json'),
        interpretation='原目标是无本体ego到Hand2可物理执行的接触动力学重定向，没有要求机器人接触数等于两个人体unsigned proximity标签。功能映射应检验物体任务/运动及承载，而不是仅复制接触数量。'),
    motivation='第三支持用于检验形态差异下承载和抗矩功能；更容易通过本身不是理由。须同时报告原ego材料点/物体姿态偏差、真实载荷及执行器努力。',
    source_pins=dict(scene=pin(P/'hand2_holding_v1/scene.py'),compiled_binding=pin(P/'hand2_holding_v1/compiled_nominal_v2/binding.json'),
        geometry_metadata=pin(P/'retarget_geometry_v2/report.json')),
    third_tip=dict(geom_id=41,name='right_finger3_tip_sensor_col',candidate_joints=list(range(8,12)),configurations=sources,
        scope='23.47/44.37mm只是当前固定姿态的凸体分离距离，不证明可达或不可达。tip_sensor几何不等于已测功能指腹/真实摩擦面。'),
    new_branch_proposal=dict(name='functional_tripod_support_v1',designated_geoms=[15,28,41],
        fixed='原腕/尺度/真关节、ctrl、tau界及材料/增益/solver不变。现scene所有phone-hand显式pair已使用相同mu/condim；不能仅改标签来制造几何。',
        candidate='真实放开第三指，检查实际filled mesh点/法向、全部phone/self几何和原生接触；重复几微米接触不能冒独立受力patch。原ego材料区域和物体姿态让步必须具名记录。',
        dynamics='新协议保留原2s、双dt、漂移/角度/速度/穿透/警告/noexternal/table门；末0.5s三pad同时Fn>0.1N覆盖>=95%。非指定刚性link禁止供力，其数值零容差须预先锁定；并列输出旧2pad判定。',
        force_scope='保真实硬件/PD界并报告每指和总法向载荷、实际tau。新增一指不代表无限预载；理想点LP仍只作提案诊断，不是软接触动态持握必要门。',
        parameter_scope='mass0.2kg、mu0.7、bbox COM/惯量及仿真重力仍是假设，不是真实参数校准。'),
    next_evidence=[
        '明确原ego重力与固定腕注册、功能映射相对原材料点/物体姿态的让步；未知不能靠新增一指掩盖。',
        '先证明第三tip的真实可达/非穿透/法向及有效力臂，再验证原生预载、控制可达和2s双dt无辅助结果；所有失败保留。',
        '若名义三指动态成功，再测有限先验/姿态邻域；不能由一个名义点声称鲁棒或唯一辨识参数。',
        '先做不依赖WM的原生基准。旧59输出只有coverage2/force5，且将第三tip计为nonpad；后续必须新建三pad数据/标签/head namespace，不能静默沿用。',
        '当前all20动作梯度按原2pad合同完成。61帧运动、独立ego轨迹、SuperMimic/CHORD同预算和硬件观测契约继续保留。'],
    premise_scope='Root报告固定5000模型TEST姿态不敌ballistic/CV、常量2pad不足0.2s；本memo未重算模型指标，也未据此调整新阈值。',
    qualification=dict(old_2pad_success=False,new_3pad_geometry_or_holding=False,real_ego_retarget=False,functional_pad_calibration=False),
    actual_new_model_FK_LP_physics_NN_training_calls=0)
out=H/'memo_v1.json'
with out.open('x',encoding='utf8') as f:json.dump(memo,f,indent=2,ensure_ascii=False,allow_nan=False)
print(json.dumps(pin(out)))
