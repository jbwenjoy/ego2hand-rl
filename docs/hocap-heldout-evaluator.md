# 留出动作的物体校验和策略评测入口

`tools/evaluate_wuji_heldout.py` 读取事前登记的源阶段摘要，并核对摘要哈希、实际源姿态、逐帧索引、规则、网格、URDF 和桌面资产。留出参考的 `frozen_recipe` 与阶段摘要中的配方摘要必须一致。开发样本的软件验证必须显式使用 `--development-fixture`，正式评测会拒绝带此开发标签的源。每个新物体使用自己的校验记录，已有开发物体的记录不能替代。[源阶段判据](hocap-heldout-source-stages.md) · [独立入口](../tools/evaluate_wuji_heldout.py)

## 调用顺序

1. 在计划和每参考阶段摘要冻结之后，分别用 `certify --device cpu` 与 `certify --device cuda:0` 为该物体做静置和 50 mm 落体校验。两次均给出 `--checkpoint`、`--parquet`、`--contract`、`--contract-sha256`、`--physics`、`--warp-cache`、`--out`。记录包含源和场景标识、名义质量、动力学模型配置、稳定载荷、落体时间、自由段误差与广义方程残差。生成的 `certificate.json` 同时记录原始数组哈希。物体质量和摩擦仍属于名义先验。
2. 使用 `evaluate` 提供相同的输入、统一验收配置 `--protocol`，以及两份 `--cpu-certificate`、`--gpu-certificate`。这个过程直接调用冻结的主评测，并对所有可评分的候选（包括短轨迹）另外执行两次新环境、相同指令的物理复算。随后对原结果和两次复算逐一采用已冻结的源阶段门限。原始主评测写在 `raw/report.json`，保留原先的腾空、静止对照、释放及精确复算结论。新的 `report.json` 单列三个实际轨迹的所有判据，只替换源阶段定义涉及的三项，并加入有接触时实际运动进展的要求。
3. 正式状态 `passed` 要求三个轨迹各自数值通过且 CPU、GPU 的物体物理证据同时有效。无法从源数据判定阶段则记录 `unscorable_source`。若可由提前终止或绝对轨迹误差独立判定失败，即使载荷存在不确定性也记录 `failed`。其他依赖未验证载荷的结果记为 `provisional_unknown_contact_forces`。轨迹和阶段的分母一并保留。
4. 标准评分结束后，独立的 `disturb --heldout-score REPORT --checkpoint ... --parquet ... --warp-cache ... --protocol configs/hocap_wuji_disturbance.json --out ...` 复用同一物体证据，比较两种方向的可动腕扰动与已存指令的开环对照。它记录 `heldout_pairing.json`，不改变主分数。

## 开发样本验证

验证过程只使用已使用过的 G04_2 开发纸盒和冻结阶段摘要，未读取预留参与者的策略结果。CPU 物体校验通过，两种初态末段桌面载荷约为 0.998、1.003 倍物重，完整平移动力学方程残差 RMS 约 0.00028、0.00036 N。单轨迹检查点与两次独立复算均在 46/140 区间终止，所有完整性条件判为失败。使用 CPU 软件验证模式时，缺少的 CUDA 证据会被明确列出。独立的评分逻辑测试拒绝了静止物体、无手载荷和继承的绝对姿态失败条件，对理想合成数据给出预期的数值通过。开发集曾通过旧门限的真实仿真记录在新阶段规则下未通过末段稳定性。旧记录的判定保持其原口径。[CPU 物体记录](../runs/heldout_eval_fixture_cpu_v4/certificate.json) · [三次实际 CPU 轨迹](../runs/heldout_eval_fixture_score_v2/report.json) · [判据映射检查](../runs/heldout_eval_fixture_gate_mapping_v2/report.json)
