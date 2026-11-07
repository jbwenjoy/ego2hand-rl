# Donor300 cached self15 label-only join v1

本包只重读已经保存的 JSON，给后续固定模型 H8 评价提供原生接触标签。不得读取权重或 NPZ、导入 Torch、运行模型、native state conversion、AD、Env、physics 或 GPU。

来源是 `full151105_gradient_empirical_gradient05_v2/reward_snapshots.jsonl` 的全452记录和 `full151105_state_prediction452_CPU_v1/group_identities452.json` 的全452身份；先核完整唯一身份，再按 H1 原顺序取300 donor，不能取文件前300行。原独审1762已逐 donor 检查 raw found25 与原生 contact15；这里独立建立新的 H1 donor label route，不复用 old152 label JSON。

每条 donor 身份固定为 A0/A2、mode=donor、branch_id=null、k=0..149；group_id 分别 k 与226+k，first_physical_pre_index=4gid、last_physical_index=4gid+3、post20_GPU_row_index=1808+gid。source_frame=200+k，terminal_observation=k+1，valid_next_policy_decision=(k<149)，no_u150=true。reward 捕获为 `after_original_reward_compute_before_Manager_forward`，state_time=`postintegration_kdt+.020`，label_time=`lastsolver_preintegration~kdt+.015`。不得将约15ms的 cached sensor truth解释为 POST20 几何、HO、人类接触或可观测力。

每路径 mapping 必须是 middle=[1,2,6,7,11,12,16,17,21,22]、distal=[3,8,13,18,23]、tips=[4,9,14,19,24]。全25 raw counts须有限非负整数；全15 contact须有限0/1。按 raw>0 的 middle10 与 (distal OR tip)5 重建，逐元素与保存contact的 IEEE FP32 bytes 对比，任何缺失、重复、类型、phase或映射不符整体失败，不筛选。

8个输入别名为 H1_config、H1_report、H1_identities、H1_saved_independent25、reward_snapshots、sensor_mapping_A0、sensor_mapping_A2、physical_saved_independent1762；samefile+SHA绑定各父证据与实际路径。协议和源码在首次合成/实际运行前固定；输出为新目录下 joined300_cached_sensor_labels.json、report.json、source_inventory.json 与来源zip。输出每行保留全部 H1 身份并添加 contact15/raw_found25/label_time/state_time，标签不作为 WM 输入。后续 H8 必须通过新的独立 raw25→15 与 H1 join 审查，才可计算标签准确性，禁止静默回退为无标签评价。

这是已知单条转笔来源两个 donor 的缓存标签资格，不是新的物理采样、独立轨迹或 ego 数据。独立的300标签与286个高度重叠的 H8 窗口不能视为独立样本。
