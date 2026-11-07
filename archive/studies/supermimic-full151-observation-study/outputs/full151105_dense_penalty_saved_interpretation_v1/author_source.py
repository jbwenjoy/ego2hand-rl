"""Author interpretation of already saved dense outputs; no new science calls.

This is an AFTER-OUTCOME descriptive artifact, not an independent audit or a
preregistration. Every one of the 1808 rows is kept in the descriptive CSV.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np

ROOT=Path('D:/workspace/supermimic-full151-observation-study')
RUN=ROOT/'outputs/full151105_solver_phase_penalty_GPU_qualified_CPU_v1'
REPORT_SHA='416d8dfcfe01a910a406ba78cdddc28628717bce5a45a917b69f6af583699bf2'
INPUTS={
 'dense_report':RUN/'report.json','dense_rows':RUN/'rows_all1808.json','dense_arrays':RUN/'all1808_proxy_scores.npz',
 'sample_buffers':RUN/'proxy_sample_buffers.npz','geometry_catalog':RUN/'geometry_ID_mapping.json',
 'dense_independent35':ROOT/'outputs/full151105_solver_phase_penalty_GPU_qualified_saved_independent_v1/report.json',
 'sparse_bonus_independent523':ROOT/'outputs/full151105_trained_bonus_saved_independent_v1/report.json',
 'WM_physical_independent656':ROOT/'outputs/full151105_WM_analysis_saved_independent_v2/report.json',
 'training_log_summary':ROOT/'outputs/full151_trained_prior_log_summary_v1/report.json',
 'prepolicy100_comparison':ROOT/'outputs/full151_schema2_three_arm_prepolicy100_comparison_CPU_v1/report.json',
 'fixed_final_three_arm_summary':ROOT/'outputs/full151_schema2_three_arm_primary2000_summary_v1/comparison.json',
 'rough_reference_SCP_receipt':ROOT/'outputs/full151_contact_coupled_scp_v1/receipt.json',
 'native_A0_independent_saved':ROOT/'outputs/full151_native_A0_saved_independent_v1/report.json',
 'current_consumer':ROOT/'src/full151_gradient/penalty_observability105_v3.py',
 'WM_CPU_loader':ROOT/'src/full151_native/world_model_checkpoint.py'}


def need(v,m):
    if not v:raise ValueError(m)


def pin(path):
    p=Path(path).resolve(strict=True);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output).resolve();need(out.parent==ROOT/'outputs' and not out.exists(),'Fresh study output required');out.mkdir()
    source=pin(__file__);(out/'author_source.py').write_bytes(Path(__file__).read_bytes())
    before={k:pin(v) for k,v in INPUTS.items()};write(out/'source_and_input_pins_before.json',{'author_source':source,'inputs':before})
    need(before['dense_report']['sha256']==REPORT_SHA,'Wrong actual dense report')
    report=read(INPUTS['dense_report']);rows=read(INPUTS['dense_rows']);catalog=read(INPUTS['geometry_catalog'])
    independent=read(INPUTS['dense_independent35']);logs=read(INPUTS['training_log_summary']);warmup=read(INPUTS['prepolicy100_comparison'])
    wm=read(INPUTS['WM_physical_independent656']);cohort=read(INPUTS['fixed_final_three_arm_summary'])
    with np.load(INPUTS['dense_arrays'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    with np.load(INPUTS['sample_buffers'],allow_pickle=False) as z:sample_ids=z['sample_geom_ids'].copy()
    checks={}
    def check(k,v):checks[k]=bool(v);need(checks[k],k)
    check('all1808_rows_and_original_array_shapes',len(rows)==1808 and a['sampled_sdf_m'].shape==(1808,1344) and a['per_geom_min_sdf_m'].shape==(1808,21))
    check('row_indices_complete_without_selection',[r['row_id'] for r in rows]==list(range(1808)))
    check('current_frozen_consumer_c489',before['current_consumer']['sha256']=='c4892a933860c0415e4a229c181a94f88755e6c529bd27c5f79bdc56be8da47a')
    check('independent_saved35_actual_pass',independent['passed']==independent['total']==35 and all(v is True for v in independent['checks'].values()))
    native=a['native_HO_depth_m']>.003;positive=a['P']>0
    check('native_and_P_labels_equal_all_saved_rows',np.array_equal(native,np.array([r['native_HO_depth_gt3mm'] for r in rows])) and np.array_equal(positive,np.array([r['proxy_P_positive'] for r in rows])))
    confusion={'both_positive':int((native&positive).sum()),'native_positive_proxy_zero':int((native&~positive).sum()),
               'proxy_positive_without_nativeHO3mm':int((~native&positive).sum()),'both_nonpositive':int((~native&~positive).sum())}
    check('confusion_all1808_not_positive_only',sum(confusion.values())==1808 and confusion=={'both_positive':42,'native_positive_proxy_zero':2,'proxy_positive_without_nativeHO3mm':0,'both_nonpositive':1764})
    strata={}
    for path in ('A0','A2'):
        for mode in ('donor','baseline','direction'):
            idx=np.array([i for i,r in enumerate(rows) if r['path']==path and r['mode']==mode],np.int64)
            strata[path+'/'+mode]={'rows':len(idx),'native_positive':int(native[idx].sum()),'P_positive':int(positive[idx].sum()),'misses':int((native[idx]&~positive[idx]).sum())}
    check('donor1200_and_replayed608_kept_separate',sum(v['rows'] for k,v in strata.items() if k.endswith('/donor'))==1200 and sum(v['rows'] for k,v in strata.items() if not k.endswith('/donor'))==608)
    table={g['id']:g for g in catalog['native_geom_catalog']};gindex={int(g):i for i,g in enumerate(a['geom_ids'])}
    missed=[]
    for i in np.flatnonzero(native&~positive):
        row=rows[int(i)];details=[]
        for pair in row['all_violating_pairs_SECONDARY']:
            g=pair['hand_geom'];sdf=float(a['per_geom_min_sdf_m'][i,gindex[g]])
            details.append({'native_pair':pair['geom'],'hand_geom_id':g,'hand_geom_name':table[g]['name'],'sample_points':int((sample_ids==g).sum()),
                'native_depth_mm':pair['depth_m']*1000,'same_geom_proxy_sample_depth_mm':max(0.,-sdf)*1000,
                'same_geom_depth_gap_mm':(pair['depth_m']+sdf)*1000,'native_above_nominal_3mm_mm':(pair['depth_m']-.003)*1000,
                'proxy_same_geom_below_nominal_3mm_mm':(.003+sdf)*1000})
        missed.append({'row_id':int(i),'path':row['path'],'k':row['k'],'source_frame':row['source_frame'],'substep':row['substep'],
            'canonical_solver_pre_time_s':row['canonical_pre_tick']*.005,'native_max_depth_mm':row['native_generated_HO_depth_m']*1000,
            'global_proxy_depth_mm':max(0.,-float(a['minimum_sdf_m'][i]))*1000,'global_proxy_minimizer_geom_ids':np.unique(sample_ids[a['sampled_sdf_m'][i]==a['minimum_sdf_m'][i]]).tolist(),
            'P':float(a['P'][i]),'BP':float(a['BP'][i]),'exact_FP32_boundary':row['proxy_min_sdf_exactly_minus_FP32_allowance'],
            'primary_exact_deepest_pairs':row['all_exact_tied_deepest_pairs'],'all_violating_pairs_details_SECONDARY':details,
            'coverage':row['primary_deepest_tie_coverage'],'classification':'covered-geom undersensitivity candidate; not a unique causal attribution'})
    check('both_misses_covered_not_exact_FP32_boundary',all(not m['coverage']['geom_set_gap_ids'] and m['coverage']['covered_proxy_undersensitivity_ids'] and not m['exact_FP32_boundary'] for m in missed))
    check('same_geom_not_cross_geom_depth_comparison',all(d['sample_points']==64 and d['same_geom_depth_gap_mm']>0 for m in missed for d in m['all_violating_pairs_details_SECONDARY']))
    check('all1800_policy_updates_logged_positive_P_in_each_prior_arm',all(logs['methods'][m]['policy_updates']==logs['methods'][m]['positive_P_mean_updates']==1800 for m in ('P_ONLY','A2')))
    check('prepolicy_divergence_scope_unchanged',warmup['WM_parameters_byte_identical_all3'] is False and warmup['failure_parameters_byte_identical_all3'] is False and warmup['Actor_Critic_parameters_original_and_identical_all3'] is True)
    check('WM_sign_result_is9_dependent_groups_not18',wm['details']['classification_counts']=={'STABLE_NONFLAT':9,'NONFLAT_UNSTABLE':0,'FLAT_OR_BELOW_NOISE':23} and all(v['eligible']==9 and v['sign_correct']==5 for v in wm['details']['agreement'].values()))
    check('all_three_original_task_conjunctions_remain_false',all(cohort['methods'][m]['task_pass'] is False for m in ('A1L','P_ONLY','A2')))
    rough=read(INPUTS['rough_reference_SCP_receipt']);a0=read(INPUTS['native_A0_independent_saved'])
    check('rough_reference_and_native_A0_failures_not_upgraded',rough['status']=='CPU_COUPLED_CANDIDATE_FAIL_RETAINED' and rough['physical_qualified'] is False and a0['native_A0_pass'] is False)
    check('no_new_model_native_or_GPU_module_import',all(m not in sys.modules for m in ('torch','mujoco','warp')))
    with (out/'all1808_description.csv').open('x',newline='',encoding='utf-8') as f:
        fields=['row_id','path','mode','k','substep','canonical_pre_tick','native_depth_mm','proxy_sample_depth_mm','P','BP','native_positive','proxy_positive','miss','primary_deepest_geom_ids','primary_gap_geom_ids','primary_covered_undersensitivity_ids']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i,row in enumerate(rows):
            cv=row['primary_deepest_tie_coverage']
            w.writerow({'row_id':i,'path':row['path'],'mode':row['mode'],'k':row['k'],'substep':row['substep'],'canonical_pre_tick':row['canonical_pre_tick'],
                'native_depth_mm':a['native_HO_depth_m'][i]*1000,'proxy_sample_depth_mm':max(0.,-float(a['minimum_sdf_m'][i]))*1000,'P':a['P'][i],'BP':a['BP'][i],
                'native_positive':bool(native[i]),'proxy_positive':bool(positive[i]),'miss':bool(native[i]&~positive[i]),
                'primary_deepest_geom_ids':json.dumps(cv['hand_geom_ids']),'primary_gap_geom_ids':json.dumps(cv['geom_set_gap_ids']),'primary_covered_undersensitivity_ids':json.dumps(cv['covered_proxy_undersensitivity_ids'])})
    write(out/'miss_details.json',missed)
    after={k:pin(v) for k,v in INPUTS.items()};check('all_saved_inputs_preserved',before==after and pin(__file__)==source)
    summary={'schema':'full151105_dense_penalty_author_saved_interpretation:1','status':'COMPLETED_AUTHOR_SAVED_INTERPRETATION_NOT_INDEPENDENT_AUDIT',
        'author_source':source,'input_pins':before,'checks':checks,'passed':sum(checks.values()),'total':len(checks),'rows':1808,'confusion':confusion,'strata':strata,'misses':missed,
        'FP32_allowance_m':float(np.float32(.003)),'nominal_to_FP32_allowance_difference_m':float(np.float32(.003))-.003,
        'recommendation':'Next candidate only: fixed existing A2 CPU WM, ALL452 matched20ms groups (300 donor+152 replayed), 1 strict loader and 1 batch452 forward/noAD; new explicit prereg/config required. No execution here.',
        'source_API_feasibility':{'native_state58':58,'actual_u20':20,'physical_xi46':46,'zero_disturbance174':174,'dynamics278':278,'total_WM298':298,'outputs':['next_state58','self15 logits, NOT HO'],
            'state_inputs':'qualified_GPU_prestate1808 row at each 4-step group start','targets':'qualified native_GPU_all2260_raw final452 post20 rows, preserving group order/identity',
            'control':'saved actual ctrl, require bitexact same four substeps; never second EMA/C2','model':'fixed actual A2 final2000, strict105 health+loader unchanged',
            'proposed_cost_not_executed':{'strict_loader_calls':1,'internal_torch_checkpoint_loads':2,'Actor_validation_constructor':1,'Actor_forward_calls':0,'internal_passive_MJB_for_Actor_validation':1,'WM_forward_calls':1,'batch_rows':452,'AD':0,'new_physics':0,'new_training':0},
            'main_limit':'Endpoint prediction errors cannot alone validate local control AD, physical/HO gradients, or method causality.'},
        'interpretation_limits':['Native positive means saved generated HO depth>3mm, not global positive/negative collision truth or force threshold.',
            '1808 rows/44 positives and608 replayed rows are dependent; Nseed1/Nsource1/Nego0.',
            'Runtime RNG/buffer invariants are producer source-backed digest booleans; no independent raw RNG reconstruction here.',
            'No target/state/proxy/WM/gradient/dynamics qualification is upgraded; old CPU conversion FAIL and rough reference/task failures remain.'],
        'new_model_loads':0,'new_MJB_FK_collision_queries':0,'new_objective_forwards':0,'new_AD_GPU_Env_physics_training':0}
    write(out/'report.json',summary)
    md=f'''全1808行同solver相位结果说明（作者事后解读，非独立审计）

固定原3mm标准：44行native生成HO深度超界，42行P>0；42行共同阳性、2行native阳性/P=0、0行proxy阳性但无观测nativeHO超界。全部1808行保留在all1808_description.csv。独立数值权威为Contact的35/35报告，本文件不替代它。

1200行fresh donor中，A0为30/29/1（native阳性/P阳性/漏检），A2为14/13/1。96行baseline和512行direction全部native≤3mm且P=0。因此旧8个局部名义状态、160个20ms末状态的P=0是稀疏时相覆盖的事实，不能外推全段proxy不响应，更不能推成训练期间P未激活。训练日志里P_ONLY和A2各1800次policy update的active P均值都>0；active加权P均值约1.08115/1.07391。日志只保存imagined active均值，不能恢复每行阳性比例、真实接触或Actor/u梯度；它与真实donor分布不同，不直接比较均值。

两次漏检都不是缺失geom：

| 路径/solver预时刻 | native最深pair | native深度mm | 同一geom采样深度mm | 同geom差mm |
|---|---|---:|---:|---:|
| A0 k11/sub4，0.235s | finger2_link3 geom27 / object geom1 | 3.030756 | 2.964228 | 0.066528 |
| A2 k5/sub3，0.110s | finger2_link3 geom27 / object geom1 | 3.573335 | 2.814320 | 0.759015 |

每个geom都有原64采样点。A0 native超界30.756μm，proxy低于3mm为35.772μm。A2最深geom27并非proxy全局最深geom：全局最深来自geom31 finger2_tip_sensor，为2.971762mm；其native次要超界pair为3.072037mm，同geom差0.100275mm。A2最深pair的native超界达0.573335mm，不能把两次都叫同一种微小边界现象。两行都不在FP32 allowance的精确等值边界；float32(0.003)与0.003只差{summary['nominal_to_FP32_allowance_difference_m']:.12g}m，不能据该scalar转换解释这些深度差。

这些是“已覆盖geom的proxy欠敏感候选”，不是唯一采样原因。native候选pair距离、有限表面点SDF/几何表示和聚合是不同量；没有新native查询或加密采样对照，不能唯一归因某一环节。全体非阳性行也不证明全局无穿透。原threshold、全部exact deepest tie-set与secondary全部超界pair均未更改。

下一最小研究行动建议（仅可行性，未执行）

优先Root提出的固定现有A2 WM、全部452个匹配20ms组的预测误差诊断，先于针对这两点修proxy或直接换prior/加训练。数据已经有300个donor组和152个baseline/direction组；每组从GPU-qualified prestate取首行，目标从同次native GPU replay的452个post20行取，actualu要求四substep逐位一致，保持phase/顺序，绝不再次EMA/C2。冻结119f CPU loader支持任意正batch的FP32 CPU state58/u20，内部固定xi46与zero174→278，再加u20→298；输出next58和self15。一次strict加载（内部2次checkpoint读取与Actor身份构造，0Actor forward）+一次batch452 WM forward即可；不需新physics、训练或AD。执行前仍须新明确协议/配置、完整原始身份与输入gates、只读normalizer/RNG记录，当前没有运行。

主结果按q/qd/p/R/v/ω原单位和path/phase/donor-versus-replayed分层保留全部452行，不按42个P阳性挑样本。这个诊断更直接回答稀疏4相位是否代表WM全段误差；它本身不把误差曲线变成物理梯度证书。现有physical/WM比较仅32个依赖方向组中9组稳定、两eps各5/9同号（含3组clip），不能当18个独立样本或HO梯度资格。下一步若要解释AD与finite-interval差异，需另预先固定分析，不能靠这次预测误差自动下结论。

当前三方法的WM/failure在尚无Actor更新和prior forward的checkpoint100就已非逐字节一致；Actor/Critic仍原始相同。因此旧单seed结果保持“相同初始参数与预算”的描述性配对，不是共同训练轨迹的严格因果对比。未来方法归因需要另设计共同WM/共同输入的控制或共同warmup，不能用本dense表补造因果证据。已知rough reference和三方法原task conjunction FAIL全部保留；不用等待完美kinematic reference才能研究动力学，也不能用proxy大多检出替代reference/动力学/任务资格。本交付不新建H1实验、不训练、不重物理。

证据与范围：report.json逐一pin dense产物、独立35、旧bonus523、WM656、全1800日志、prepolicy100与fixed-final三臂总结；source和输入哈希前后不变。没有模型、目标、MJB、FK/query、AD、GPU、Env或物理调用。
'''
    (out/'diagnostic.md').write_text(md,encoding='utf-8')
    write(out/'artifact_pins.json',{f.name:pin(f) for f in out.iterdir() if f.is_file()})
    print(json.dumps({'passed':summary['passed'],'total':summary['total'],'report':pin(out/'report.json'),'note':pin(out/'diagnostic.md')},indent=2))


if __name__=='__main__':main()
