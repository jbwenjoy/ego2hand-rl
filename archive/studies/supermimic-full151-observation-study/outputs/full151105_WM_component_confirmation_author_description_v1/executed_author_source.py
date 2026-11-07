"""Author description only: saved native deltas, no model/Log/FW/AD or physics."""
from pathlib import Path
import argparse,csv,hashlib,json,os
ROOT=Path('D:/workspace/supermimic-full151-observation-study')
RUN=ROOT/'outputs/full151105_WM_component_confirmation_CPU_v1'
OUT=ROOT/'outputs/full151105_WM_component_confirmation_author_description_v1'
REPORT_SHA='82b88d40d26aa0a940b09777a40e741c62dfecf2c2e14ba5bbbe51040b100908'
ARMS=('pointwise','pointwise_plus_common_pre_response')
BLOCKS={'q':(0,20,'rad'),'qd':(20,40,'rad/s'),'p':(40,43,'m'),
        'logR3':(43,46,'rad'),'v':(46,49,'m/s'),'omega':(49,52,'rad/s')}
RAW={'q':(0,20,'rad'),'qd':(20,40,'rad/s'),'p':(40,43,'m'),
     'R_raw9':(43,52,'dimensionless'),'v':(52,55,'m/s'),'omega':(55,58,'rad/s')}


def pin(p):return {'path':str(Path(p).resolve()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def csvwrite(p,rows):
    with p.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',action='store_true');args=ap.parse_args()
    if not args.run:print(json.dumps({'status':'AUTHOR_SAVED_DESCRIPTION_DECLARED_NOT_RUN','script':pin(__file__)}));return
    assert not OUT.exists();assert pin(RUN/'report.json')['sha256']==REPORT_SHA
    sources={name:pin(RUN/name) for name in ('report.json','protocol.json','all152_identities.json','all64_pairs.json','pointwise_evaluation_arrays.npz','pointwise_plus_common_pre_response_evaluation_arrays.npz')}
    manifest=read(RUN/'output_pins.json')
    for name,p in sources.items():assert manifest[name]['sha256']==p['sha256'] and Path(manifest[name]['path']).samefile(p['path'])
    OUT.mkdir();(OUT/'executed_author_source.py').write_bytes(Path(__file__).read_bytes())
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import numpy as np
    import torch
    torch.set_num_threads(4)
    r=read(RUN/'report.json');identities=read(RUN/'all152_identities.json');pairs=read(RUN/'all64_pairs.json')
    all8=[];all6=[];all48=[];raw6=[];global_roundoff={}
    labels=[('A0','A2')[s//4]+'/'+str((24,74,124,148)[s%4]) for s in range(8)]
    byarm={}
    for arm in ARMS:
        with np.load(RUN/(arm+'_evaluation_arrays.npz'),allow_pickle=False) as f:e={k:f[k].copy() for k in f.files}
        mi,pl=e['minus_indices'],e['plus_indices'];h=e['actual_du_rad_F64'];s52=e['scale52_FP32']
        assert mi.shape==pl.shape==h.shape==(64,) and (h>0).all()
        assert all(pairs[p]['pair_id']==p and pairs[p]['minus_branch_id']==int(mi[p]) and pairs[p]['plus_branch_id']==int(pl[p]) for p in range(64))
        pc=e['pred_common52_FP32'].astype('f8');tc=e['target_common52_FP32'].astype('f8')
        response_error=(pc[pl]-pc[mi])-(tc[pl]-tc[mi])
        normalized=(response_error/s52.astype('f8'))/h[:,None]  # h/(1rad), per frozen recipe.
        with torch.no_grad():
            point_elements=(torch.from_numpy(e['point_delta52_FP32'])/torch.from_numpy(s52)).square()
            native_point=float(point_elements.mean())
            point8=[float(point_elements[s*19:(s+1)*19].mean()) for s in range(8)]
        error=e['pred152_FP32'].astype('f8')-e['target152_FP32'].astype('f8')
        assert response_error.shape==normalized.shape==(64,52) and e['point_delta52_FP32'].shape==(152,52)
        assert native_point==r['native52_frozen_criterion_metrics'][arm]['pointwise_loss']
        weighted=0.
        for block,(lo,hi,unit) in BLOCKS.items():
            z=normalized[:,lo:hi];mse=float(np.mean(z*z));dims=hi-lo;contribution=dims*mse/52;weighted+=contribution
            all6.append({'arm':arm,'block':block,'dimensions':dims,'pair_count':64,'component_count':64*dims,'state_unit':unit,
                'normalized_component_RMS_dimensionless':float(np.sqrt(mse)),'normalized_block_MSE_dimensionless':mse,'weight_dimensions_over52':dims/52,
                'contribution_to_all52_MSE':contribution,'raw_finite_response_error_component_RMS':float(np.sqrt(np.mean(response_error[:,lo:hi]**2))),
                'raw_coordinate_secant_error_component_RMS_per_control_rad':float(np.sqrt(np.mean((response_error[:,lo:hi]/h[:,None])**2)))})
            for snapshot in range(8):
                q=z[snapshot*8:(snapshot+1)*8];raw=response_error[snapshot*8:(snapshot+1)*8,lo:hi]
                all48.append({'arm':arm,'snapshot_id':snapshot,'label':labels[snapshot],'block':block,'dimensions':dims,'pair_count':8,'state_unit':unit,
                    'normalized_component_RMS_dimensionless':float(np.sqrt(np.mean(q*q))),'normalized_block_MSE_dimensionless':float(np.mean(q*q)),
                    'raw_finite_response_error_component_RMS':float(np.sqrt(np.mean(raw*raw))),
                    'raw_coordinate_secant_error_component_RMS_per_control_rad':float(np.sqrt(np.mean((raw/h[snapshot*8:(snapshot+1)*8,None])**2)))})
        global_roundoff[arm]={'native_FP32_point_recomputed':native_point,'all64x52_F64_MSE':float(np.mean(normalized*normalized)),
            'dimensions_weighted_six_block_MSE':weighted,'original_producer_MSE':r['native52_frozen_criterion_metrics'][arm]['common_pre_pair_loss'],
            'weighted_minus_producer_F64_roundoff':weighted-r['native52_frozen_criterion_metrics'][arm]['common_pre_pair_loss']}
        for block,(lo,hi,unit) in RAW.items():
            z=error[:,lo:hi];row={'arm':arm,'block':block,'unit':unit,'rows':152,'dimensions':hi-lo,'component_RMSE':float(np.sqrt(np.mean(z*z))),
                'vector_or_Frobenius_RMS':float(np.sqrt(np.mean(np.sum(z*z,axis=-1)))) if block in ('p','v','omega','R_raw9') else ''}
            raw6.append(row)
        byarm[arm]={'point8':point8,'response8':[float(np.mean(normalized[s*8:(s+1)*8]**2)) for s in range(8)],'error58':error}
    for s,label in enumerate(labels):
        row={'snapshot_id':s,'label':label,'point_rows_per_arm':19,'pairs_per_arm':8,'dependent_endpoints':16}
        for arm in ARMS:
            row[arm+'_point_MSE_FP32']=byarm[arm]['point8'][s];row[arm+'_normalized_response_MSE_F64']=byarm[arm]['response8'][s]
            for block,(lo,hi,unit) in RAW.items():
                z=byarm[arm]['error58'][s*19:(s+1)*19,lo:hi]
                row[arm+'_'+block+'_component_RMSE']=float(np.sqrt(np.mean(z*z)))
        all8.append(row)
    csvwrite(OUT/'all8_snapshot_metrics.csv',all8);csvwrite(OUT/'all6_native52_response_blocks.csv',all6)
    csvwrite(OUT/'all8x6_native52_response_blocks.csv',all48);csvwrite(OUT/'all6_raw58_state_error_units.csv',raw6)
    # These are author descriptions, not an independent validity/accuracy audit.
    g=r['native52_frozen_criterion_metrics'];a,b=ARMS
    md=['# 全量 component confirmation：作者保存数据描述','',
        '**原冻结确认条件未满足。** 加响应项的全64对归一化响应 MSE 从 %.9g 降到 %.9g（下降 %.3f%%），但全152 native FP32 point MSE 从 %.9g 升到 %.9g（增加 %.3f%%）。因此原判据仍为 `local_component_confirmation=False`；这是新测试的 tradeoff，不以训练 loss 改善代替。'%(g[a]['common_pre_pair_loss'],g[b]['common_pre_pair_loss'],100*(1-g[b]['common_pre_pair_loss']/g[a]['common_pre_pair_loss']),g[a]['pointwise_loss'],g[b]['pointwise_loss'],100*(g[b]['pointwise_loss']/g[a]['pointwise_loss']-1)),
        '', '本描述只做保存数组算术：无权重读取、模型/Log前向、AD或物理。FP32 point由保存的 native point_delta52/s52 按原 Torch CPU4 规则归约；响应先把 native FP32 common-pre52 转F64，再做正负端点差、除s52、除实际du/(1rad)。两者不与raw58单位误差混合。',
        '', '## 全8个 snapshot（全部19分支和8对/点）','',
        '|snapshot|point-only point MSE|+response point MSE|point-only response MSE|+response response MSE|', '|---|---:|---:|---:|---:|']
    for row in all8:md.append('|%s|%.8g|%.8g|%.8g|%.8g|'%(row['label'],row[a+'_point_MSE_FP32'],row[b+'_point_MSE_FP32'],row[a+'_normalized_response_MSE_F64'],row[b+'_normalized_response_MSE_F64']))
    md+=['','## 全6个 native52 响应块（64对；每分量归一化 MSE）','',
         '|块/维数/原单位|point-only MSE|+response MSE|point-only raw有限响应残差RMS|+response raw有限响应残差RMS|','|---|---:|---:|---:|---:|']
    for block,(lo,hi,unit) in BLOCKS.items():
        v=[next(z for z in all6 if z['arm']==arm and z['block']==block) for arm in ARMS]
        md.append('|%s / %d / %s|%.8g|%.8g|%.8g|%.8g|'%(block,hi-lo,unit,v[0]['normalized_block_MSE_dimensionless'],v[1]['normalized_block_MSE_dimensionless'],v[0]['raw_finite_response_error_component_RMS'],v[1]['raw_finite_response_error_component_RMS']))
    md+=['','归一化残差为 `(model finite response − physical finite response)/s52/(actualdu/1rad)`，无量纲。全局 MSE 为各块 MSE 按维数20/20/3/3/3/3除52加权；不是六块等权。完整 component RMS、贡献、原单位残差及每控制rad的secant残差见CSV。全8×6×2臂均保留，未先筛成功组或flat。','',
         '## raw58 预测误差（另列原单位，不参与确认门）','',
         '|块/原单位|point-only component RMSE|+response component RMSE|','|---|---:|---:|']
    for block,(_,_,unit) in RAW.items():
        v=[next(z for z in raw6 if z['arm']==arm and z['block']==block) for arm in ARMS]
        md.append('|%s / %s|%.8g|%.8g|'%(block,unit,v[0]['component_RMSE'],v[1]['component_RMSE']))
    md+=['','位置向量RMS为 %.6f→%.6f mm；Rraw9是无量纲矩阵元素/Frobenius误差，native52的logR3是rad，二者不可换称旋转角误差。'%(r['raw58_state_error_original_units'][a]['p']['vector_RMS']*1000,r['raw58_state_error_original_units'][b]['p']['vector_RMS']*1000),
         '', '全部152/64与新相位24/74/124/148均保留。8个中心、同中心的两epsilon和各分量相依，不扩大独立N；不报告CI、回归或总赢家。输入是native snapshot_x8重复19的共同pre设计，branch PRE full58未直捕，受工程540/primitive恢复条件限制，wholehidden仍UNKNOWN。同源单seed已见路径上的fresh扰动测试不等未见状态/ego泛化；原物理参考与三臂任务FAIL、物理梯度未合格均不变。',
         '', '作者描述，不是独立审计；原确认报告SHA `'+REPORT_SHA+'`。Root另作全量图及结论；Contact独立24项复算（SHA `2ca69685e2bc63e95e3dde1edefa4829858c47b7a4d62e3275185eda56cc3b9f`）另列，不与作者描述混并。']
    (OUT/'description.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    result={'schema':'full151105_WM_component_confirmation_author_description:1','status':'COMPLETED_AUTHOR_SAVED_DESCRIPTION_NOT_INDEPENDENT_AUDIT',
        'source_pins':sources,'author_script':pin(__file__),'coverage':{'snapshots':8,'point_rows_per_arm':152,'pairs_per_arm':64,'native52_blocks':6,'snapshot_block_rows_twoarms':96},
        'global_52_weighted_accounting':global_roundoff,'original_decision':r['decision'],'new_models_weights_Log_FW_AD_physics_GPU':0,
        'point_arithmetic_only_Torch_CPU':True,'input_files_unchanged':all(pin(RUN/name)==v for name,v in sources.items())}
    (OUT/'description.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (OUT/'output_pins.json').write_text(json.dumps({p.name:pin(p) for p in OUT.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'description':pin(OUT/'description.md'),'report':pin(OUT/'description.json')}))


if __name__=='__main__':main()
