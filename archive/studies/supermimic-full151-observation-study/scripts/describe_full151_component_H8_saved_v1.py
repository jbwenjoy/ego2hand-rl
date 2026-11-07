"""Saved JSON author description only. No NumPy/Torch/model/array/FW/physics."""
from pathlib import Path
import argparse,csv,hashlib,json,math,sys
ROOT=Path('D:/workspace/supermimic-full151-observation-study')
RUN=ROOT/'outputs/full151105_three_component_H8_donor_CPU_v1'
OUT=ROOT/'outputs/full151105_component_H8_author_description_v1'
REPORT_SHA='c676c80037a056f7f51d7877bcbdaeffa5575fafb360fae3c2148ba01087deba'
ROLES=('initial_unmodified_A2','pointwise_final1000','pointwise_plus_response_final1000')
UNITS={'q':'rad','qd':'rad/s','p':'m','R_raw9':'dimensionless','v':'m/s','omega':'rad/s'}
SHORT={'initial_unmodified_A2':'initial','pointwise_final1000':'point','pointwise_plus_response_final1000':'response'}


def pin(p):return {'path':str(Path(p).resolve()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def csvwrite(p,rows):
    with p.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ratio(value,reference):return value/reference if reference!=0 else None


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',action='store_true');args=ap.parse_args()
    if not args.run:print(json.dumps({'status':'AUTHOR_JSON_DESCRIPTION_DECLARED_NOT_RUN','source':pin(__file__)}));return
    source=pin(RUN/'report.json');assert source['sha256']==REPORT_SHA and not OUT.exists()
    r=json.loads((RUN/'report.json').read_text(encoding='utf-8-sig'));stats=r['statistics']
    table={(z['model_role'],z['horizon'],z['path'],z['block']):z for z in stats}
    expected={(role,h,path,block) for role in ROLES for h in range(1,9) for path in ('ALL','A0','A2') for block in UNITS}
    assert len(stats)==len(table)==432 and set(table)==expected
    assert all(z['unit']==UNITS[z['block']] and z['window_count']==(286 if z['path']=='ALL' else 143)
               and all(math.isfinite(z[k]) for k in ('component_RMSE','MAE','maximum_absolute')) for z in stats)
    OUT.mkdir();(OUT/'executed_author_source.py').write_bytes(Path(__file__).read_bytes())
    csvwrite(OUT/'all432_report_statistics.csv',stats)
    growth=[];relative=[];heterogeneity=[]
    for z in stats:
        role,h,path,block=z['model_role'],z['horizon'],z['path'],z['block'];v=z['component_RMSE'];base=table[role,1,path,block]['component_RMSE'];q=ratio(v,base)
        growth.append({'model_role':role,'horizon':h,'path':path,'block':block,'unit':z['unit'],'window_count':z['window_count'],
            'component_RMSE':v,'h1_component_RMSE':base,'absolute_change_from_h1':v-base,'ratio_to_h1':q,
            'percent_change_from_h1':None if q is None else (q-1)*100,'relative_defined':base!=0})
    for before,after in ((ROLES[0],ROLES[1]),(ROLES[0],ROLES[2]),(ROLES[1],ROLES[2])):
        for h in range(1,9):
            for path in ('ALL','A0','A2'):
                for block,unit in UNITS.items():
                    a=table[before,h,path,block]['component_RMSE'];b=table[after,h,path,block]['component_RMSE'];q=ratio(b,a)
                    relative.append({'reference_role':before,'comparison_role':after,'horizon':h,'path':path,'block':block,'unit':unit,
                        'reference_component_RMSE':a,'comparison_component_RMSE':b,'absolute_difference':b-a,'comparison_over_reference':q,
                        'percent_change':None if q is None else (q-1)*100,'relative_defined':a!=0})
    for role in ROLES:
        for block,unit in UNITS.items():
            a=table[role,8,'A0',block]['component_RMSE'];b=table[role,8,'A2',block]['component_RMSE']
            heterogeneity.append({'model_role':role,'horizon':8,'block':block,'unit':unit,'A0_windows':143,'A2_windows':143,
                'A0_component_RMSE':a,'A2_component_RMSE':b,'A2_over_A0':ratio(b,a),'relative_defined':a!=0})
    csvwrite(OUT/'all432_horizon_growth_from_h1.csv',growth);csvwrite(OUT/'all432_within_cohort_model_comparisons.csv',relative)
    csvwrite(OUT/'all54_h8_statistics.csv',[z for z in stats if z['horizon']==8]);csvwrite(OUT/'all18_h8_path_heterogeneity.csv',heterogeneity)
    # Short MD displays every unit at h8, every horizon's position vector RMS,
    # every role/unit growth endpoint, and every role/unit path comparison.
    lines=['# 三模型 H8：全量保存统计的作者描述','',
        '**探索性、非独立审计；没有新的科学通过门或总体赢家。** 本文只读已完成报告的432条统计，不打开rollout数组，不导入Torch/NumPy、不加载权重、不前向或运行物理。',
        '', '286个相互重叠的固定donor控制windows来自A0/A2各143起点；每个h的ALL为286，路径各143。三模型使用同一cohort；ALL不与其两个子路径重复累加。8个h共6864条模型输出相依，不作为独立N或CI。',
        '', '## 全部模型的 ALL h8（分量RMSE，原单位）','',
        '|块/单位|initial|point final1000|response final1000|response相对point变化|','|---|---:|---:|---:|---:|']
    for block,unit in UNITS.items():
        a,b,d=[table[role,8,'ALL',block]['component_RMSE'] for role in ROLES]
        q=ratio(d,b);s='undefined' if q is None else '%.3f%%'%((q-1)*100)
        lines.append('|%s / %s|%.9g|%.9g|%.9g|%s|'%(block,unit,a,b,d,s))
    pos=[table[role,8,'ALL','p']['vector_or_Frobenius_RMS']*1000 for role in ROLES]
    lines+=['','ALL h8位置**向量**RMS为 initial %.6f、point %.6f、response %.6f mm；它与上表p分量RMSE相差sqrt(3)。Rraw9是矩阵元素/无量纲Frobenius误差，绝不解释为旋转角。'%tuple(pos),
        '', '在ALL h8，两个校准模型的六个分量RMSE均低于initial。response相对point在q/qd/v较低、p/Rraw9/omega较高；这是逐块的tradeoff，不能通过跨单位投票或求和形成总赢家，也不推唯一遗忘原因。',
        '', '## 全h的位置向量RMS（mm，仅本H8 cohort）','', '|h|initial|point|response|','|---|---:|---:|---:|']
    for h in range(1,9):lines.append('|%d|%.7g|%.7g|%.7g|'%(h,*[table[role,h,'ALL','p']['vector_or_Frobenius_RMS']*1000 for role in ROLES]))
    lines+=['','## 所有模型/单位的 ALL h1→h8 分量RMSE变化','',
        '|role|块/单位|h1|h8|h8/h1|相邻h增加/减少/相等次数|','|---|---|---:|---:|---:|---:|']
    growth_details=[]
    for role in ROLES:
        for block,unit in UNITS.items():
            values=[table[role,h,'ALL',block]['component_RMSE'] for h in range(1,9)]
            delta=[values[i+1]-values[i] for i in range(7)];counts=(sum(v>0 for v in delta),sum(v<0 for v in delta),sum(v==0 for v in delta))
            q=ratio(values[-1],values[0]);s='undefined' if q is None else '%.6g'%q
            lines.append('|%s|%s / %s|%.7g|%.7g|%s|%d/%d/%d|'%(SHORT[role],block,unit,values[0],values[-1],s,*counts))
            growth_details.append({'model_role':role,'block':block,'unit':unit,'ALL_all8_component_RMSE':values,'adjacent_increase_decrease_equal':counts})
    lines+=['','不要求增长单调；固定起点的目标时相随h推进，目标观测范围从h1的1..143变为h8的8..150。逐h比较描述自回归积累及其目标时相，不是控制了时间分布的因果证明。完整ALL/A0/A2所有h增长及相对误差保存在CSV，零基准分母只标undefined、不填0。',
        '', '## h8全部路径异质性（分量RMSE；同单位内比较）','',
        '|role|块/单位|A0|A2|A2/A0|','|---|---|---:|---:|---:|']
    for z in heterogeneity:
        q=z['A2_over_A0'];s='undefined' if q is None else '%.6g'%q
        lines.append('|%s|%s / %s|%.7g|%.7g|%s|'%(SHORT[z['model_role']],z['block'],z['unit'],z['A0_component_RMSE'],z['A2_component_RMSE'],s))
    lines+=['','## 解释限度与文件','',
        '三个角色分别是未修改的initial A2组件，以及相同起点/norm/数据的两种final1000校准；initial不是伪造final1000或重新加载旧105wrapper。固定外部actualu的open-loop状态诊断不等于Actor重选动作、EMA/记忆、prior/reward或survival的native H8 imagination，更不是policy-gradient replay。zero174是既有consumer假设，CPU FP32不等于原GPU AMP。',
        '', '数据是已见单谱系/单seed donor路径；不能与旧all452 H1、新fresh152确认或任务梯度stable组混cohort。原确认tradeoff FAIL、物理参考/任务FAIL与hidden state UNKNOWN不变。这里的误差变化不另立accuracy pass阈值、泛化或梯度资格。',
        '', '- `all432_report_statistics.csv`：完整原报告统计；`all54_h8_statistics.csv`保留h8的ALL/A0/A2全部角色/单位。',
        '- `all432_horizon_growth_from_h1.csv`：每个h/path/单位的同模型变化；`all432_within_cohort_model_comparisons.csv`：三种明确配对对照，每次只比较同h/path/单位。',
        '- `all18_h8_path_heterogeneity.csv`：全部h8路径对照。没有按误差、成功组或flat筛行。',
        '', '源报告SHA `'+REPORT_SHA+'`；Contact独立24项审计 `ab43a6e5efb150eccccc792215f3f6c4e4d22bd795b5f2bae15e83d749dfc4b8`另列，本文为作者描述而非独立复算。']
    (OUT/'description.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    result={'schema':'full151105_component_H8_author_description:1','status':'COMPLETED_ALL432_AUTHOR_JSON_DESCRIPTION_NOT_INDEPENDENT_AUDIT',
        'source_report':source,'source_protocol':r['protocol'],'author_source':pin(__file__),
        'coverage':{'model_roles':list(ROLES),'horizons':8,'paths':['ALL','A0','A2'],'unit_blocks':UNITS,'all_statistics_rows':432,'h8_rows':54,'relative_comparison_rows':432},
        'ALL_growth_by_model_unit':growth_details,'no_new_accuracy_gate':True,'no_overall_winner':True,'new_array_Torch_model_weight_FW_AD_physics_GPU':0,
        'source_unchanged':pin(RUN/'report.json')==source}
    (OUT/'description.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (OUT/'output_pins.json').write_text(json.dumps({p.name:pin(p) for p in OUT.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf-8')
    assert not any(k in sys.modules for k in ('torch','numpy','mujoco','warp'))
    print(json.dumps({'status':result['status'],'description':pin(OUT/'description.md'),'report':pin(OUT/'description.json')}))


if __name__=='__main__':main()
