"""Saved-only author description; fixed six-block exact-tie peak rules.

Peak rule is frozen before reading prediction values: primary is maximum per-row
component RMS over ALL452 rows of each block, retaining EVERY exact tie.
Secondary is maximum absolute component over each block, retaining all (row,
feature) exact ties. No isclose, subset selection, mixed-unit score or new model.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np

ROOT=Path('D:/workspace/supermimic-full151-observation-study')
RUN=ROOT/'outputs/full151105_state_prediction452_CPU_v1'
CONFIG_SHA='ba8c3669506acfb9fe01441770927dcc656508c94d14a50826b1901b4bc8aa65'
BLOCKS={'q':(0,20,'rad',1000,'mrad'),'qd':(20,40,'rad/s',1,'rad/s'),'p':(40,43,'m',1000,'mm'),
        'R':(43,52,'dimensionless',1,'dimensionless'),'v':(52,55,'m/s',1,'m/s'),'omega':(55,58,'rad/s',1,'rad/s')}
INPUTS={'report':RUN/'report.json','arrays':RUN/'prediction_arrays.npz','identities':RUN/'group_identities452.json','statistics':RUN/'statistics.json',
 'raw_RNG_before':RUN/'RNG_before.npz','raw_RNG_after':RUN/'RNG_after.npz',
 'independent452':ROOT/'outputs/full151105_state_prediction452_saved_independent_v1/report.json',
 'dense_rows':ROOT/'outputs/full151105_solver_phase_penalty_GPU_qualified_CPU_v1/rows_all1808.json',
 'dense_independent35':ROOT/'outputs/full151105_solver_phase_penalty_GPU_qualified_saved_independent_v1/report.json',
 'WM_physical_independent656':ROOT/'outputs/full151105_WM_analysis_saved_independent_v2/report.json',
 'prepolicy100_comparison':ROOT/'outputs/full151_schema2_three_arm_prepolicy100_comparison_CPU_v1/report.json',
 'rough_reference':ROOT/'outputs/full151_contact_coupled_scp_v1/receipt.json'}


def need(v,m):
    if not v:raise ValueError(m)


def pin(path):
    p=Path(path).resolve(strict=True);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args();out=Path(args.output).resolve()
    need(out.parent==ROOT/'outputs' and not out.exists(),'Fresh author-description output');out.mkdir()
    source=pin(__file__);(out/'author_source.py').write_bytes(Path(__file__).read_bytes())
    inputs={k:pin(v) for k,v in INPUTS.items()}
    write(out/'analysis_rule_freeze.json',{'source':source,'input_pins':inputs,
        'primary_peak':'ALL452 per-row sqrt(mean(F64signederror_block**2)), maximum with ALL EXACT ties; six blocks fixed before numerical read.',
        'secondary_peak':'ALL452xblockcomponents maximum absolute error, all EXACT row/feature ties.',
        'HO_join':'ALL452 groups mechanically map four physical rows 4i..4i+3; interval maximum of saved generatedHO depths, including all negative groups. This is not post20 groundtruth.'})
    r=read(INPUTS['report']);ids=read(INPUTS['identities']);dense=read(INPUTS['dense_rows']);checks={}
    def check(name,v):checks[name]=bool(v);need(checks[name],name)
    check('fixed_actual_case_not_selection',r['config']['sha256']==CONFIG_SHA and r['prediction_rows']==r['WM_batch_size']==452)
    check('all452_order_and_identity_preserved',len(ids)==452 and [x['group_id'] for x in ids]==list(range(452)))
    with np.load(INPUTS['arrays'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    pred,target=a['predicted_x452_FP32'],a['target_x452_FP32']
    check('full_finite_FP32_shape',pred.shape==target.shape==(452,58) and pred.dtype==target.dtype==np.float32 and np.isfinite(pred).all() and np.isfinite(target).all())
    e=pred.astype(np.float64)-target.astype(np.float64)
    check('saved_raw_errors_exact',np.array_equal(e,a['signed_error452x58_F64']) and np.array_equal(abs(e),a['absolute_error452x58_F64']))
    check('one_load_two_reads_one_full_forward',r['counts']=={'strict_WM_loader':{'attempted':1,'returned':1},'checkpoint_torch_loads':{'attempted':2,'returned':2},'WM_forward':{'attempted':1,'returned':1}})
    independent=read(INPUTS['independent452']);check('independent_saved25_pass',independent['passed']==independent['total']==25 and all(v is True for v in independent['checks'].values()))
    def stats(ix):
        d=e[ix];result={'rows':len(d),'blocks':{}}
        for name,(lo,hi,unit,factor,display) in BLOCKS.items():
            v=d[:,lo:hi];sq=v*v;value={'units':unit,'component_RMSE':float(np.sqrt(sq.mean())),'MAE':float(abs(v).mean()),'maxabs':float(abs(v).max()),
                'per_feature_RMSE':np.sqrt(sq.mean(0)).tolist(),'per_feature_MAE':abs(v).mean(0).tolist(),'per_feature_maxabs':abs(v).max(0).tolist(),
                'display_units':display,'display_component_RMSE':float(np.sqrt(sq.mean()))*factor}
            if name in ('p','v','omega'):value['vector_RMS']=float(np.sqrt(sq.sum(-1).mean()))
            if name=='R':value['raw9_Frobenius_RMS']=float(np.sqrt(sq.sum(-1).mean()))
            result['blocks'][name]=value
        return result
    statistics={'all452':stats(np.arange(452))};coverage={}
    for path in ('A0','A2'):
        for mode in ('donor','baseline','direction'):
            ix=np.array([i for i,v in enumerate(ids) if v['path']==path and v['mode']==mode],np.int64)
            statistics[path+'/'+mode]=stats(ix);coverage[path+'/'+mode]=len(ix)
    check('all_six_pathmode_strata_retained',coverage=={'A0/donor':150,'A0/baseline':12,'A0/direction':64,'A2/donor':150,'A2/baseline':12,'A2/direction':64})
    check('all1808_HO_rows_available_for_mechanical_join',len(dense)==1808)
    joined=[]
    for i,identity in enumerate(ids):
        four=dense[4*i:4*i+4]
        need(all(v['row_id']==4*i+j and v['path']==identity['path'] and v['mode']==identity['mode'] and v['k']==identity['k'] and v['branch_id']==identity['branch_id'] and v['substep']==j+1 for j,v in enumerate(four)),'HO mechanical identity join differs')
        depth=max(v['native_generated_HO_depth_m'] for v in four)
        joined.append({'group_id':i,'max_over_four_solver_PRE_HO_depth_m':depth,'any_four_solver_HO_gt3mm':bool(depth>.003),
            'positive_solver_substep_count':sum(v['native_HO_depth_gt3mm'] for v in four)})
    check('HO_join_keeps_all452_including_negatives',len(joined)==452 and sum(x['positive_solver_substep_count'] for x in joined)==44)
    def identity(i):return {**ids[i],**joined[i]}
    peaks={};row_metrics={}
    for name,(lo,hi,unit,factor,display) in BLOCKS.items():
        v=e[:,lo:hi];rms=np.sqrt((v*v).mean(-1));absolute=abs(v);peak=float(rms.max());maximum=float(absolute.max())
        exact=np.flatnonzero(rms==peak);locations=np.argwhere(absolute==maximum)
        row_metrics[name]=rms
        peaks[name]={'units':unit,'display_units':display,'maximum_row_component_RMS':peak,'display_maximum_row_component_RMS':peak*factor,
            'all_exact_primary_ties':[{**identity(int(i)),'component_RMS':float(rms[i])} for i in exact],
            'maximum_absolute_component':maximum,'all_exact_secondary_row_feature_ties':[{**identity(int(i)),'feature_index':lo+int(j),'block_feature_index':int(j),'signed_error':float(v[i,j])} for i,j in locations],
            'no_isclose_or_subset_selection':True}
    with (out/'all452_descriptive_curves.csv').open('x',newline='',encoding='utf-8') as f:
        fields=list(ids[0])+['max_over_four_solver_PRE_HO_depth_m','any_four_solver_HO_gt3mm','positive_solver_substep_count']+[name+'_component_RMS_'+values[2] for name,values in BLOCKS.items()]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i,v in enumerate(ids):
            row={**v,**{k:value for k,value in joined[i].items() if k!='group_id'}}
            row.update({name+'_component_RMS_'+values[2]:float(row_metrics[name][i]) for name,values in BLOCKS.items()});w.writerow(row)
    with np.load(INPUTS['raw_RNG_before'],allow_pickle=False) as z:before_rng={k:z[k].copy() for k in z.files}
    with np.load(INPUTS['raw_RNG_after'],allow_pickle=False) as z:after_rng={k:z[k].copy() for k in z.files}
    check('saved_raw_RNG_vectors_byte_equal',set(before_rng)==set(after_rng) and all(before_rng[k].dtype==after_rng[k].dtype and before_rng[k].shape==after_rng[k].shape and before_rng[k].tobytes()==after_rng[k].tobytes() for k in before_rng))
    check('no_model_or_native_backend_import',all(m not in sys.modules for m in ('torch','mujoco','warp')))
    check('all_inputs_and_author_source_unchanged',{k:pin(v) for k,v in INPUTS.items()}==inputs and pin(__file__)==source)
    report={'schema':'full151105_state452_author_saved_description:1','status':'COMPLETED_AUTHOR_DESCRIPTION_NOT_INDEPENDENT_AUDIT','author_source':source,'input_pins':inputs,
        'checks':checks,'passed':sum(checks.values()),'total':len(checks),'statistics':statistics,'primary_and_secondary_exact_peaks':peaks,
        'mechanical_HO_join':{'all_groups':452,'positive_groups':sum(v['any_four_solver_HO_gt3mm'] for v in joined),'positive_solver_substeps':44,
            'phase':'Maximum of four native solver PREintegration depths over interval, not post20 state geometry, not a new label.'},
        'new_model_or_objective_or_FK_MJB_or_AD_GPU_physics':0,
        'limits':['All452 dependent groups are kept;300donor/24baseline/128direction. No new selection or independent-N claim.',
            'State error is predicted CPUFP32 minus saved nativeGPUFP32; R raw9 differences are dimensionless, omega is Reader/wrist frame from BODY raw velocities.',
            'One-step forward errors do not validate control derivatives, physical or HO gradients, task quality or causal method benefit.',
            'The optional HO interval descriptor does not inspect nonhand/environment, selfcollision, landing or drop mechanisms; no peak is assigned a cause from this join.',
            'Producer weights/normalizer invariants remain runtime source-backed assertions; raw serialized CPU RNG vectors alone are independently compared here.',
            'Known-rough reference and prepolicy100 trajectory/WM divergence remain. No autoregressive or historical native GPUAMP replay is performed.']}
    write(out/'report.json',report);write(out/'all_six_block_peaks.json',peaks)
    lines=['全452状态误差：作者保存数据说明（不是独立审计）','',
        '峰值规则在读取预测值前固化：对全部452行、全部6个block取逐行分量RMS最大值，保留所有精确并列；次要表保留最大绝对分量的所有(row,feature)精确并列。没有isclose、挑组或混单位总分。','',
        '| block | units | all452 component RMSE | MAE | max absolute component |','|---|---|---:|---:|---:|']
    for name,value in statistics['all452']['blocks'].items():lines.append(f"| {name} | {value['units']} | {value['component_RMSE']:.9g} | {value['MAE']:.9g} | {value['maxabs']:.9g} |")
    lines+=['','完整六path/mode统计在report.json；全部452曲线在all452_descriptive_curves.csv。以下只是一页峰值索引，所有精确并列完整保存在all_six_block_peaks.json。','',
        '| block | primary peak(component RMS) | original group ID/path/mode/k |','|---|---:|---|']
    for name,p in peaks.items():
        desc='; '.join(f"{v['group_id']}/{v['path']}/{v['mode']}/k{v['k']}" for v in p['all_exact_primary_ties'])
        lines.append(f"| {name} | {p['display_maximum_row_component_RMS']:.9g} {p['display_units']} | {desc} |")
    lines+=['',f"机械HO join保留全部452组，其中{report['mechanical_HO_join']['positive_groups']}组的4个solver PRE量最大值>3mm，合计44个positive solver substep；这不是post20几何标签。全部阴性组同时保留。",'',
        '这里只读既存预测、target、身份和标量数组；没有运行模型、目标、MJB/FK/query、AD、GPU或物理。R不变成角度，q不wrap，不对预测state投影/修复。',
        'CPU ampFalse one-step forward error、20ms末state与区间solver PRE量的时相要分开；该报告没有梯度资格或方法收益结论。']
    (out/'description.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(out/'artifact_pins.json',{f.name:pin(f) for f in out.iterdir() if f.is_file()})
    print(json.dumps({'passed':report['passed'],'total':report['total'],'report':pin(out/'report.json'),'description':pin(out/'description.md'),
        'all452_component_RMSE':{k:v['component_RMSE'] for k,v in statistics['all452']['blocks'].items()},
        'primary_exact_tie_counts':{k:len(v['all_exact_primary_ties']) for k,v in peaks.items()}},indent=2))


if __name__=='__main__':main()
