"""Author description of ALL8/128/64 saved outputs, no new research run.

Before reading values: retain all endpoints; flag entire-vector numeric du==0
and raw u bit-equivalence separately. Preserve every response and comparison to
ALL3 baseline replicates. RMS ratios have no floor: a zero denominator is None.
No stable subset, old9, derivative accuracy gate, score or new model is used.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path('D:/workspace/supermimic-full151-observation-study')
RUN=ROOT/'outputs/full151105_saved_action_response_CPU_v1'
REPORT_SHA='b7e951f25dd0d45228fd26c80a2b5eac9c56b4cb9248a52cad6209a2cf154689'
CONFIG_SHA='f07239456b8fe2c6cc10eb785e63018447742265472fd81ae24da5d8dc592461'
BLOCKS={'q':(0,20,'rad',1000,'mrad'),'qd':(20,40,'rad/s',1,'rad/s'),'p':(40,43,'m',1000,'mm'),
 'R':(43,52,'dimensionless',1,'dimensionless'),'v':(52,55,'m/s',1,'m/s'),'omega':(55,58,'rad/s',1,'rad/s')}
FILES={'report':RUN/'report.json','raw152':RUN/'all152_branch_raw.npz','decomposition':RUN/'decomposition_arrays.npz',
 'guards':RUN/'visible_and_control_guard_masks.npz','metadata152':RUN/'all152_branch_metadata.json','pairs64':RUN/'all64_pair_records.json',
 'independent22':ROOT/'outputs/full151105_saved_action_response_saved_independent_v1/report.json'}


def require(v,m):
    if not v:raise ValueError(m)


def pin(p):
    p=Path(p).resolve(strict=True);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p,d):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(d,f,indent=2,allow_nan=False)


def rms(a):
    a=np.asarray(a);return float(np.sqrt(np.mean(a*a))) if a.size else None


def ratio(n,d):return None if d is None or d==0 else float(n/d)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',required=True);args=ap.parse_args();out=Path(args.output).resolve()
    require(out.parent==ROOT/'outputs' and not out.exists(),'Fresh author output');out.mkdir()
    source=pin(__file__);inputs={k:pin(v) for k,v in FILES.items()};(out/'author_source.py').write_bytes(Path(__file__).read_bytes())
    write(out/'description_rule_freeze.json',{'source':source,'input_pins':inputs,'scope':__doc__})
    require(inputs['report']['sha256']==REPORT_SHA,'Fixed actual saved run')
    report=read(FILES['report']);metadata=read(FILES['metadata152']);pairs=read(FILES['pairs64']);independent=read(FILES['independent22'])
    with np.load(FILES['raw152'],allow_pickle=False) as z:raw={k:z[k].copy() for k in z.files}
    with np.load(FILES['decomposition'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    with np.load(FILES['guards'],allow_pickle=False) as z:guards={k:z[k].copy() for k in z.files}
    checks={}
    def check(k,v):checks[k]=bool(v);require(checks[k],k)
    check('actual_config_report_scope',report['config']['sha256']==CONFIG_SHA and report['fixed_visible_state_interpretation_permitted'] is True)
    check('complete8_152_128_64_no_filter',report['counts']=={'snapshots':8,'all_branches':152,'baseline_rows':24,'perturbation_endpoints':128,'all_epsilon_pairs':64,'defined_pairs':64,'undefined_pairs_kept':0})
    check('independent22_all_true',independent['passed']==independent['total']==22 and all(v is True for v in independent['checks'].values()))
    check('all_original_guards_passed',all(v.dtype==np.bool_ and v.all() for v in guards.values()))
    pred,target,u=raw['predicted_y152_FP32'],raw['target_y152_FP32'],raw['actual_u152_FP32']
    check('raw_all152_FP32_shapes',pred.shape==target.shape==(8,19,58) and u.shape==(8,19,20) and pred.dtype==target.dtype==u.dtype==np.float32)
    pm=pred[:,:3].astype(np.float64).mean(1);tm=target[:,:3].astype(np.float64).mean(1)
    check('all8_baseline_means_and_bias',np.array_equal(pm,a['baseline_predicted_mean_F64']) and np.array_equal(tm,a['baseline_target_mean_F64']) and np.array_equal(pm-tm,a['constant_baseline_error_F64']))
    check('all128_centered_responses',np.array_equal(pred[:,3:].astype(np.float64)-pm[:,None],a['centered_model_response_F64']) and np.array_equal(target[:,3:].astype(np.float64)-tm[:,None],a['centered_physical_response_F64']))
    check('all64_pair_metadata_kept',len(pairs)==64 and {r['pair_id'] for r in pairs}==set(range(64)) and sum(r['legacy_clipped'] for r in pairs)==12 and sum(r['asymmetric_about_baseline'] for r in pairs)==12)
    du=u[:,3:].astype(np.float64)-raw['original_snapshot_u_FP32'][:,None].astype(np.float64)
    zero=np.all(du==0,axis=-1);bit_same=np.all(u[:,3:].view(np.uint32)==raw['original_snapshot_u_FP32'][:,None].view(np.uint32),axis=-1)
    pbase=pred[:,3:,None].astype(np.float64)-pred[:,:3,None].transpose(0,2,1,3).astype(np.float64)
    tbase=target[:,3:,None].astype(np.float64)-target[:,:3,None].transpose(0,2,1,3).astype(np.float64)
    pred_bits_equal=pred[:,3:,None].view(np.uint32)==pred[:,:3,None].transpose(0,2,1,3).view(np.uint32)
    target_bits_equal=target[:,3:,None].view(np.uint32)==target[:,:3,None].transpose(0,2,1,3).view(np.uint32)
    np.savez_compressed(out/'all128_encoded_control_response.npz',actual_du_from_original_u0_rad_F64=du,all20_numeric_zero=zero,all20_raw_bits_same_u0=bit_same,
        predicted_response_F64=a['centered_model_response_F64'],physical_response_F64=a['centered_physical_response_F64'],response_error_F64=a['centered_response_error_F64'],
        predicted_endpoint_minus_each_baseline_F64=pbase,physical_endpoint_minus_each_baseline_F64=tbase,
        predicted_endpoint_bit_equal_each_baseline=pred_bits_equal,physical_endpoint_bit_equal_each_baseline=target_bits_equal)
    defined=a['secant_defined_mask'];summary={};by_snapshot=[]
    for name,(lo,hi,unit,factor,display) in BLOCKS.items():
        true=rms(a['centered_physical_response_F64'][...,lo:hi]);model=rms(a['centered_model_response_F64'][...,lo:hi]);err=rms(a['centered_response_error_F64'][...,lo:hi])
        ts=rms(a['physical_coordinate_secant_F64'][defined][:,lo:hi]);ps=rms(a['model_coordinate_secant_F64'][defined][:,lo:hi]);es=rms(a['coordinate_secant_error_F64'][defined][:,lo:hi])
        summary[name]={'state_units':unit,'display_units':display,'display_factor':factor,'baseline_bias_RMS':rms(a['constant_baseline_error_F64'][...,lo:hi]),
            'physical_centered_response_RMS':true,'model_centered_response_RMS':model,'centered_response_error_RMS':err,'response_error_to_physical_RMS_descriptive':ratio(err,true),
            'physical_coordinate_secant_RMS':ts,'model_coordinate_secant_RMS':ps,'coordinate_secant_error_RMS':es,'secant_error_to_physical_RMS_descriptive':ratio(es,ts),
            'baseline_physical_range_max':float(a['baseline_physical_range_F64'][:,lo:hi].max()),'baseline_model_range_max':float(a['baseline_model_range_F64'][:,lo:hi].max()),
            'zero_control_endpoint_count':int(zero.sum()),'zero_control_physical_response_RMS':rms(a['centered_physical_response_F64'][zero][:,lo:hi]),
            'zero_control_model_response_RMS':rms(a['centered_model_response_F64'][zero][:,lo:hi]),
            'physical_pair_numeric_zero_components':int(a['physical_pair_numeric_zero'][...,lo:hi].sum()),'model_pair_numeric_zero_components':int(a['model_pair_numeric_zero'][...,lo:hi].sum())}
        for s in range(8):
            td=rms(a['centered_physical_response_F64'][s,:,lo:hi]);ed=rms(a['centered_response_error_F64'][s,:,lo:hi])
            by_snapshot.append({'snapshot_id':s,'path':('A0','A2')[s//4],'k':(0,49,99,149)[s%4],'block':name,'units':unit,
                'baseline_bias_RMS':rms(a['constant_baseline_error_F64'][s,lo:hi]),'physical_response_RMS':td,'model_response_RMS':rms(a['centered_model_response_F64'][s,:,lo:hi]),
                'response_error_RMS':ed,'response_error_to_physical_RMS_descriptive':ratio(ed,td),'all16_endpoints':16,'zero_actual_control_endpoints':int(zero[s].sum())})
    endpoint_rows=[]
    for s in range(8):
        for off in range(16):
            row=metadata[s*19+3+off]
            entry={k:row[k] for k in ('row_id','snapshot_id','path','k','j','joint_index','epsilon_index','epsilon_rad','sign','group_id')}
            entry.update(all20_actual_du_numeric_zero=bool(zero[s,off]),all20_actual_u_bitexact_to_original_u0=bool(bit_same[s,off]),
                physical_endpoint_bitexact_each_baseline=json.dumps(np.all(target_bits_equal[s,off],axis=-1).tolist()),
                model_endpoint_bitexact_each_baseline=json.dumps(np.all(pred_bits_equal[s,off],axis=-1).tolist()))
            for b,(lo,hi,_,_,_) in BLOCKS.items():
                entry[b+'_physical_response_RMS']=rms(a['centered_physical_response_F64'][s,off,lo:hi]);entry[b+'_model_response_RMS']=rms(a['centered_model_response_F64'][s,off,lo:hi])
            endpoint_rows.append(entry)
    with (out/'all128_endpoints.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(endpoint_rows[0]));w.writeheader();w.writerows(endpoint_rows)
    with (out/'all8_by6_blocks.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(by_snapshot[0]));w.writeheader();w.writerows(by_snapshot)
    check('all128_and_all8x6_retained',len(endpoint_rows)==128 and len(by_snapshot)==48)
    check('no_backend_or_model_import',all(m not in sys.modules for m in ('torch','mujoco','warp')))
    check('all_input_and_source_bytes_preserved',{k:pin(v) for k,v in FILES.items()}==inputs and pin(__file__)==source)
    result={'schema':'full151105_saved_response_author_description:1','status':'AUTHOR_SAVED_DESCRIPTION_NOT_INDEPENDENT_AUDIT','source':source,'input_pins':inputs,'checks':checks,
        'passed':sum(checks.values()),'total':len(checks),'blocks_all8_128_64':summary,'all8_by6':by_snapshot,
        'encoded_zero_control':{'all_endpoint_rows':128,'numeric_zero20_rows':int(zero.sum()),'raw_bits_equal20_rows':int(bit_same.sum()),
            'zero_control_original_branch_ids':[r['row_id'] for r in endpoint_rows if r['all20_actual_du_numeric_zero']]},
        'limits':['RMS ratios are after-outcome descriptive values without floors, thresholds or stable filtering. Zero physical RMS gives None.',
            'F64 baseline coordinate means are statistics, not model or physical states; raw FP32 encoding and each of3 repeats remain separate.',
            'Same visible58/u is not complete hidden-state equality.3baseline ranges are empirical observations, not noise bounds.',
            'Finite asymmetric/clipped coordinate chords are not unique physical Jacobians; no old5/9, score/goal or synchrony reinterpretation.',
            'CPUshadow one-step knownrough single-source/seed and prepolicy100 divergence remain; no causal method-benefit claim.'],
        'new_model_load_forward_AD_MJB_Env_GPU_physics_training':0}
    write(out/'report.json',result)
    text=['保存数据响应分解：作者描述（非独立审计）','',
        '全部8snapshot、152branches、128端点、64pairs保留。新指标只描述actual-coordinate finite secants；不替换旧nominalFD或5/9 taskscalar结果。', '',
        '| block | display units | baseline bias RMS | physical centered response RMS | model centered response RMS | response error RMS | error/physical RMS |',
        '|---|---|---:|---:|---:|---:|---:|']
    for b,v in summary.items():
        f=v['display_factor'];text.append(f"| {b} | {v['display_units']} | {v['baseline_bias_RMS']*f:.9g} | {v['physical_centered_response_RMS']*f:.9g} | {v['model_centered_response_RMS']*f:.9g} | {v['centered_response_error_RMS']*f:.9g} | {v['response_error_to_physical_RMS_descriptive']} |")
    text+=['','| block | physical coordinate secant RMS | model coordinate secant RMS | secant error RMS | error/physical RMS |','|---|---:|---:|---:|---:|']
    for b,v in summary.items():text.append(f"| {b} ({v['state_units']}/control rad) | {v['physical_coordinate_secant_RMS']:.9g} | {v['model_coordinate_secant_RMS']:.9g} | {v['coordinate_secant_error_RMS']:.9g} | {v['secant_error_to_physical_RMS_descriptive']} |")
    text+=['',f"全128端点机械标记：{int(zero.sum())}个完整20维actualdu数值全0，{int(bit_same.sum())}个actualu原始bits与u0全相同。所有非零控制行也保留，零控制行和三个baseline的完整58维原始编码差分别存于NPZ；不得把这些有限重复的范围宣称为真实noise界。",'',
        'all8_by6_blocks.csv保留全部snapshot分组，all128_endpoints.csv保留每一个端点。ratio没有epsilon/floor，physical RMS=0时为None。',
        '没有新模型、forward、AD、score、MJB/FK/query、GPU、Env、物理或训练；后续假设应由Root决定，本描述不创建新prereg/实现。']
    (out/'description.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    write(out/'artifact_pins.json',{p.name:pin(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({'passed':result['passed'],'total':result['total'],'report':pin(out/'report.json'),'description':pin(out/'description.md'),'encoded_zero_control':result['encoded_zero_control'],
        'block_summary':{k:{m:v[m] for m in ('baseline_bias_RMS','physical_centered_response_RMS','model_centered_response_RMS','centered_response_error_RMS','response_error_to_physical_RMS_descriptive','secant_error_to_physical_RMS_descriptive')} for k,v in summary.items()}},indent=2))


if __name__=='__main__':main()
