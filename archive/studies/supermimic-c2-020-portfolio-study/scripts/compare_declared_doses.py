"""Read-only checkpoint-level20minus5 descriptions; no pooling or best dose."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def stats(x):return {'mean':statistics.mean(x),'median':statistics.median(x),'min':min(x),'max':max(x)}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--current',type=Path,default=ROOT/'outputs/analysis_v1/comparison.json');p.add_argument('--output',type=Path,default=ROOT/'outputs/dose_comparison_v1');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    protocol=read(ROOT/'analysis_protocol_v1.json');prior=protocol['prior5mrad_comparison'];baseline=Path(prior['path'])
    if sha(baseline)!=prior['sha256']:raise ValueError('Frozen5mrad comparison changed')
    header={'schema':'declared_C2_checkpoint_dose_description:1','source_sha256':sha(__file__),'protocol_sha256':sha(ROOT/'analysis_protocol_v1.json'),'baseline5_sha256':sha(baseline),'gpu_used':False,'N_training_seeds_per_method':3,'process_repeats_per_checkpoint_and_dose':3,'cohorts_pooled':False,'best_dose_selected':False,'causal_dose_effect_claimed':False}
    if not a.current.exists():write(a.output/'comparison.json',{**header,'status':'PENDING_20COHORT','pairs':None});print('PENDING_20COHORT');return
    old,new=read(baseline),read(a.current)
    if old.get('status')!='COMPLETE' or new.get('status')!='COMPLETE':write(a.output/'comparison.json',{**header,'status':'PENDING_OR_INVALID_COHORT','pairs':None});print('PENDING_OR_INVALID_COHORT');return
    left,right=old['checkpoint_summaries'],new['checkpoint_summaries']
    if set(left)!=set(right) or len(left)!=9:raise ValueError('Nine fixed checkpoint identities differ')
    # Equality is checked for physical/reference inputs, not the intervention or
    # realized feedback action trajectory/numerical noise after the same t0.
    checks={'fixed_checkpoints':all(left[k]['checkpoint_sha256']==right[k]['checkpoint_sha256'] for k in left),'both_cohorts27_valid_capacity_zero':all(r['status']=='VALID_CAPACITY_ZERO' for d in (old,new) for r in d['results'])}
    for key in ('reference.npz','reference.json','model.mjb'):
        checks['same_'+key]=len({r['hashes'][key] for d in (old,new) for r in d['results']})==1
    ldir,rdir=(Path(d['results'][0]['directory']) for d in (old,new))
    with np.load(ldir/'control_targets_50hz.npz',allow_pickle=False) as l,np.load(rdir/'control_targets_50hz.npz',allow_pickle=False) as r:
        checks['same_full_t0_controls_and_physics']=all(np.array_equal(l[k],r[k]) for k in ['initial_qpos','initial_qvel','initial_ctrl','initial_cached_target_rad','u_prev_rad','position_ctrl_ids'])
    with np.load(ldir/'rollout.npz',allow_pickle=False) as l,np.load(rdir/'rollout.npz',allow_pickle=False) as r:
        checks['same_public_physical_t0']=all(np.array_equal(l[k][0],r[k][0]) for k in ['joint_pos_rad','joint_vel_rad_s','joint_target_rad','object_pose_w','object_lin_vel_w_m_s','object_ang_vel_w_rad_s','wrist_pose_w'])
    if not all(checks.values()):raise ValueError('Checkpoint/reference/capacity binding differs')
    pairs=[]
    for model_id,l in left.items():
        r=right[model_id]
        if l['method']!=r['method'] or l['training_seed']!=r['training_seed'] or not l['valid_three_processes'] or not r['valid_three_processes']:raise ValueError('Incomplete/mismatched checkpoint summaries')
        pairs.append({'model_id':model_id,'method':l['method'],'training_seed':l['training_seed'],'checkpoint_sha256':l['checkpoint_sha256'],'process_passes5':l['process_passes'],'process_passes20':r['process_passes'],'class5':l['outcome_class'],'class20':r['outcome_class'],'metrics5':l['metrics'],'metrics20':r['metrics'],'delta20minus5_process_mean':{k:r['metrics'][k]['mean']-l['metrics'][k]['mean'] for k in l['metrics']},'delta20minus5_process_worst':{k:r['metrics'][k]['worst']-l['metrics'][k]['worst'] for k in l['metrics']}})
    methods={}
    for method in ('A1L','P_ONLY','A2'):
        group=[p for p in pairs if p['method']==method];methods[method]={'N_training_seeds':3,'seed_process_pass_profiles5':old['method_summaries'][method]['seed_process_pass_profile'],'seed_process_pass_profiles20':new['method_summaries'][method]['seed_process_pass_profile'],'all3pass_training_seeds5':old['method_summaries'][method]['training_seeds_all3pass'],'all3pass_training_seeds20':new['method_summaries'][method]['training_seeds_all3pass'],'descriptive_delta_process_means':{k:stats([p['delta20minus5_process_mean'][k] for p in group]) for k in group[0]['delta20minus5_process_mean']}}
    data={**header,'status':'COMPLETE_SEPARATE_COHORT_DESCRIPTION','current20_sha256':sha(a.current),'binding_checks':checks,'pairs':pairs,'methods':methods,'limits':protocol['dose_comparison'],'no_CI':True,'no_frame_N':True,'m0':None,'a_brake':None}
    write(a.output/'comparison.json',data)
    fields=['model_id','method','training_seed','process_passes5','process_passes20']+['delta_mean_'+k for k in pairs[0]['delta20minus5_process_mean']]+['delta_worst_'+k for k in pairs[0]['delta20minus5_process_worst']]
    with (a.output/'checkpoint_dose_differences.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for pair in pairs:w.writerow({**{k:pair[k] for k in fields[:5]},**{'delta_mean_'+k:v for k,v in pair['delta20minus5_process_mean'].items()},**{'delta_worst_'+k:v for k,v in pair['delta20minus5_process_worst'].items()}})
    lines=['Separate20minus5 checkpoint descriptions; no pooled54job inference or best-dose selection.','Both doses retain all27 observations and nine fixed final checkpoints. Each method N=3training seeds, not9or18processes.']
    for pair in pairs:lines.append(f"{pair['model_id']}: process passes5={pair['process_passes5']}/3,20={pair['process_passes20']}/3; mean deltas={pair['delta20minus5_process_mean']}; worst deltas={pair['delta20minus5_process_worst']}.")
    lines.append(protocol['dose_comparison']);(a.output/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':data['status'],'method_profiles':{k:{'5':v['seed_process_pass_profiles5'],'20':v['seed_process_pass_profiles20']} for k,v in methods.items()}},indent=2))

if __name__=='__main__':main()
