"""Independently decode TRAIN packets; retain rejected rows and NEVER_FIT CHECK."""
import argparse,json,subprocess,sys,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import mujoco
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.mechanics.warp_packet_decode import decode_saved_case
from supermimic_research.mechanics.controlled_pen_records import pack_case
from supermimic_research.learning.pen_interventions import prefix_noise_schedule,seeded_training_profiles,validate_full_rank_cohort


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    rp=a.report.resolve();run=json.loads(rp.read_text());cp=verify_file_pin(run['config']);cfg=tomllib.loads(cp.read_text());out=a.output.resolve()
    full=run['schema']=='trained_pen_same_source_response_probe:3'
    if full:full_directions,full_profiles=validate_full_rank_cohort(cfg)
    roles=['TRAIN_CONTROLLED_PHYSICAL_RESPONSE','VALIDATE_CONTROLLED_PHYSICAL_RESPONSE_NEVER_FIT'] if full else ['TRAIN_CONTROLLED_PHYSICAL_RESPONSE']
    if out.exists() or not out.is_relative_to(ROOT/'runs') or run['schema'] not in ['trained_pen_same_source_response_probe:2','trained_pen_same_source_response_probe:3'] or run['role'] not in roles or run['role']!=cfg['role'] or run['restore_mode']!='STRICT' or run['CHECK_response_rows_used']!=0 or run['passed']!=run['total']:raise ValueError('Fresh strict TRAIN report;CHECK cannot be relabelled')
    verify_source_snapshot(json.loads(verify_file_pin(run['source_receipt']).read_text()),live=True)
    out.mkdir();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),cp,verify_file_pin(run['protocol'])]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    checks={};metadata=[];rows=[];models={};physics={};states={};all_gates=[]
    directions,profiles=(full_directions,full_profiles) if full else seeded_training_profiles(cfg['direction_seed'],cfg['directions'],cfg['repeats'])
    checks['directions_exact']=np.array_equal(directions,np.load(verify_file_pin(run['directions']))['directions'])
    noise=dict(np.load(verify_file_pin(run['prefix_noise'])))
    for seed in cfg['noise_seeds']:checks[f'noise{seed}_exact']=np.array_equal(noise[f'seed{seed}'],prefix_noise_schedule(seed,max(cfg['source_phases'])-1,cfg['prefix_raw_sigma']))
    producers=[json.loads(verify_file_pin(p).read_text()) for p in run['producers']]
    checks['all_four_distinct_episodes']=len(producers)==4 and len({p['episode'] for p in producers})==4
    for p in producers:
        actions=p['prefix_actions'];schedule=noise[f'seed{p["noise_seed"]}'];key=p['episode']
        checks[key+'_uninterrupted_clock']=p['no_post_start_resets'] and len(actions)*8==p['native_integrations'] and [r['phase'] for r in actions]==list(range(1,len(actions)+1)) and p['final_phase']==len(actions)+1
        checks[key+'_exogenous_noise']=all(np.array_equal(np.array(r['submitted_raw'],np.float32),np.array(r['mean_raw'],np.float32)+schedule[r['control_index']]) and r['clamped_components']==int((abs(np.array(r['submitted_raw']))>1).sum()) for r in actions)
        audit=p['actor_audit'];checks[key+'_weights_only_W1']=audit['full_W1_model_arrays']==404 and audit['full_W1_model_scalar_match'] and not audit['old_native_or_optimizer_loaded']
    for source in run['sources']:
        group=[c for c in run['cases'] if c['source_name']==source['name']]
        checks[source['name']+'_all_profiles_retained']=[c['profile'] for c in group]==profiles
        if not source['available']:continue
        state=torch.load(verify_file_pin(source['source']),map_location='cpu',weights_only=False);states[source['name']]=state
        verify_source_snapshot(state['native']['sources'],live=True)
        for pin in state['native']['engine_sources']:verify_file_pin(pin)
        key=source['model']['path']
        if key not in models:models[key]=mujoco.MjModel.from_binary_path(str(verify_file_pin(source['model'])));physics[key]=dict(np.load(verify_file_pin(source['physical_context'])))
    noise_state_audits=[]
    for actor in cfg['actors']:
        for phase in cfg['source_phases']:
            entries=[s for s in run['sources'] if s['actor']==actor and s['phase']==phase and s['available']]
            if len(entries)!=2:continue
            arrays=[states[s['name']]['native']['native']['data']['arrays'] for s in entries]
            q=float(np.max(abs(arrays[0]['qpos']-arrays[1]['qpos'])));v=float(np.max(abs(arrays[0]['qvel']-arrays[1]['qvel'])))
            checks[f'{actor}_P{phase}_noise_histories_change_state']=q>0 or v>0
            noise_state_audits.append(dict(actor=actor,phase=phase,sources=[s['name'] for s in entries],qpos_max_abs_difference=q,qvel_max_abs_difference=v))
    for ci,c in enumerate(run['cases']):
        if not c['available']:continue
        z=dict(np.load(verify_file_pin(c['trace'])));w=dict(np.load(verify_file_pin(c['window'])));source=states[c['source_name']];ns=source['native']['native']['data']['arrays'];m=models[c['model']['path']]
        metrics,decoded=decode_saved_case(c,z,m,physics[c['model']['path']])
        packet=pack_case(m,c,z,decoded,np.array(c['qids']),np.array(c['dofs']),np.array(c['aids']),z['initial_warmstart'],c['physics'])
        if packet['x'].shape!=(1,102) or packet['y'].shape!=(1,90):raise ValueError('Single original20ms transition per branch')
        state0=mujoco.MjData(m);mujoco.mj_setState(m,state0,z['initial_native_state'],mujoco.mjtState.mjSTATE_INTEGRATION)
        world=decoded['world_COM_wrench'].mean(0);R=source['source_qpos_R'];Rc=source['source_cached_R'];truth=np.r_[R.T@world[:3],R.T@world[3:]];cached=np.r_[Rc.T@world[:3],Rc.T@world[3:]]
        local=dict(source_qpos=np.array_equal(z['pre_qpos'][0],ns['qpos'][0]),source_qvel=np.array_equal(z['pre_qvel'][0],ns['qvel'][0]),
            actual_initial_warm=np.array_equal(z['initial_warmstart'],ns['qacc_warmstart'][0]),source_phase=int(z['pre_action_phase'][0])==source['phase'],
            held_PD=np.array_equal(z['post_ctrl'][:,c['aids']],np.tile(c['actual_PD20'],(8,1))),clock=bool(np.max(abs(z['post_GPU_time']-(float(ns['time'][0])+np.arange(1,9)*.0025)))<1e-6),
            fixed_mount=bool(np.array_equal(z['post_mocap_pos'],np.tile(state0.mocap_pos,(8,1,1))) and np.array_equal(z['post_mocap_quat'],np.tile(state0.mocap_quat,(8,1,1)))),
            true_source_BODY_label=bool(np.allclose(packet['y'][0,78:84],truth,atol=1e-10,rtol=1e-10)),
            resident_mean_wrench=bool(np.allclose(w['wrench_sum'][5]/8,cached,atol=1e-9,rtol=1e-9)),
            group_normal=bool(np.allclose(packet['y'][0,84:90],w['Fn6_sum']/8,atol=1e-9,rtol=1e-9)),
            finite=bool(all(np.isfinite(packet[key]).all() for key in ['x','y','heads'])),
            no_aid=not np.any(z['post_qfrc_applied']) and not np.any(z['post_xfrc_applied']),role=c['role']==run['role'])
        local.update({'EFC_'+key:bool(value) for key,value in metrics['checks'].items()})
        geometry=packet['geometry_flags'][0];legal=bool(np.all(geometry<=[cfg['object_penetration_m'],cfg['self_penetration_m'],cfg['joint_excursion_rad']]))
        valid=bool(c['qualified'] and packet['valid'][0] and all(local.values()) and legal)
        metadata.append(dict(row_index=len(rows),case_index=ci,name=c['name'],source_name=c['source_name'],actor=c['actor'],episode=c['episode'],noise_seed=c['noise_seed'],phase=c['phase'],profile=c['profile'],
            valid=valid,online_quality=c['qualified'],packet_quality=bool(packet['valid'][0]),geometry_legal=legal,geometry=geometry.tolist(),checks=local,decoder_metrics=metrics))
        rows.append({key:packet[key][0] for key in ['x','y','heads']});all_gates.append(local)
    data={key:np.array([r[key] for r in rows]) for key in ['x','y','heads']};data['valid']=np.array([r['valid'] for r in metadata],dtype=bool)
    if not rows:raise ValueError('No available rows;retain raw failure evidence')
    groups=[]
    for source in run['sources']:
        ids=[i for i,r in enumerate(metadata) if r['source_name']==source['name']]
        if ids:
            checks[source['name']+'_identical_state_warm']=all(np.array_equal(data['x'][i,:81],data['x'][ids[0],:81]) for i in ids)
            for direction in range(cfg['directions']):
                for epsilon in cfg['PD_eps_rad']:
                    signs={str(sign):[i for i in ids if metadata[i]['profile']['direction']==direction and metadata[i]['profile']['epsilon']==epsilon and metadata[i]['profile']['sign']==sign] for sign in (1,-1)}
                    complete=all(len(v)==cfg['repeats'] for v in signs.values())
                    if not complete and not full:raise ValueError('Available source must retain every declared signed repeat')
                    pair_ids=signs['1']+signs['-1'];group=dict(source_name=source['name'],direction=direction,epsilon=epsilon,rows=signs,qualified=bool(complete and data['valid'][pair_ids].all()))
                    if full:group['complete']=complete
                    groups.append(group)
    checks['declared_sources']=len(run['sources'])==(cfg['expected_sources'] if full else 24);checks['declared_cases']=len(run['cases'])==(cfg['expected_cases'] if full else 816)
    checks['all_available_rows_retained']=len(rows)==sum(c['available'] for c in run['cases'])
    checks['cost_accounted']=run['actual_native_integrations']==sum(p['native_integrations'] for p in producers)+8*len(rows) and run['optimizer_steps']==0
    checks['no_CHECK_response_rows']=run['CHECK_response_rows_used']==0
    np.savez_compressed(out/'response_rows.npz',**data)
    verify_source_snapshot(receipt,live=True);verify_source_snapshot(json.loads(verify_file_pin(run['source_receipt']).read_text()),live=True)
    result=dict(schema='controlled_pen_response_preparation:2' if full else 'controlled_pen_TRAIN_preparation:1',probe=file_pin(rp),config=file_pin(cp),protocol=run['protocol'],source_receipt=file_pin(out/'source_receipt.json'),
        rows=file_pin(out/'response_rows.npz'),metadata=metadata,response_groups=groups,noise_state_audits=noise_state_audits,directions=run['directions'],physics=next(s['physics'] for s in run['sources'] if s['available']),
        role=run['role'],scope=run['scope'],checks={k:bool(v) for k,v in checks.items()},passed=sum(checks.values()),total=len(checks),
        total_rows=len(rows),qualified_rows=int(data['valid'].sum()),rejected_rows=[r['row_index'] for r in metadata if not r['valid']],qualified_response_groups=sum(g['qualified'] for g in groups),
        row_gate_passed=sum(sum(g.values()) for g in all_gates),row_gate_total=sum(len(g) for g in all_gates),
        physical_qualification='ALL_AVAILABLE_ROWS_PASS' if data['valid'].all() else 'REJECTED_ROWS_RETAINED_AND_MASKED',
        actual_native_integrations=0,optimizer_steps=0,CHECK_response_rows_used=0,scales_fitted=False)
    if full:result.update(direction_basis=cfg['direction_basis'],direction_rank=int(np.linalg.matrix_rank(directions)),direction_condition=float(np.linalg.cond(directions)),validation_rows_used_for_TRAIN=0,unavailable_case_indices=[i for i,c in enumerate(run['cases']) if not c['available']])
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],rows=result['total_rows'],qualified=result['qualified_rows'],rejected=result['rejected_rows'],response_groups=[result['qualified_response_groups'],len(groups)],row_gates=[result['row_gate_passed'],result['row_gate_total']])),flush=True)
    if not all(checks.values()):raise ValueError('Retain preparation contract discrepancy')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
