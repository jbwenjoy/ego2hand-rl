"""Independent physical packets, finite responses and frozen-model PD JVPs."""
import argparse,json,subprocess,sys,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.mechanics.warp_packet_decode import decode_saved_case
from supermimic_research.mechanics.controlled_pen_records import pack_case
from supermimic_research.learning.pen_response import model_from_arm
from supermimic_research.learning.pen_native_response import compare_native_response,vector_comparison,repeated_contrast_spread
from supermimic_research.learning.native_pen_ppo import tensor_tree_equal


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());rp=(cp.parent/cfg['probe_report']).resolve();run=json.loads(rp.read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or run['passed']!=run['total']:raise ValueError('Fresh declared physical-response review')
    hp=(cp.parent/cfg['wm_health']).resolve();health=json.loads(hp.read_text());parents_path=(cp.parent/cfg['wm_parents']).resolve();parents=json.loads(parents_path.read_text())
    prepared=json.loads(verify_file_pin(health['prepared']).read_text())
    if prepared['parents']!=file_pin(parents_path):raise ValueError('WM parent provenance mismatch')
    checkpoint=torch.load(verify_file_pin(health['checkpoint']),map_location='cpu',weights_only=False)
    additional=None;duplicate_direct=None;model_sources={mode:health['checkpoint'] for mode in checkpoint['arms']}
    if 'additional_wm_health' in cfg:
        additional_path=(cp.parent/cfg['additional_wm_health']).resolve();additional=json.loads(additional_path.read_text())
        if additional['prepared']!=health['prepared'] or additional['update']!=2000 or additional['passed']!=additional['total']:raise ValueError('Matched healthy final2000 coupled-model dataset required')
        extra=torch.load(verify_file_pin(additional['checkpoint']),map_location='cpu',weights_only=False)
        duplicate_direct=tensor_tree_equal(extra['arms']['DIRECT']['model'],checkpoint['arms']['DIRECT']['model'])
        checkpoint['arms']=dict(checkpoint['arms'],IMPULSE=extra['arms']['IMPULSE']);model_sources['IMPULSE']=additional['checkpoint']
    old_case=next(c for c in parents['cases'] if c['controller']=='STATE_ONLY' and c['dt']==.0025)
    old_arrays=dict(np.load(verify_file_pin(old_case['fixed_model_arrays'])));old_scalars=json.loads(verify_file_pin(old_case['fixed_model_scalars']).read_text())
    out.mkdir();protocol=(cp.parent/cfg['protocol']).resolve();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),cp,protocol]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    states={};models={};contexts={};frame_audits=[];checks={};rows=[];metadata=[];qualities=[];row_checks=[]
    if additional is not None:checks['matched_duplicate_DIRECT_weights']=duplicate_direct
    for entry in run['sources']:
        if not entry['available']:continue
        source=torch.load(verify_file_pin(entry['source']),map_location='cpu',weights_only=False);states[entry['name']]=source;native=source['native']['native']['model']
        same_arrays=set(old_arrays)==set(native['arrays']) and all(np.array_equal(value,native['arrays'][key]) for key,value in old_arrays.items())
        same_scalars=old_scalars==native['scalars'];checks[entry['name']+'_full_WM_plant_arrays']=same_arrays;checks[entry['name']+'_full_WM_plant_scalars']=same_scalars
        key=entry['model']['path']
        if key not in models:models[key]=mujoco.MjModel.from_binary_path(str(verify_file_pin(entry['model'])));contexts[key]=dict(np.load(verify_file_pin(entry['physical_context'])))
        frame_audits.append(dict(name=entry['name'],phase=entry['phase'],rotation_gap_rad=entry['cached_rotation_gap_rad'],COM_gap_m=entry['cached_COM_gap_m']))
    for index,c in enumerate(run['cases']):
        if not c['available']:continue
        z=dict(np.load(verify_file_pin(c['trace'])));w=dict(np.load(verify_file_pin(c['window'])));source=states[c['source_name']];m=models[c['model']['path']];p=contexts[c['model']['path']]
        metrics,decoded=decode_saved_case(c,z,m,p)
        packet=pack_case(m,c,z,decoded,np.array(c['qids']),np.array(c['dofs']),np.array(c['aids']),z['initial_warmstart'],c['physics'])
        mean_world=decoded['world_COM_wrench'].mean(0);Rq=source['source_qpos_R'];Rc=source['source_cached_R']
        true_label=np.r_[Rq.T@mean_world[:3],Rq.T@mean_world[3:]];cached_label=np.r_[Rc.T@mean_world[:3],Rc.T@mean_world[3:]]
        ns=source['native']['native']['data']['arrays'];t0=float(ns['time'][0]);sig=mujoco.mjtState.mjSTATE_INTEGRATION;initial=mujoco.MjData(m);mujoco.mj_setState(m,initial,z['initial_native_state'],sig)
        local=dict(source_qpos=np.array_equal(z['pre_qpos'][0],ns['qpos'][0]),source_qvel=np.array_equal(z['pre_qvel'][0],ns['qvel'][0]),
            actual_initial_warm=np.array_equal(z['initial_warmstart'],ns['qacc_warmstart'][0]),source_phase=int(z['pre_action_phase'][0])==source['phase'],
            held_PD=np.array_equal(z['post_ctrl'][:,c['aids']],np.tile(c['actual_PD20'],(8,1))),clock=bool(np.max(abs(z['post_GPU_time']-(t0+np.arange(1,9)*.0025)))<1e-6),
            fixed_mount=bool(np.array_equal(z['post_mocap_pos'],np.tile(initial.mocap_pos,(8,1,1))) and np.array_equal(z['post_mocap_quat'],np.tile(initial.mocap_quat,(8,1,1)))),
            no_aid=not np.any(z['post_qfrc_applied']) and not np.any(z['post_xfrc_applied']),role=c['role']=='CHECK_TRAINED_ACTOR_PHYSICAL_RESPONSE_NEVER_FIT',
            true_source_BODY_label=bool(np.allclose(packet['y'][0,78:84],true_label,atol=1e-10,rtol=1e-10)),
            resident_cached_axes=bool(np.array_equal(w['source_R'],Rc)),resident_mean_wrench=bool(np.allclose(w['wrench_sum'][5]/8,cached_label,atol=1e-9,rtol=1e-9)),
            group_normal=bool(np.allclose(packet['y'][0,84:90],w['Fn6_sum']/8,atol=1e-9,rtol=1e-9)))
        local.update({'EFC_'+key:bool(value) for key,value in metrics['checks'].items()})
        geometry=packet['geometry_flags'][0];legal=bool(np.all(geometry<=np.array([.002,.0002,.001])))
        valid=bool(c['qualified'] and packet['valid'][0] and all(local.values()) and legal)
        rows.append({key:packet[key][0] for key in ['x','y','heads']});qualities.append(valid);metadata.append(dict(case_index=index,name=c['name'],source_name=c['source_name'],actor=c['actor'],phase=c['phase'],profile=c['profile'],valid=valid,
            packet_quality=bool(packet['valid'][0]),online_quality=c['qualified'],geometry_legal=legal,decoder_metrics=metrics,cached_vs_true_label_L2=float(np.linalg.norm(cached_label-true_label)),checks=local));row_checks.append(local)
    data={key:np.array([row[key] for row in rows]) for key in ['x','y','heads']};data['valid']=np.array(qualities)
    np.savez_compressed(out/'response_rows.npz',**data)
    checks.update({f'row{i}_'+k:bool(v) for i,local in enumerate(row_checks) for k,v in local.items()})
    checks['declared_sources']=len(run['sources'])==12;checks['declared_branches']=len(run['cases'])==324
    models_by_mode={};predictions={};forward={};physics_match={}
    torch.set_num_threads(4);device=torch.device('cuda:0')
    for mode,saved in checkpoint['arms'].items():
        checks[mode+'_shared_frozen_y_scale']=np.array_equal(saved['scales']['y_scale'],next(iter(checkpoint['arms'].values()))['scales']['y_scale'])
        model=model_from_arm(saved,device);model.eval();models_by_mode[mode]=model
        physics_match[mode]=all(e['physics']==saved['physics'] for e in run['sources'] if e['available']);checks[mode+'_physics']=physics_match[mode]
        with torch.no_grad():prediction=model.prediction(torch.tensor(data['x'],device=device,dtype=torch.float32)).cpu().numpy()
        predictions[mode]=prediction;scale=np.asarray(saved['scales']['y_scale']);mask=data['valid'];forward[mode]={}
        for name,sl in [('object12',slice(40,52)),('wrench6',slice(78,84)),('Fn6',slice(84,90))]:
            error=(prediction[:,sl]-data['y'][:,sl])/scale[sl];forward[mode][name]=dict(valid_rows=int(mask.sum()),normalized_MSE=float(np.mean(error[mask]**2)) if mask.any() else None)
    np.savez_compressed(out/'frozen_model_predictions.npz',**predictions)
    base_scale=np.asarray(next(iter(checkpoint['arms'].values()))['scales']['y_scale']);blocks=dict(object12=slice(40,52),wrench6=slice(78,84),Fn6=slice(84,90));pairs=[];amplitudes=[];base_spreads=[];derivatives={};pair_index={}
    directions=np.load(verify_file_pin(run['directions']))['directions']
    for entry in run['sources']:
        source_name=entry['name'];ids=[i for i,c in enumerate(metadata) if c['source_name']==source_name]
        if not ids:continue
        baseline=[i for i in ids if metadata[i]['profile']['direction']<0]
        if len(baseline)!=3:raise ValueError('All baseline repeats must be retained')
        bx=data['x'][baseline[0]:baseline[0]+1];checks[source_name+'_identical_state_warm']=all(np.array_equal(data['x'][i,:81],bx[0,:81]) for i in ids)
        base_spreads.append(dict(source_name=source_name,blocks={name:float(np.max(np.linalg.norm((data['y'][baseline,sl]-data['y'][baseline,sl].mean(0))/base_scale[sl],axis=1))) for name,sl in blocks.items()}))
        for direction in range(2):
            for mode,model in models_by_mode.items():
                x=torch.tensor(bx,device=device,dtype=torch.float32);tangent=torch.zeros_like(x);tangent[:,81:101]=torch.tensor(directions[direction],device=device,dtype=torch.float32)
                _,jvp=torch.autograd.functional.jvp(model.prediction,x,tangent);derivatives[(source_name,direction,mode)]=jvp[0].detach().cpu().numpy()
            for epsilon in [.002,.005]:
                selected=lambda sign:[i for i in ids if metadata[i]['profile']['direction']==direction and metadata[i]['profile']['epsilon']==epsilon and metadata[i]['profile']['sign']==sign]
                plus,minus=selected(1),selected(-1)
                if len(plus)!=3 or len(minus)!=3:raise ValueError('Retain all paired repeats')
                qualified=bool(data['valid'][plus+minus].all());physical_fd=(data['y'][plus].mean(0)-data['y'][minus].mean(0))/(2*epsilon)
                row=dict(source_name=source_name,actor=entry['actor'],phase=entry['phase'],direction=direction,epsilon=epsilon,rows=plus+minus,qualified=qualified,blocks={},models={})
                for name,sl in blocks.items():row['blocks'][name]=repeated_contrast_spread(data['y'][plus,sl],data['y'][minus,sl],base_scale[sl])
                for mode,saved in checkpoint['arms'].items():
                    scale=np.asarray(saved['scales']['y_scale']);row['models'][mode]={}
                    for name,sl in blocks.items():
                        comparison,_=compare_native_response(data['y'][plus,sl],data['y'][minus,sl],derivatives[(source_name,direction,mode)][sl],scale[sl],epsilon)
                        row['models'][mode][name]=comparison
                pair_index[(source_name,direction,epsilon)]=len(pairs);pairs.append(row);derivatives[(source_name,direction,epsilon)]=physical_fd
            amp=dict(source_name=source_name,direction=direction,blocks={})
            for name,sl in blocks.items():
                comparison=vector_comparison(derivatives[(source_name,direction,.002)][sl]/base_scale[sl],derivatives[(source_name,direction,.005)][sl]/base_scale[sl])
                amp['blocks'][name]=comparison;stable=comparison['relative_L2_error']<=cfg['native_amplitude_relative_error'] and comparison['cosine'] is not None and comparison['cosine']>=cfg['native_amplitude_cosine']
                for epsilon in [.002,.005]:
                    row=pairs[pair_index[(source_name,direction,epsilon)]];other=pairs[pair_index[(source_name,direction,.005 if epsilon==.002 else .002)]]
                    screen=bool(stable and row['qualified'] and other['qualified'] and row['blocks'][name]['contrast_exceeds_observed_spread'] and other['blocks'][name]['contrast_exceeds_observed_spread'])
                    row['blocks'][name]['two_amplitude_consistency_screen']=screen
                    for mode in checkpoint['arms']:
                        c=row['models'][mode][name];c['candidate_diagnostic_screen']=bool(screen and physics_match[mode] and c['relative_L2_error']<cfg['model_AD_relative_error'] and c['cosine'] is not None and c['cosine']>=cfg['model_AD_cosine'])
            amplitudes.append(amp)
    aggregate={}
    for mode in checkpoint['arms']:
        aggregate[mode]={}
        for name,sl in blocks.items():
            usable=[r for r in pairs if r['qualified']];num=den=0.
            for row in usable:
                key=(row['source_name'],row['direction']);scale=base_scale[sl];pred=derivatives[(*key,mode)][sl]/scale;target=derivatives[(*key,row['epsilon'])][sl]/scale
                num+=float(np.sum((pred-target)**2));den+=float(np.sum(target**2))
            aggregate[mode][name]=dict(qualified_pairs=len(usable),AD_relative_L2_error=float(np.sqrt(num/max(den,1e-24))),zero_response_relative_error=1.,
                signal_pairs=sum(r['blocks'][name]['contrast_exceeds_observed_spread'] for r in usable),two_amplitude_screen_pairs=sum(r['blocks'][name]['two_amplitude_consistency_screen'] for r in usable),
                candidate_model_screen_pairs=sum(r['models'][mode][name]['candidate_diagnostic_screen'] for r in usable))
    checks={k:bool(v) for k,v in checks.items()};verify_source_snapshot(receipt,live=True)
    result=dict(schema='trained_pen_physical_response_independent_review:1',config=file_pin(cp),protocol=file_pin(protocol),probe=file_pin(rp),wm_health=file_pin(hp),wm_checkpoint=health['checkpoint'],
        wm_parents=file_pin(parents_path),wm_parent_model_arrays=old_case['fixed_model_arrays'],wm_parent_model_scalars=old_case['fixed_model_scalars'],source_receipt=file_pin(out/'source_receipt.json'),
        rows=file_pin(out/'response_rows.npz'),predictions=file_pin(out/'frozen_model_predictions.npz'),metadata=metadata,qualified_rows=int(data['valid'].sum()),total_rows=len(rows),frame_audits=frame_audits,
        frozen_model_forward=forward,response_pairs=pairs,amplitude_comparisons=amplitudes,baseline_spreads=base_spreads,aggregate=aggregate,checks=checks,passed=sum(checks.values()),total=len(checks),
        model_checkpoint_sources=model_sources,actual_native_integrations=0,optimizer_steps=0,role=run['role'],scope=cfg['scope'])
    if additional is not None:result['additional_wm_health']=file_pin(additional_path)
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],qualified_rows=result['qualified_rows'],aggregate=aggregate,
        cached_rotation_gap_range=[min(c['rotation_gap_rad'] for c in frame_audits),max(c['rotation_gap_rad'] for c in frame_audits)])),flush=True)
    if not all(checks.values()):raise ValueError('Retain independent physical packet/plant readback discrepancy')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
