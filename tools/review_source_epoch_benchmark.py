"""Independent packet and matched-repeat response checks for source epochs."""
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


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());bp=(cp.parent/cfg['benchmark']).resolve();run=json.loads(bp.read_text());hp=(cp.parent/cfg['scale_health']).resolve();health=json.loads(hp.read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or run['passed']!=run['total']:raise ValueError('Fresh source-epoch physical readback')
    checkpoint=torch.load(verify_file_pin(health['checkpoint']),map_location='cpu',weights_only=False);scale=checkpoint['arms']['DIRECT']['scales']['y_scale']
    out.mkdir();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),cp]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    models={};contexts={};sources={};checks={};rows=[];metadata=[]
    for entry in run['sources']:
        sources[entry['name']]=torch.load(verify_file_pin(entry['source']),map_location='cpu',weights_only=False)
        key=entry['model']['path']
        if key not in models:models[key]=mujoco.MjModel.from_binary_path(str(verify_file_pin(entry['model'])));contexts[key]=dict(np.load(verify_file_pin(entry['physical_context'])))
    for i,c in enumerate(run['cases']):
        z=dict(np.load(verify_file_pin(c['trace'])));m=models[c['model']['path']];source=sources[c['source_name']];metrics,wrench=decode_saved_case(c,z,m,contexts[c['model']['path']])
        packet=pack_case(m,c,z,wrench,np.array(c['qids']),np.array(c['dofs']),np.array(c['aids']),z['initial_warmstart'],c['physics']);ns=source['native']['native']['data']['arrays']
        local=dict(source_q=np.array_equal(z['pre_qpos'][0],ns['qpos'][0]),source_v=np.array_equal(z['pre_qvel'][0],ns['qvel'][0]),warm=np.array_equal(z['initial_warmstart'],ns['qacc_warmstart'][0]),
            phase=int(z['pre_action_phase'][0])==source['phase'],no_aid=not np.any(z['post_qfrc_applied']) and not np.any(z['post_xfrc_applied']),
            geometry=bool(np.all(packet['geometry_flags'][0]<=np.array([.002,.0002,.001]))),packet_quality=bool(packet['valid'][0]),
            recorded_restore=c['restore_exact'],PD_unclipped=c['PD_error_rad']<=2e-7,native_count=c['native_integrations']==8,
            role=c['role']=='CHECK_SOURCE_EPOCH_EQUIVALENCE_NEVER_FIT')
        local.update({'EFC_'+k:bool(v) for k,v in metrics['checks'].items()});checks.update({f'row{i}_'+k:bool(v) for k,v in local.items()})
        rows.append(packet['y'][0]);metadata.append(dict(name=c['name'],source_name=c['source_name'],mode=c['mode'],profile=c['profile'],checks=local,
            x=packet['x'][0],PD=packet['x'][0,81:101],raw=z['actions20'][0]))
    y=np.array(rows);np.savez_compressed(out/'responses.npz',y=y,x=np.array([m['x'] for m in metadata]));comparisons=[]
    blocks=dict(hand40=slice(0,40),object12=slice(40,52),warm26=slice(52,78),wrench6=slice(78,84),Fn6=slice(84,90))
    for source_name in sources:
        group=[i for i,m in enumerate(metadata) if m['source_name']==source_name];keys=sorted({(metadata[i]['profile']['direction'],metadata[i]['profile']['epsilon'],metadata[i]['profile']['sign']) for i in group})
        for j,key in enumerate(keys):
            select=lambda mode:[i for i in group if metadata[i]['mode']==mode and tuple(metadata[i]['profile'][k] for k in ['direction','epsilon','sign'])==key]
            left,right=select('STRICT'),select('READ_LOCKED_EPOCH')
            if len(left)!=3 or len(right)!=3:raise ValueError('Complete paired repeat groups required')
            checks[f'{source_name}_G{j}_same_input']=all(np.array_equal(metadata[i]['x'],metadata[left[0]]['x']) and np.array_equal(metadata[i]['raw'],metadata[left[0]]['raw']) for i in left+right)
            for name,sl in blocks.items():
                p=y[left,sl]/scale[sl];q=y[right,sl]/scale[sl];difference=float(np.linalg.norm(p.mean(0)-q.mean(0)))
                radius=float(np.max(np.linalg.norm(p-p.mean(0),axis=1))+np.max(np.linalg.norm(q-q.mean(0),axis=1)))
                within=difference<=radius+cfg['normalized_envelope_floor'];checks[f'{source_name}_G{j}_{name}_observed_envelope']=within
                comparisons.append(dict(source_name=source_name,profile_key=list(key),block=name,normalized_mean_difference_L2=difference,observed_radius_sum=radius,floor=cfg['normalized_envelope_floor'],within=within))
    # Keep the JSON compact by pinning the numeric inputs rather than embedding x/raw repeatedly.
    for row in metadata:
        for key in ['x','PD','raw']:row.pop(key)
    checks.update(declared_rows=len(rows)==324,declared_groups=len(comparisons)==270,epoch_committed=run['epoch']['validated'] and not run['epoch']['active'])
    verify_source_snapshot(receipt,live=True)
    result=dict(schema='source_epoch_independent_physical_review:1',config=file_pin(cp),benchmark=file_pin(bp),scale_health=file_pin(hp),scale_checkpoint=health['checkpoint'],source_receipt=file_pin(out/'source_receipt.json'),
        responses=file_pin(out/'responses.npz'),metadata=metadata,comparisons=comparisons,checks=checks,passed=sum(checks.values()),total=len(checks),
        envelope_passed=sum(c['within'] for c in comparisons),envelope_total=len(comparisons),actual_native_integrations=0,optimizer_steps=0,scope=cfg['scope'])
    write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],envelope=[result['envelope_passed'],result['envelope_total']],failures=[c for c in comparisons if not c['within']][:5])),flush=True)
    if not all(checks.values()):raise ValueError('Preserve source-epoch physical/noise equivalence failure; do not relax the registered floor')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
