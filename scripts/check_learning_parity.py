"""Narrow CPU neural-port parity; two saved features, no optimization/native.

The old WM is loaded by file spec only here. Production learning uses package
imports. Actor/Critic confirmation uses byte-copy ledger + saved paired100
states/health evidence, without constructing or forwarding those networks.
"""
from pathlib import Path
import argparse,importlib.util,json,os,sys,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def need(v,msg):
    if not v:raise ValueError(msg)

def run(manifest):
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import torch
    torch.set_num_threads(4);torch.set_default_dtype(torch.float32)
    need(not torch.cuda.is_initialized(),'CPU only')
    from supermimic_research.artifacts import ArtifactRegistry,file_pin
    from supermimic_research.learning import world_model as new
    from supermimic_research.runtime import structured
    registry=ArtifactRegistry(manifest)
    result=json.loads(registry.resolve('warm_WM_result').read_text());health=json.loads(registry.resolve_pin(result['health']).read_text())
    need(result['iteration']==5000 and result['checkpoint']==health['checkpoint'] and all(v is True for v in health['checks'].values()),'Fixed healthy warm5000')
    protocol=json.loads(registry.resolve_pin(result['protocol']).read_text())
    old_source=protocol['sources']['trainer'];path=registry.resolve_pin(old_source)
    spec=importlib.util.spec_from_file_location('validation_legacy_WM224a',path);old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    need(old_source['sha256']=='224a0c9f47e196c626e862aa52d46402a454a6d16f0b497d3720ae74e25ca539','Frozen WM implementation')
    need(protocol['inputs']['original_acquisition']==registry.pin('old_original_TRAIN') and protocol['inputs']['original_pair_report']==registry.pin('old_pair_labels'),'Registry original data roles match warm experiment')
    checkpoint=torch.load(registry.resolve_pin(result['checkpoint']),map_location='cpu',weights_only=False)
    need(checkpoint['schema']==old.ARTIFACT and checkpoint['iteration']==5000,'Own fixed5000 artifact')
    # Saved prepared cache has an explicit filename in the frozen source, under
    # its recorded output, not a regenerated geometry or a new data selection.
    prepared=Path(protocol['output'])/'prepared_FIT_targets_and_features.npz'
    with np.load(prepared,allow_pickle=False) as z:
        features=z['features366_F64'][[0,1]].copy();gold_normalized=z['features366_FP32_normalized'][[0,1]].copy()
        norm_fixture={k:z[k].copy() for k in ['input_mean','input_std','head_mean','head_std']}
    norm=checkpoint['normalizers'];physical=checkpoint['physical_context']
    for key,v in norm_fixture.items():need(np.array_equal(norm[key].numpy(),v),'Saved normalization bytes '+key)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2718)
        a=old.make_model(torch).to(device='cpu',dtype=torch.float32)
        b=new.make_model(new.WorldModelConfig(hidden=(128,128)),device='cpu',dtype='float32')
    for model in [a,b]:model.load_state_dict(checkpoint['model_state'],strict=True);model.eval();model.requires_grad_(False)
    need(new.raw_tree_equal(a.state_dict(),b.state_dict()) and new.raw_tree_equal(a.state_dict(),checkpoint['model_state']),'Strict layers.* parameter readback')
    output={};calls=dict(own_CPU_checkpoint_loads=2,WM_constructors=2,WM_forward=0,CPU_input_AD_grad_calls=0,Actor_Critic_constructors=0,Actor_Critic_forward=0,optimizer_constructors=0,optimizer_updates=0,native_model_FK_forward_steps=0,CUDA_calls=0)
    F=torch.tensor(features,dtype=torch.float64);x=F[:,:58];h=F[:,78];n=F[:,79]
    for tag,model,codec,headsfn in [('old',a,old.loadmod('learning_parity_legacy_structured',registry.resolve_pin(protocol['sources']['structured_codec'])),old.heads_from_physical),('new',b,structured,new.heads_from_physical)]:
        u=F[:,58:78].clone().requires_grad_(True)
        current=torch.cat([F[:,:58],u,F[:,78:]],dim=-1);normalized=((current-norm['input_mean'])/norm['input_std']).float()
        need(np.array_equal(normalized.detach().numpy(),gold_normalized),'Same saved normalized features at base action')
        raw=model(normalized);calls['WM_forward']+=1
        head=raw[:,:178].double()*norm['head_std']+norm['head_mean']
        decoded=codec.decode_physical_heads(x,headsfn(head),(h*n).double(),physical['mass'],physical['I'],physical['g'])
        projected=(raw*torch.linspace(.001,.002,222)).sum()+(head*torch.linspace(.003,.004,178)).sum()+(decoded[:,40:43]*torch.tensor([.2,-.1,.3])).sum()
        gu=torch.autograd.grad(projected,u)[0];calls['CPU_input_AD_grad_calls']+=1
        direction=torch.cos(torch.arange(20,dtype=torch.float64))
        need(torch.isfinite(gu).all() and torch.isfinite(decoded).all(),'Finite CPU input AD and decode')
        output[tag]=dict(raw222=raw.detach(),physical178=head.detach(),decoded58=decoded.detach(),u20_gradient=gu.detach(),u_direction=(gu*direction).sum(-1).detach())
    comparisons={}
    for key,oldvalue in output['old'].items():
        value=output['new'][key]
        need(oldvalue.dtype==value.dtype and oldvalue.shape==value.shape and torch.allclose(oldvalue,value,rtol=1e-12,atol=1e-12),key+' parity')
        comparisons[key]=dict(shape=list(value.shape),dtype=str(value.dtype),max_abs_error=float((oldvalue-value).abs().max()))
    need(all(p.grad is None for model in [a,b] for p in model.parameters()),'No parameter gradients or training')
    # No extra Actor/Critic construction/forward: code bytes and own saved
    # serialized-state/health fixtures are sufficient for this narrow port.
    ledger_path=ROOT/'provenance/dex_chord_import.json';ledger=json.loads(ledger_path.read_text())
    from dex_supermimic.rl.dex_supermimic import networks,common
    bound={}
    for key,module in [('networks',networks),('common',common)]:
        actual=Path(module.__file__).resolve();entry=next(v for v in ledger['copied_files'] if Path(v['destination']['path']).resolve()==actual)
        need(entry['byte_equal'] is True and file_pin(actual)==entry['destination'] and entry['source']['sha256']==entry['destination']['sha256'],'Actual new imported Dex copy binding '+key)
        bound[key]=file_pin(actual)
    pairpath=registry.resolve('paired100_checkpoint');pair=torch.load(pairpath,map_location='cpu',weights_only=False)
    need(pair['schema']=='phone_tripod_sharedWM_A1A2_training_checkpoint:1' and pair['paired_round']==100 and pair['shared_WM_updates']==200,'Only fixed paired100 migration fixture')
    paired_protocol=json.loads(registry.resolve_pin(pair['protocol']).read_text())
    paired_output=Path(paired_protocol['output']);paired_health=paired_output/'health_0100.json';ph=json.loads(paired_health.read_text())
    need(ph['checkpoint']==file_pin(pairpath) and ph['passed']==ph['total']==15 and all(v is True for v in ph['checks'].values()),'Existing real paired100 health, not new prediction claims')
    norms={}
    for mode,arm in pair['arms'].items():
        for statekey,normkey in [('stateactor','actor'),('statecritic','critic'),('statecritic_target','critic_target')]:
            state=arm[statekey];saved={k[len('norm.'):]:v for k,v in state.items() if k.startswith('norm.')}
            need(new.raw_tree_equal(saved,arm['norm'][normkey]) and all(torch.isfinite(v).all() for v in state.values()),'Serialized own weights/norm raw readback '+mode+'/'+statekey)
            norms[mode+'/'+statekey]=dict(weight_tensor_count=len(state),normalizer_tensor_count=len(saved),raw_digest=new.tensor_digest(state))
    evidence=paired_output/'health_readback_0100.npz'
    with np.load(evidence,allow_pickle=False) as z:
        pairs=[key[:-7] for key in z.files if key.endswith('_before')]
        need(len(pairs)==5,'Expected stored WM plus2Actor+2Critic readback fixtures')
        for key in pairs:
            aa,bb=z[key+'_before'],z[key+'_after'];need(aa.dtype==bb.dtype and aa.shape==bb.shape and aa.tobytes()==bb.tobytes(),'Saved prediction health fixture '+key)
    need('mujoco' not in sys.modules and not torch.cuda.is_initialized(),'No native/GPU')
    checks={k:True for k in ['fixed_warm5000_and_saved_features_norm_context','strict_same_layers_weights','raw222_head178_decoded58_CPU_forward_parity','fixed_u_input_AD_finite_parity_no_parameter_gradient','new_actual_Dex_imports_bound_to_identical_copy_ledger','paired100_serialized_norms_and_stored_prediction_health_fixtures']}
    return dict(schema='phone_learning_migration_parity:1',status='PASS_NARROW_CPU_WM_PORT_AND_SAVED_ACTOR_CRITIC_COPY_PARITY',checks=checks,passed=len(checks),total=len(checks),
        artifact_manifest=file_pin(manifest),warm_result=registry.pin('warm_WM_result'),warm_checkpoint=result['checkpoint'],legacy_WM_source=old_source,new_WM_source=file_pin(new.__file__),
        prepared_features=file_pin(prepared),fixed_feature_rows=[0,1],feature_source_role='Existing FIT cache only for parity, no fitting or data reselection',comparisons=comparisons,
        input_AD_direction=output['new']['u_direction'].tolist(),paired100_checkpoint=file_pin(pairpath),paired100_health=file_pin(paired_health),stored_prediction_fixture=file_pin(evidence),stored_prediction_pairs=pairs,
        copied_Dex_imports=bound,serialized_Actor_Critic_states=norms,copy_ledger=file_pin(ledger_path),calls=calls,validation_source=file_pin(__file__),
        scope=['Two actual saved366 features and one unchanged fixed warm5000 checkpoint. Old and new WM each perform one CPU batch2 forward and one fixed scalar input-u AD.',
        'Actor/Critic code bytes/import locations, paired100 serialized raw states/norms and existing before/after prediction health fixtures are checked. No new Actor/Critic forward or constructor.',
        'No optimizer, parameter gradient, training, native model/physics or CUDA. This is migration parity, not evidence of learned model accuracy, physical-action gradients, contact, holding or policy benefit.'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=ROOT/'configs/artifacts.toml');p.add_argument('--output',type=Path,default=ROOT/'provenance/learning_parity.json');a=p.parse_args()
    need(not a.output.exists(),'Fresh validation output; retain earlier outcomes')
    try:r=run(a.manifest)
    except BaseException as e:
        a.output.write_text(json.dumps(dict(schema='phone_learning_migration_parity:1',status='FAILED_PRESERVED_NO_RETRY',error=repr(e),traceback=traceback.format_exc()),indent=2),encoding='utf8');raise
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(dict(report=str(a.output),passed=r['passed'],total=r['total'])))
