"""New105 consumer contract fixtures and installed CPU networks, never a run audit.

The only old real checkpoint read below is a rejection test; no weights from it
are installed. Positive network tests use fresh synthetic installed modules.
"""
from __future__ import annotations
import argparse,copy,hashlib,os,sys,time,traceback
import numpy as np
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_native.io import pin,write,read,loadnp
from full151_native.metadata import FORMAT_VERSION,MODE,VARIANT,DESCRIPTOR_SCHEMA
from full151_native.controller import _digest,PURE_PATH,PURE_SHA256,KIND
from full151_native.checkpoint import validate_checkpoint_namespace,validate_controller_metadata,_actor_payload,_tensor_digest,_validated_bundle
from full151_native.world_model_checkpoint import _wm_payload,_physical_xi,_make_frozen_wrapper,validate_health_metadata,FIELDS,HEALTH_SCHEMA


def main(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    before=time.perf_counter();checks={};details={}
    sources={k:pin(p) for k,p in {
        'checker':Path(__file__),'actor_consumer':ROOT/'src/full151_native/checkpoint.py',
        'WM_consumer':ROOT/'src/full151_native/world_model_checkpoint.py','controller':ROOT/'src/full151_native/controller.py',
        'bundle':ROOT/'src/full151_native/bundle.py','io':ROOT/'src/full151_native/io.py','metadata':ROOT/'src/full151_native/metadata.py','pure_projection':PURE_PATH}.items()}
    def require(name,value):
        checks[name]=bool(value)
        if not checks[name]:raise AssertionError(name)
    def reject(name,call):
        try:call()
        except (ValueError,KeyError,RuntimeError,TypeError) as e:checks[name]=True;details[name]=str(e);return
        raise AssertionError('Expected rejection: '+name)
    try:
        import torch
        import mujoco
        from dex_supermimic.rl.dex_supermimic.networks import Actor,WorldModel
        torch.set_num_threads(1)
        require('fresh_CPU_mask_and_CUDA_uninitialized',os.environ['CUDA_VISIBLE_DEVICES']=='-1' and not torch.cuda.is_initialized() and not torch.cuda.is_available())
        forbidden={}
        def forbidden_call(name):
            def call(*a,**kw):
                forbidden[name]+=1;raise AssertionError('Consumer CPU checker forbids '+name)
            return call
        for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_collision','mj_geomDistance','mj_contactForce','mj_kinematics','mj_comPos'):
            forbidden[name]=0;setattr(mujoco,name,forbidden_call(name))
        rng=torch.get_rng_state().clone()
        fixture_file=output/'synthetic_bundle_marker.json';write(fixture_file,{'synthetic_fixture':True,'physical_qualified':False})
        descriptor={'schema':DESCRIPTOR_SCHEMA,'mode':MODE,'retarget_variant_id':VARIANT,'input_bundle':pin(fixture_file),
                    'physical_reference_qualified':False,'known_rough_reference':True,'reference_states':151,'control_intervals':150,
                    'human_contact_observation_known':False,'source_identity':{'source_window':[200,351]},'source_pins':{}}
        descriptor['input_signature_sha256']=_digest(descriptor)
        root=dict(descriptor,trained_method='A1L',training_seed=0,run_role='engineering_smoke',config_sha256='b'*64,
                  execution_source_pins={'controller':sources['controller'],'pure_projection':sources['pure_projection']})
        controller={'schema':'full151_knownrough_baseline_EMA_C2:1','kind':KIND,'mode':MODE,'retarget_variant_id':VARIANT,
                    'format_version':FORMAT_VERSION,'exact_resume_supported':False,'root_descriptor':root,'root_descriptor_sha256':_digest(root),
                    'baseline_shape':[150,20],'baseline_u150_FP32_sha256':'c'*64,'active_intervals':[0,150],'terminal_observation':150,
                    'inactive_imagination':'finite u149 placeholder; runtime must keep survival0 and mask dead losses',
                    'margin_rad':.020,'pure_projection_sha256':PURE_SHA256,'controller_source_sha256':sources['controller']['sha256']}
        controller['reference_controller_signature']=_digest(controller)
        state={'format_version':105,'config':{'seed':0,'wm_hidden':(512,512,512)},'reference_signature':controller['reference_controller_signature'],
               'full151_controller':controller,'full151_runtime_evidence':{'fixture_only':True,'natural_end':150},
               'full151_initial_parameter_fingerprints':{k:{'sha256':'a'*64,'tensors':1,'parameters':1} for k in ('wm','fail_head','actor','critic')},'contact_dim':15}
        def mutate_header(key,value):
            s=copy.deepcopy(state);s[key]=value;return s
        def mutate_controller(key,value,*,resign=False):
            s=copy.deepcopy(state);s['full151_controller'][key]=value
            if resign:
                c=s['full151_controller'];c['root_descriptor_sha256']=_digest(c['root_descriptor'])
                c['reference_controller_signature']=_digest({k:v for k,v in c.items() if k!='reference_controller_signature'});s['reference_signature']=c['reference_controller_signature']
            return s
        require('positive_synthetic_signed105_metadata_only',validate_controller_metadata(state,descriptor,'A1L')[0] is controller)
        for version in (3,103,104):reject('reject_old_format_'+str(version),lambda version=version:validate_checkpoint_namespace(mutate_header('format_version',version)))
        reject('reject_boolean_format',lambda:validate_checkpoint_namespace(mutate_header('format_version',True)))
        old_relabel={'format_version':105,'rough_controller':{},'rough_runtime_evidence':{}}
        reject('reject_integer_only_relabelled104',lambda:validate_checkpoint_namespace(old_relabel))
        contaminated=copy.deepcopy(state);contaminated['rough_controller']={}
        reject('reject_mixed_legacy_namespace',lambda:validate_checkpoint_namespace(contaminated))
        for field,value in [('full151_runtime_evidence',{}),('full151_initial_parameter_fingerprints',{})]:reject('reject_missing_'+field,lambda field=field,value=value:validate_checkpoint_namespace(mutate_header(field,value)))
        fp=copy.deepcopy(state['full151_initial_parameter_fingerprints']);fp['actor']['parameters']=True
        reject('reject_bool_fingerprint_count',lambda:validate_checkpoint_namespace(mutate_header('full151_initial_parameter_fingerprints',fp)))
        for key,value in [('baseline_shape',[15,20]),('active_intervals',[0,15]),('terminal_observation',15),('mode','rough_observation_prefix_v1'),('retarget_variant_id','old_variant'),('inactive_imagination','finite u14 placeholder')]:
            reject('reject_old_or_wrong_'+key,lambda key=key,value=value:validate_checkpoint_namespace(mutate_controller(key,value)))
        reject('reject_signature_mutation',lambda:validate_controller_metadata(mutate_controller('margin_rad',.005),descriptor,'A1L'))
        reject('reject_wrong_arm',lambda:validate_controller_metadata(state,descriptor,'P_ONLY'))
        reject('reject_unknown_arm',lambda:validate_controller_metadata(state,descriptor,'UNKNOWN'))
        s=copy.deepcopy(state);s['config']['seed']=False
        reject('reject_bool_seed',lambda:validate_controller_metadata(s,descriptor,'A1L'))
        r=copy.deepcopy(root);r['schema']='rough_prefix_controller_input_descriptor:1'
        reject('reject_old_descriptor_even_resigned',lambda:validate_controller_metadata(mutate_controller('root_descriptor',r,resign=True),descriptor,'A1L'))
        r=copy.deepcopy(root);r['physical_reference_qualified']=True
        reject('reject_geometry_or_physics_promotion',lambda:validate_controller_metadata(mutate_controller('root_descriptor',r,resign=True),descriptor,'A1L'))
        r=copy.deepcopy(root);r['execution_source_pins']={}
        reject('reject_missing_actual_execution_sources',lambda:validate_controller_metadata(mutate_controller('root_descriptor',r,resign=True),descriptor,'A1L'))
        bad=dict(sources['controller']);bad['sha256']='0'*64;r=copy.deepcopy(root);r['execution_source_pins']['controller']=bad
        reject('reject_changed_source_hash',lambda:validate_controller_metadata(mutate_controller('root_descriptor',r,resign=True),descriptor,'A1L'))
        late=copy.deepcopy(state);late['full151_controller']['latest_actualu_evidence']={'fixture_only':True,'collect_call':2}
        require('latest_actualu_marker_excluded_from_reference_digest',validate_controller_metadata(late,descriptor,'A1L')[0] is late['full151_controller'])
        for label,value in [('nonfinite',float('nan')),('nonordinary',np.float32(1))]:
            r=copy.deepcopy(root);r['extra']=value
            reject('reject_'+label+'_descriptor',lambda r=r:_digest(r))
        with torch.random.fork_rng(devices=[]):
            actor=Actor(obs_dim=155);wm=WorldModel(dynamics_dim=278,contact_dim=15,hidden=(512,512,512),amp=False)
            actor_state=copy.deepcopy(actor.state_dict());wm_state=copy.deepcopy(wm.state_dict())
            state['modules']={'actor':actor_state,'wm':wm_state}
            require('installed_Actor_payload_ABI',_actor_payload(state) is actor_state)
            require('installed_WM_payload_ABI',_wm_payload(state) is wm_state)
            actor2=Actor(obs_dim=155);actor2.load_state_dict(_actor_payload(state),strict=True);actor2.eval().requires_grad_(False)
            require('installed_Actor_strict_load_digest_unchanged',_tensor_digest(actor2.state_dict())==_tensor_digest(actor_state))
            y=actor2(torch.zeros((2,155),dtype=torch.float32))
            require('installed_Actor_CPU_finite155_to20',y.shape==(2,20) and bool(torch.isfinite(y).all()) and y.device.type=='cpu')
            w2=WorldModel(dynamics_dim=278,contact_dim=15,hidden=(512,512,512),amp=False);w2.load_state_dict(_wm_payload(state),strict=True);w2.eval().requires_grad_(False)
            require('installed_WM_strict_load_digest_unchanged',_tensor_digest(w2.state_dict())==_tensor_digest(wm_state))
            wrapped=_make_frozen_wrapper(w2,np.zeros(46,dtype=np.float32))
            x=torch.zeros((2,58),dtype=torch.float32);x[:,43:52]=torch.eye(3).reshape(1,9);x.requires_grad_(True)
            u=torch.zeros((2,20),dtype=torch.float32,requires_grad=True);x1,contact=wrapped(x,u)
            gx,gu=torch.autograd.grad(x1.square().sum()+contact.square().sum(),(x,u))
            require('installed_WM_CPU_shape_finite_and_input_autograd',x1.shape==(2,58) and contact.shape==(2,15) and bool(torch.isfinite(gx).all()) and bool(torch.isfinite(gu).all()))
            require('installed_WM_parameters_frozen',not any(p.requires_grad for p in wrapped.parameters()))
            for name,input_x,input_u in [('wrong_state',x[:,:57],u),('wrong_control',x,u[:,:19]),('F64',x.double(),u.double()),('nan',x,torch.full_like(u,float('nan')))]:reject('reject_WM_input_'+name,lambda input_x=input_x,input_u=input_u:wrapped(input_x,input_u))
        require('module_construction_CPU_RNG_unchanged',torch.equal(rng,torch.get_rng_state()))
        s=copy.deepcopy(state);s['modules']['actor']['norm._var'][0,0]=-1
        reject('reject_actor_negative_variance',lambda:_actor_payload(s))
        s=copy.deepcopy(state);s['modules']['actor']['net.0.weight']=torch.zeros((512,207))
        reject('reject_actor_wrong_observation_ABI',lambda:_actor_payload(s))
        s=copy.deepcopy(state);s['modules']['actor']['net.2.bias'][0]=float('nan')
        reject('reject_actor_nonfinite_unlisted_tensor',lambda:_actor_payload(s))
        for field,value in [('contact_dim',14),('config',{'seed':0,'wm_hidden':[512,512]})]:reject('reject_WM_'+field,lambda field=field,value=value:_wm_payload(mutate_header(field,value)|{'modules':state['modules']}))
        s=copy.deepcopy(state);s['modules']['wm']['norm.count']=torch.tensor(0,dtype=torch.int64)
        reject('reject_WM_wrong_normalizer_count_dtype',lambda:_wm_payload(s))
        s=copy.deepcopy(state);s['modules']['wm']['delta_norm.var'][0]=-1
        reject('reject_WM_negative_variance',lambda:_wm_payload(s))
        s=copy.deepcopy(state);s['modules']['wm']['net.9.weight']=torch.zeros((68,512))
        reject('reject_WM_wrong_contact_head',lambda:_wm_payload(s))
        context=loadnp('D:/workspace/supermimic-chord-study/outputs/a1l_paired_v2_clip007_2000iter_seed1/compiled/fixed_physics_context.npz')
        small={k:np.ascontiguousarray(v[:1].copy()) for k,v in context.items()}
        small['reader_context']=small['reader_context'].astype(np.float32)
        require('actual8_CPU_arrays_bitexact_positive',np.array_equal(_physical_xi(small,small,1),small['reader_context'][0]))
        damaged={k:v.copy() for k,v in small.items()};damaged['geom_pos'].flat[0]+=np.float32(.001)
        reject('reject_mutated_actual8',lambda:_physical_xi(damaged,small,1))
        damaged={k:v.copy() for k,v in small.items()};damaged['reader_context'].flat[0]=np.nextafter(damaged['reader_context'].flat[0],np.float32(np.inf))
        reject('reject_one_ulp_physicalxi_drift',lambda:_physical_xi(damaged,small,1))
        fake_auditor=output/'SYNTHETIC_audit_full151_checkpoint_cpu.py';fake_auditor.write_text('# CPU contract fixture only; not a real105 health producer\n',encoding='utf-8')
        fake_checkpoint={'path':str(output/'NONEXISTENT_model_final.pt'),'sha256':'d'*64}
        health={'schema':HEALTH_SCHEMA,'status':'PASS_SAVED_ENGINEERING_HEALTH','checks':{'SYNTHETIC_ONLY':True},'passed':1,'total':1,'seed':0,'iteration':2000,'final_iteration_scope':True,'run_dir':str(output),'method':'A1L','checkpoint':fake_checkpoint,'auditor':pin(fake_auditor)}
        hkw=dict(checkpoint_pin=fake_checkpoint,expected_method='A1L',training_run=output,expected_iteration=2000,expected_health_auditor=pin(fake_auditor))
        require('future105_health_schema_fixture_only',validate_health_metadata(health,**hkw) is health)
        bad=copy.deepcopy(health);bad['schema']='rough104_checkpoint_CPU_health:1'
        reject('reject_old104_health_schema',lambda:validate_health_metadata(bad,**hkw))
        bad=copy.deepcopy(health);bad['checks']['SYNTHETIC_ONLY']=False
        reject('reject_failed_health',lambda:validate_health_metadata(bad,**hkw))
        reject('reject_missing_explicit_new105_auditor',lambda:validate_health_metadata(health,**(hkw|{'expected_health_auditor':None})))
        old_auditor=pin('D:/workspace/supermimic-rough-prefix-learning-study/scripts/audit_rough_checkpoint_cpu_v3.py')
        reject('reject_old104_auditor_even_with_new_schema',lambda:validate_health_metadata(health,**(hkw|{'expected_health_auditor':old_auditor})))
        old_path=Path('D:/workspace/supermimic-rough-prefix-learning-study/outputs/primary_a1l_seed0_v1/train/model_final.pt')
        old=torch.load(old_path,map_location='cpu',weights_only=False);details['old_real_checkpoint_rejection_only']=pin(old_path)
        reject('actual_old104_checkpoint_header_rejected',lambda:validate_checkpoint_namespace(old))
        old['format_version']=105
        reject('actual_old104_integer_relabel_rejected',lambda:validate_checkpoint_namespace(old))
        from full151_native.bundle import load_bundle
        bundle_path=ROOT/'outputs/full151_native_input_v1/bundle.json'
        b=load_bundle(bundle_path)
        require('actual_new_bundle_passive_reload_without_Reader_or_FK',_validated_bundle(b)['descriptor']==b['descriptor'])
        changed=copy.deepcopy(b);changed['baseline_u150'][10,1]=np.nextafter(changed['baseline_u150'][10,1],np.float32(np.inf))
        reject('reject_caller_mutated_actual_baseline',lambda:_validated_bundle(changed))
        changed=copy.deepcopy(b);changed['arrays']['init']['q_init_requested_FP32_rad'][0]+=np.float32(.01)
        reject('reject_caller_mutated_actual_init',lambda:_validated_bundle(changed))
        changed=copy.deepcopy(b);changed['descriptor']['source_identity']['local_reference_range_half_open']=[0,16]
        reject('reject_caller_shortened_reference_identity',lambda:_validated_bundle(changed))
        details['actual_bundle']=pin(bundle_path);details['actual_descriptor']=b['descriptor']['input_signature_sha256']
        require('no_new_FK_collision_or_physics_called',not any(forbidden.values()))
        details['forbidden_calls']=forbidden
        require('all_sources_unchanged',all(pin(p['path'])==p for p in sources.values()))
        require('CPU_only_no_CUDA_initialization',not torch.cuda.is_initialized())
        status='PASS_SYNTHETIC_CONTRACT_AND_INSTALLED_CPU_MODULES_ONLY'
    except Exception as e:
        status='CHECKER_FAILURE_RETAINED';details['exception']=repr(e);details['traceback']=traceback.format_exc()
    report={'schema':'full151105_checkpoint_consumers_CPU_contract:1','status':status,'sources':sources,'checks':checks,'passed':sum(checks.values()),'total':len(checks),'details':details,'wall_s':time.perf_counter()-before,
            'mode':MODE,'retarget_variant_id':VARIANT,'actual_new105_checkpoint_load':'NOT_RUN_NO_ACTUAL105_CHECKPOINT_EXISTS_A0_ONLY',
            'actual105_training_health':'PENDING_NEW_PRODUCER_NOT_IMPLEMENTED','installed_network_tests':'Fresh synthetic modules; no training weights migration; autograd finite does not establish physical gradients.',
            'positive_metadata_health_fixtures':'Explicitly synthetic structural contracts, not evidence of a real run or producer.',
            'knownrough_scope':'SourceSCP and geometry failures remain retained. No human contact labels; native contact15 is self-collision.',
            'Env_constructed':False,'physics_steps':0,'geometry_queries':0,'WM_training_or_saved180_forward':False,'GPU_used':False}
    write(output/'report.json',report);print({'status':status,'passed':report['passed'],'total':report['total'],'report':pin(output/'report.json')},flush=True)
    return 0 if status.startswith('PASS_') else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();raise SystemExit(main(a.output))
