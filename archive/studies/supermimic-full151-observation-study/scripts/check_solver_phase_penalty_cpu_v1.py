"""Pure/synthetic native state ABI and hook checks; no actual dense data/model."""
from __future__ import annotations
import argparse,ast,copy,json,os,pickle,hashlib,random,subprocess,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151_gradient import penalty_observability105_v1 as a


def synthetic_groups():
    samples=[];rewards=[];clock=np.empty(601,np.float32);clock[0]=0
    for i in range(600):clock[i+1]=np.float32(clock[i]+np.float32(.005))
    for pi,path in enumerate(('A0','A2')):
        slots=[(k,None,'donor') for k in range(150)]
        for ki,k in enumerate((0,49,99,149)):
            slots.extend((k,pi*76+ki*19+i,'baseline' if i<3 else 'direction') for i in range(19))
        for k,bid,mode in slots:
            rewards.append({'path':path,'k':k,'branch_id':bid})
            for sub in range(1,5):
                samples.append({'path':path,'phase':k,'branch_id':bid,'mode':mode,'substep':sub,'source_frame':200+k,
                   'canonical_pre_tick':4*k+sub-1,'canonical_post_tick':4*k+sub,'pre_time':float(clock[4*k+sub-1]),'post_time':float(clock[4*k+sub]),
                   'pre_qpos':[0.]*27,'pre_qvel':[0.]*26,'post_qpos':[0.]*27,'post_qvel':[0.]*26,'external_force_max':0.,
                   'phase_contract':'solver quantities preintegration; q/qvel/time postintegration; before scene.update and Manager forward'})
    return samples,rewards


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args();out=Path(args.output).resolve()
    a.require(not out.exists(),'Fresh CPU proof directory required');out.mkdir(parents=True);(out/'executed_sources').mkdir()
    pins={k:a.pin(v) for k,v in a.OWN.items()}
    for k,v in pins.items():(out/'executed_sources'/(k+'_'+Path(v['path']).name)).write_bytes(Path(v['path']).read_bytes())
    a.write(out,'source_before.json',pins);checks={}
    def check(n,v):
        checks[n]=bool(v)
        if not checks[n]:raise AssertionError(n)
    def rejects(n,fn):
        try:fn()
        except (ValueError,KeyError):check(n,True)
        else:check(n,False)
    try:
        for v in pins.values():ast.parse(Path(v['path']).read_text(encoding='utf-8-sig'))
        check('sources_AST_valid_and_frozen_before_tests',True)
        check('fresh_noTorch_Warp_before_fixture','torch' not in sys.modules and 'warp' not in sys.modules)
        p=a.read(a.PREREG);check('exact_protocol1ec18',a.pin(a.PREREG)['sha256']==a.PREREG_SHA)
        declaration=a.declaration();check('default_declaration_no_data_or_models',declaration['state_conversions']==declaration['NPZ_opened']==declaration['objective_constructs']==0)
        h={'schema':a.SCHEMA,'role':a.ROLE,'protocol':{},'output':str(ROOT/'outputs/full151105_solver_phase_penalty_CPU_fixture_NEVER_EXECUTED'),
           'CPU_run_authorized':False,'cpu_threads':1,'source_pins':{'fixture':{}},'cpu_ready':{}}
        a.validate_header(h);check('pure_header_positive',True)
        for key,val in [('schema','old104'),('role','physics'),('CPU_run_authorized',1),('cpu_threads',True),('output',str(ROOT/'src'))]:
            rejects('header_reject_'+key,lambda key=key,val=val:a.validate_header({**h,key:val}))
        rejects('unknown_header_field',lambda:a.validate_header({**h,'GPU_run_authorized':True}))
        rejects('header_missing_cpu_ready',lambda:a.validate_header({k:v for k,v in h.items() if k!='cpu_ready'}))
        exact=np.array([.25,-.5,1],np.float32)
        check('exact_FP32_JSON_roundtrip',np.array_equal(a.exact_fp32(exact.tolist(),(3,),'synthetic'),exact))
        rejects('unrepresentable_F64_not_silently_repaired',lambda:a.exact_fp32([.1],(1,),'bad'))
        rejects('FP32_NaN_rejected',lambda:a.exact_fp32([np.nan],(1,),'bad'))
        rejects('arithmetic_scaleaware_bound_not_relaxed',lambda:a.arithmetic_check([1e5+1],[1e5],'too large'))
        check('arithmetic_8eps_scaleaware_positive',a.arithmetic_check([1e5+.01],[1e5],'allowed')>.009)
        mapping={'hand_qpos':np.arange(7,27,dtype=np.int64),'hand_dof':np.arange(6,26,dtype=np.int64),'object_qpos_start':0,'object_dof_start':0}
        a.validate_mapping(mapping);check('complete27_26_mapping_positive',True)
        rejects('mapping_overlap',lambda:a.validate_mapping({**mapping,'hand_qpos':np.arange(20)}))
        rejects('mapping_duplicate',lambda:a.validate_mapping({**mapping,'hand_qpos':np.full(20,7)}))
        rejects('mapping_boolean_start',lambda:a.validate_mapping({**mapping,'object_qpos_start':False}))
        rejects('mapping_float_indices',lambda:a.validate_mapping({**mapping,'hand_dof':np.arange(6,26,dtype=float)}))
        hand={10,11};objects={2};covered={10}
        contacts=[{'geom':[10,2],'dist':-.004},{'geom':[2,11],'dist':-.004},{'geom':[10,11],'dist':-.1},{'geom':[2,3],'dist':-.2}]
        cv=a.pair_coverage(contacts,hand,objects,covered,{10:0.})
        check('HO_not_self_or_environment_depth',cv['native_generated_HO_depth_m']==.004 and cv['native_generated_HO_count']==2)
        check('all_exact_tied_deepest_pairs_retained',len(cv['all_exact_tied_deepest_pairs'])==2)
        check('primary_coverage_categories_can_overlap',cv['primary_deepest_tie_coverage']['geom_set_gap_ids']==[11] and cv['primary_deepest_tie_coverage']['covered_proxy_undersensitivity_ids']==[10])
        varied=contacts+[{'geom':[10,2],'dist':-.0035}]
        vv=a.pair_coverage(varied,hand,objects,covered,{10:0.})
        check('all_violating_pairs_secondary_not_primary_argmax',len(vv['all_violating_pairs_SECONDARY'])==3 and len(vv['all_exact_tied_deepest_pairs'])==2)
        check('exact3mm_not_above_gate',not a.pair_coverage([{'geom':[10,2],'dist':-.003}],hand,objects,covered)['native_HO_depth_gt3mm'])
        empty={'nacon':0,'nefc':0,'raw_wp_contact':{'geom':[],'worldid':[],'type':[],'efc_address':[]},'contacts':[]}
        eg,et=a.native_contact_abi(empty,20);check('legal_JSON_empty_contacts_not_unknown',eg.shape==(0,2) and et.shape==(0,))
        bad=copy.deepcopy(empty);bad['raw_wp_contact']['geom']=None
        rejects('missing_contact_not_filled_empty',lambda:a.native_contact_abi(bad,20))
        def contact_row(n,nefc):
            return {'nacon':n,'nefc':nefc,'raw_wp_contact':{'geom':[[10,2]]*n,'worldid':[0]*n,'type':[0]*n,'efc_address':[[0]*6 for _ in range(n)]},'contacts':[{'geom':[10,2],'dist':-.004,'force6':[0.]*6} for _ in range(n)]}
        full=contact_row(192,4096);fg,ft=a.native_contact_abi(full,20)
        check('exact_capacity_without_overflow_is_legal_array',fg.shape==(192,2) and ft.shape==(192,))
        check('zero_force_and_zero_type_not_filtered_from_HO',a.pair_coverage(full['contacts'],hand,objects,covered)['native_HO_depth_gt3mm'])
        overflow=contact_row(193,4096);rejects('over_capacity_rejected',lambda:a.native_contact_abi(overflow,20))
        unknown=contact_row(1,0);unknown['raw_wp_contact']['geom'][0]=[99,2]
        rejects('unknown_native_geom_rejected',lambda:a.native_contact_abi(unknown,20))
        order=contact_row(1,0);order['contacts'][0]['geom']=[2,10]
        rejects('CPU_WP_pair_order_rejected',lambda:a.native_contact_abi(order,20))
        s,r=synthetic_groups();groups,rmap=a.group_rows(s,r)
        check('synthetic_all1808_452_exactschedule',len(groups)==452 and len(rmap)==452)
        bads=copy.deepcopy(s);bads[1]['pre_qpos'][2]=1.
        rejects('within_group_prepost_discontinuity_rejected',lambda:a.group_rows(bads,r))
        bads=copy.deepcopy(s);bads[0]['source_frame']=201
        rejects('sourceframe_identity_rejected',lambda:a.group_rows(bads,r))
        bads=copy.deepcopy(s);bads[0]['canonical_pre_tick']=False
        rejects('boolean_canonical_tick_rejected',lambda:a.group_rows(bads,r))
        bads=copy.deepcopy(s);bads[0]['post_time']+=1e-8
        rejects('clock_not_source_FP32_recursion_rejected',lambda:a.group_rows(bads,r))
        # True subprocess default: never opens any NPZ or importsTorch.
        run=subprocess.run([sys.executable,'-B',str(a.OWN['CLI'])],capture_output=True,text=True,timeout=20,env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1'})
        (out/'default_stdout.json').write_text(run.stdout,encoding='utf-8');(out/'default_stderr.txt').write_text(run.stderr,encoding='utf-8')
        a.require(run.returncode==0,run.stderr);d=json.loads(run.stdout)
        check('actual_defaultCLI_noTorch_noNPZ_no_conversion',d['Torch_imported'] is False and d['Warp_imported'] is False and d['NPZ_opened']==d['state_conversions']==0)
        # Native unbound state on synthetictensors, never a model/Reader/Env.
        import torch,mujoco
        torch.set_num_threads(1);denied={'calls':0};originals={}
        def deny(*args,**kwargs):denied['calls']+=1;raise RuntimeError('Native physics/model/query forbidden in CPUfixture')
        names=('mj_step','mj_forward','mj_kinematics','mj_collision','mj_fwdPosition','mj_inverse','mj_jac','mj_geomDistance')
        for n in names:
            if hasattr(mujoco,n):originals[n]=getattr(mujoco,n);setattr(mujoco,n,deny)
        originals['MjModel']=mujoco.MjModel
        class ForbiddenModel:
            from_binary_path=staticmethod(deny)
            from_xml_string=staticmethod(deny)
            from_xml_path=staticmethod(deny)
        mujoco.MjModel=ForbiddenModel
        try:
            def rng():return hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state()),protocol=5)+torch.get_rng_state().numpy().tobytes()).hexdigest()
            before=rng();q=np.zeros((2,27),np.float32);v=np.zeros((2,26),np.float32)
            q[:,:3]=[1,2,3];q[:,3]=q[:,6]=np.float32(np.sqrt(.5));q[:,7:]=np.arange(20,dtype=np.float32)/100
            v[:,:6]=[.4,-.2,.1,1,2,3];v[:,6:]=np.arange(20,dtype=np.float32)/10
            pp=np.broadcast_to(np.array([.1,.2,.3],np.float32),(2,3)).copy()
            R=np.broadcast_to(np.array([[1,0,0],[0,0,-1],[0,1,0]],np.float32),(2,3,3)).copy()
            x=a.native_state58(q,v,pp,R,mapping,p['source_pins']['native_reader']).numpy()
            check('actual_unbound_reader_state58_shape_and_q_bits',x.shape==(2,58) and x[:,:20].tobytes()==q[:,7:].tobytes() and x[:,20:40].tobytes()==v[:,6:].tobytes())
            expected=np.array([.9,2.7,-1.8],np.float32)
            check('nonidentity_palm_position_closedform',np.allclose(x[:,40:43],expected,rtol=0,atol=4e-7))
            check('bodyomega_to_world_to_palm_closedform',np.allclose(x[:,55:58],[-2,3,-1],rtol=0,atol=1e-6))
            check('world_linear_velocity_to_palm_closedform',np.allclose(x[:,52:55],[.4,.1,.2],rtol=0,atol=1e-7))
            expectedR=np.array([[0,-1,0],[0,0,1],[-1,0,0]],np.float32)
            check('rowmajor_relative_rotation_closedform',np.allclose(x[:,43:52].reshape(2,3,3),expectedR,rtol=0,atol=3e-7))
            a.validate_x58(x,x.copy(),'self fixture');check('x58_positive_validator',True)
            xx=x.copy();xx[0,0]+=np.float32(1e-5)
            rejects('q_bits_mismatch_rejected',lambda:a.validate_x58(xx,x,'bad'))
            xx=x.copy();xx[0,55]+=.01
            rejects('angular_arithmetic_mismatch_rejected',lambda:a.validate_x58(xx,x,'bad'))
            qq=q.copy();qq[:,3]*=1.02
            rejects('nonunit_freequat_not_repaired',lambda:a.native_state58(qq,v,pp,R,mapping,p['source_pins']['native_reader']))
            rejects('wrong_state58_sourcepin_rejected',lambda:a.native_state58(q,v,pp,R,mapping,p['source_pins']['native_state_pack']))
            rr=R.copy();rr[:,0,:]*=2
            rejects('improper_recorded_frame_rejected',lambda:a.native_state58(q,v,pp,rr,mapping,p['source_pins']['native_reader']))
            # Existing hook + actual unbound state, no actualproxy construction.
            class ClosedProxy(torch.nn.Module):
                def __init__(self):super().__init__();self.calls=0
                def forward(self,x):
                    self.calls+=1;b=len(x);dist=(-.003-.001*x[:,0:1]).expand(b,3)
                    return {'support':x[:,1:2].expand(b,64),'presence':torch.ones((b,6)),
                        'minimum_signed_distance_m':dist.amin(-1),'sampled_signed_distance_m':dist,'soft_contact_weight':torch.ones((b,3)),
                        'projected_contact_com_o_m':torch.zeros((b,3,3)),'inward_normal_o':torch.ones((b,3,3))}
            class ClosedPenalty(torch.nn.Module):
                def __init__(self):super().__init__();self.geometry_proxy=ClosedProxy()
                def forward(self,x,m):
                    d=self.geometry_proxy(x);P=((-d['sampled_signed_distance_m']-.003).clamp_min(0).amax(-1)/.001).square();S=d['support'].mean(-1)
                    return -.1*P,{'geometry_intent_penetration_prior':P,'geometry_intent_support_match':S,'geometry_intent_phase_weight':torch.ones_like(P)}
            proxy=ClosedPenalty();state=torch.ones((2,58),dtype=torch.float32)
            with torch.no_grad():score,terms,capture=a.common.evaluate_objective(proxy,state,np.array([0,149]))
            check('existing_proxy_hook_no_extra_forward',proxy.geometry_proxy.calls==1 and capture['sampled_signed_distance_m'].shape==(2,3) and len(proxy.geometry_proxy._forward_hooks)==0)
            check('closed_P_scalar_and_penalty_exact_formula',np.allclose(terms['geometry_intent_penetration_prior'].numpy(),1,atol=4e-7) and np.allclose(score.numpy(),-.1,atol=1e-7))
            allowance=np.float32(.003)
            check('FP32_allowance_boundary_is_reported_not_gate_relaxation',float(allowance)!=.003 and bool(np.float32(-.003)==-allowance))
            check('no_actual_native_API_or_model_calls',denied['calls']==0)
            check('fixtures_CPU_CUDA_never_initialized',not torch.cuda.is_initialized())
            check('fixtures_RNG_unchanged',rng()==before)
        finally:
            for n,value in originals.items():setattr(mujoco,n,value)
        check('no_actual_binding_or_WM_import','full151_intent.binding_v3' not in sys.modules and 'full151_native.world_model_checkpoint' not in sys.modules and 'warp' not in sys.modules)
        text=Path(a.__file__).read_text(encoding='utf-8');tree=ast.parse(text)
        check('source_gates_before_single_factory',text.index("write(out,'converter_qualification.json'")<text.index("call('objective_factory'"))
        check('source_primary_P_ONLY_factory_not_A2_factory','from full151_intent.binding_v3 import build_penalty_only_objective' in text and 'build_contact_objective' not in text)
        check('source_stream_schedule15_no_u150','range(0,1808,128)' in text and "counts['objective_forward']['returned']==15" in text)
        check('source_native58_exact_unbound_call','result=SimReader.state(fake)' in text)
        for k,v in pins.items():check('executed_source_preserved_'+k,a.pin(v['path'])==v)
        proof={'schema':'full151105_dense_penalty_CPU_contract_proof:1','status':'PASS_SYNTHETIC_NATIVE_STATE_AND_HOOK_ONLY','checks':checks,'passed':sum(checks.values()),'total':len(checks),
            'source_pins':pins,'protocol':a.pin(a.PREREG),'scope':{'actual_physical_state_conversions':0,'actual_MJB_loads':0,'actual_objective_constructs':0,'actual_full05_data_reads':0,
            'synthetic_unbound_native_state58_rows':2,'native_imports_Torch_MuJoCo':True,'native_API_deny_calls':denied['calls'],'actual_WM_Actor_forward_or_weight_load':0,
            'actual_Env_physics_GPU':0,'actual_converter_or_dense_proxy_qualified':False}}
        a.write(out,'report.json',proof)
        ready={'schema':'full151105_dense_penalty_CPU_contract_READY:1','status':'PASS_SYNTHETIC_NATIVE_STATE_AND_HOOK_ONLY','source_pins':pins,'cpu_report':a.pin(out/'report.json'),
            'cpu_checks':{'passed':proof['passed'],'total':proof['total']},'protocol':a.pin(a.PREREG),'CPU_scientific_run_authorized':False,'GPU_run_authorized':False,
            'api':{'entry':str(a.OWN['CLI']),'module':str(a.OWN['consumer']),'default':'no args declaration; --config JSON hashonly','actual':'Root separate authorization; CUDA_VISIBLE_DEVICES=-1 --config JSON --run','schema':a.SCHEMA,'role':a.ROLE,'keys':sorted(a.KEYS)}}
        a.write(out,'READY.json',ready)
        print(json.dumps({'passed':proof['passed'],'total':proof['total'],'report':a.pin(out/'report.json'),'READY':a.pin(out/'READY.json'),'source_pins':pins},indent=2))
    except BaseException as error:
        a.write(out,'failure.json',{'status':'FAILED_CPU_FIXTURE_PRESERVED','error':repr(error),'checks_completed':checks,'source_pins':pins,'actual_physical_work':False});raise


if __name__=='__main__':main()
