"""Source/JSON-only preregistration. No state conversion or proxy evaluation."""
from __future__ import annotations
import ast
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path('D:/workspace/supermimic-full151-observation-study')
DEX=Path('D:/workspace/dex-supermimic-research')
ROUGH=Path('D:/workspace/supermimic-rough-prefix-learning-study')
RUN=ROOT/'outputs/full151105_gradient_empirical_gradient05_v2'
OUT=ROOT/'outputs/full151105_solver_phase_penalty_source_feasibility_v2'
PROTOCOL=ROOT/'protocols/full151105_solver_phase_penalty_observability_CPU_prereg_v2.json'


def require(v,m):
    if not v:raise ValueError(m)


def pin(p):
    p=Path(p).resolve(strict=True);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return {'path':str(p),'sha256':h.hexdigest()}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)


def main():
    require(not OUT.exists() and not PROTOCOL.exists(),'Fresh no-overwrite preregistration')
    paths={'writer':Path(__file__),'worker':ROOT/'src/full151_gradient/serial_probe105_v2.py',
           'native_reader':DEX/'src/dex_supermimic/rl/dex_supermimic/sim.py',
           'Pen_reader_subclass':DEX/'src/dex_supermimic/rl/dex_supermimic/pen.py',
           'native_state_pack':DEX/'src/dex_supermimic/rl/dex_supermimic/state.py',
           'native_rotation':DEX/'src/dex_supermimic/rl/dex_supermimic/rotations.py',
           'native_task_cfg':DEX/'src/dex_supermimic/tasks/pen_spin/pen_spin_env_cfg.py',
           'Warp_forward':DEX/'.venv/Lib/site-packages/mujoco_warp/_src/forward.py',
           'Warp_kinematics_source_NOT_executed':DEX/'.venv/Lib/site-packages/mujoco_warp/_src/smooth.py',
           'Warp_readback':DEX/'.venv/Lib/site-packages/mujoco_warp/_src/io.py',
           'measured_contacts':ROUGH/'vendor/measured_contacts.py',
           'binding3':ROOT/'src/full151_intent/binding_v3.py',
           'R_objective':ROUGH/'vendor/frozen_contact_math/contact_objective.py',
           'R_penalty':ROUGH/'vendor/frozen_contact_math/penalty_only_objective.py',
           'R_geometry':ROUGH/'vendor/frozen_contact_math/contact_geometry.py',
           'bundle_loader':ROOT/'src/full151_native/bundle.py',
           'native_actual_context':ROOT/'src/full151_native/native.py',
           'old104_codec_SOURCE_ONLY_not_105_enrollment':ROUGH/'src/rough_prefix/saved_state_codec_v3.py'}
    pins={k:pin(p) for k,p in paths.items()};texts={k:p.read_text(encoding='utf-8-sig') for k,p in paths.items()}
    for t in texts.values():ast.parse(t)
    checks={}
    def fact(n,v):checks[n]=bool(v);require(v,n)
    fact('worker_actual_a24c',pins['worker']['sha256']=='a24c89da47f4aad7acfc9d9dfd1f053518cdc878230716bb2c857127ec13e592')
    fact('existing_actual_binding3_324a',pins['binding3']['sha256']=='324a64bb3603f0e00b950c204468f932053a9daa11c96559f35c141975b0598e')
    fact('sample_saved_pre_and_post_raw',all(k in texts['worker'] for k in ("pre_qpos=prev['qpos']","pre_qvel=prev['qvel']","post_qpos=now['qpos']","post_qvel=now['qvel']")))
    fact('substep_samples_immediately_after_original_step',"original_step();report['episode_step_returned']+=1\n                                    now=raw()" in texts['worker'])
    start=texts['Warp_forward'].index('def step(m: Model, d: Data):')
    tail=texts['Warp_forward'][start:texts['Warp_forward'].index('def step1(',start)]
    fact('Warp_forward_and_solver_before_integration',tail.index('forward(m, d)')<tail.index('implicit(m, d)') and 'solver.solve(m, d)' in texts['Warp_forward'])
    fact('task_implicitfast_not_RK4', 'integrator="implicitfast"' in texts['native_task_cfg'])
    fact('readback_copies_raw_and_derived_without_recompute',all(k in texts['Warp_readback'] for k in ('result.qpos[:] = d.qpos.numpy()[world_id]','result.xpos[:] = d.xpos.numpy()[world_id]','result.xquat[:] = d.xquat.numpy()[world_id]','result.contact.dist[:ncon] = d.contact.dist.numpy()[ncon_filter]')))
    fact('freejoint_derived_pose_normalizes_pre_qpos',all(k in texts['Warp_kinematics_source_NOT_executed'] for k in ('if jnt_type_ == JointType.FREE:','xquat = wp.normalize(xquat)','xpos_out[worldid, bodyid] = xpos')))
    fact('reader_raw_state_is_palm_frame_and_bodyomega_to_world',all(k in texts['native_reader'] for k in ('obj_q = d.qpos[:, fq]','obj_v = d.qvel[:, fv]','mv(inv, mv(r, obj_v[:, 3:6]))')))
    fact('all_endpoint_captures_record_actual_reader_frame',all(k in texts['worker'] for k in ('state20_reader_palm_pos=reader.palm_pos.cpu().numpy()','state20_reader_palm_rot=reader.palm_rot.cpu().numpy()')))
    pen=next(n for n in ast.parse(texts['Pen_reader_subclass']).body if isinstance(n,ast.ClassDef) and n.name=='PenSimReader')
    fact('PenSimReader_inherits_state_no_override',any(isinstance(b,ast.Name) and b.id=='SimReader' for b in pen.bases) and not any(isinstance(n,ast.FunctionDef) and n.name=='state' for n in pen.body))
    fact('native_quat_normalization_is_original_reader_semantics','q = F.normalize(q, dim=-1)' in texts['native_rotation'])
    fact('all_category_penetration_is_not_HO_only', 'penetration_by_category_m' in texts['measured_contacts'] and 'penetration = max(penetration, -float(c.dist), 0.0)' in texts['measured_contacts'])
    fact('P_ONLY_actual_P_eager_source', 'bonus = -.1 * penalty' in texts['R_penalty'] and '_, base_terms = self.base(x1_state58, memory)' in texts['R_penalty'])
    fact('sample_geom_IDs_and_positions_are_registered_buffers',all(k in texts['R_geometry'] for k in ('"sample_geom_ids"','"sample_points_body"','"hand_collision_geoms"','"object_geoms"')))
    with (RUN/'physical_samples.jsonl').open(encoding='utf-8') as f:first=json.loads(next(f))
    fact('first_row_format_only_27_26_7',len(first['pre_qpos'])==len(first['post_qpos'])==27 and len(first['pre_qvel'])==len(first['post_qvel'])==26 and len(first['object_solver_pose'])==7)
    fact('first_row_declares_preintegration_solver_contract',first['phase_contract']=='solver quantities preintegration; q/qvel/time postintegration; before scene.update and Manager forward')
    bundle_path=ROOT/'outputs/full151_native_input_v1/bundle.json';bundle=read(bundle_path)
    inputs={'physical_config':pin(ROOT/'configs/full151105-gradient-empirical_gradient05-v2.json'),
            'physical_samples':pin(RUN/'physical_samples.jsonl'),'reward_snapshots':pin(RUN/'reward_snapshots.jsonl'),
            'physical_arrays':pin(RUN/'physical_arrays.npz'),'branch_index':pin(RUN/'branch_index.json'),
            'snapshot_records':pin(RUN/'snapshot_records.json'),'donor_records':pin(RUN/'donor_records.json'),
            'physical_saved_independent1762':pin(ROOT/'outputs/full151105_full_serial_saved_review_v1/report.json'),
            'objective_spec':pin(ROOT/'configs/full151-v2-declared-H123-GPU-objective-spec-v1.json'),
            'previous_bonus_independent523':pin(ROOT/'outputs/full151105_trained_bonus_saved_independent_v1/report.json'),
            'previous_bonus_report':pin(ROOT/'outputs/full151105_trained_bonus_CPU_v1/report.json'),
            'previous_bonus_factory_geometry_manifest':pin(ROOT/'outputs/full151105_trained_bonus_CPU_v1/loader_factory_receipts.json'),
            'full151_bundle':pin(bundle_path),'native_data':bundle['source_pins']['native_data'],'reference_arrays':bundle['source_pins']['reference_arrays'],
            'init_arrays':bundle['source_pins']['init_arrays'],'nominal_model':bundle['model'],'actual_context':bundle['context']}
    for name in ('A0','A2'):
        for f in ('initial_readback.json','runtime_reset_readback.json','control_metadata.json','controller_descriptor.json','actual_context_readbacks.npz','expected_context_fields.npz','actual8field_copy.json','nominal_model.mjb'):
            inputs[name+'__'+f]=pin(RUN/('donor_'+name)/f)
    protocol={'schema':'full151105_penalty_solver_phase_observability_CPU_preregistration:1','revision':2,
      'revision_history':{'previous':pin(ROOT/'protocols/full151105_solver_phase_penalty_observability_CPU_prereg_v1.json'),'change':'Make all452 saved actual-reader palm frames the primary converter authority, rather than inferring every frame from startup mocap. No denseconversion/scoring or numerical threshold change.'},
      'status':'FROZEN_SOURCE_FEASIBILITY_ONLY_NOT_CONVERTED_NOT_SCORED','created_UTC':datetime.now(timezone.utc).isoformat(),
      'CPU_run_authorized':False,'GPU_run_authorized':False,'training_authorized':False,'source_pins':pins,'inputs':inputs,
      'question':'At identical saved solver pre-state, does the exact trained sampled penalty observe native generated hand-object depth above the original3mm gate? Distinguish finite snapshot coverage from proxy observability, without rerunning physics.',
      'prior_knowledge':'Previous160 physical+8 WM next20 proxyP were all0; training prior logs can be positive but are different imaginedstates. This is a new sequential exploratory diagnostic, not a blind confirmatory claim or old FD rerun.',
      'all_rows_schedule':{'physical_samples':1808,'control_intervals':452,'paths':['A0','A2'],'each_path_rows':904,
        'donor_rows':1200,'baseline_branch_rows':96,'perturbation_branch_rows':512,'four_substeps':[1,2,3,4],
        'ordering':'original physical_samples JSONL order; preserve path/mode/k/branch_id/substep/canonical tick/sourceframe, no sample selection',
        'dependent_replicates':'2fresh saved-action-source paths;152 restored branches include24 baselines. Repeated states/steps/branches not independentN. No archived closed-loop reproduction claim.'},
      'phase_contract':{'raw_pre':'pre_qpos27/pre_qvel26 from previous_raw; interval starts with raw() before env.step; subsequent substep previous_raw is priorpost',
        'native_solver':'Warp.step calls forward/position/collision/solver on prestate, then implicit integration. Contactdist and xpose remain pre-derived while qpos/qvel/time have advanced.',
        'readback':'get_data_into copies qpos/qvel and xpos/xquat/contact separately; it does not recompute FK. Sampling is before scene.update/Managerforward.',
        'forbidden_mismatch':'Never compare15ms lastsolver depth with20ms postintegration proxy. Every1808 proxy input is the corresponding PRE qpos/qvel, not post or nearest endpoint.',
        'reward_memory':'P_ONLY eagerly calls fullA2base; start0/end151/time=integercontrol k for all4substeps. Penalty itself unmasked by declaredintent. k149 active, no u150, no source confidence or contact label invented.'},
      'converter_prerequisites_ZERO_PROXY_UNTIL_ALL_PASS':{
        'new105_adapter':'Use original unbound SimReader.state on explicitly shaped CPU qpos/qvel/index/palm views, no SimReader/Env construction and no native FK. Old104 codec only source precedent, not enrolled whole16/61 loader.',
        'passive_MJB_only':'New authorized process may read frozen nominal MJB constants/index arrays; assert nq27,nv26,20ordered named handjoint QA/VA,object freejoint7/6,COM0,no plugin,implicitfast fixedsource. IDs from MJB and source bindings, not guessed slices.',
        'fixed_palm':'Primary converter authority is reward_snapshots state20_reader_palm_pos/rot recorded from the SAME actual SimReader for all452 groups. Assert constant perpath and properFP32matrix, bind native wrist151 fixed table and both path startup/reset/mocap records to it. For all4prestates use that groups recorded matrix/position directly, not a newly rounded quaternion reconstruction. Passive MJB must assert right_palm_link identity child/no joint of world mocap_base id0. Physical_samples does not store per-substep mocap, so do not claim1808 direct rootreadbacks; fixed-root structure/source and recorded reader-frame invariance are separate evidence.',
        'mapping':'Native pack[q20,qd20,RpT(pobj-ppalm),RpT Robj,rowmajor9,RpT v_world,RpT Robj omega_body]. Original native quat_to_mat normalizes wxyz; require input norm within1e-5 first. Native free angular is BODY, not world.',
        'reference_validation':'Compare converted lastpostq/postv of ALL452 fourstep groups to corresponding reward_snapshots.physics58. Handq/qd exactFP32 bits; p/R/v/angular arithmetic bound8*eps32*max(1,abs(reference),abs(converted)) componentwise, frozen NEW_PROPOSED arithmetic rule, not physical tolerance.',
        'pre_anchor_validation':'For all152 branch firstpre and8selecteddonor firstpre compare converted native58 to saved snapshot_x[same path/k] with sameq/qd exact and arithmetic bound; do not borrow endpointk+1.',
        'solver_pose_validation':'All1808 object_solver_pose positions versus PRE objectfreeqpos (world freebody), rotations normalize wxyz sign-invariant comparison viaR; same8eps32componentwise bound. This is raw pose arithmetic, not FK/geometry; if any fail stop before proxy.',
        'timeline':'452 groups, eachsubstep1..4, tickpre=4k+i-1/post=4k+i;within-group preq/qv equals precedingpost bitwise, expectedFP32clock fixed. Source-action restore boundaries not concatenated into one time trajectory.',
        'guards':'Finite allrows,actual4substep ctrl fixedknown,original physical/capacity/identity/Saved1762 pins allcurrent. Missing mapping,pose,type,NaN,overflowUNKNOWN or source mismatch ->FAIL/PENDING; no dropped/far-state filtering or repair.',
        'converter_batch_budget':'One native pure tensor state conversion over2260 rows (1808pre +452known post endpoints), pluspose arithmetic checks; no model/WM/Actor/objective calls yet. This is not dynamics or hiddenstate restoration.'},
      'native_depth_label':{'primary':'max(max(0,-contact.dist)) across WORLD0 generated contacts whose exact geom pair is object×enabledhand, strict >.003m. This is HO-only slice of original3mm physical gate, not whole-task score.',
        'exact_object_mask':'geoms with bodyid==object/cube as in pinnedworker; enumerate ids/types/names. Noother object/environment contact substituted.',
        'exact_hand_mask':'pinnedworker regex robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col), enabled contype orconaffinity; enumerate all IDs; compare actual proxy sample_geom_ids/metadata and report excluded/native-only geoms.',
        'no_wrong_scalar':'measured_hand_object.penetration_m is maximumALLcategories, not HO. Recompute HO percontact and compare saved penetration_by_category_m.hand_object. Save self/other depths separately.',
        'raw_mapping':'CPUcontacts world-filtered by get_data_into must align original rawWP geom/worldid order. get_data_into reorders CPUefc addresses; do not use CPUefc as original constraint identity.',
        'no_positive_force_filter':'Primary generated-depth label includes all candidate types and zero-normal-force contacts, consistent with original quality metric. Force>1e-6/efc/typeCONSTRAINT subset is only separately labeled secondary, no primaryrow dropping.',
        'missing_or_empty':'validempty generatedHO -> observed candidate depth0 with explicitcontactcount0, not proof of no global penetration. Malformed/unknown geom/type/world orcapacity ->UNKNOWN and stop before score; never fillmissing0.'},
      'actual_geometry_binding':{'factory':'one full151_intent.binding_v3.build_penalty_only_objective(exact trainedspec08d9,devicecpu); keep eagerA2forward unchanged, training64basis, originalPformula',
        'same_actual_geometry':'nominal MJB plusactual host geom_size/geom_pos/geom_friction, same8field readbacks bothpaths; immutable nominalgeomquat/body/joint/mesh topology. Compare existing bonus geometry manifest/sampling hash and register sample_geom_ids/sample_points_body/sample_body_ids buffers for audit.',
        'scope':'1344 fixedsamples=64pereligiblegeom in existingexactfactory, compiledmesh convexhull/capsule; object exterior union1cylinder+2markerboxes. This is sampled proxy, not native narrowphase equivalence.',
        'COMradius':'sameCOM0 and nominal enclosingradius once; do not subtractCOM twice or substitute crosssectionradius/native friction. P is geometric and needs no new LP/mask/intent orforceclosure assumption.'},
      'future_scoring_budget_NOT_AUTHORIZED':{'factory_calls':1,'batch_size':128,'objective_forwards':15,'batch_rows':[128]*14+[16],
        'scored_rows':1808,'new_WM_or_Actor_load_or_forward':0,'new_AD':0,'new_Env_Warp_physics_nativeFK_collision_query':0,
        'passive_costs':'Source-bound MJBconstant read+deterministicfactory buffer/surface sampling allowed only under futureseparate RootCPUconfig; original pureTorch FK inside proxy is expected, not a new nativeFK/query.',
        'capture':'Read-only hook on existinggeometry_proxy forward, returnNone and remove finally. Savefull1808x1344 SDF, actualrawP/BP, minSDF,proxy samplebuffers/geom IDs, pergeom minimumSDF andoptionalexistingweights/support64; no extra geometryforward.',
        'readonly':'CPU FP32/AMPFalse,CUDA_VISIBLE_DEVICES=-1/notinitialized; allparamsfrozen andbuffers/RNG/Python/NumPy/TorchCPU/inputfiles beforeafterunchanged. Noautograd needed for observability. Savepartial andstopfirstfailure; no retry/tune/resample.'},
      'fixed_analysis':{'original_threshold_m':.003,'planning_margin_used':False,
        'proxy_observed':'actual rawP>0 and separately min_sampled_SDF<-.003 exactsource arithmetic; keep equality at3mm explicit',
        'continuous':'allrows nativeHO depthm versusproxy sampledepthmax(0,-minSDF), rawPdimensionless andweightedBP=-.1P; no reweighting to conceal zeros',
        'tables':'All1808 perstep CSV/NPZ, fullconfusion counts and allcontinuous scatter, bypath anddonor/baseline/perturbation separately; rowswithin3mm boundary retained, noCI orindependentNs.',
        'miss_breakdown':'For nativeHOdepth>3mm andproxyP0, use deepestnativepairhandgeomID: absentproxygeom ->GEOM_SET_COVERAGE_GAP; present andpergeom sampledSDF>=-3mm ->SAME_POSE_PROXY_UNDERSENSITIVITY_CANDIDATE. Cannot uniquely separate finite sampling/objectexterior/narrowphase semantics without laterindependent method.',
        'false_positive_caveat':'ProxyP>0 withnative observedHOdepth<=3mm isDISAGREEMENT_WITH_GENERATED_CONTACTS, not global geometricfalsepositive because contactabsence is not exhaustivegeometrytruth.',
        'coverage_question':'Denseprestate positives withold8nextstateP0 support limitedsparse/phasecoverage as one explanation; denseproxy misses at validatedsamepose indicate proxy observability problem. Both maycoexist; do not force exclusive causalstory.'},
      'limits':['Fresh donor saved-action replay is not original closed-loop archive reproduction. This covers onlytwo fixed action-source paths and restored dependentbranches.',
        'No WM/Actor/physicalgradient orwholehiddenstate qualification; no native forcegradient ortruehumancontact label.',
        'Same-state depth/proxy disagreement is not proof of a unique implementation bug or lack of trainingpenalty signal. Existinglocalproxy/compiledqualification failures remain retained.',
        'Do not reuse next20ms endpoint score with15ms/otherprestep contactdist. Converter/pose guard failure is a blocker, not permission to choose a convenient phase.',
        'This document is sourcefeasible/preregistered only; newconverter/hook/checker/sourceclosure and RootseparateCPUauthorization required before actual2260conversions or1808scores.']}
    OUT.mkdir(parents=True);(OUT/'source.py').write_bytes(Path(__file__).read_bytes());write(PROTOCOL,protocol)
    report={'schema':'full151105_solver_phase_penalty_source_feasibility:1','status':'PASS_SOURCE_FEASIBLE_WITH_UNEXECUTED_CONVERTER_GATES',
        'checks':checks,'passed':sum(checks.values()),'total':len(checks),'source_pins':pins,'input_pins':inputs,'protocol':pin(PROTOCOL),
        'source_review_limit':'Read source/JSONmetadata and firstrawrow FORMAT only; no dense state arithmetic orgeometry, no numeric converter/pose calibration has run.',
        'actual_work':{'new_state_conversions':0,'NPZ_arrays_loaded':0,'MJB_model_loads':0,'objective_constructs':0,'proxy_scores':0,'WM_Actor_forward_AD_Env_physics_GPU':0},
        'physical_gradient_qualified':False,'wholehiddenstate_qualified':False,'actual_converter_or_dense_proxy_ready':False}
    write(OUT/'report.json',report)
    print(json.dumps({'report':pin(OUT/'report.json'),'protocol':pin(PROTOCOL),'passed':report['passed'],'total':report['total'],'writer':pins['writer']},indent=2))


if __name__=='__main__':main()
