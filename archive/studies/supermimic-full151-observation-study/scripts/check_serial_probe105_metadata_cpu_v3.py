"""Worker2 planning, counters and synthetic review-metadata contracts only."""
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import hashlib
import importlib
import json
import sys
import numpy as np

ROOT=Path('D:/workspace/supermimic-full151-observation-study')
SOURCE=ROOT/'src/full151_gradient/serial_probe105_v2.py'
OUT=ROOT/'outputs/full151105_serial_probe_v2_preparation_CPU_v1'

def pin(path):
    path=Path(path).resolve(strict=True)
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def write_fixture(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)

def main():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir();before=OUT/'source_before';before.mkdir()
    for path in (SOURCE,Path(__file__)):(before/path.name).write_bytes(path.read_bytes())
    sys.path.insert(0,str(ROOT/'src'))
    m=importlib.import_module('full151_gradient.serial_probe105_v2')
    from full151_gradient import transaction105 as txmodule
    from full151_native.metadata import write
    checks={};details={}
    def check(name,value):
        if not value:raise AssertionError(name)
        checks[name]=True
    def rejects(name,fn):
        try:fn()
        except (m.InventoryReviewRequired,ValueError,KeyError,FileNotFoundError) as exc:
            checks[name]=True;details[name]=repr(exc);return
        raise AssertionError('Expected rejection: '+name)
    discovery=m.execution_plan('inventory_discovery_all');full=m.execution_plan('serial_full')
    check('discovery_two_full_donors',discovery['donors']==['A0','A2'] and discovery['donor_intervals']==150 and discovery['donor_state_rows']==601)
    check('discovery_all8captures',discovery['capture_phases']==[0,49,99,149] and discovery['snapshot_count']==8 and sum(v['scheduled'] for v in discovery['snapshot_slots'])==8)
    check('discovery1200_no_scheduled_branch',discovery['episode_substeps']==1200 and discovery['branch_count']==0 and not any(v['scheduled'] for v in discovery['branch_slots']))
    check('discovery152_NOT_RUN_not_taskfalse',len(discovery['branch_slots'])==152 and all(v['status']=='NOT_RUN' and v['actualu'] is None and v['r_k'] is None and not v['restorequalified'] for v in discovery['branch_slots']))
    check('full152and1808unchanged',full['branch_count']==152 and full['episode_substeps']==1808 and len(full['branch_slots'])==152 and all(v['scheduled'] for v in full['branch_slots']))
    check('source_labels_exact8',list(m.SNAPSHOT_LABELS)==['A0/0','A0/49','A0/99','A0/149','A2/0','A2/49','A2/99','A2/149'])
    check('same_snapshot_grid_andterminal',discovery['snapshot_ticks']==full['snapshot_ticks']==[0,196,396,596] and discovery['next_ticks']==full['next_ticks']==[4,200,400,600])
    check('donor_branch_budget_equation',all(p['episode_substeps']==len(p['donors'])*p['donor_intervals']*4+p['branch_count']*4 for p in (discovery,full)))
    rejects('legacy_phase0_not_worker2',lambda:m.execution_plan('engineering_restore_smoke'))
    rejects('unknown_mode',lambda:m.execution_plan('unknown'))
    arrays,dirs=m.allocate_artifacts()
    check('fixed_8_152_array_ABI',arrays['snapshot_x'].shape==(8,58) and arrays['branch_x_next'].shape==(152,58) and dirs['center_offset'].shape==(8,4,2,20))
    check('no_vacuous_restore_mask',not arrays['completed'].any() and not arrays['restore_qualified'].any() and not arrays['restore_inventory_bitexact'].any())
    scope=m.completion_scope('inventory_discovery_all')
    check('discovery_status_not_physical_probe',scope['status']=='COMPLETED_INVENTORY_DISCOVERY' and scope['inventory_discovery_only'])
    check('discovery_gradient_hidden_never_qualified',scope['gradient_qualified'] is False and scope['whole_graph_hidden_state_qualified'] is False)
    check('discovery_baseline_scope_NOT_RUN',scope['empirical_visible_state_baseline_scope']=='NOT_RUN_INVENTORY_ONLY')
    check('discovery_cap_clock_donor_scope',scope['capacity_scope']=='Actual fresh donors plusinitialization only' and scope['clock_scope']=='Actual fresh donor sequentialFP32 clock only')
    # These are plain Python callbacks, not NativeSnapshot/Env methods.
    names=('transaction_begin','transaction_finish','native_snapshot_restore')
    counters={name+suffix:0 for name in names for suffix in ('_attempted','_returned')}
    callbacks=[]
    fake=SimpleNamespace(begin=lambda value:callbacks.append('begin'),finish=lambda value:callbacks.append('finish'),native=SimpleNamespace(restore=lambda:callbacks.append('fake_restore')))
    m.instrument_transaction_calls(fake,counters)
    check('instrument_install_has_no_calls',not callbacks and not any(counters.values()))
    fake.begin('x');fake.finish('x')
    check('counter_records_actual_begin_finish',counters['transaction_begin_returned']==counters['transaction_finish_returned']==1 and counters['native_snapshot_restore_attempted']==0)
    fake.native.restore()
    check('counter_fake_restore_not_assumed',counters['native_snapshot_restore_attempted']==counters['native_snapshot_restore_returned']==1)
    # Metadata-only fixtures are explicitly labelled and never native evidence.
    bundlepin=pin(ROOT/'outputs/full151_native_input_v1/bundle.json')
    workerpin=pin(SOURCE);txpin=pin(txmodule.__file__)
    fixture_serial=0
    def fixture(name,mutate_review=None,mutate_desc=None,mutate_top=None):
        nonlocal fixture_serial
        fixture_serial+=1;folder=OUT/'metadata_fixtures'/(str(fixture_serial)+'_'+name)
        tables={}
        for label in m.SNAPSHOT_LABELS:
            location=folder/label.replace('/','_');phase=int(label.split('/')[1])
            leaf={'path':'native_entry[0].value[0]','type':'CPU_METADATA_FIXTURE.Foreign','reason':'Synthetic metadata contract only, not real native object or qualification'}
            desc={'schema':'full151105_probe_transaction_inventory:1','label':label,'strict_native_types_checked':True,'CPU_METADATA_FIXTURE_ONLY':True,
                'profile':{'phase':phase,'boundary':'pre_action','format_version':105,'mode':txmodule.MODE,'retarget_variant_id':txmodule.VARIANT,'strict_native_types_checked':True},
                'opaque_identity_only':[{'path':leaf['path'],'type':leaf['type'],'id':0}]}
            top={'worker_source':workerpin,'transaction_source':txpin,'CPU_METADATA_FIXTURE_ONLY':True,
                 'native_entries':[{'entry':0,'type':'CPU_METADATA_FIXTURE.Owner','members':[{'key':'foreign','type':leaf['type'],'identity':0}]}]}
            if mutate_desc and label=='A0/0':mutate_desc(desc)
            if mutate_top and label=='A0/0':mutate_top(top)
            write_fixture(location/'transaction_description.json',desc);write_fixture(location/'inventory_topology.json',top)
            tables[label]={'inventory':{'description':pin(location/'transaction_description.json'),'topology':pin(location/'inventory_topology.json')},
                           'allowed_identity_only':[leaf],'source_reviewed_closure_globals':True}
        review={'schema':'full151105_foreign_inventory_per_snapshot_source_review:2','worker':workerpin,'transaction':txpin,'input_bundle':bundlepin,
            'source_reviewed_closure_globals':True,'approved_probe_modes':['serial_full'],'allowed_snapshot_labels':list(m.SNAPSHOT_LABELS),
            'phase0_only':False,'conditional_engineering_only':True,'whole_graph_hidden_state_qualified':False,'per_snapshot':tables,
            'source_pins':{'worker':workerpin,'transaction':txpin},'CPU_METADATA_FIXTURE_ONLY':True}
        if mutate_review:mutate_review(review)
        write_fixture(folder/'review.json',review)
        return {'probe_mode':'serial_full','input_bundle':bundlepin,'foreign_inventory_review':pin(folder/'review.json')},review
    valid_config,valid_review=fixture('valid')
    check('valid_all8_metadata_joins',len(m.validate_review_manifest(valid_config)['per_snapshot'])==8)
    current=SimpleNamespace(label='A2/149',describe=lambda:{'opaque_identity_only':[{'path':'native_entry[0].value[0]','type':'CPU_METADATA_FIXTURE.Foreign'}]})
    check('current_snapshot_selects_exact_table',m.reviewed_foreign_inventory(valid_config,current,None)['snapshot_label']=='A2/149')
    rejects('unexpected_current_label_no_fallback',lambda:m.reviewed_foreign_inventory(valid_config,SimpleNamespace(label='A0/1'),None))
    rejects('current_exact_path_drift_rejected',lambda:m.reviewed_foreign_inventory(valid_config,SimpleNamespace(label='A2/149',describe=lambda:{'opaque_identity_only':[{'path':'native_entry[1].value[0]','type':'CPU_METADATA_FIXTURE.Foreign'}]}),None))
    for name,mutator in (
        ('old_flat_schema1',lambda j:j.update(schema='full151105_foreign_inventory_source_review:1')),
        ('missing_label',lambda j:j['per_snapshot'].pop('A2/149')),
        ('extra_label',lambda j:j['per_snapshot'].update({'A0/150':copy.deepcopy(j['per_snapshot']['A0/0'])})),
        ('reordered_labels',lambda j:j['allowed_snapshot_labels'].reverse()),
        ('phase0_only',lambda j:j.update(phase0_only=True)),
        ('discovery_scope_not_allowed',lambda j:j.update(approved_probe_modes=['serial_full','inventory_discovery_all'])),
        ('hidden_qualification_forbidden',lambda j:j.update(whole_graph_hidden_state_qualified=True)),
        ('missing_closure_review',lambda j:j.update(source_reviewed_closure_globals=False)),
        ('wrong_worker',lambda j:j.update(worker=pin(ROOT/'src/full151_gradient/serial_probe105_v1.py'))),
        ('missing_source_pins',lambda j:j.update(source_pins={})),
        ('duplicate_paths',lambda j:j['per_snapshot']['A0/0']['allowed_identity_only'].append(copy.deepcopy(j['per_snapshot']['A0/0']['allowed_identity_only'][0]))),
        ('wildcard_path',lambda j:j['per_snapshot']['A0/0']['allowed_identity_only'][0].update(path='native_entry[*]')),
    ):
        config,_=fixture(name,mutate_review=mutator)
        rejects('manifest_'+name,lambda c=config:m.validate_review_manifest(c))
    for name,mutator in (
        ('label_mismatch',lambda j:j.update(label='A2/149')),
        ('phase_mismatch',lambda j:j['profile'].update(phase=49)),
        ('bool_phase',lambda j:j['profile'].update(phase=False)),
        ('terminal_boundary',lambda j:j['profile'].update(boundary='terminal_observation')),
        ('wrong_format',lambda j:j['profile'].update(format_version=104)),
        ('wrong_mode',lambda j:j['profile'].update(mode='rough_prefix_old')),
        ('wrong_variant',lambda j:j['profile'].update(retarget_variant_id='old16')),
        ('surrogate_type_scope',lambda j:j['profile'].update(strict_native_types_checked=False)),
    ):
        config,_=fixture(name,mutate_desc=mutator)
        rejects('description_'+name,lambda c=config:m.validate_review_manifest(c))
    for name,mutator in (
        ('wrong_topology_worker',lambda j:j.update(worker_source=txpin)),
        ('opaque_owner_missing',lambda j:j.update(native_entries=[])),
        ('direct_leaf_type_mismatch',lambda j:j['native_entries'][0]['members'][0].update(type='CPU_METADATA_FIXTURE.Wrong')),
    ):
        config,_=fixture(name,mutate_top=mutator)
        rejects('topology_'+name,lambda c=config:m.validate_review_manifest(c))
    source=SOURCE.read_text(encoding='utf-8-sig');tree=ast.parse(source)
    probe=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_probe')
    guarded=False
    for n in ast.walk(probe):
        if isinstance(n,ast.For) and isinstance(n.iter,ast.Name) and n.iter.id=='capture_phases' and n.body:
            first=n.body[0]
            if isinstance(first,ast.If) and ast.unparse(first.test)=="mode == 'inventory_discovery_all'" and len(first.body)==1 and isinstance(first.body[0],ast.Continue):guarded=True
    check('actual_branch_loop_discovery_skip_present',guarded)
    check('separate_serializer_andrestore_counters',all(name in source for name in ('integration_get_attempted','integration_set_attempted','native_snapshot_restore_attempted','transaction_begin_attempted')))
    check('no_frameworks_after_metadata',not any(name in sys.modules for name in ('torch','warp','mujoco','mjlab')))
    check('executed_worker1_preserved',pin(ROOT/'src/full151_gradient/serial_probe105_v1.py')['sha256']=='79207ad27252157036915360696442924176b22692464a3ea855c879656c104d')
    source_pins={'serial_probe105_v2':workerpin,'checker_v3':pin(__file__),'transaction105':txpin,
        'metadata105':pin(ROOT/'src/full151_native/metadata.py'),'io105':pin(ROOT/'src/full151_native/io.py'),
        'proposal2':pin(ROOT/'protocols/full151_same_warp_control_gradient_proposal_v2.json')}
    report={'schema':'full151105_serial_worker_v2_CPU_contract:1','status':'PASS_CPU_PLANNING_AND_SYNTHETIC_METADATA_ONLY','passed':len(checks),'total':len(checks),'checks':checks,'details':details,
        'source_pins':source_pins,'synthetic_fixture_scope':'Eight-label metadata joins with explicit CPU_METADATA_FIXTURE_ONLY flags; not actual native inventory, source-review approval or physical qualification.',
        'actual_Env_instances':0,'actual_model_loads':0,'actual_native_snapshot_restores':0,'actual_physics_steps':0,'WM_calls':0,'GPU_calls':0,
        'discovery_plan':discovery,'full_plan':full,'discovery_completion_vocabulary':scope,'current_actual_metadata_preflight':'Root defaultfront separately pending',
        'guard_scope':'Discovery avoids branch calls by fixed control flow plus zero-call postcondition; instrumentation is not a general interception/rejection before arbitrary external calls.'}
    write(OUT/'report.json',report)
    ready={'status':'CPU_PREPARATION_ONLY_NOT_NATIVE_QUALIFIED','source_pins':source_pins,'cpu_report':pin(OUT/'report.json'),'cpu_checks':{'passed':len(checks),'total':len(checks)},
        'api':{'entry':'full151_gradient.serial_probe105_v2.run_probe(bundle,config,out,emit)','metadata':'validate_config(config,bundle=None)',
            'modes':['inventory_discovery_all','serial_full'],'review_schema':'full151105_foreign_inventory_per_snapshot_source_review:2'},
        'qualification':'Native inventory-all andper-snapshot physical/replay/clock/capacity qualification are not executed by this checker.'}
    write(OUT/'READY.json',ready)
    print(json.dumps({'status':report['status'],'passed':len(checks),'total':len(checks),'report':pin(OUT/'report.json'),'READY':pin(OUT/'READY.json')}))

if __name__=='__main__':main()
