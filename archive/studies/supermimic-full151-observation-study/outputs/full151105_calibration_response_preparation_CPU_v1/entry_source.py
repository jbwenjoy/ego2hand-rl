"""Fresh-phase calibration response entry DRAFT; default metadata only."""
from pathlib import Path
import argparse,ast,datetime,json,os,shutil,sys,time,traceback,zipfile
STARTED=time.perf_counter();ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'vendor'))
from full151_native.io import read,pin,sha,checked,jsonable
from full151_native.metadata import write
SCHEMA='full151105_fresh_calibration_response_physical_run:1'
ROLE='fresh_response_calibration'
PHASES=[24,74,124,148]
MODES={'serial_full':{'donor_intervals':300,'snapshots':8,'baseline_branches':24,'perturbation_branches':128,'planned_episode_substeps':1808},
       'inventory_discovery_all':{'donor_intervals':300,'snapshots':8,'baseline_branches':0,'perturbation_branches':0,'planned_episode_substeps':1200}}
WORKER=ROOT/'src/full151_gradient/serial_probe105_calibration_v1.py'
SUPERVISOR=ROOT/'scripts/run_full151_calibration_response_v1.py'

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def positive(value):
    r=read(checked(value));c=r.get('checks')
    if not c or any(type(v) is not bool or not v for v in c.values()) or type(r.get('passed')) is not int or type(r.get('total')) is not int or r['passed']!=r['total'] or r['total']!=len(c):raise ValueError('Strict positive prerequisite')
    return r
def same(a,b):return a['sha256']==b['sha256'] and checked(a).samefile(checked(b))

def preflight(path):
    c=read(path)
    if c['schema']!=SCHEMA or c['role']!=ROLE or c['probe_mode'] not in MODES:raise ValueError('New calibration-response namespace required')
    if type(c['evaluation_seed']) is not int or c['evaluation_seed']!=0:raise ValueError('Fixed seed0')
    if type(c['GPU_run_authorized']) is not bool or c['training_authorized'] is not False or c['automatic_retry'] is not False or c['compile_fallback'] is not False:raise ValueError('Explicit GPU bool/no training/retry/fallback')
    if (c['soft_worker_seconds'],c['hard_process_seconds'])!=(180,240):raise ValueError('Fixed soft180/hard240')
    output=Path(c['output']).resolve()
    if output.parent!=(ROOT/'outputs').resolve() or output.exists():raise ValueError('Fresh exact study output required')
    pins=c['source_pins']
    if type(pins) is not dict or not pins:raise ValueError('Source closure required')
    verified={}
    def member(value):
        target=checked(value);st=target.stat();key=(st.st_dev,st.st_ino,value['sha256'])
        if key not in verified or not target.samefile(verified[key]):raise ValueError('Current dependency not in frozen source closure: '+str(target))
        return target
    for value in pins.values():
        target=checked(value);st=target.stat();verified[(st.st_dev,st.st_ino,value['sha256'])]=target
    for source in (Path(__file__),SUPERVISOR,WORKER,ROOT/'src/full151_gradient/transaction105.py'):
        member(pin(source));ast.parse(source.read_text(encoding='utf-8'))
    protocol=read(member(c['protocol']));proposal=read(member(c['proposal']));member(c['input_bundle'])
    if protocol.get('schema')!='full151105_fresh_calibration_response_execution_protocol:1' or protocol.get('mode_counts')!=MODES or protocol.get('snapshot_phases')!=PHASES:raise ValueError('New exact phase/count protocol required')
    allowed=protocol.get('allowed_probe_modes')
    if type(allowed) is not list or not allowed or len(set(allowed))!=len(allowed) or any(m not in MODES for m in allowed) or c['probe_mode'] not in allowed:raise ValueError('Protocol does not authorize this mode; discovery does not authorize full152')
    if not same(protocol['proposal'],c['proposal']) or not same(protocol['input_bundle'],c['input_bundle']):raise ValueError('Protocol/proposal/input aliases differ')
    if proposal.get('schema')!='full151105_fresh_calibration_response_physical_proposal:1' or proposal.get('role')!=ROLE or proposal.get('status')!='FROZEN_RESPONSE_ACQUISITION_DESIGN_NOT_SELF_AUTHORIZATION' or proposal.get('GPU_authorized_by_this_proposal') is not False or proposal.get('training_authorized') is not False or proposal.get('physical_reference_qualified') is not False:raise ValueError('New frozen response proposal only; no inherited qualification or self-authorization')
    member(proposal['method_proposal']);member(proposal['parent_worker'])
    for key in ('transaction_CPU_READY','transaction_CPU_report','schema2_WM_CPU_report','serial_CPU_READY','serial_CPU_report'):member(c['readiness'][key])
    ready=read(checked(c['readiness']['transaction_CPU_READY']));tr=positive(c['readiness']['transaction_CPU_report']);positive(c['readiness']['schema2_WM_CPU_report'])
    if tr['status']!='PASS_CPU_SURROGATE_INVENTORY_ONLY' or not same(ready['cpu_report'],c['readiness']['transaction_CPU_report']):raise ValueError('Actual transaction preparation required')
    for value in ready['source_pins'].values():member(value)
    if not same(ready['source_pins']['transaction105'],pin(ROOT/'src/full151_gradient/transaction105.py')):raise ValueError('Current CPU-tested transaction differs')
    serial_ready=read(checked(c['readiness']['serial_CPU_READY']));positive(c['readiness']['serial_CPU_report'])
    if not same(serial_ready['cpu_report'],c['readiness']['serial_CPU_report']):raise ValueError('New worker CPU proof alias mismatch')
    for value in serial_ready['source_pins'].values():member(value)
    if not any(same(pin(WORKER),v) for v in serial_ready['source_pins'].values()):raise ValueError('New worker is absent from its current CPU plan proof')
    for donor in c['donor_inputs'].values():
        directory=Path(donor['directory']).resolve(strict=True)
        for name,v in donor['input_pins'].items():
            if not checked(v).samefile(directory/name):raise ValueError('Donor source directory/name mismatch')
    if c['probe_mode']=='inventory_discovery_all':
        if c.get('foreign_inventory_review') is not None or c.get('inventory_discovery_health') is not None or c.get('calibration_recipe') is not None:raise ValueError('Discovery has no foreign authorization/previous discovery/recipe-derived response release')
    else:
        discovery=positive(c['inventory_discovery_health'])
        if discovery.get('probe_mode')!='inventory_discovery_all' or discovery.get('completed_inventory_snapshots')!=8 or discovery.get('episode_step_returned')!=1200 or discovery.get('restored_branches')!=0 or discovery.get('physical_gradient_qualified') is not False or discovery.get('whole_hidden_state_qualified') is not False:raise ValueError('New eight-snapshot1200/zero-restore empirical discovery required')
        if not same(discovery['input_bundle'],c['input_bundle']):raise ValueError('Actual same bundle required')
        dc=read(checked(discovery['config']));dr=read(checked(discovery['producer_report']))
        if dc.get('schema')!=SCHEMA or dc.get('role')!=ROLE or dc.get('probe_mode')!='inventory_discovery_all' or not same(dc['proposal'],c['proposal']):raise ValueError('Old discovery cannot be relabeled as calibration discovery')
        if dr.get('schema')!='full151105_calibration_response_worker_report:1' or dr.get('role')!=ROLE or dr.get('snapshot_phases')!=PHASES:raise ValueError('New-phase actual discovery report required')
        if not any(same(pin(WORKER),v) for v in dc['source_pins'].values()):raise ValueError('Discovery used a different worker')
        review=read(member(c['foreign_inventory_review']))
        if review.get('schema')!='full151105_foreign_inventory_per_snapshot_source_review:2':raise ValueError('Exact per-label source review required')
        if not same(review['source_inventory_saved_review'],c['inventory_discovery_health']) or not same(review['source_inventory_config'],discovery['config']) or not same(review['source_inventory_report'],discovery['producer_report']):raise ValueError('Foreign manifest must bind actual new discovery')
        if not Path(review['source_inventory_run']).samefile(discovery['run_dir']):raise ValueError('Actual discovery run identity differs')
        labels=[f'{name}/{k}' for name in ('A0','A2') for k in PHASES]
        if set(review['per_snapshot'])!=set(labels) or set(discovery['inventory_snapshot_pins'])!=set(labels):raise ValueError('All eight new labels required; no union/fallback')
        for label in labels:
            for field in ('description','topology'):
                if not same(review['per_snapshot'][label]['inventory'][field],discovery['inventory_snapshot_pins'][label][field]):raise ValueError('New per-label provenance mismatch')
        member(c['calibration_recipe'])
        if protocol.get('calibration_recipe') is None or not same(c['calibration_recipe'],protocol['calibration_recipe']):raise ValueError('Full responses require an independently frozen calibration recipe')
        recipes=protocol.get('calibration_source_pins')
        if type(recipes) is not dict or not recipes:raise ValueError('All calibration WM/recipe code must be frozen before response acquisition')
        for value in recipes.values():member(value)
    from full151_gradient.serial_probe105_calibration_v1 import validate_config,execution_plan
    validate_config(c,None);plan=execution_plan(c['probe_mode']);expected=MODES[c['probe_mode']]
    if plan['episode_substeps']!=expected['planned_episode_substeps'] or plan['snapshot_count']!=8 or plan['branch_count']!=expected['baseline_branches']+expected['perturbation_branches']:raise ValueError('Independent mode/worker count mismatch')
    if any(k in sys.modules for k in ('torch','warp','mujoco','mujoco_warp')):raise AssertionError('Metadata-only preflight imported a native framework')
    return c,None,{'status':'CPU_INPUT_SOURCE_PREFLIGHT_ONLY','config_sha256':sha(path),'entry_sha256':sha(__file__),'role':c['role'],'probe_mode':c['probe_mode'],
      'planned_counts':expected,'snapshot_phases':PHASES,'source_pins':len(pins),'Torch_imported':False,'Warp_imported':False,'Mujoco_imported':False,'GPU_started':False,'physics_steps':0,
      'CPU_proof_scope':'Metadata and pure planning only; no new native inventory, restore, baseline or hidden-state qualification'}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path);ap.add_argument('--run',action='store_true');a=ap.parse_args();path=a.config.resolve(strict=True)
    if not a.run:os.environ['CUDA_VISIBLE_DEVICES']='-1'
    c,_,proof=preflight(path)
    if not a.run:print(json.dumps(proof));return
    if c['GPU_run_authorized'] is not True:raise ValueError('Root explicit GPU authorization absent')
    out=Path(c['output']).resolve()
    if a.output is None or a.output.resolve()!=out or out.exists():raise ValueError('Fresh exact output; no resume/overwrite')
    out.mkdir(parents=True,exist_ok=False);runtime={'status':'initializing','pid':os.getpid(),'parent_pid':os.getppid(),'started_utc':utc(),'plan_sha256':sha(path),'source_sha256':sha(__file__),'role':c['role'],'probe_mode':c['probe_mode'],'automatic_retry':False,'optimizer_updates':0,'physical_reference_qualified':False}
    write(out/'runtime.json',runtime);shutil.copy2(path,out/'config.json');write(out/'CPU_preflight.json',proof);folder=out/'source';folder.mkdir()
    with zipfile.ZipFile(folder/'execution_sources.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for i,v in enumerate(c['source_pins'].values()):archive.write(checked(v),f'{i:04d}_{Path(v["path"]).name}')
    write(folder/'inventory.json',c['source_pins'])
    def emit(**updates):runtime.update(updates,worker_elapsed_seconds=time.perf_counter()-STARTED);write(out/'runtime.json',runtime)
    try:
        from full151_native.bundle import load_bundle
        bundle=load_bundle(checked(c['input_bundle']),c['input_bundle']['sha256'])
        from full151_gradient.serial_probe105_calibration_v1 import run_probe
        result=run_probe(bundle,{**c,'_worker_started_perf':STARTED},out,emit)
        write(out/'result.json',jsonable(result))
        if result.get('status') not in ('COMPLETED_PHYSICAL_PROBE','COMPLETED_INVENTORY_DISCOVERY'):raise ValueError('Incomplete/failed response worker; retain raw result')
        runtime.update(status='completed_full151_calibration_response_probe',result_status=result.get('status'),episode_steps_returned=result.get('episode_step_returned','SEE_PROBE_COUNTERS'),step_count_scope='Episode returns only; construction/graph capture API counts separate')
        write(out/'completion.json',{'schema':'full151105_calibration_response_worker_completion:1','config_sha256':sha(path),'result':jsonable(result),'qualification_scope':'Identity/completion does not establish true gradient, whole-hidden-state restoration or calibration benefit'})
    except BaseException as exc:
        runtime.update(status='failed_full151_calibration_response_probe',error=repr(exc));write(out/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'no_retry':True});raise
    finally:runtime.update(finished_utc=utc(),wall_seconds=time.perf_counter()-STARTED);write(out/'runtime.json',runtime)
    print(json.dumps({'status':result['status'],'probe_mode':c['probe_mode'],'episode_steps_returned':result.get('episode_step_returned'),'snapshots':result.get('completed_snapshot_count'),'branches':result.get('completed_branch_count'),'report':str(out/'physical_report.json')}))

if __name__=='__main__':main()
