"""Explicit bounded105 physical-probe entry; default is metadata only."""
from pathlib import Path
import argparse,ast,datetime,json,os,shutil,sys,time,traceback,zipfile
STARTED=time.perf_counter();ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'vendor'))
from full151_native.io import read,pin,sha,checked,canonical,jsonable
from full151_native.metadata import write
SCHEMA='full151105_same_warp_physical_probe_run:2'
MODES={'serial_full':{'donor_intervals':300,'snapshots':8,'baseline_branches':24,'perturbation_branches':128,'planned_episode_substeps':1808},
       'inventory_discovery_all':{'donor_intervals':300,'snapshots':8,'baseline_branches':0,'perturbation_branches':0,'planned_episode_substeps':1200}}
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def positive(value):
    r=read(checked(value));c=r.get('checks')
    if not c or not all(v is True for v in c.values()) or r.get('passed')!=r.get('total') or r['total']!=len(c):raise ValueError('Incomplete positive prerequisite')
    return r
def same(a,b):return a['sha256']==b['sha256'] and checked(a).samefile(checked(b))
def preflight(path):
    c=read(path)
    if c['schema']!=SCHEMA or c['role']!='full151105_serial_physical_probe' or c['probe_mode'] not in MODES:raise ValueError('Explicit105 physical probe namespace required')
    if type(c['evaluation_seed']) is not int or c['evaluation_seed']!=0:raise ValueError('Frozen evaluation_seed0 required')
    if c['training_authorized'] is not False or type(c['GPU_run_authorized']) is not bool or c['automatic_retry'] is not False or c['compile_fallback'] is not False:raise ValueError('No training/retry/fallback')
    if (c['soft_worker_seconds'],c['hard_process_seconds'])!=(180,240):raise ValueError('Finite proposal budget changed')
    if not Path(c['output']).resolve().is_relative_to((ROOT/'outputs').resolve()):raise ValueError('Study-owned output required')
    for v in c['source_pins'].values():checked(v)
    indexed={(str(checked(v).resolve()).casefold(),v['sha256']) for v in c['source_pins'].values()}
    for source in (Path(__file__),ROOT/'scripts/run_full151_gradient_v2.py',ROOT/'src/full151_gradient/serial_probe105_v2.py',ROOT/'src/full151_gradient/transaction105.py'):
        if (str(source.resolve()).casefold(),sha(source)) not in indexed:raise ValueError('Actual probe source unarchived')
        ast.parse(source.read_text(encoding='utf-8'))
    protocol=read(checked(c['protocol']));proposal=read(checked(c['proposal']))
    if protocol['schema']!='full151105_physical_probe_execution_protocol:2' or protocol['mode_counts']!=MODES or protocol['input_bundle']!=c['input_bundle'] or protocol['proposal']!=c['proposal']:raise ValueError('Common protocol/count/input mismatch')
    if proposal['schema']!='full151105_same_Warp_control_gradient_diagnostic_proposal:2' or proposal['execution_status']!='NOT_EXECUTED' or proposal['physical_reference_qualified'] is not False:raise ValueError('Phase-corrected frozen proposal2 required; no qualification upgrade')
    for key in ('transaction_CPU_READY','transaction_CPU_report','schema2_WM_CPU_report','serial_CPU_READY','serial_CPU_report'):
        if (str(checked(c['readiness'][key]).resolve()).casefold(),c['readiness'][key]['sha256']) not in indexed:raise ValueError('Readiness not archived')
    ready=read(checked(c['readiness']['transaction_CPU_READY']));tr=positive(c['readiness']['transaction_CPU_report']);positive(c['readiness']['schema2_WM_CPU_report'])
    if tr['status']!='PASS_CPU_SURROGATE_INVENTORY_ONLY' or not same(ready['cpu_report'],c['readiness']['transaction_CPU_report']):raise ValueError('Only actual CPU component preparation accepted')
    for value in ready['source_pins'].values():
        if (str(checked(value).resolve()).casefold(),value['sha256']) not in indexed:raise ValueError('CPU transaction proof source unarchived or changed')
    if not same(ready['source_pins']['transaction105'],pin(ROOT/'src/full151_gradient/transaction105.py')):raise ValueError('Current transaction differs from CPU-tested component')
    serial_ready=read(checked(c['readiness']['serial_CPU_READY']));positive(c['readiness']['serial_CPU_report'])
    if not same(serial_ready['cpu_report'],c['readiness']['serial_CPU_report']):raise ValueError('Serial CPU proof/report alias mismatch')
    for value in serial_ready['source_pins'].values():
        if (str(checked(value).resolve()).casefold(),value['sha256']) not in indexed:raise ValueError('Serial CPU proof source unarchived or changed')
    current_serial=pin(ROOT/'src/full151_gradient/serial_probe105_v2.py')
    if not any(same(current_serial,v) for v in serial_ready['source_pins'].values()):raise ValueError('Current serial source was not CPU planning-tested')
    for donor in c['donor_inputs'].values():
        directory=Path(donor['directory']).resolve(strict=True)
        for name,v in donor['input_pins'].items():
            if checked(v).resolve()!=directory/name:raise ValueError('Donor basename/directory pin differs')
    if c['foreign_inventory_review'] is not None:checked(c['foreign_inventory_review'])
    if c['probe_mode']=='inventory_discovery_all' and c['foreign_inventory_review'] is not None:raise ValueError('Discovery-all captures fresh donors only, with no restore/foreign authorization')
    if c['probe_mode']=='serial_full':
        if c.get('restore_smoke_health') is None or c.get('inventory_discovery_health') is None:raise ValueError('Actual empirical phase0 and complete8 inventory saved reviews required')
        smoke=positive(c['restore_smoke_health']);discovery=positive(c['inventory_discovery_health'])
        if smoke.get('empirical_phase0_baseline_passed') is not True or smoke.get('physical_gradient_qualified') is not False or smoke.get('whole_hidden_state_qualified') is not False:raise ValueError('Typed empirical baseline scope required, never a hidden-state certificate')
        if discovery.get('probe_mode')!='inventory_discovery_all' or discovery.get('completed_inventory_snapshots')!=8 or discovery.get('episode_step_returned')!=1200 or discovery.get('restored_branches')!=0 or discovery.get('physical_gradient_qualified') is not False:raise ValueError('Complete actual8snapshot zero-restore donor inventory proof required')
        for evidence in (smoke,discovery):
            if not same(evidence['input_bundle'],c['input_bundle']):raise ValueError('Same actual native input bundle required')
        if c['foreign_inventory_review'] is None:raise ValueError('Exact per-snapshot source review required before full152')
        review=read(checked(c['foreign_inventory_review']))
        if review.get('schema')!='full151105_foreign_inventory_per_snapshot_source_review:2':raise ValueError('No phase0 flat manifest fallback')
        if not same(review['source_inventory_saved_review'],c['inventory_discovery_health']) or not same(review['source_inventory_config'],discovery['config']) or not same(review['source_inventory_report'],discovery['producer_report']):raise ValueError('Per-snapshot review must join the same actual discovery review/config/report')
        if Path(review['source_inventory_run']).resolve()!=Path(discovery['run_dir']).resolve():raise ValueError('Inventory run provenance mismatch')
        labels=[f'{name}/{k}' for name in ('A0','A2') for k in (0,49,99,149)]
        if set(review['per_snapshot'])!=set(labels) or set(discovery['inventory_snapshot_pins'])!=set(labels):raise ValueError('Exactly8 actual inventory labels required')
        for label in labels:
            for field in ('description','topology'):
                if not same(review['per_snapshot'][label]['inventory'][field],discovery['inventory_snapshot_pins'][label][field]):raise ValueError('Per-snapshot source records differ from independently reviewed actual inventory')
    from full151_native.bundle import load_bundle
    bundle=load_bundle(checked(c['input_bundle']),c['input_bundle']['sha256'])
    from full151_gradient.serial_probe105_v2 import validate_config
    validate_config(c,bundle)
    from full151_gradient.serial_probe105_v2 import execution_plan
    plan=execution_plan(c['probe_mode']);expected=MODES[c['probe_mode']]
    if plan['episode_substeps']!=expected['planned_episode_substeps'] or plan['snapshot_count']!=expected['snapshots'] or plan['branch_count']!=expected['baseline_branches']+expected['perturbation_branches']:raise ValueError('Independent front/worker scheduling contract differs')
    if any(k in sys.modules for k in ('torch','warp','mujoco_warp')):raise AssertionError('Metadata preflight initialized framework')
    return c,bundle,{'status':'CPU_INPUT_SOURCE_PREFLIGHT_ONLY','config_sha256':sha(path),'entry_sha256':sha(__file__),'role':c['role'],'probe_mode':c['probe_mode'],'planned_counts':MODES[c['probe_mode']],'source_pins':len(c['source_pins']),'Torch_imported':False,'Warp_imported':False,'GPU_started':False,'physics_steps':0,'CPU_proof_scope':'Metadata+surrogate interfaces; native inventory/restore/baselines/clock NOT_YET_QUALIFIED'}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path);ap.add_argument('--run',action='store_true');a=ap.parse_args();path=a.config.resolve(strict=True)
    if not a.run:os.environ['CUDA_VISIBLE_DEVICES']='-1'
    c,bundle,proof=preflight(path)
    if not a.run:print(json.dumps(proof,indent=2));return
    if c['GPU_run_authorized'] is not True:raise ValueError('Root explicit boundedGPU authorization absent')
    out=Path(c['output']).resolve()
    if a.output is None or a.output.resolve()!=out or out.exists():raise ValueError('Fresh exact output only; never resume or overwrite')
    out.mkdir(parents=True,exist_ok=False);runtime={'status':'initializing','pid':os.getpid(),'parent_pid':os.getppid(),'started_utc':utc(),'plan_sha256':sha(path),'source_sha256':sha(__file__),'role':c['role'],'probe_mode':c['probe_mode'],'automatic_retry':False,'optimizer_updates':0,'physical_reference_qualified':False}
    write(out/'runtime.json',runtime);shutil.copy2(path,out/'config.json');write(out/'CPU_preflight.json',proof);folder=out/'source';folder.mkdir()
    with zipfile.ZipFile(folder/'execution_sources.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for i,v in enumerate(c['source_pins'].values()):archive.write(checked(v),f'{i:04d}_{Path(v["path"]).name}')
    write(folder/'inventory.json',c['source_pins'])
    def emit(**updates):runtime.update(updates,worker_elapsed_seconds=time.perf_counter()-STARTED);write(out/'runtime.json',runtime)
    try:
        from full151_gradient.serial_probe105_v2 import run_probe
        result=run_probe(bundle,{**c,'_worker_started_perf':STARTED},out,emit)
        write(out/'result.json',jsonable(result))
        if result.get('status') not in ('COMPLETED_ENGINEERING_RESTORE_SMOKE','COMPLETED_PHYSICAL_PROBE','COMPLETED_INVENTORY_DISCOVERY','INVENTORY_REVIEW_REQUIRED'):raise ValueError('Probe returned noncomplete/failed status; retained raw result')
        runtime.update(status='completed_full151_gradient_probe',result_status=result.get('status'),episode_steps_returned=result.get('episode_step_returned','SEE_PROBE_COUNTERS'),step_count_scope='Episode API returns; constructor/capture counters separate')
        write(out/'completion.json',{'schema':'full151105_physical_probe_worker_completion:1','config_sha256':sha(path),'result':jsonable(result),'qualification_scope':'Must inspect result/independent saved review; terminal identity never qualifies gradient'})
    except BaseException as exc:
        runtime.update(status='failed_full151_gradient_probe',error=repr(exc));write(out/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'no_retry':True});raise
    finally:runtime.update(finished_utc=utc(),wall_seconds=time.perf_counter()-STARTED);write(out/'runtime.json',runtime)
    print(json.dumps(jsonable(result),indent=2))
if __name__=='__main__':main()
