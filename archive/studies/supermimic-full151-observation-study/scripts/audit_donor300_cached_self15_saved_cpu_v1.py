"""Independent donor300 JSON/IEEE-f32 audit. Standard library only."""
import argparse,hashlib,json,math,struct,sys,zipfile
from pathlib import Path
ROOT=Path('D:/workspace/supermimic-full151-observation-study')
ALIASES={'H1_config','H1_report','H1_identities','H1_saved_independent25','reward_snapshots','sensor_mapping_A0','sensor_mapping_A2','physical_saved_independent1762'}
IDENTITY={'group_id','path','mode','k','branch_id','source_frame','terminal_observation','valid_next_policy_decision','no_u150','first_physical_pre_index','last_physical_index','post20_GPU_row_index'}
def need(x,s):
    if not x:raise ValueError(s)
def pin(p):
    p=Path(p).resolve(strict=True);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def checked(v):
    need(type(v) is dict and set(v)=={'path','sha256'} and Path(v['path']).is_absolute(),'Absolute pin');p=Path(v['path']).resolve(strict=True);need(pin(p)['sha256']==v['sha256'],'SHA '+str(p));return p
def same(a,b):return a['sha256']==b['sha256'] and checked(a).samefile(checked(b))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def passed(v):return v['passed']==v['total']==len(v['checks']) and bool(v['checks']) and all(x is True for x in v['checks'].values())
def key(r):
    need(r['path'] in ('A0','A2') and type(r['k']) is int and 0<=r['k']<150,'Path/k')
    need(r['mode'] in ('donor','baseline','direction') and (r['branch_id'] is None if r['mode']=='donor' else type(r['branch_id']) is int),'Mode/branchID');return r['path'],r['mode'],r['k'],r['branch_id']
def f32(v):return struct.pack('<f',float(v))
def mapping(raw):
    need(type(raw) is list and len(raw)==25 and all(type(v) in (int,float,bool) and math.isfinite(v) and v>=0 and int(v)==v for v in raw),'25 nonnegative finite integer counts')
    # Independent finger/offset enumeration, not producer index-list reuse.
    return [float(raw[5*f+t]>0) for f in range(5) for t in (1,2)]+[float(raw[5*f+3]>0 or raw[5*f+4]>0) for f in range(5)]
def join(g,r):
    need(set(g)==IDENTITY and key(g)==key(r) and g['mode']=='donor','Exact H1 donor schema/join');k=g['k'];gid=k+(226 if g['path']=='A2' else 0)
    expected={'group_id':gid,'source_frame':200+k,'terminal_observation':k+1,'first_physical_pre_index':4*gid,'last_physical_index':4*gid+3,'post20_GPU_row_index':1808+gid}
    need(all(type(g[n]) is int and g[n]==v for n,v in expected.items()),'H1 original integer addresses/phases')
    need(g['valid_next_policy_decision'] is (k<149) and g['no_u150'] is True,'H1 terminal149')
    need(all(type(r[n]) is int and r[n]==g[n] for n in ('source_frame','terminal_observation')) and r['valid_next_policy_decision'] is g['valid_next_policy_decision'] and r['no_u150'] is True,'Raw reward identity phase')
    need(r['label_time']=='lastsolver_preintegration~kdt+.015' and r['state_time']=='postintegration_kdt+.020' and r['capture']=='after_original_reward_compute_before_Manager_forward','15ms cached/20ms state')
    raw=r['raw_found'];contact=r['contact'];need(type(raw) is list and len(raw)==1,'Single world rawfound')
    y=mapping(raw[0]);need(type(contact) is list and len(contact)==15 and all(type(v) in (int,float) and math.isfinite(v) and v in (0,1) for v in contact),'Binary native15')
    need(all(f32(a)==f32(b) for a,b in zip(y,contact)),'Raw25/native15 all original IEEEf32 bytes')
    return {**g,'contact15':contact,'raw_found25':raw[0],'label_time':r['label_time'],'state_time':r['state_time']}
def fixture(out):
    out.mkdir(exist_ok=False);(out/'executed_auditor.py').write_bytes(Path(__file__).read_bytes());write(out/'source_before_fixture.json',{'auditor':pin(__file__),'actual_inputs_read':0})
    raw=[0.]*25;raw[1]=2.;raw[3]=1.;raw[4]=3.;raw[24]=1.;y=mapping(raw)
    ok=y==[1.]+[0.]*9+[1.,0.,0.,0.,1.] and all(f32(a)==f32(b) for a,b in zip(y,[1]+[0]*9+[1,0,0,0,1])) and f32(-0.)!=f32(0.)
    write(out/'report.json',{'schema':'full151105_donor300_independent_pure_contract:1','checks':{'stride5_integer_counts_distal_OR_tip_order_and_signedzero_bytes':ok},'passed':int(ok),'total':1,'auditor':pin(__file__),'actual_JSON_NPZ_models_read':0})
    write(out/'READY.json',{'auditor':pin(__file__),'cpu_report':pin(out/'report.json'),'cpu_checks':{'passed':int(ok),'total':1}});need(ok,'Single fixture');print(json.dumps({'READY':pin(out/'READY.json')}))
def audit(protocol,run,out):
    out.mkdir(exist_ok=False);(out/'executed_auditor.py').write_bytes(Path(__file__).read_bytes());write(out/'source_before_actual.json',{'auditor':pin(__file__),'actual_label_JSON_read':False})
    p=read(protocol);need(p['schema']=='full151105_donor300_cached_self15_saved_CPU_protocol:1' and p['role']=='all300_donor_cached_native15_sensor_label_join_only','New label namespace')
    need(p['CPU_run_authorized'] is True and p['GPU_run_authorized'] is p['training_authorized'] is p['automatic_retry'] is False and p['expected_rows']==300 and p['no_model_or_NPZ_reads'] is True,'Authorized JSON-only bounded run')
    i=p['inputs'];need(set(i)==ALIASES,'Exact8 parents')
    for v in [*i.values(),*p['source_pins'].values(),p['source_contract']]:checked(v)
    need(p['source_pins']['producer']['sha256']=='930ed92160f52e7a4710d7cc4d19b222ef80f9aff965fe5a79fb2478a36102d0' and p['source_contract']['sha256']=='4f91fc4400c2481d53554e8510d5cfc31fd97a9830d758d2ad84eaf806ec1fce','Frozen source and contract')
    r=read(run/'report.json');need(r['schema']=='full151105_donor300_cached_self15_saved_analysis:1' and r['status']=='COMPLETED_ALL300_DONOR_CACHED_SENSOR_LABEL_JOIN_ONLY' and passed(r),'Completed producer')
    need(same(r['protocol'],pin(protocol)) and r['input_pins']==i and r['source_pins']==p['source_pins'] and read(run/'protocol.json')==p and read(run/'source_inventory.json')==p['source_pins'],'Exact external protocol/maps/copied semantics')
    label=r['outputs']['joined300_cached_sensor_labels.json'];need(checked(label).samefile(run/'joined300_cached_sensor_labels.json'),'Owned label route')
    h=read(checked(i['H1_report']));hs=read(checked(i['H1_saved_independent25']));ph=read(checked(i['physical_saved_independent1762']))
    need(passed(hs) and passed(ph) and hs['engineering_qualified'] is ph['engineering_qualified'] is True,'Positive prior25/1762')
    need(h['schema']=='full151105_state_prediction452_CPU_analysis:1' and h['status']=='COMPLETED_FIXED_A2_ALL452_STATE_ERROR_NOT_MODEL_QUALIFICATION' and same(h['config'],i['H1_config']) and same(hs['producer_report'],i['H1_report']) and same(hs['config'],i['H1_config']),'H1 source/independent lineage')
    need(same(h['input_pins']['reward_snapshots'],i['reward_snapshots']) and same(h['input_pins']['physical_saved_independent1762'],i['physical_saved_independent1762']) and Path(hs['run_dir']).samefile(checked(i['H1_report']).parent) and Path(ph['run_dir']).samefile(checked(i['reward_snapshots']).parent),'One H1/full05 route')
    need(checked(i['H1_identities']).samefile(checked(i['H1_report']).parent/'group_identities452.json') and same(hs['auditor'],p['source_pins']['H1_saved_auditor']) and same(ph['auditor'],p['source_pins']['serial_saved_auditor']),'Qualified actual source pin/identity path')
    for path in ('A0','A2'):
        m=read(checked(i['sensor_mapping_'+path]));need(m['middle']==[5*f+t for f in range(5) for t in (1,2)] and m['distal']==[5*f+3 for f in range(5)] and m['tips']==[5*f+4 for f in range(5)] and m['native_sensor']=='finger_collision' and m['state_phase']=='postintegration20ms' and m['sensor_phase']=='lastsolver preintegration15ms before Managerforward','Original actual mapping/phase')
        need(checked(i['sensor_mapping_'+path]).samefile(checked(i['reward_snapshots']).parent/('donor_'+path)/'sensor_label_mapping.json'),'Actual mapping path')
    identities=read(checked(i['H1_identities']));reward=[json.loads(line) for line in checked(i['reward_snapshots']).read_text(encoding='utf-8').splitlines()];need(type(identities) is list and len(identities)==len(reward)==452,'Complete452')
    lookup={}
    for row in reward:need(key(row) not in lookup,'Unique raw key');lookup[key(row)]=row
    need(all(type(g['group_id']) is int and g['group_id']==n for n,g in enumerate(identities)) and len({key(g) for g in identities})==452 and {key(g) for g in identities}==set(lookup),'All452 exact identity coverage')
    donors=[g for g in identities if g['mode']=='donor'];need([(g['path'],g['k']) for g in donors]==[(a,k) for a in ('A0','A2') for k in range(150)],'H1-ordered all300, never first300')
    wanted=[join(g,lookup[key(g)]) for g in donors];actual=read(checked(label));need(type(actual) is list and len(actual)==300,'All300 output rows')
    for a,b in zip(actual,wanted):need(set(a)==IDENTITY|{'contact15','raw_found25','label_time','state_time'} and a==b and all(f32(x)==f32(y) for x,y in zip(a['contact15'],b['contact15'])),'All16 fields and4500 original FP32 contact bytes')
    with zipfile.ZipFile(run/'execution_sources.zip') as z:
        expected={name+'/'+Path(v['path']).name:v for name,v in p['source_pins'].items()};need(set(z.namelist())==set(expected) and all(hashlib.sha256(z.read(n)).hexdigest()==v['sha256'] for n,v in expected.items()),'Exact source archive bytes')
    for v in [*i.values(),*p['source_pins'].values(),*r['outputs'].values()]:checked(v)
    need(not any(n in sys.modules for n in ('numpy','torch','mujoco','warp','dex_supermimic')),'No model/array/native imports')
    checks={s:True for s in ('exact8_parent_current_SHA_samefile_and_new_protocol','all452_unique_source_H1_keys_and_original_order','all300_donor_H1_integer_group_phase_terminal_join','all4500_raw25_stride5_native15_IEEEf32_bytes','all300_16field_label_output_original_values','source_ZIP_and_JSON_only_no_qualification_upgrade')}
    result={'schema':'full151105_donor300_cached_self15_saved_independent:1','status':'PASS_SAVED_ALL300_DONOR_RAW25_TO_NATIVE15_AND_H1_JOIN','engineering_qualified':True,'checks':checks,'passed':len(checks),'total':len(checks),'auditor':pin(__file__),'producer_report':pin(run/'report.json'),'protocol':pin(protocol),'label_JSON':label,'H1_identities':i['H1_identities'],'run_dir':str(run),'source_pins':{'auditor':pin(__file__),**p['source_pins']},'counts':{'source_reward_rows':452,'donor_rows':300,'native15_values':4500,'NPZ_Torch_model_FW_AD_physics':0},'scope':'Cached lastsolver~15ms sensor labels, POST20 state identity source; not post20 geometry/HO/human truth, new sampling, independent trajectories or model accuracy. H1 array correspondence remains prior25/1762 source evidence.'}
    write(out/'report.json',result);print(json.dumps({'report':pin(out/'report.json'),'passed':len(checks),'total':len(checks)}))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path);ap.add_argument('--producer-dir',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--self-check',action='store_true');a=ap.parse_args()
    if a.self_check:fixture(a.output.resolve());return
    need(a.protocol is not None and a.producer_dir is not None,'Saved paths required')
    try:audit(a.protocol.resolve(strict=True),a.producer_dir.resolve(strict=True),a.output.resolve())
    except Exception as error:
        if a.output.exists() and not(a.output/'failure.json').exists():write(a.output/'failure.json',{'error':repr(error),'auditor':pin(__file__),'automatic_retry':False})
        raise
if __name__=='__main__':main()
