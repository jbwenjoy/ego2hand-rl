"""Label-only JSON qualification; standard library only, no weights or arrays."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time
import zipfile

ROOT=Path('D:/workspace/supermimic-full151-observation-study')
CONTRACT=ROOT/'docs/donor300_cached_self15_label_contract_v1.md'
CONTRACT_SHA='4f91fc4400c2481d53554e8510d5cfc31fd97a9830d758d2ad84eaf806ec1fce'
SCHEMA='full151105_donor300_cached_self15_saved_CPU_protocol:1'
PREFIX='full151105_donor300_cached_self15_saved_CPU_'
INPUTS={'H1_config','H1_report','H1_identities','H1_saved_independent25','reward_snapshots','sensor_mapping_A0','sensor_mapping_A2','physical_saved_independent1762'}
MIDDLE=[1,2,6,7,11,12,16,17,21,22]
DISTAL=[3,8,13,18,23]
TIPS=[4,9,14,19,24]
IDENTITY={'group_id','path','mode','k','branch_id','source_frame','terminal_observation','valid_next_policy_decision','no_u150','first_physical_pre_index','last_physical_index','post20_GPU_row_index'}

def require(value,message):
    if not value:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def pin(path):return {'path':str(Path(path).resolve(strict=True)),'sha256':sha(path)}
def checked(value):
    require(type(value) is dict and set(value)=={'path','sha256'} and type(value['path']) is str and Path(value['path']).is_absolute() and type(value['sha256']) is str and len(value['sha256'])==64,'Exact absolute SHA pin')
    p=Path(value['path']).resolve(strict=True);require(sha(p)==value['sha256'],'Saved pin changed: '+str(p));return p
def same(a,b):return checked(a).samefile(checked(b)) and a['sha256']==b['sha256']
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
def allchecks(x):require(x['passed']==x['total']==len(x['checks']) and x['checks'] and all(v is True for v in x['checks'].values()),'All parent saved checks passed')
def no_backend():require(not any(k in sys.modules for k in ('numpy','torch','mujoco','warp')),'Standard-library JSON only')

def validate_mapping(m):
    require(m['middle']==MIDDLE and m['distal']==DISTAL and m['tips']==TIPS and m['native_sensor']=='finger_collision' and m['state_phase']=='postintegration20ms' and m['sensor_phase']=='lastsolver preintegration15ms before Managerforward','Fixed raw25 to cached15 mapping and phases')

def identity_key(r):
    require(r['path'] in ('A0','A2') and type(r['k']) is int and 0<=r['k']<150,'Known path and integer k')
    require(r['mode'] in ('donor','baseline','direction') and (r['branch_id'] is None if r['mode']=='donor' else type(r['branch_id']) is int),'Donor null/branch integer identity')
    return r['path'],r['mode'],r['k'],r['branch_id']

def validate_donor(g):
    require(set(g)==IDENTITY and g['mode']=='donor' and g['branch_id'] is None,'Exact H1 donor identity schema')
    path,_,k,_=identity_key(g);gid=k+(226 if path=='A2' else 0)
    require(all(type(g[key]) is int for key in ('group_id','source_frame','terminal_observation','first_physical_pre_index','last_physical_index','post20_GPU_row_index')),'H1 strict integer identity fields')
    require(g['group_id']==gid and g['source_frame']==200+k and g['terminal_observation']==k+1 and g['first_physical_pre_index']==4*gid and g['last_physical_index']==4*gid+3 and g['post20_GPU_row_index']==1808+gid and g['valid_next_policy_decision'] is (k<149) and g['no_u150'] is True,'H1 donor group/endpoint phase binding')

def join_one(g,r,m):
    validate_donor(g);validate_mapping(m);require(identity_key(g)==identity_key(r),'Exact donor identity join')
    for key in ('source_frame','terminal_observation'):
        require(type(r[key]) is int and r[key]==g[key],'Reward integer phase '+key)
    require(r['valid_next_policy_decision'] is g['valid_next_policy_decision'] and r['no_u150'] is True and r['label_time']=='lastsolver_preintegration~kdt+.015' and r['state_time']=='postintegration_kdt+.020' and r['capture']=='after_original_reward_compute_before_Manager_forward','Cached15/POST20 native phase, no u150')
    raw=r['raw_found'];contact=r['contact']
    require(type(raw) is list and len(raw)==1 and type(raw[0]) is list and len(raw[0])==25 and all(type(v) in (int,float,bool) and math.isfinite(v) and v>=0 and int(v)==v for v in raw[0]),'Finite nonnegative integer raw25 counts')
    require(type(contact) is list and len(contact)==15 and all(type(v) in (int,float) and math.isfinite(v) and v in (0,1) for v in contact),'Exact finite binary15')
    found=[v>0 for v in raw[0]];reconstructed=[float(found[j]) for j in MIDDLE]+[float(found[d] or found[t]) for d,t in zip(DISTAL,TIPS)]
    require(all(struct.pack('<f',float(a))==struct.pack('<f',b) for a,b in zip(contact,reconstructed)),'Raw25 reconstructs all15 FP32 bytes')
    return {**g,'contact15':contact,'raw_found25':raw[0],'label_time':r['label_time'],'state_time':r['state_time']}

def join_all(identities,reward,mappings):
    require(type(identities) is list and len(identities)==452 and type(reward) is list and len(reward)==452,'All452 source identity and cached reward rows')
    lookup={};identity_keys=[]
    for r in reward:
        key=identity_key(r);require(key not in lookup,'Duplicate original reward identity');lookup[key]=r
    require([g['group_id'] for g in identities]==list(range(452)),'Exact H1 global original row order')
    for g in identities:identity_keys.append(identity_key(g))
    require(len(set(identity_keys))==452 and set(identity_keys)==set(lookup),'Complete unique452 source identity coverage')
    donors=[g for g in identities if g['mode']=='donor']
    require([(g['path'],g['k']) for g in donors]==[(path,k) for path in ('A0','A2') for k in range(150)],'Exactly300 donor identity rows in H1 order')
    return [join_one(g,lookup[identity_key(g)],mappings[g['path']]) for g in donors]

def metadata(protocol_path,output=None):
    no_backend();p=read(protocol_path)
    require(p['schema']==SCHEMA and p['role']=='all300_donor_cached_native15_sensor_label_join_only' and p['output_prefix']==PREFIX,'New label-only namespace')
    require(type(p['CPU_run_authorized']) is bool and p['training_authorized'] is p['GPU_run_authorized'] is False and p['automatic_retry'] is False and p['expected_rows']==300 and p['no_model_or_NPZ_reads'] is True,'Explicit JSON-only bounded operation')
    i=p['inputs'];require(set(i)==INPUTS,'Exact8 source semantic inputs')
    for v in [*i.values(),*p['source_pins'].values()]:checked(v)
    require(same(p['source_contract'],{'path':str(CONTRACT),'sha256':CONTRACT_SHA}) and same(p['source_pins']['producer'],pin(__file__)),'Current producer and frozen scientific contract')
    h=read(checked(i['H1_report']));hs=read(checked(i['H1_saved_independent25']));allchecks(hs)
    require(h['schema']=='full151105_state_prediction452_CPU_analysis:1' and h['status']=='COMPLETED_FIXED_A2_ALL452_STATE_ERROR_NOT_MODEL_QUALIFICATION' and same(h['config'],i['H1_config']),'Qualified H1 native target lineage')
    require(hs['schema']=='full151105_state_prediction452_saved_independent:1' and hs['engineering_qualified'] is True and same(hs['producer_report'],i['H1_report']) and same(hs['config'],i['H1_config']) and Path(hs['run_dir']).samefile(checked(i['H1_report']).parent),'Qualified H1 independent25 binding')
    require(checked(i['H1_identities']).samefile(checked(i['H1_report']).parent/'group_identities452.json'),'Exact H1 original identity route')
    require(same(h['input_pins']['reward_snapshots'],i['reward_snapshots']) and same(h['input_pins']['physical_saved_independent1762'],i['physical_saved_independent1762']),'Same actual H1/full05 cached source')
    physical=read(checked(i['physical_saved_independent1762']));allchecks(physical)
    require(physical['schema']=='full151105_full_serial_saved_independent:1' and physical['engineering_qualified'] is True and Path(physical['run_dir']).samefile(checked(i['reward_snapshots']).parent) and same(physical['auditor'],p['source_pins']['serial_saved_auditor']),'Original saved1762 includes300 donor raw25 reconstruction')
    require(same(hs['auditor'],p['source_pins']['H1_saved_auditor']),'Pinned H1 independent source')
    for path in ('A0','A2'):
        require(checked(i['sensor_mapping_'+path]).samefile(checked(i['reward_snapshots']).parent/('donor_'+path)/'sensor_label_mapping.json'),'Same original per-path mapping route');validate_mapping(read(checked(i['sensor_mapping_'+path])))
    out=None if output is None else Path(output).resolve()
    if out is not None:require(out.parent.samefile(ROOT/'outputs') and out.name.startswith(PREFIX) and not out.exists(),'Fresh source-label output only')
    return p,out

def synthetic():
    no_backend();m={'middle':MIDDLE,'distal':DISTAL,'tips':TIPS,'native_sensor':'finger_collision','state_phase':'postintegration20ms','sensor_phase':'lastsolver preintegration15ms before Managerforward'}
    g={'group_id':375,'path':'A2','mode':'donor','k':149,'branch_id':None,'source_frame':349,'terminal_observation':150,'valid_next_policy_decision':False,'no_u150':True,'first_physical_pre_index':1500,'last_physical_index':1503,'post20_GPU_row_index':2183}
    raw=[0.]*25;raw[2]=2.;raw[4]=1.;r={**g,'contact':[0.,1.]+[0.]*8+[1.]+[0.]*4,'raw_found':[raw],'label_time':'lastsolver_preintegration~kdt+.015','state_time':'postintegration_kdt+.020','capture':'after_original_reward_compute_before_Manager_forward'}
    joined=join_one(g,r,m);rejected=0
    for field,value in [('terminal_observation',149),('contact',[0.]*15)]:
        bad=copy.deepcopy(r);bad[field]=value
        try:join_one(g,bad,m)
        except ValueError:rejected+=1
    require(joined['group_id']==375 and joined['raw_found25'][2]==2 and rejected==2,'Terminal/cached-label synthetic positive and two rejection cases')
    return {'positive_terminal_mapping':True,'reject_phase':True,'reject_label':True,'actual_saved_inputs_read':0}

def execute(protocol_path,output):
    started=time.perf_counter();protocol_pin=pin(protocol_path);p,out=metadata(protocol_path,output);require(p['CPU_run_authorized'] is True,'Root actual saved JSON extraction authorization');out.mkdir()
    write(out/'protocol.json',p);write(out/'source_inventory.json',p['source_pins'])
    with zipfile.ZipFile(out/'execution_sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for key,value in p['source_pins'].items():z.write(checked(value),key+'/'+checked(value).name)
    try:
        i=p['inputs'];identities=read(checked(i['H1_identities']));reward=[json.loads(line) for line in checked(i['reward_snapshots']).read_text(encoding='utf-8').splitlines()]
        joined=join_all(identities,reward,{path:read(checked(i['sensor_mapping_'+path])) for path in ('A0','A2')});write(out/'joined300_cached_sensor_labels.json',joined)
        for value in [*i.values(),*p['source_pins'].values()]:checked(value)
        require(same(protocol_pin,pin(protocol_path)),'Run protocol unchanged');no_backend();checks={'qualified_H1_and_full05_source_samefile_SHA':True,'unique_all452_source_identity_coverage':True,'exact300_donor_order_and_terminal_phase':True,'all300_raw25_to15_FP32_bytes':True,'JSON_only_no_NPZ_Torch_model_native':True}
        report={'schema':'full151105_donor300_cached_self15_saved_analysis:1','status':'COMPLETED_ALL300_DONOR_CACHED_SENSOR_LABEL_JOIN_ONLY','protocol':protocol_pin,'source_contract':pin(CONTRACT),'input_pins':i,'source_pins':p['source_pins'],'checks':checks,'passed':len(checks),'total':len(checks),'outputs':{'joined300_cached_sensor_labels.json':pin(out/'joined300_cached_sensor_labels.json')},'rows':300,'cached_sensor_values':4500,'counts':{'NPZ_reads_Torch_loads_model_FW_AD_physics_GPU':0},'scope':'Cached lastsolver~15ms native sensor labels only; not POST20 geometry/HO/human, no independent trajectories.','elapsed_seconds':time.perf_counter()-started}
        write(out/'report.json',report);return {'status':report['status'],'report':pin(out/'report.json'),'labels':report['outputs']['joined300_cached_sensor_labels.json']}
    except BaseException as error:
        write(out/'failure.json',{'schema':'full151105_donor300_cached_self15_saved_failure:1','error':repr(error),'automatic_retry':False});raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol');parser.add_argument('--output');parser.add_argument('--execute',action='store_true');parser.add_argument('--synthetic',action='store_true');a=parser.parse_args()
    if a.synthetic:print(json.dumps(synthetic()));return
    require(a.protocol is not None,'Protocol required')
    if a.execute:require(a.output is not None,'Fresh output required');value=execute(Path(a.protocol).resolve(strict=True),a.output)
    else:p,_=metadata(Path(a.protocol).resolve(strict=True));value={'schema':SCHEMA,'status':'METADATA_ONLY_NO_LABEL_EXTRACTION','actual_rows_read':0}
    print(json.dumps(value))

if __name__=='__main__':main()
