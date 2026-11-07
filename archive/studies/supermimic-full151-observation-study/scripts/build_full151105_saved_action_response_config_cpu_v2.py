"""Fresh saved-action-response CPU config only; never decode or analyze data.

Default writes nothing. --authorize writes one fresh config, not an execution.
FilePins is reused verbatim from the qualified state452 authorizer.
"""
from __future__ import annotations
import argparse,hashlib,importlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'protocols/full151105_saved_action_response_CPU_prereg_v1.json'
PROTOCOL_SHA='dc49c538cb996c46360917c3cf480f34ed410175b84af0e52ef82aad5be48221'
OWN={'consumer':ROOT/'src/full151_gradient/action_response452105_v2.py',
     'CLI':ROOT/'scripts/diagnose_full151_action_response_saved_cpu_v2.py',
     'checker':ROOT/'scripts/check_full151_action_response_saved_cpu_v2.py'}
CONFIG=ROOT/'configs/full151105-saved-action-response-CPU-v1.json'
OUTPUT=ROOT/'outputs/full151105_saved_action_response_CPU_v1'
REVIEW=ROOT/'outputs/full151105_saved_action_response_builder_source_only_v2/READY_source_only.json'
DEDUP_PREDECESSOR=ROOT/'scripts/build_full151105_state_prediction452_config_cpu_v1.py'
DEDUP_SHA='141504b6652871dfa50b8d082a56e91fa502d9bade51cff76cf82bf84b7a8500'
META_REVIEW=ROOT/'outputs/full151105_saved_action_response_metadata_preflight_v2/report.json'
META_REVIEW_SHA='892d982c688ccd2ad93f12dc0170af1632801b8c6ca1ee847c55732235467b0e'
READY_SCHEMA='full151105_saved_action_response_CPU_READY:1'
PROOF_SCHEMA='full151105_saved_action_response_CPU_contract:1'
STATUS='PASS_SYNTHETIC_RESPONSE_PAIR_GUARDS_NO_MODEL'


def need(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


class FilePins:
    """One physical identity is hashed once; different copies never merge.

    samestat/dev+ino is the filesystem identity used by samefile. Every merge
    is also explicitly checked with Path.samefile. A changed file or identity
    collision fails closed; SHA equality alone is never a deduplication key.
    """
    def __init__(self):
        self.cache={};self.source_map={};self.members={}

    def pin(self,path):
        p=Path(path);need(p.is_absolute() and p.is_file(),'Absolute existing file required: '+str(p))
        resolved=p.resolve(strict=True);st=resolved.stat()
        need(st.st_ino!=0,'Stable file identity unavailable: '+str(p))
        identity=(st.st_dev,st.st_ino);signature=(st.st_size,st.st_mtime_ns)
        if identity in self.cache:
            old=self.cache[identity]
            need(resolved.samefile(old['resolved']) and signature==old['signature'],'File changed or identity collision: '+str(p))
            sha=old['sha256']
        else:
            h=hashlib.sha256()
            with resolved.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            end=resolved.stat();need((end.st_dev,end.st_ino,end.st_size,end.st_mtime_ns)==(st.st_dev,st.st_ino,*signature),'File changed during hash')
            sha=h.hexdigest();self.cache[identity]={'resolved':resolved,'signature':signature,'sha256':sha}
        return {'path':str(p),'sha256':sha},identity

    def checked(self,value):
        need(type(value) is dict and set(value)=={'path','sha256'} and type(value['path']) is str and type(value['sha256']) is str and len(value['sha256'])==64,'Exact absolute path/SHA pin required')
        current,identity=self.pin(value['path']);need(current['sha256']==value['sha256'],'Pinned bytes changed: '+value['path'])
        return Path(value['path']),identity

    def same(self,left,right):
        a,ia=self.checked(left);b,ib=self.checked(right)
        return ia==ib and left['sha256']==right['sha256'] and a.samefile(b)

    def add(self,value):
        p,identity=self.checked(value);key=(*identity,value['sha256'])
        if key in self.members:
            need(p.samefile(Path(self.members[key]['path'])),'Never merge different physical copies')
            return
        self.members[key]=value
        self.source_map['file_%04d'%len(self.source_map)]=value  # First recorded spelling retained.

    def add_map(self,values):
        need(type(values) is dict and values,'Nonempty flat source map required')
        for value in values.values():self.add(value)


def positive(report):
    c=report.get('checks');need(type(c) is dict and c and all(type(x) is bool and x for x in c.values()),'Strict alltrue current CPU proof required')
    need(type(report.get('passed')) is int and type(report.get('total')) is int and report['passed']==report['total']==len(c),'Actual CPU counts differ')


def build(ready_path,authorize=False):
    need(type(authorize) is bool,'Explicit CPU authorization bool required')
    need(not any(k in sys.modules for k in ('torch','warp','mujoco','mujoco_warp')),'Fresh metadata process only')
    need(not any(v.startswith('PENDING_') for v in (READY_SCHEMA,PROOF_SCHEMA,STATUS)),'Wait for actual narrow current-source CPU READY')
    pins=FilePins();ready_path=Path(ready_path).resolve(strict=True);ready=read(ready_path)
    pins.checked(ready['cpu_report']);proof=read(ready['cpu_report']['path']);positive(proof)
    need(ready.get('schema')==READY_SCHEMA and proof.get('schema')==PROOF_SCHEMA and ready.get('status')==proof.get('status')==STATUS,'Wrong response-analysis CPU proof')
    need(ready['cpu_checks']=={'passed':proof['passed'],'total':proof['total']},'READY/proof count mismatch')
    need(ready.get('CPU_scientific_run_authorized') is False and ready.get('GPU_run_authorized') is False,'Synthetic preparation is not execution authority')
    for key in ('actual_saved_response_runs','actual_saved_array_reads','actual_model_FW_AD_MJB_physics'):
        need(type(proof.get(key)) is int and proof[key]==0,'CPU proof executed actual analysis: '+key)
    protocol_pin,_=pins.pin(PROTOCOL)
    need(protocol_pin['sha256']==PROTOCOL_SHA and pins.same(ready['protocol'],protocol_pin) and pins.same(proof['protocol'],protocol_pin),'Frozen saved-response protocol differs')
    need(set(ready['source_pins'])==set(proof['source_pins'])==set(OWN),'Exact three tested source keys required')
    for key,path in OWN.items():
        value,_=pins.pin(path)
        need(pins.same(ready['source_pins'][key],value) and pins.same(proof['source_pins'][key],value),'Executed CPU source drift: '+key)
    dependencies=ready.get('dependency_pins',{});proof_dependencies=proof.get('dependency_pins',{})
    need(type(dependencies) is dict and type(proof_dependencies) is dict and set(dependencies)==set(proof_dependencies),'CPU dependency keys differ')
    for key,value in dependencies.items():need(pins.same(value,proof_dependencies[key]),'CPU dependency proof mismatch: '+key)
    p=read(PROTOCOL)
    need(p['schema']=='full151105_saved_action_response_CPU_preregistration:1' and p['cpu_threads']==1,'Exact saved-only scope/threads required')
    need(all(p[k] is False for k in ('CPU_run_authorized','GPU_run_authorized','training_authorized')),'Preregistration is not execution authorization')
    for value in p['inputs'].values():pins.checked(value)
    meta_pin,_=pins.pin(META_REVIEW);need(meta_pin['sha256']==META_REVIEW_SHA,'Actual parent metadata proof changed')
    meta=read(META_REVIEW)
    need(meta['schema']=='full151105_saved_action_response_actual_parent_metadata:1' and meta['status']=='PASS_ACTUAL_PARENT_HASH_JSON_ONLY','Actual parent metadata-only join required')
    consumer_pin,_=pins.pin(OWN['consumer']);need(pins.same(meta['consumer'],consumer_pin) and pins.same(meta['protocol'],protocol_pin),'Parent metadata/source/protocol differs')
    for key in ('NPZ_decode_attempted','actual_response_decomposition','model_weight_load_FW_AD_MJB_Env_GPU_physics'):
        need(type(meta[key]) is int and meta[key]==0,'Parent metadata executed numeric analysis')
    need(meta['Torch_imported'] is False,'Parent metadata imported Torch')
    # Only the proved consumer pure JSON/hash/source API, never its run method.
    sys.path.insert(0,str(ROOT/'src'))
    core=importlib.import_module('full151_gradient.action_response452105_v2')
    actual,_=pins.pin(core.__file__);expected,_=pins.pin(OWN['consumer']);need(pins.same(actual,expected),'Unexpected consumer import')
    pins.add_map(core.required_sources(p));pins.add_map(p['source_pins']);pins.add_map(p['inputs'])
    pins.add_map(ready['source_pins'])
    if dependencies:pins.add_map(dependencies)
    pins.add(pins.pin(ready_path)[0]);pins.add(ready['cpu_report']);pins.add(protocol_pin);pins.add(meta_pin)
    dedup,_=pins.pin(DEDUP_PREDECESSOR);need(dedup['sha256']==DEDUP_SHA,'Qualified dedup predecessor changed');pins.add(dedup)
    source_record=read(REVIEW)
    need(source_record['schema']=='full151105_saved_action_response_builder_source_only:1' and source_record['AST_parse_only'] is True and source_record['builder_executed'] is False,'Builder source-only AST receipt required')
    selfpin,_=pins.pin(Path(__file__).resolve());need(pins.same(source_record['source_pins']['builder'],selfpin),'Builder changed since source-only freeze')
    pins.add(selfpin);pins.add(pins.pin(REVIEW)[0])
    need(not CONFIG.exists() and not OUTPUT.exists(),'Fresh config/output only, no overwrite/retry')
    c={'schema':core.SCHEMA,'role':core.ROLE,'protocol':protocol_pin,'output':str(OUTPUT),'CPU_run_authorized':authorize,'cpu_threads':1,'source_pins':pins.source_map,'cpu_ready':pins.pin(ready_path)[0]}
    need(set(c)==core.KEYS and c['schema']==p['runplan_schema'] and c['role']==p['role'],'Actual eight-key API differs')
    need(ready['api']['schema']==core.SCHEMA and ready['api']['role']==core.ROLE and ready['api']['keys']==sorted(core.KEYS),'READY API differs')
    api_module,_=pins.pin(ready['api']['module']);api_entry,_=pins.pin(ready['api']['entry']);own_entry,_=pins.pin(OWN['CLI'])
    need(pins.same(api_module,actual) and pins.same(api_entry,own_entry),'READY consumer/CLI route differs')
    core.validate_header(c)
    need(not any(k in sys.modules for k in ('torch','warp','mujoco','mujoco_warp')),'Metadata imported backend')
    for value in pins.source_map.values():pins.checked(value)
    if authorize:
        with CONFIG.open('x',encoding='utf-8',newline='\n') as out:json.dump(c,out,indent=2);out.write('\n')
    return {'schema':'full151105_saved_action_response_builder_declaration:1','status':'CREATED_EXPLICITLY_AUTHORIZED_CPU_CONFIG_ONLY' if authorize else 'DECLARED_NO_CONFIG_WRITTEN',
        'config':pins.pin(CONFIG)[0] if authorize else {'path':str(CONFIG),'written':False},'output':str(OUTPUT),'protocol':protocol_pin,'cpu_ready':pins.pin(ready_path)[0],
        'CPU_run_authorized':authorize,'unique_source_file_count':len(pins.source_map),'source_dedup':'same physical file identity+SHA, not SHA-only; semantic input aliases retained in protocol',
        'Torch_imported':False,'Warp_imported':False,'prediction_NPZ_opened':0,'responses_or_secants_computed':0,'models_FW_AD_MJB_Env_physics_GPU':0}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ready',required=True,type=Path);p.add_argument('--authorize',action='store_true',help='Create a fresh CPU-authorized config only, never execute analysis.')
    a=p.parse_args();print(json.dumps(build(a.ready,a.authorize),indent=2))


if __name__=='__main__':main()
