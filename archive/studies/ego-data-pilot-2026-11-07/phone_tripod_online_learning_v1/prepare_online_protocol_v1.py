"""Prepare/bind SOURCE CPUFalse online protocol without Torch or arrays.

--prepare creates the initial pending-final template once. --bind fills actual
final root artifacts (or own resume), retains all execution flags False.
Root separately reviews and authorizes a fresh actual protocol.
"""
from pathlib import Path
import argparse,copy,hashlib,json,runpy,sys
H=Path(__file__).resolve().parent;P=H.parent
def pin(path):
    p=Path(path).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)
def runner():return runpy.run_path(str(H/'train_online_matched_v1.py'))
def prepare():
    m=runner();seed=read(P/'phone_tripod_supermimic_v1/protocol_CPUFalse_v1.json');template=P/'phone_tripod_online_collection_v1/protocol_CPUFalse_v3.json';c=read(template)
    sources=dict(c['sources']);sources['trainer']=pin(H/'train_online_matched_v1.py')
    recipe=dict(schema='phone_tripod_online_matched_learning_recipe:1',recipe=m['RECIPE'],first_learning_checkpoint='cycle1/update10; health+ETA exits; target3cycles100 remains incomplete',counter_semantics='online-only Adam/update counts, explicit offline lineage1000 paired/WM1100; fresh5 optimizers reset only atnewstage',scope='Actualclosedloopcycle source, not executed; no policy benefit/ego generalization or physical-gradient qualification')
    write(H/'recipe_v1.json',recipe)
    p=dict(schema=m['SCHEMA'],role='new3cycle_native_common_replay_sharedWM_matched_A1_A2_learning',CPU_run_authorized=False,GPU_run_authorized=False,native_collection_authorized=False,device='cuda:0',output=str(H/'gpu_run1'),recipe=m['RECIPE'],recipe_pin=pin(H/'recipe_v1.json'),sources=sources,
        inputs=dict(training_seed_protocol=pin(P/'phone_tripod_supermimic_v1/protocol_CPUFalse_v1.json'),root_training_protocol=None,root_result=None,root_health=None,root_terminal=None,collector_template=pin(template),task_bundle=seed['inputs']['task_bundle'],geometry_catalog=seed['inputs']['geometry_catalog']),resume_checkpoint=None,resume_health=None,preparation=pin(__file__),
        data_scope='Native newONLINE_TRAIN commonpool only plusoffline originalTRAIN684 andtripodTRAIN153 WMreplay. No oldmonitor/CHECK/freshfinal/TEST opens or fitting. Botharms useallonlinepool/per-mode reward, same rowbatch; allpartial/failure/alias retained.',
        physical_scope='Fixednativeparameters/scene/wrist, noaid, unchangedReader safety andPOSTH taskterminal. Exploration.95Actor+.05namedprofile.0.4s acquisition isnot2s learned-policy evaluation.',
        resume_scope='No recollection of completedcycle; dataset/protocol/raw receipts pinchain, stable(poolSHA,row/episode)identity mapping, previousprocess fixedlogprefixbytes retained; restore WM/allarms/target/norm/5Adam/CPUandselectedCUDA/sampleRNG')
    write(H/'protocol_CPUFalse_v1.json',p);m['metadata_preflight'](H/'protocol_CPUFalse_v1.json')
    assert 'torch' not in sys.modules and 'mujoco' not in sys.modules
    write(H/'metadata_CPUFalse_v1.json',dict(schema='phone_tripod_online_learning_metadata:1',status='PASS_SOURCE_METADATA_PENDING_FINAL1000',protocol=pin(H/'protocol_CPUFalse_v1.json'),source=sources['trainer'],Torch_imported=False,MuJoCo_imported=False,NN_native_array_payload_calls=0))
    print(json.dumps(dict(protocol=pin(H/'protocol_CPUFalse_v1.json'),source=sources['trainer'],recipe=pin(H/'recipe_v1.json'),metadata=pin(H/'metadata_CPUFalse_v1.json'))))
def bind(args):
    m=runner();p=copy.deepcopy(read(args.base))
    for key,arg in [('root_training_protocol',args.root_protocol),('root_result',args.root_result),('root_health',args.root_health),('root_terminal',args.root_terminal)]:p['inputs'][key]=pin(arg)
    p['output']=str((H/args.output_name).resolve());p.update(CPU_run_authorized=False,GPU_run_authorized=False,native_collection_authorized=False)
    if args.resume_checkpoint is not None:
        if args.resume_health is None:raise ValueError('Resume requires actual own health')
        p['resume_checkpoint']=pin(args.resume_checkpoint);p['resume_health']=pin(args.resume_health)
    write(args.protocol_output,p);m['metadata_preflight'](args.protocol_output)
    assert 'torch' not in sys.modules and 'mujoco' not in sys.modules
    print(json.dumps(dict(bound_CPUFalse=pin(args.protocol_output),newoutput=p['output'],actual_run_authorized=False)))
if __name__=='__main__':
    a=argparse.ArgumentParser();g=a.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--bind',action='store_true')
    a.add_argument('--base',type=Path,default=H/'protocol_CPUFalse_v1.json');a.add_argument('--root-protocol',type=Path);a.add_argument('--root-result',type=Path);a.add_argument('--root-health',type=Path);a.add_argument('--root-terminal',type=Path);a.add_argument('--output-name',default='gpu_run1');a.add_argument('--protocol-output',type=Path);a.add_argument('--resume-checkpoint',type=Path);a.add_argument('--resume-health',type=Path);args=a.parse_args()
    if args.prepare:prepare()
    else:bind(args)
