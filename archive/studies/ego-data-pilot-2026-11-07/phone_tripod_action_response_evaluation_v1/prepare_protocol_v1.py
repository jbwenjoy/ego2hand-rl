"""CPUFalse preparation/final binding only, no Torch/native/array payload."""
from pathlib import Path
import argparse,copy,hashlib,json,runpy,sys
H=Path(__file__).resolve().parent;P=H.parent
def pin(path):
    p=Path(path).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,v):
    with Path(path).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
def prepare():
    m=runpy.run_path(str(H/'evaluate_WM_action_response_v1.py'));seed=read(P/'phone_tripod_supermimic_v1/protocol_CPUFalse_v1.json');warm=P/'phone_geometry_momentum_wm_v1/cpu_resume1/result.json';wr=read(warm)
    sources=dict(seed['sources']);sources['paired_trainer']=sources.pop('trainer');sources.update(statistics=pin(P/'phone_geometry_momentum_evaluation_v1/evaluate_fixed5000.py'),probe_runner=pin(P/'phone_tripod_action_response_v1/run_native_v1.py'),probe_common=pin(P/'phone_tripod_action_response_v1/common_v1.py'))
    review=P/'phone_tripod_action_response_v1/saved_integrity_review_v1/report.json'
    p=dict(schema=m['SCHEMA'],source=pin(H/'evaluate_WM_action_response_v1.py'),sources=sources,rules=m['RULES'],CPU_run_authorized=False,GPU_run_authorized=False,training_authorized=False,output=str(H/'completed_v1'),
        inputs=dict(warm_result=pin(warm),warm_health=wr['health'],warm_training_protocol=wr['protocol'],final_training_protocol=None,final_result=None,final_health=None,final_terminal=None,probe_report=pin(P/'phone_tripod_action_response_v1/native_completed_v1/report.json'),probe_terminal=pin(P/'phone_tripod_action_response_v1/process_run1/terminal.json'),probe_saved_review=pin(review) if review.exists() else None,geometry_catalog=seed['inputs']['geometry_catalog'],task_bundle=seed['inputs']['task_bundle'],model=read(seed['inputs']['task_bundle']['path'])['model']),
        role='fixed_two_WM_heldout_tripod_H1_response_and_model_only_AD_FD_diagnostic',
        scope='Known local MONITOR source states, permanentlyheldout newabsolutecontrol responses. Someactions exceed currentActor+EMA range (retain descriptor), not policygradient proof. Nofit/selection/newnative/ego claim. Eightcontext20axes/signs/unavailable/partial allretained. Marginalcoverage22 never fabricates simultaneoustriple. ModelFD.0001 differs fromnativefinite.01 and is selfconsistency only.',
        preparation=pin(__file__))
    write(H/'protocol_CPUFalse_v1.json',p);m['metadata_preflight'](H/'protocol_CPUFalse_v1.json');assert 'torch' not in sys.modules and 'mujoco' not in sys.modules
    write(H/'metadata_CPUFalse_v1.json',dict(schema='phone_tripod_action_response_evaluation_metadata:1',status='PASS_METADATA_PENDING_FIXEDFINAL1000',protocol=pin(H/'protocol_CPUFalse_v1.json'),source=p['source'],Torch_imported=False,MuJoCo_imported=False,NPZ_NN_AD_native_calls=0))
    print(json.dumps(dict(protocol=pin(H/'protocol_CPUFalse_v1.json'),source=p['source'],metadata=pin(H/'metadata_CPUFalse_v1.json'))))
def bind(args):
    m=runpy.run_path(str(H/'evaluate_WM_action_response_v1.py'));p=copy.deepcopy(read(args.base))
    for key,path in [('final_training_protocol',args.final_protocol),('final_result',args.final_result),('final_health',args.final_health),('final_terminal',args.final_terminal),('probe_saved_review',args.probe_saved_review)]:p['inputs'][key]=pin(path)
    p['CPU_run_authorized']=False;p['output']=str((H/args.output_name).resolve());write(args.protocol_output,p);m['metadata_preflight'](args.protocol_output)
    assert 'torch' not in sys.modules and 'mujoco' not in sys.modules
    print(json.dumps(dict(bound_CPUFalse=pin(args.protocol_output),output=p['output'])))
if __name__=='__main__':
    a=argparse.ArgumentParser();g=a.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--bind',action='store_true');a.add_argument('--base',type=Path,default=H/'protocol_CPUFalse_v1.json');a.add_argument('--final-protocol',type=Path);a.add_argument('--final-result',type=Path);a.add_argument('--final-health',type=Path);a.add_argument('--final-terminal',type=Path);a.add_argument('--probe-saved-review',type=Path);a.add_argument('--output-name',default='completed_v1');a.add_argument('--protocol-output',type=Path);args=a.parse_args()
    if args.prepare:prepare()
    else:bind(args)
