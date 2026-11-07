"""Route the same frozen experiment from healthy paired100 to fixed1000."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
parent=H/'protocol_CPU_GPU_True_run1.json';p=read(parent);hp=H/'gpu_run1/health_0100.json';health=read(hp);cp=H/'gpu_run1/paired_checkpoint_0100.pt'
assert health['passed']==health['total']==15 and all(health['checks'].values()) and health['paired_round']==100 and health['shared_WM_updates']==200
assert pin(cp)==health['checkpoint'] and pin(cp)['sha256']=='9c773938665bffe4d52893cb8a9abfc53e4d4bf4d103a2beaf36383464f6b7eb'
review=read(H/'root_saved_first100_review_v2/report.json');assert review['passed']==review['total']==18 and all(review['checks'].values())
assert p['resume_checkpoint'] is None and p['resume_health'] is None and p['device']=='cuda:0'
for v in [*p['inputs'].values(),*p['sources'].values(),p['recipe_pin']]:assert pin(v['path'])==v
assert p['sources']['trainer']['sha256']=='a1bcf9f820e00d7da4ac515dd81d45d39a2c59abd07c85319d1346db384b2439'
p['CPU_run_authorized']=False;p['GPU_run_authorized']=False;p['output']=str(H/'gpu_resume1');p['resume_checkpoint']=pin(cp);p['resume_health']=pin(hp)
p['resume_lineage']=dict(parent_protocol=pin(parent),parent_payload_review=pin(H/'root_saved_first100_review_v2/report.json'),preparer=pin(__file__),previous_first100_pause_fulfilled=True,user_goal_resumed=True,
    scope='Same source/data/recipe, restore sharedWM+bothActor/Critic/target/norms+fiveoptimizers+CPU/CUDA/samplerRNG at paired100. Skip common100warmup. Additional900rounds to1000; no newexperiment or repeatedfirst100pause. Training counts lineage; setup/clone/load scopes separate.')
p.pop('root_review',None)
out=H/'protocol_CPU_GPUFalse_resume1.json'
with out.open('x',encoding='utf8') as f:json.dump(p,f,indent=2,allow_nan=False)
launcher=(H/'execute_GPU_once_v1.py').read_text(encoding='utf8')
launcher=launcher.replace("H/'protocol_CPU_GPU_True_run1.json'","H/'protocol_CPU_GPU_True_resume1.json'").replace("H/'process_run1'","H/'process_resume1'")
old='First actual matched offline A1/A2 learning: commonWM100 then1000paired rounds target. Fresh Actor/Critic1733, old WM5000 weights+frozen modelnormalizers with newoptimizers. First healthy paired100+ETA exits; Root archives and pausesGoal. No native online, no method/task benefit claim. Hardware samples whole-machine; torch peak is worker-owned.'
new='Resume same frozen paired100 experiment through1000, additional900paired rounds and no repeatedcommon100. Restore both policies/critics/targets, sharedWM, fiveAdam states and rawRNG. Priorfirst100pause fulfilled; no newexperiment or earlypause at500. No nativeonline or benefit qualification. Whole-machine hardware snapshots include setup/health.'
assert old in launcher;launcher=launcher.replace(old,new)
lp=H/'execute_GPU_resume_once_v1.py'
with lp.open('x',encoding='utf8',newline='\n') as f:f.write(launcher)
print(json.dumps(dict(resume_CPUFalse=pin(out),launcher=pin(lp))))
