"""Stdlib-only runtime identity fixture, no physical or reference inputs."""
import argparse,hashlib,json,os,sys,time
from datetime import datetime,timezone
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--run',action='store_true');a=ap.parse_args()
assert a.run
assert json.loads(a.plan.read_text())['schema']=='CPU_identity_fixture_plan:1'
a.output.mkdir(parents=True,exist_ok=False)
r={'status':'initializing','pid':os.getpid(),'parent_pid':os.getppid(),'started_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':hashlib.sha256(a.plan.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'sys_executable':sys.executable,'base_executable':sys._base_executable,'CPU_fixture_only':True,'physics_steps':0}
(a.output/'runtime.json').write_text(json.dumps(r,indent=2))
time.sleep(2.5)
r['status']='completed_CPU_identity_fixture';r['finished_utc']=datetime.now(timezone.utc).isoformat();r['Torch_imported']='torch' in sys.modules;r['Warp_imported']='warp' in sys.modules or 'mujoco_warp' in sys.modules
(a.output/'runtime.json').write_text(json.dumps(r,indent=2))
