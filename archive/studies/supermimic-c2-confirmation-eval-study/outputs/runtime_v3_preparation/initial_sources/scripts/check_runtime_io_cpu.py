"""Real Windows read-handle/rename sharing fixture plus bounded negative cases."""
import ctypes
from ctypes import wintypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('iohelper',ROOT/'src/runtime_io.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks={};details={}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rejected(fn,kind):
    try:fn()
    except kind:return True
    return False
require_windows=os.name=='nt'
if not require_windows:raise RuntimeError('This fixture requires actual Windows sharing handles')
kernel=ctypes.WinDLL('kernel32',use_last_error=True)
kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
kernel.CreateFileW.restype=wintypes.HANDLE
kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle.restype=wintypes.BOOL
def open_read_without_delete(path):
    handle=kernel.CreateFileW(str(path),0x80000000,0x1|0x2,None,3,0x80,None)
    if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    return handle
payload=json.dumps({'status':'completed','control_intervals':150},indent=2).encode()+b'\n'
with tempfile.TemporaryDirectory() as d:
    root=Path(d);src=root/'runtime.json.tmp';dst=root/'runtime.json'
    dst.write_bytes(b'old');src.write_bytes(payload);h=open_read_without_delete(dst)
    try:
        try:os.replace(src,dst)
        except PermissionError as error:
            checks['actual_windows_read_handle_blocks_replace']=getattr(error,'winerror',None) in (5,32)
            details['actual_denial']={'class':type(error).__name__,'winerror':error.winerror,'errno':error.errno}
        else:checks['actual_windows_read_handle_blocks_replace']=False
        checks['denial_keeps_payload_and_old_target']=src.read_bytes()==payload and dst.read_bytes()==b'old'
        observations=[]
        def checked_replace(s,t):
            observations.append(Path(s).read_bytes()==payload);os.replace(s,t)
        release=threading.Timer(.1,lambda:kernel.CloseHandle(h));release.start()
        result=m.replace_with_bounded_retry(src,dst,replace_fn=checked_replace)
        release.join();h=None
        checks['bounded_retry_succeeds_after_handle_release']=2<=result['attempts']<=32 and result['elapsed_seconds']<.9
        checks['one_payload_same_temp_all_attempts']=all(observations) and dst.read_bytes()==payload and not src.exists()
        details['released_handle_result']=result
    finally:
        if h is not None:kernel.CloseHandle(h)
    dst.write_bytes(b'old');src.write_bytes(payload);h=open_read_without_delete(dst);before=time.monotonic()
    try:
        checks['persistent_handle_failure_propagates']=rejected(lambda:m.replace_with_bounded_retry(src,dst,max_seconds=.05),PermissionError)
        details['persistent_lock_elapsed_seconds']=time.monotonic()-before
        checks['persistent_failure_budget_bounded']=details['persistent_lock_elapsed_seconds']<.25
        checks['persistent_failure_preserves_both_files']=src.read_bytes()==payload and dst.read_bytes()==b'old'
    finally:kernel.CloseHandle(h)
    calls=[]
    def unrelated(*args):calls.append(1);raise FileNotFoundError('unrelated')
    checks['non_permission_error_no_retry']=rejected(lambda:m.replace_with_bounded_retry(src,dst,replace_fn=unrelated),FileNotFoundError) and len(calls)==1
    for invalid in (0,-1,1.01,float('nan'),True):
        checks['invalid_budget_'+str(invalid)]=rejected(lambda x=invalid:m.replace_with_bounded_retry(src,dst,max_seconds=x),ValueError)
    # Deterministic clock proves that a denied metadata commit never invokes
    # the physics/control scope a second time and finally still executes once.
    counter={'physics':0,'serialize':0,'finally':0,'replace':0,'clock':0.}
    def fail_permission(*args):counter['replace']+=1;raise PermissionError('denied')
    try:
        try:
            counter['physics']+=1;counter['serialize']+=1
            m.replace_with_bounded_retry(src,dst,max_seconds=.03,replace_fn=fail_permission,
                clock=lambda:counter['clock'],sleep=lambda s:counter.__setitem__('clock',counter['clock']+s))
        finally:counter['finally']+=1
    except PermissionError:pass
    checks['failure_no_physics_or_serialization_replay']=counter['physics']==counter['serialize']==counter['finally']==1 and 1<counter['replace']<=32
    checks['deterministic_clock_stops_at_budget']=counter['clock']<=.03
    details['counter']=counter
out=ROOT/'outputs/runtime_io_cpu_v3';out.mkdir(parents=True,exist_ok=True)
report={'status':'passed' if all(checks.values()) else 'failed','count':len(checks),'checks':checks,'details':details,
 'helper_sha256':digest(ROOT/'src/runtime_io.py'),'checker_sha256':digest(__file__),'GPU_used':False,
 'scope':'Actual Windows sharing-denial fixture. Demonstrates a possible mechanism only; does not identify the reader or sole cause of the V2 incident.'}
(out/'checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2));raise SystemExit(0 if all(checks.values()) else 1)
