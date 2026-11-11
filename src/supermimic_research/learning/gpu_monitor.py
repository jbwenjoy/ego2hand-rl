"""Bounded device-wide telemetry in a thread, outside CUDA update synchronization."""
import subprocess,threading,time


class GPUMonitor:
    def __init__(self):
        self.samples=[];self.errors=[];self.stop_event=threading.Event();self.thread=threading.Thread(target=self._loop,daemon=True)
    def _loop(self):
        while not self.stop_event.is_set():
            start=time.perf_counter()
            try:
                result=subprocess.run(['nvidia-smi','--id=0','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                self.samples.append(dict(start=start,end=time.perf_counter(),values=[float(s.strip()) for s in result.stdout.strip().split(',')]))
            except (OSError,ValueError,subprocess.SubprocessError) as error:self.errors.append(str(error))
            self.stop_event.wait(.2)
    def start(self):self.thread.start()
    def stop(self):self.stop_event.set();self.thread.join(timeout=6)
