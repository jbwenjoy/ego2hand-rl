"""Actual Newton context layout: padded gradient and absent Mgrad workspace."""
from .pen_solve_exit_observer import SolveExitGraphObserver as OriginalObserver


def expected_layout(nv,nv_pad,solver,worlds):
    if (nv,nv_pad,int(solver),worlds)!=(26,28,2,1):
        raise ValueError('Saved Hand2 W1 Newton layout26/28 required')
    return {'grad':(1,28),'Mgrad':(1,0),'search':(1,26)}


class SolveExitGraphObserverV2(OriginalObserver):
    def __init__(self,sim,**kwargs):
        super().__init__(sim,**kwargs)
        shapes=expected_layout(self.model.nv,self.model.nv_pad,self.model.opt.solver,self.data.nworld)
        with self.wp.ScopedDevice(sim.wp_device):
            for key in ['grad','Mgrad']:
                self.buffers['context.'+key]=self.wp.empty(shapes[key],dtype=float)

    def _emit(self,context):
        from .pen_solve_exit_kernels import count_solve_exit
        self.context_owners.append(context)
        for key,target in self.buffers.items():
            group,path=key.split('.',1);source=self.data if group=='data' else context
            for name in path.split('.'):source=getattr(source,name)
            if (source.shape,source.dtype,source.device)!=(target.shape,target.dtype,target.device):
                raise ValueError('Actual padded/empty context layout differs: '+key)
            if source.size:
                if source.ptr==target.ptr:raise ValueError('Nonempty observer buffer aliases source')
                self.wp.copy(target,source)
        self.wp.launch(count_solve_exit,dim=1,inputs=[self.executions])
