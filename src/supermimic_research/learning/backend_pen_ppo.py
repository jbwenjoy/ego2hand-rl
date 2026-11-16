"""PPO/native save/load guards for explicit collision-backend identity."""
from .pen_support_ppo import SupportPenPPO
from supermimic_research.runtime.backend_pen_runtime import BackendSupportPenRuntime
from supermimic_research.runtime.explicit_collision_backend import active_backend


def flatten_contract(value,path=(('attr','_collision_backend_contract'),)):
    if isinstance(value,dict):
        return {p:v for k,child in value.items() for p,v in flatten_contract(child,path+(('item',k),)).items()}
    if isinstance(value,list):
        return {p:v for i,child in enumerate(value) for p,v in flatten_contract(child,path+(('item',i),)).items()}
    if type(value) not in [str,int,float,bool,type(None)]:raise ValueError('Backend identity must be a JSON scalar tree')
    return {path:value}


def require_saved_backend(state,contract):
    if state['metadata'].get('collision_backend')!=contract or state['native']['metadata'].get('collision_backend')!=contract:
        raise ValueError('Explicit matching collision backend checkpoint required;old native metadata cannot be rebound')
    prefix=(('attr','_collision_backend_contract'),)
    actual={p:v for p,v in state['native']['python']['scalars'].items() if p[:1]==prefix}
    expected=flatten_contract(contract)
    if actual.keys()!=expected.keys() or any(type(actual[p])!=type(v) or actual[p]!=v for p,v in expected.items()):
        raise ValueError('Owned backend identity scalars disagree with attested metadata')


class BackendSupportPenPPO(SupportPenPPO):
    def __init__(self,adapter,config,arm):
        if not isinstance(adapter,BackendSupportPenRuntime):raise ValueError('Explicit backend runtime required')
        super().__init__(adapter,config,arm)

    def preflight(self,state):
        backend=active_backend(self.adapter._collision_backend_contract);backend.verify(cold=True)
        require_saved_backend(state,self.adapter._collision_backend_contract)
        return super().preflight(state)

    def checkpoint(self,sources):
        active_backend(self.adapter._collision_backend_contract).verify(cold=True)
        return super().checkpoint(sources)
