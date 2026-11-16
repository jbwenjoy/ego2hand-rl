import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from supermimic_research.artifacts import file_pin
from supermimic_research.runtime.pen_solve_exit_observer import SolveExitHook, SolveExitGraphObserver, require_closed_full54


class SolveExitBinding(unittest.TestCase):
    def make(self):
        model,data,ctx=object(),object(),object();events=[]
        def original(m,d,c,compact=False):
            self.assertIs(m,model);self.assertIs(d,data);events.append('original');return 17
        module=SimpleNamespace(_solve=original)
        return module,model,data,ctx,events,original

    def test_original_precedes_copy_and_return_is_preserved(self):
        mod,m,d,c,events,original=self.make()
        with SolveExitHook(mod,m,d,lambda ctx:events.append('copy')) as hook:
            self.assertEqual(mod._solve(m,d,c),17)
        self.assertEqual(events,['original','copy']);self.assertEqual(hook.calls,1);self.assertIs(mod._solve,original)

    def test_original_failure_does_not_copy_and_restores(self):
        mod,m,d,c,events,_=self.make()
        def fail(*args,**kwargs):raise RuntimeError('original failed')
        mod._solve=fail
        with self.assertRaises(RuntimeError):
            with SolveExitHook(mod,m,d,lambda ctx:events.append('copy')):mod._solve(m,d,c)
        self.assertIs(mod._solve,fail);self.assertEqual(events,[])

    def test_copy_failure_restores(self):
        mod,m,d,c,events,original=self.make()
        def copy(ctx):raise RuntimeError('copy failed')
        with self.assertRaises(RuntimeError):
            with SolveExitHook(mod,m,d,copy):mod._solve(m,d,c)
        self.assertIs(mod._solve,original);self.assertEqual(events,['original'])

    def test_wrong_model_and_compact_reject_before_execution(self):
        mod,m,d,c,events,original=self.make()
        with SolveExitHook(mod,m,d,lambda ctx:None):
            for args in [(object(),d,c,False),(m,object(),c,False),(m,d,c,True)]:
                with self.assertRaises(ValueError):mod._solve(*args)
        self.assertEqual(events,[]);self.assertIs(mod._solve,original)

    def test_nested_and_changed_bindings_restore(self):
        mod,m,d,c,events,original=self.make()
        with self.assertRaises(ValueError):
            with SolveExitHook(mod,m,d,lambda ctx:None):
                with self.assertRaises(ValueError):
                    with SolveExitHook(mod,m,d,lambda ctx:None):pass
                mod._solve=lambda *args:None
        self.assertIs(mod._solve,original)


class Full54Prerequisite(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.config=self.root/'cfg.toml'
        self.config.write_text('required_closed_producer="producer/report.json"\nrequired_closed_review="review.json"\n')
        (self.root/'producer').mkdir();self.p=self.root/'producer/report.json';self.r=self.root/'review.json'

    def write(self,path,d):path.write_text(json.dumps(d))

    def fixture(self):
        cases=[dict(name='case'+str(i),native_queries=1,stop_reason='BASE_PHYSICAL_INVALID') for i in range(54)]
        receipt=self.root/'receipt.json';self.write(receipt,{'fixture':'CPU GUARD ONLY'})
        registration=self.root/'registration.json';self.write(registration,dict(cases=cases))
        producer=dict(schema='pen_full54_native_screened_rollout:1',passed=8,total=8,checks={str(i):True for i in range(8)},cases=cases,source_receipt=file_pin(receipt),registration=file_pin(registration),actual_native_integrations=432,attempted_native_integrations=432,optimizer_steps=0,attempted_optimizer_steps=0)
        self.write(self.p,producer)
        review=dict(schema='pen_full54_native_screened_saved_review:3',passed=5,total=5,checks={str(i):True for i in range(5)},complete=True,reviewed_cases=cases,run=file_pin(self.p),registration=file_pin(registration),source_receipt=file_pin(receipt))
        self.write(self.r,review)
        return producer,review

    def test_missing_reports_reject_without_gpu(self):
        with self.assertRaises(ValueError):require_closed_full54(self.config)

    def test_closed_workflow_accepts_physical_stops_as_outcomes(self):
        self.fixture();proof=require_closed_full54(self.config)
        self.assertEqual(proof['producer'],file_pin(self.p))

    def test_partial_or_empty_checks_reject(self):
        for update in [dict(complete=False),dict(passed=0,total=0,checks={})]:
            _,r=self.fixture();r.update(update);self.write(self.r,r)
            with self.assertRaises(ValueError):require_closed_full54(self.config)

    def test_failure_file_and_wrong_review_owner_reject(self):
        self.fixture();failure=self.p.parent/'failure.json';self.write(failure,{'failure':True})
        with self.assertRaises(ValueError):require_closed_full54(self.config)
        failure.unlink();_,r=self.fixture();r['run']['sha256']='0'*64;self.write(self.r,r)
        with self.assertRaises(ValueError):require_closed_full54(self.config)

    def test_duplicate_order_and_nonzero_optimizer_reject(self):
        for key,value in [('cases',[dict(name='duplicate',native_queries=1)]*54),('optimizer_steps',1)]:
            p,r=self.fixture();p[key]=value;self.write(self.p,p);r['run']=file_pin(self.p);self.write(self.r,r)
            with self.assertRaises(ValueError):require_closed_full54(self.config)


class ExternalCopiesCPU(unittest.TestCase):
    def make(self):
        import warp as wp
        import numpy as np
        observer=object.__new__(SolveExitGraphObserver)
        observer.wp=wp;observer.context_owners=[]
        observer.data=SimpleNamespace(qacc=wp.array(np.array([[1.,2.]],np.float32),device='cpu'))
        context=SimpleNamespace(grad=wp.array(np.array([[3.,4.]],np.float32),device='cpu'))
        observer.buffers={'data.qacc':wp.empty_like(observer.data.qacc),'context.grad':wp.empty_like(context.grad)}
        observer.executions=wp.zeros(1,dtype=int,device='cpu')
        return observer,context

    def test_actual_CPU_copies_and_counter_leave_sources_unchanged(self):
        import numpy as np
        observer,context=self.make()
        before=[observer.data.qacc.numpy().copy(),context.grad.numpy().copy()]
        observer._emit(context)
        self.assertEqual(observer.count(),1)
        np.testing.assert_array_equal(observer.buffers['data.qacc'].numpy(),before[0])
        np.testing.assert_array_equal(observer.buffers['context.grad'].numpy(),before[1])
        np.testing.assert_array_equal(observer.data.qacc.numpy(),before[0])
        np.testing.assert_array_equal(context.grad.numpy(),before[1])
        self.assertIs(observer.context_owners[0],context)

    def test_aliasing_destination_rejects_before_source_write(self):
        import numpy as np
        observer,context=self.make()
        before=observer.data.qacc.numpy().copy()
        observer.buffers['data.qacc']=observer.data.qacc
        with self.assertRaises(ValueError):observer._emit(context)
        self.assertEqual(observer.count(),0)
        np.testing.assert_array_equal(observer.data.qacc.numpy(),before)


if __name__=='__main__':unittest.main()
