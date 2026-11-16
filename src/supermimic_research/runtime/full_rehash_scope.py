"""Explicit reversible cold-verifier binding for a bounded diagnostic process.

Only frozen_sources.verify_file_pin and source_validation_epoch.verify_file_pin
are rebound. All file bytes are rehashed on every call;epoch=None stays default.
No artifact default,collision binding,installed file or native state changes.
"""
import copy
import hashlib
import marshal
from pathlib import Path
import threading
from .. import artifacts,frozen_sources,strict_file_verification
from . import source_validation_epoch

ORIGINAL=artifacts.verify_file_pin
CANDIDATE=strict_file_verification.verify_file_pin_single_resolve
_ACTIVE=None
_MODULES=(frozen_sources,source_validation_epoch)


def code_hash(fn):return hashlib.sha256(marshal.dumps(fn.__code__)).hexdigest()


def implementation_pins():
    return [artifacts.file_pin(Path(__file__)),artifacts.file_pin(Path(strict_file_verification.__file__))]


class FullRehashVerifierScope:
    def __init__(self,mode,pins):
        if mode not in ['ORIGINAL_FULL_REHASH','SINGLE_RESOLVE_FULL_REHASH']:
            raise ValueError('Explicit full-rehash mode required;no source epoch')
        wanted=[str(Path(__file__).resolve()),str(Path(strict_file_verification.__file__).resolve())]
        if [p['path'] for p in pins]!=wanted:raise ValueError('Exact implementation source pins required')
        self.mode=mode;self.pins=copy.deepcopy(pins);self.binding=self._dispatch;self.active=False;self.used=False
        self.calls=0;self.boundary_hash_calls=0;self.validated=False;self.postcheck_error=None
        self.original_code=code_hash(ORIGINAL);self.candidate_code=code_hash(CANDIDATE)

    def _check(self):
        if _ACTIVE is not self or not self.active or any(m.verify_file_pin is not self.binding for m in _MODULES):
            raise ValueError('Full-rehash verifier scope binding changed')
        if artifacts.verify_file_pin is not ORIGINAL or strict_file_verification.verify_file_pin_single_resolve is not CANDIDATE:
            raise ValueError('Default/candidate implementation binding changed')
        if code_hash(ORIGINAL)!=self.original_code or code_hash(CANDIDATE)!=self.candidate_code:
            raise ValueError('Verifier implementation code changed')

    def _hash_implementations(self):
        for pin in self.pins:ORIGINAL(pin);self.boundary_hash_calls+=1

    def _dispatch(self,pin):
        self._check();self.calls+=1
        return (ORIGINAL if self.mode=='ORIGINAL_FULL_REHASH' else CANDIDATE)(pin)

    def __enter__(self):
        global _ACTIVE
        if threading.current_thread() is not threading.main_thread() or _ACTIVE is not None or self.used:
            raise ValueError('Exclusive single-use main-thread verifier scope required')
        if any(m.verify_file_pin is not ORIGINAL for m in _MODULES) or artifacts.verify_file_pin is not ORIGINAL:
            raise ValueError('Unexpected verifier binding before scope')
        self._hash_implementations();self.used=True;self.active=True;_ACTIVE=self
        for module in _MODULES:module.verify_file_pin=self.binding
        try:self._check()
        except BaseException:
            for module in _MODULES:module.verify_file_pin=ORIGINAL
            self.active=False;_ACTIVE=None;raise
        return self

    def __exit__(self,kind,error,tb):
        global _ACTIVE
        problem=None
        try:self._check();self._hash_implementations()
        except BaseException as caught:problem=caught;self.postcheck_error=dict(type=type(caught).__name__,message=str(caught))
        finally:
            for module in _MODULES:module.verify_file_pin=ORIGINAL
            self.active=False;_ACTIVE=None
        self.validated=kind is None and problem is None
        if problem is not None and kind is None:raise problem
        return False

    def audit(self):
        return dict(mode=self.mode,validated=self.validated,active=self.active,full_rehash_calls=self.calls,
            implementation_boundary_hash_calls=self.boundary_hash_calls,original_code=self.original_code,candidate_code=self.candidate_code,
            rebound_symbols=['frozen_sources.verify_file_pin','source_validation_epoch.verify_file_pin'],
            default_unchanged=artifacts.verify_file_pin is ORIGINAL,bindings_restored=all(m.verify_file_pin is ORIGINAL for m in _MODULES),
            source_epoch_used=False,postcheck_error=self.postcheck_error,
            scope='Explicit bounded process binding only. Both modes fully reread/hash each pin. Entry/exit code-source hashes charged separately;not an atomic filesystem snapshot or installed/default loader change.')
