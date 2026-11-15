"""Explicit Windows read-locked source validation for bounded restore batches.

Default loaders still hash on every load. An epoch hashes at entry/exit, keeps
read-sharing-only handles, and checks handle/path identity before each use.
Caller outputs are provisional until successful context exit. This is not a
security boundary against malicious native code or preexisting writable maps.
"""
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from supermimic_research.artifacts import verify_file_pin
from supermimic_research.frozen_sources import verify_source_snapshot


def _blob(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _stamp(path):
    s=os.stat(path,follow_symlinks=True)
    return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns


class _FileInfo(ctypes.Structure):
    _fields_=[('attributes',wintypes.DWORD),('created',wintypes.FILETIME),('accessed',wintypes.FILETIME),('written',wintypes.FILETIME),
        ('volume',wintypes.DWORD),('size_hi',wintypes.DWORD),('size_lo',wintypes.DWORD),('links',wintypes.DWORD),('index_hi',wintypes.DWORD),('index_lo',wintypes.DWORD)]


@dataclass(frozen=True)
class _Binding:
    path:str
    handle:int
    identity:tuple
    stamp:tuple


class SourceValidationEpoch:
    def __init__(self,manifest,engine_sources):
        self._manifest_blob=_blob(manifest);self._engines_blob=_blob(engine_sources);self._bindings=[];self._active=False;self._used=False
        self.validated=False;self.verify_calls=0;self.boundary_hash_s=0.;self.lock_s=0.;self._api=None

    def _information(self,handle):
        info=_FileInfo()
        if not self._api.GetFileInformationByHandle(handle,ctypes.byref(info)):raise ctypes.WinError(ctypes.get_last_error())
        return info.volume,info.index_hi,info.index_lo,info.size_hi,info.size_lo,info.written.dwHighDateTime,info.written.dwLowDateTime

    def _close(self):
        errors=[]
        for b in reversed(self._bindings):
            if not self._api.CloseHandle(b.handle):errors.append(ctypes.get_last_error())
        self._bindings=[];self._active=False
        if errors:raise OSError('Failed to close epoch source handles: '+str(errors))

    def _hash_all(self):
        start=time.perf_counter();verify_source_snapshot(json.loads(self._manifest_blob),live=True)
        for pin in json.loads(self._engines_blob):verify_file_pin(pin)
        self.boundary_hash_s+=time.perf_counter()-start

    def _identities(self):
        for b in self._bindings:
            if self._information(b.handle)!=b.identity or _stamp(b.path)!=b.stamp:raise ValueError('Read-locked source identity changed: '+b.path)

    def __enter__(self):
        if os.name!='nt':raise ValueError('Read-locked source epoch requires the Windows backend')
        if self._used or self._active:raise ValueError('Source epoch is single-use')
        self._used=True;api=ctypes.WinDLL('kernel32',use_last_error=True);self._api=api
        api.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE];api.CreateFileW.restype=wintypes.HANDLE
        api.GetFileInformationByHandle.argtypes=[wintypes.HANDLE,ctypes.POINTER(_FileInfo)];api.GetFileInformationByHandle.restype=wintypes.BOOL
        api.CloseHandle.argtypes=[wintypes.HANDLE];api.CloseHandle.restype=wintypes.BOOL
        manifest=json.loads(self._manifest_blob);pins=[p for r in manifest['records'] for p in [r['observed'],r['snapshot']]]+json.loads(self._engines_blob)
        unique={}
        for pin in pins:
            path=str(Path(pin['path']).resolve(strict=True));signature=(pin['bytes'],pin['sha256'])
            if path!=pin['path']:raise ValueError('Epoch requires canonical full source paths')
            if path in unique and unique[path]!=signature:raise ValueError('Conflicting source pins for one path')
            unique[path]=signature
        start=time.perf_counter()
        try:
            for path in sorted(unique):
                extended=('\\\\?\\UNC\\'+path[2:]) if path.startswith('\\\\') else '\\\\?\\'+path
                handle=api.CreateFileW(extended,0x80000000,0x1,None,3,0x80,None) # GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING
                if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
                try:binding=_Binding(path,handle,self._information(handle),_stamp(path))
                except BaseException:
                    api.CloseHandle(handle);raise
                self._bindings.append(binding)
            self.lock_s=time.perf_counter()-start;self.file_count=len(self._bindings)
            self._hash_all();self._identities();self._active=True
            return self
        except BaseException:
            self._close();raise

    def verify(self,manifest,engine_sources):
        if not self._active:raise ValueError('Source epoch is not active or has expired')
        if _blob(manifest)!=self._manifest_blob or _blob(engine_sources)!=self._engines_blob:raise ValueError('Source/engine manifest is not bound to this epoch')
        self._identities();self.verify_calls+=1

    def __exit__(self,exc_type,exc,tb):
        completed=False
        try:
            self._identities();self._hash_all()
            completed=exc_type is None
        finally:self._close()
        self.validated=completed
        return False

    def audit(self):
        return dict(contract='windows_read_locked_source_epoch:1',validated=self.validated,active=self._active,source_files=getattr(self,'file_count',0),verify_calls=self.verify_calls,
            lock_wall_s=self.lock_s,boundary_hash_wall_s=self.boundary_hash_s,
            scope='Full SHA at entry/exit; retained read-sharing handles and per-use file identity/manifest checks. Transaction outputs valid only after successful exit; no persistent checkpoint conversion.')


def validate_state_sources(manifest,engine_sources,epoch=None):
    if epoch is None:
        verify_source_snapshot(manifest,live=True)
        for pin in engine_sources:verify_file_pin(pin)
    else:
        if type(epoch) is not SourceValidationEpoch:raise ValueError('Explicit SourceValidationEpoch required')
        epoch.verify(manifest,engine_sources)
