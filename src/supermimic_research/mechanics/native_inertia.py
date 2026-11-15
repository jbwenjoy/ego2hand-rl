"""Independent lower-sparse inertia decoding from saved native row metadata."""
import numpy as np


def expand_sparse_inertia(values,rownnz,rowadr,colind,nv):
    values,rownnz,rowadr,colind=map(np.asarray,[values,rownnz,rowadr,colind])
    if values.ndim!=1 or rownnz.shape!=(nv,) or rowadr.shape!=(nv,) or colind.shape!=values.shape:
        raise ValueError('Explicit lower-sparse mass layout')
    out=np.zeros((nv,nv),dtype=np.float64);used=[]
    for row in range(nv):
        start,count=int(rowadr[row]),int(rownnz[row]);end=start+count
        if start<0 or count<=0 or end>len(values):raise ValueError('Invalid mass row range')
        cols=colind[start:end]
        if np.any(cols<0) or np.any(cols>row) or len(set(map(int,cols)))!=count or row not in cols:
            raise ValueError('Unique lower columns including diagonal required')
        used.extend(range(start,end));out[row,cols]=values[start:end];out[cols,row]=values[start:end]
    if sorted(used)!=list(range(len(values))) or not np.isfinite(out).all():raise ValueError('Full finite sparse mass coverage')
    return out
