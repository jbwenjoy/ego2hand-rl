"""Explicit HOT3D RGB projection/RLE diagnostics, not independent pose recovery."""
import numpy as np
from scipy.ndimage import binary_erosion,distance_transform_edt


def fisheye624(points,parameters):
    p=np.asarray(points,dtype=np.float64);k=np.asarray(parameters,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=3 or k.shape!=(15,) or not np.isfinite(p).all() or not np.isfinite(k).all() or np.any(p[:,2]<=0):
        raise ValueError('Finite front-facing camera points and Aria15 calibration required')
    f,cx,cy=k[:3];radial_coeff=k[3:9];p1,p2,s1,s2,s3,s4=k[9:]
    r=np.hypot(p[:,0],p[:,1]);theta=np.arctan2(r,p[:,2]);scale=np.divide(theta,r,out=np.zeros_like(r),where=r>0)
    a,b=p[:,0]*scale,p[:,1]*scale;t2=theta*theta
    radial=1+sum(c*t2**(i+1) for i,c in enumerate(radial_coeff));x,y=a*radial,b*radial
    q=x*x+y*y;xy=x*y
    u=x+2*p2*xy+p1*(q+2*x*x)+s1*q+s2*q*q
    v=y+2*p1*xy+p2*(q+2*y*y)+s3*q+s4*q*q
    return np.column_stack((f*u+cx,f*v+cy))


def decode_hot3d_mask(value):
    h,w=value['height'],value['width'];rle=np.asarray(value['rle'])
    if type(h) is not int or type(w) is not int or h<=0 or w<=0 or rle.ndim!=1 or len(rle)%2 or (len(rle) and rle.dtype.kind not in 'iu'):
        raise ValueError('Positive dimensions and integer start/length pairs required')
    mask=np.zeros(h*w,dtype=np.bool_);previous=0
    for start,length in rle.reshape(-1,2):
        lo=int(start)-1;hi=lo+int(length)
        if length<=0 or lo<previous or lo<0 or hi>h*w:raise ValueError('Ordered nonoverlapping one-based HOT3D runs required')
        mask[lo:hi]=True;previous=hi
    return mask.reshape(h,w)


def rasterize(uv,faces,shape):
    from PIL import Image,ImageDraw
    uv=np.asarray(uv);faces=np.asarray(faces)
    if uv.ndim!=2 or uv.shape[1]!=2 or not np.isfinite(uv).all() or faces.ndim!=2 or faces.shape[1]!=3 or faces.dtype.kind not in 'iu' or (len(faces) and (faces.min()<0 or faces.max()>=len(uv))):raise ValueError('Finite indexed projected triangles required')
    image=Image.new('1',(shape[1],shape[0]));draw=ImageDraw.Draw(image)
    for triangle in uv[faces]:draw.polygon([tuple(v) for v in triangle],fill=1)
    return np.asarray(image,dtype=np.bool_)


def compare_masks(reference,candidate):
    a=np.asarray(reference,dtype=np.bool_);b=np.asarray(candidate,dtype=np.bool_)
    if a.shape!=b.shape or a.ndim!=2:raise ValueError('Aligned HxW masks required')
    union=a|b;intersection=a&b
    result=dict(reference_pixels=int(a.sum()),candidate_pixels=int(b.sum()),intersection_pixels=int(intersection.sum()),
        missed_pixels=int((a&~b).sum()),extra_pixels=int((b&~a).sum()),IoU=float(intersection.sum()/union.sum()) if union.any() else 1.)
    if a.any() and b.any():
        yy,xx=np.where(union);y0,y1=max(0,int(yy.min())-2),min(len(a),int(yy.max())+3);x0,x1=max(0,int(xx.min())-2),min(a.shape[1],int(xx.max())+3)
        aa=a[y0:y1,x0:x1];bb=b[y0:y1,x0:x1];ab=aa&~binary_erosion(aa);bc=bb&~binary_erosion(bb)
        distances=np.r_[distance_transform_edt(~ab)[bc],distance_transform_edt(~bc)[ab]]
        result.update(symmetric_boundary_max_px=float(distances.max()),symmetric_boundary_p95_px=float(np.quantile(distances,.95)))
    else:result.update(symmetric_boundary_max_px=None,symmetric_boundary_p95_px=None)
    return result
