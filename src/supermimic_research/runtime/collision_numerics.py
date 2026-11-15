"""Explicit collision-only numerical policy; physical parameter keys rejected."""
import math


def validate_collision_numerics(options):
    if not isinstance(options,dict) or set(options)!={'ccd_iterations','ccd_tolerance'}:
        raise ValueError('Both explicit CCD iteration cap and tolerance required; no physical keys')
    for key,value in options.items():
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError('Finite positive CCD numerical values required')
    if int(options['ccd_iterations'])!=options['ccd_iterations']:raise ValueError('Integer CCD iteration cap')
    return dict(ccd_iterations=int(options['ccd_iterations']),ccd_tolerance=float(options['ccd_tolerance']))


def collision_numerical_receipt(host,device):
    return dict(host=dict(ccd_iterations=int(host.opt.ccd_iterations),ccd_tolerance=float(host.opt.ccd_tolerance)),
                GPU=dict(ccd_iterations=int(device.opt.ccd_iterations),ccd_tolerance=float(device.opt.ccd_tolerance.numpy()[0])))


def set_collision_numerics(host,device,options):
    requested=validate_collision_numerics(options)
    before=collision_numerical_receipt(host,device)
    host.opt.ccd_iterations=requested['ccd_iterations'];host.opt.ccd_tolerance=requested['ccd_tolerance']
    device.opt.ccd_iterations=requested['ccd_iterations'];device.opt.ccd_tolerance.fill_(requested['ccd_tolerance'])
    return dict(requested=requested,before=before,after=collision_numerical_receipt(host,device),
                scope='Explicit host/device CCD numerical override only; model-wide array/scalar diffs must be audited separately.')
