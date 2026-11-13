"""Explicit numerical solver settings; physical parameter keys are rejected."""
import math


def validate_numerical_options(options):
    if not isinstance(options, dict):
        raise ValueError('Explicit numerical settings table')
    integers = {'iterations', 'ls_iterations'}
    reals = {'tolerance', 'ls_tolerance'}
    if set(options) - integers - reals:
        raise ValueError('Only iteration/termination numerical settings are permitted')
    result = {}
    for key, value in options.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Finite positive numerical settings, no boolean coercion')
        if key in integers and int(value) != value:
            raise ValueError('Integer numerical iteration limits')
        result[key] = int(value) if key in integers else float(value)
    return result


def set_model_numerical_options(host, device, options):
    values = validate_numerical_options(options)
    for key, value in values.items():
        setattr(host.opt, key, value)
        if key in ['iterations', 'ls_iterations']:
            setattr(device.opt, key, value)
        else:
            getattr(device.opt, key).fill_(value)
    return values


def numerical_receipt(host, device):
    keys = ['iterations', 'ls_iterations', 'tolerance', 'ls_tolerance']
    return dict(host={k: int(getattr(host.opt,k)) if k in ['iterations','ls_iterations'] else float(getattr(host.opt,k)) for k in keys},
                GPU={k: int(getattr(device.opt, k)) if k in ['iterations', 'ls_iterations']
                     else float(getattr(device.opt, k).numpy()[0]) for k in keys})
