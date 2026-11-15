"""Predeclared first-control physical-response probes, never fitting data."""
import numpy as np


def intervention_profiles():
    directions = np.sin(np.pi*(np.arange(20)+.5)[None]*np.arange(1,3)[:,None]/20)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    rows = []
    for repeat in range(3):
        rows.append(dict(name=f'BASE_R{repeat}', repeat=repeat, direction=-1, epsilon=0., sign=0))
        for direction in range(2):
            for epsilon in (.002, .005):
                for sign in ((1,-1) if repeat % 2 == 0 else (-1,1)):
                    rows.append(dict(name=f'D{direction+1}_E{epsilon}_S{sign}_R{repeat}',
                        repeat=repeat, direction=direction, epsilon=epsilon, sign=sign))
    return directions, rows


def first_raw_action(delta20, direction20, epsilon, sign):
    # Physical EMA starts at delta. action scale=.5, EMA alpha=.5, so the
    # first executed PD target perturbation is .25 times the raw perturbation.
    raw = (2*np.asarray(delta20) + 4*sign*epsilon*np.asarray(direction20)).astype(np.float32)
    if raw.shape != (20,) or not np.isfinite(raw).all() or np.any(abs(raw)>1):
        raise ValueError('Finite bounded first raw20; clipping invalidates derivative probe')
    return raw
