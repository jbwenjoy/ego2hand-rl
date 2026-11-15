"""Check checkpoint RNG formats on private generators, never global state."""
import copy
import random
import numpy as np
import torch


def validate_rng(state,*,cuda_count=None):
    count=torch.cuda.device_count() if cuda_count is None else cuda_count
    try:
        if set(state)!= {'python','numpy','torch','cuda'} or len(state['cuda'])!=count:raise ValueError('RNG device/key layout')
        random.Random(0).setstate(copy.deepcopy(state['python']))
        np.random.RandomState(0).set_state(copy.deepcopy(state['numpy']))
        torch.Generator(device='cpu').set_state(state['torch'])
        for i,value in enumerate(state['cuda']):torch.Generator(device=f'cuda:{i}').set_state(value)
    except (ValueError,TypeError,RuntimeError,KeyError) as error:
        raise ValueError('Invalid checkpoint RNG format: '+str(error)) from error
