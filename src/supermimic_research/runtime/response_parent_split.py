"""Frozen development parent-phase roles, never candidate-level random splits."""


def parent_roles(spec):
    expected={'TRAIN':[1,2,3,5,6,7],'CHECK':[8,9,10],'excluded':[0,4],'repeats':3}
    if spec!=expected:raise ValueError('Predeclared fresh parent split and3 repeats required')
    return {k:role for role in ['TRAIN','CHECK'] for k in spec[role]}
