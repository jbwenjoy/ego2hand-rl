"""Fresh disjoint action directions for predeclared actual source branches."""
import numpy as np


def branch_profiles(config):
    if (config['anchors'] != [0,2,4,6] or config['dt'] != [.005,.0025]
            or config['PD_target_eps_rad'] != [.005,.02]
            or config['train_direction_seed'] == config['check_direction_seed']
            or config['train_direction_count'] != 12 or config['check_direction_count'] != 4):
        raise ValueError('Explicit fixed256 branch cohort and disjoint role seeds')
    directions = {}; profiles = []
    for role in ['TRAIN','CHECK']:
        seed, count = [config[role.lower()+'_'+key] for key in ['direction_seed','direction_count']]
        values = np.random.default_rng(seed).normal(size=(count,20))
        values /= np.linalg.norm(values,axis=1,keepdims=True)
        directions[role] = values
        for anchor in config['anchors']:
            for direction in range(count):
                epsilon = config['PD_target_eps_rad'][direction % 2]
                for sign in [1,-1]:
                    profiles.append(dict(name=f'{role}_A{anchor}_D{direction}_S{sign}',role=role,
                        direction_seed=seed,direction=direction,anchor=anchor,epsilon=epsilon,sign=sign))
    return directions,profiles
