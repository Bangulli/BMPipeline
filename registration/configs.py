### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib
from typing import Union

### External Imports ###

### Internal Imports ###
import cost_functions as cf
import regularization as reg

########################


affine_config = {
    'device': "cuda:0",
    'echo': True,
    'registration_shape': (1, 1, 150, 150, 150),
    'num_levels_aff': 3,
    'used_levels_aff' : 3,
    'num_iters_aff': [100, 100, 100],
    'learning_rate_aff' : 0.002,
    'cost_function_aff' : cf.mi_global,
    'cost_function_params_aff' : {},
    'return_best': True
}

masked_affine_config = {
    'device': "cuda:0",
    'echo': True,
    'registration_shape': (1, 1, 150, 150, 150),
    'num_levels_aff': 3,
    'used_levels_aff' : 3,
    'num_iters_aff': [100, 100, 100],
    'learning_rate_aff' : 0.002,
    'cost_function_aff' : cf.mi_masked_global,
    'cost_function_params_aff' : {},
    'return_best': True
}

affine_deformable_config = {
    'device': "cuda:0",
    'echo': True,
    'registration_shape': (1, 1, 150, 150, 150),
    'num_levels_aff': 3,
    'used_levels_aff' : 3,
    'num_iters_aff': [100, 100, 100],
    'learning_rate_aff' : 0.002,
    'cost_function_aff' : cf.mi_global,
    'cost_function_params_aff' : {},
    'num_levels' : 2,
    'used_levels' : 2,
    'num_iters': [150, 200],
    'learning_rate' : 0.002,
    'cost_function' : cf.mi_local,
    'cost_function_params' : {},
    'alpha' : [1.0, 1.5],
    'regularization_function': reg.diffusion_relative,
    'regularization_function_params': {},
    'penalty_function' : None,
    'penalty_function_params' : {},
}
