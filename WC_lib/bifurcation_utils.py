import numba
import numpy as np
from copy import deepcopy as dcp 
from model_utils import parameter_index_mapping

def get_1d_bifurcation_params(
                                default_parameters,
                                bifurcation_parameter_name="theta_e",
                                bifurcation_range=[2.5,3.5],
                                n_points=8
        ):
    bif_space = np.linspace(
                                bifurcation_range[0],
                                bifurcation_range[1],
                                n_points
                )
    parameters = np.tile(default_parameters,(n_points,1))
    param_index = parameter_index_mapping[bifurcation_parameter_name]
    parameters[:,param_index] = bif_space
    return parameters
