from numerical_model.params import params
import numpy as np

def observe(state):
    nx = int(params.nx)

    return np.copy(state[:,:nx])