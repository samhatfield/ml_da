from numerical_model.params import params
import numpy as np

def observe(state, row=None):
    nx = int(params.nx)

    if row is None:
        return np.copy(state[...,:nx])
    else:
        return np.copy(state[...,row])
