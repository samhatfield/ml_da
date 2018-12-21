from numerical_model.models import step_two_layer as truth_step, step_one_layer as fcst_step
from numerical_model.params import params
import numpy as np

def spin_up():
    nx, ny = int(params.nx), int(params.ny)

    truth = np.zeros(nx+nx*ny)
    truth[0] = 1.0
    for _ in range(5000):
        truth = truth_step(truth)
    return truth

def gen_ensemble(truth, n_ens):
    from numpy.random import rand

    nx = int(params.nx)
    n_steps = truth.shape[0]
    return np.array([truth[int(n_steps*rand()),:nx] for _ in range(n_ens)])
