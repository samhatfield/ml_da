from numerical_model.models import step_two_layer as truth_step, step_one_layer as fcst_step
from numerical_model.params import params
from setup import spin_up, gen_ensemble
from output import output
from observation import observe
import numpy as np
from numpy.random import randn
from math import sqrt
import matplotlib.pyplot as plt

nx, ny = int(params.nx), int(params.ny)

n_steps = 200
obs_err_var = 0.2
n_ens = 20
assim_freq = 1
write_freq = 1

step_member = fcst_step

# Define and spin up truth
truth = np.zeros((n_steps,nx))
truth_full = spin_up()
truth[0,:] = truth_full[:nx]

# Generate truth run
for i in range(1,n_steps):
    truth_full = truth_step(truth_full)
    truth[i,:] = truth_full[:nx]

# Extract observations
obs = observe(truth)
obs_covar = obs_err_var * np.eye(nx)
for i in range(n_steps):
    obs[i,:] += sqrt(obs_err_var)*randn(nx)

# Generate initial ensemble
ensemble = gen_ensemble(truth, n_ens)

# # Run assimilation cycle
for i in range(n_steps):
    if i % 10 == 0:
        print(f"Step {i}")

    # Analysis step
    # if i%assim_freq == 0:
    #     ensemble = assimilate(ensemble, obs[:,i], obs_covar)

    # Write output
    if i % write_freq == 0:
        output(ensemble, truth[i,:])

    # Forecast step
    for i in range(n_ens):
        ensemble[i,:] = step_member(ensemble[i,:])