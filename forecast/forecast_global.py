from numerical_model.models import step_two_layer as truth_step, step_one_layer as fcst_step
import numpy as np
from numerical_model.params import params
from ai.forecaster import Forecaster
import matplotlib.pyplot as plt

nx, ny = int(params.nx), int(params.ny)

# Number of forecasts, length of forecasts and spacing between initial conditions drawn from model
# spin-up (in timesteps)
n_fcsts = 10
fcst_len = 500
spacing = 5000

# Set initial condition for truth
truth = np.zeros(nx+nx*ny)
truth[0] = 8.0

# Initialise variables for storing forecast errors
num_model_err_store = np.zeros(fcst_len)
neur_model_err_store = np.zeros(fcst_len)

# Construct neural net forecaster object
nn_fcster = Forecaster(1)

for i in range(n_fcsts):
    print(f'Running forecast {i+1} of {n_fcsts}')

    # Get initial condition by running model for a bit
    for _ in range(spacing):
        truth = truth_step(truth)

    # Run truth
    truth_fcst = np.zeros((nx,fcst_len))
    truth_state = truth[:]
    for j in range(fcst_len):
        truth_fcst[:nx,j] = truth_state[:nx]
        truth_state = truth_step(truth_state)

    # Run numerical model forecast
    num_model_fcst = np.zeros((nx,fcst_len))
    num_model_state = truth[:nx]
    for j in range(fcst_len):
        num_model_fcst[:,j] = num_model_state[:]
        num_model_state = fcst_step(num_model_state)

    # Run neural net forecast
    neur_model_fcst = np.zeros((nx,fcst_len))
    neur_model_state = np.zeros((1,nx))
    neur_model_state[0,:] = truth[:nx]
    for j in range(fcst_len):
        neur_model_fcst[:,j] = neur_model_state[0,:]
        neur_model_state = nn_fcster.step(neur_model_state)

    # Compute L1 error of both forecasts
    num_model_err = np.sum(np.abs(num_model_fcst - truth_fcst), axis=0)/nx
    neur_model_err = np.sum(np.abs(neur_model_fcst - truth_fcst), axis=0)/nx

    # Accumulate the average
    num_model_err_store += num_model_err
    neur_model_err_store += neur_model_err

# Compute average forecast errors
num_model_err_store /= n_fcsts
neur_model_err_store /= n_fcsts

plt.plot(truth_fcst[0,:], label='Truth')
plt.plot(num_model_fcst[0,:], label='Numerical model')
plt.plot(neur_model_fcst[0,:], label='Neural net')
plt.legend()

plt.figure()
plt.plot(num_model_err_store, label='Numerical model')
plt.plot(neur_model_err_store, label='Neural net')
plt.legend()

plt.show()
