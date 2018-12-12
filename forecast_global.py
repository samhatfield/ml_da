from lorenz96 import TwoLayerLorenz96Model
import numpy as np
from neural_net import build_model
from model_params import nx, ny
import matplotlib.pyplot as plt

# Number of forecasts, length of forecasts and spacing between initial conditions drawn from model
# spin-up (in timesteps)
n_fcsts = 1
fcst_len = 500
spacing = 200

# Build model and load weights
model = build_model(nx, nx)
model.load_weights('weights', by_name=False)

# Normalisation factors for input layer
max_train = 30.0
min_train = -20.0

# Build numerical model and set initial condition
num_model = TwoLayerLorenz96Model()
state = np.zeros(nx+nx*ny)
state[0] = 8.0

print(f'Performing {n_fcsts} forecasts')

for i in range(n_fcsts):
    # Get initial condition by running model for a bit
    for _ in range(spacing):
        state = num_model.step(state)
    init = state
    
    # Run numerical model forecast
    num_model_fcst = np.zeros((nx,fcst_len))
    num_model_state = init[:]
    for j in range(fcst_len):
        num_model_fcst[:,j] = num_model_state[:nx]
        num_model_state = num_model.step(num_model_state)
        
    # Run neural net forecast
    tends = np.zeros((3,nx))
    neur_model_fcst = np.zeros((nx,fcst_len))
    neur_model_state = init[:nx]
    model_input = np.zeros((1,nx))
    for j in range(fcst_len):
        tends = np.roll(tends, 1, axis=0)
           
        # Get tendency from neural net
        model_input[0,:] = 2.0*(neur_model_state - min_train)/(max_train - min_train) - 1.0
        tends[0,:] = model.predict(model_input, batch_size=1)[0,:]
        
        # Adams-Bashforth
        if j == 0:
            tend = tends[0,:]
        elif j == 1:
            tend = 1.5*tends[0,:] - 0.5*tends[1,:]
        else:
            tend = (23.0/12.0)*tends[0,:] - (4.0/3.0)*tends[1,:] + (5.0/12.0)*tends[2,:]
        neur_model_state += tend
        neur_model_fcst[:,j] = neur_model_state[:nx]

plt.plot(num_model_fcst[0,:], label='Numerical model')
plt.plot(neur_model_fcst[0,:], label='Neural net')
plt.legend()
plt.show()
