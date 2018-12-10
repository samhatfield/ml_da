from lorenz96 import step
from model_params import nx, ny, δt
import numpy as np

# Length of run
N = 2010000

# Initialise state vector array
state_1 = state_2 = np.zeros(nx+nx*ny)
state_1[0] = 8.0

# Generate training data
with open('training_data.txt', 'w') as training_data_file:
    for i in range(N):
        if i%10000 == 0:
            print(f"Time step {i}")
            
        # Step forward once
        state_2 = step(state_1)
        
        if i%(1.0/δt) == 0:
            # Form training data and write to file
            training_data_line = np.concatenate((state_1[:nx], state_2[:nx] - state_1[:nx]))
            training_data_file.write(' '.join([str(f) for f in training_data_line]) + '\n')
        
        state_1 = state_2