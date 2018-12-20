import numpy as np
from neural_net import build_model
from model_params import nx

# Load raw training data
raw_training_data = np.loadtxt('training_data.txt')

# Get number of training pairs
print(f"Training with {raw_training_data.shape[0]} training pairs")

# Extract training data into inputs and outputs
x_train = raw_training_data[:,:nx]
y_train = raw_training_data[:,nx:]

# Renormalise input data
max_train = 30.0
min_train = -20.0
x_train = 2.0*(x_train - min_train)/(max_train - min_train) - 1.0

# Build model
model = build_model(nx, nx)

# Train!
model.fit(x_train, y_train, epochs=200,batch_size=128,validation_split=0.2)

model.save_weights("weights")