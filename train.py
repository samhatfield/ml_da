import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from model_params import nx

# Load raw training data
raw_training_data = np.loadtxt('tmp.txt')

# Get number of training pairs
n_run  = raw_training_data.shape[0]
print(f"Training with {n_run} training pairs")

# Extract training data into inputs and outputs
x_train = raw_training_data[:,:nx]
y_train = raw_training_data[:,nx:]

# Renormalise input data
max_train = 30.0
min_train = -20.0
x_train = 2.0*(x_train - min_train)/(max_train - min_train) - 1.0

# Build multilayer perceptron with tanh activation functions
model = Sequential()
model.add(Dense(8, input_dim=8, activation='tanh'))
model.add(Dense(100, activation='tanh'))
model.add(Dense(100, activation='tanh'))
model.add(Dense(100, activation='tanh'))
model.add(Dense(100, activation='tanh'))
model.add(Dense(8, activation='tanh'))
model.compile(loss='mean_absolute_error', optimizer='SGD', metrics=['mae'])

# Train!
model.fit(x_train, y_train, epochs=200,batch_size=128,validation_split=0.2)

model.save_weights("weights")