from keras.models import Sequential
from keras.layers import Dense

# Number of hidden layers and nodes per hidden layer
n_hidden_layers = 4
n_per_hidden_layer = 100

def build_model(n_input, n_output):
    # Build multilayer perceptron with tanh activation functions
    model = Sequential()
    model.add(Dense(n_input, input_dim=n_input, activation='tanh'))
    for _ in range(n_hidden_layers):
        model.add(Dense(n_per_hidden_layer, activation='tanh'))
    model.add(Dense(n_output, activation='tanh'))
    model.compile(loss='mean_absolute_error', optimizer='SGD', metrics=['mae'])
    
    return model