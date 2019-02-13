"""
Build the Keras model from the given model parameters.
"""
def build_model(n_input, n_output, n_hidden_layers, n_per_hidden_layer):
    from keras.models import Sequential
    from keras.layers import Dense

    # Build multilayer perceptron with tanh activation functions
    model = Sequential()
    model.add(Dense(n_input, input_dim=n_input, activation='relu'))
    for _ in range(n_hidden_layers):
        model.add(Dense(n_per_hidden_layer, activation='relu'))
    model.add(Dense(n_output, activation='linear'))
    model.compile(loss='mae', optimizer='adam')

    return model

"""
Take the history object returned by keras.models.fit and output the columns to a text file.
"""
def save_history(filename, history):
    import numpy as np

    output = np.vstack([
        history.history["val_loss"],
        history.history["loss"],
    ])
    np.savetxt(filename, output.T)