def build_model(n_input, n_output, n_hidden_layers, n_per_hidden_layer):
    from keras.models import Sequential
    from keras.layers import Dense

    # Build multilayer perceptron with tanh activation functions
    model = Sequential()
    model.add(Dense(n_input, input_dim=n_input, activation='tanh'))
    for _ in range(n_hidden_layers):
        model.add(Dense(n_per_hidden_layer, activation='tanh'))
    model.add(Dense(n_output, activation='tanh'))
    model.compile(loss='mean_absolute_error', optimizer='SGD', metrics=['mae'])

    return model

def save_history(filename, history):
    import numpy as np

    output = np.vstack([
        history.history["val_loss"],
        history.history["val_mean_absolute_error"],
        history.history["loss"],
        history.history["mean_absolute_error"]
    ])
    np.savetxt(filename, output.T)