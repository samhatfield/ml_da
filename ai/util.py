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
    model.compile(loss='mse', optimizer='adam', metrics=[corr])

    return model

"""
Take the history object returned by keras.models.fit and output the columns to a text file.
"""
def save_history(filename, history):
    import numpy as np

    output = np.vstack([
        history.history["val_loss"],
        history.history["loss"],
        history.history["val_corr"],
        history.history["corr"]
    ])
    np.savetxt(filename, output.T)

"""
Keras metric for computing the correlation between the actual value (x) predicted value (y).
"""
def corr(x, y):
    from keras import backend as b

    x_m, y_m = b.mean(x), b.mean(y)
    return b.sum((y - y_m)*(x - x_m))/b.sqrt(b.sum(b.square(y - y_m))*(b.sum(b.square(x - x_m))))