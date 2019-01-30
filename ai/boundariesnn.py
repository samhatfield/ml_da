import numpy as np

class BoundariesNN:
    # Name of weights file
    weights_file = "boundariesnn.hdf"

    # Number of hidden layers and nodes per hidden layer
    n_hidden_layers = 2
    n_per_hidden_layer = 40

    @staticmethod
    def train(BoundariesNN):
        from iris import load_cubes
        from keras.models import Sequential
        from keras.layers import Dense

        # Load training data
        q, u, v = load_cubes("training_data.nc", ["pv", "u", "v"])

        # Convert to raw NumPy arrays
        q, u, v = q.data, u.data, v.data

        # Get dimensions
        n_time, n_lev, _, n_lon = q.shape

        # Compute number of training pairs
        # 2 (top and bottom) * number of time steps (minus 1) * number of layers
        # * number of longitudes
        n_train = 2*(n_time-1)*n_lev*n_lon

        # Number of input and output variables to neural net
        n_input = 6*n_lev*3
        n_output = 3*n_lev

        print(f"Training with {n_train} training pairs, dimensions: ({n_input}, {n_output})")

        # Define input and output arrays
        train_in  = np.zeros((n_train,n_input))
        train_out = np.zeros((n_train,n_output))

        i = 0
        for t in range(n_time-1):
            for x in range(n_lon):
                train_in[i,:6*n_lev]         = BoundariesNN.get_stencil_top(q[t,...], x)
                train_in[i,6*n_lev:12*n_lev] = BoundariesNN.get_stencil_top(u[t,...], x)
                train_in[i,12*n_lev:]        = BoundariesNN.get_stencil_top(v[t,...], x)

                train_out[i,:2]  = q[t+1,:,0,x] - q[t,:,0,x]
                train_out[i,2:4] = u[t+1,:,0,x] - u[t,:,0,x]
                train_out[i,4:]  = v[t+1,:,0,x] - v[t,:,0,x]

                i+=1

                train_in[i,:6*n_lev]         = BoundariesNN.get_stencil_bottom(q[t,...], x)
                train_in[i,6*n_lev:12*n_lev] = BoundariesNN.get_stencil_bottom(u[t,...], x)
                train_in[i,12*n_lev:]        = BoundariesNN.get_stencil_bottom(v[t,...], x)

                train_out[i,:2]  = q[t+1,:,-1,x] - q[t,:,-1,x]
                train_out[i,2:4] = u[t+1,:,-1,x] - u[t,:,-1,x]
                train_out[i,4:]  = v[t+1,:,-1,x] - v[t,:,-1,x]

                i+=1

        print("Training data prepared")

        # Build multilayer perceptron with tanh activation functions
        model = Sequential()
        model.add(Dense(n_input, input_dim=n_input, activation='tanh'))
        for _ in range(BoundariesNN.n_hidden_layers):
            model.add(Dense(BoundariesNN.n_per_hidden_layer, activation='tanh'))
        model.add(Dense(n_output, activation='tanh'))
        model.compile(loss='mean_absolute_error', optimizer='SGD', metrics=['mae'])

        model.fit(train_in, train_out, epochs=200, batch_size=128, validation_split=0.2)

        model.save_weights(BoundariesNN.weights_file)

    @staticmethod
    def get_stencil_top(full_array, long):
        top = full_array[:,:2,:]
        stencil = np.roll(top,1-long,axis=2)[...3]
        return stencil.flatten()

    @staticmethod
    def get_stencil_bottom(full_array, long):
        top = full_array[:,-2:,:]
        stencil = np.roll(top,1-long,axis=2)[...3]
        stencil = stencil[:,::-1,:]
        return stencil.flatten()
