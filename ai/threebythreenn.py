import numpy as np

class ThreeByThreeNN:
    # Name of weights file
    weights_file = "threebythreenn.hdf"

    # Number of hidden layers and nodes per hidden layer
    n_hidden_layers = 2
    n_per_hidden_layer = 40

    @staticmethod
    def train(training_data):
        # Split up training data
        q, u, v = training_data

        # Get dimensions
        n_time, n_lev, n_lat, n_lon = q.shape
        print(f"{n_time} timesteps, {n_lev} levels, {n_lat} latitudes, {n_lon} longitudes")

        # Compute number of training pairs
        # number of time steps (minus 1) * number of layers
        # * number of latitudes (minus top and bottom) * number of longitudes
        n_train = (n_time-1)*n_lev*(n_lat-2)*n_lon

        # Number of input and output variables to neural net
        n_input = 9*n_lev*3
        n_output = 3*n_lev

        print(f"Training with {n_train} training pairs, dimensions: ({n_input}, {n_output})")

        # Define input and output arrays
        train_in  = np.zeros((n_train,n_input))
        train_out = np.zeros((n_train,n_output))

        i = 0
        for t in range(n_time-1):
            for x in range(n_lon):
                for y in range(1,n_lat-1):
                    train_in[i,:9*n_lev]         = ThreeByThreeNN.get_stencil(q[t,...], x, y, n_lon)
                    train_in[i,9*n_lev:18*n_lev] = ThreeByThreeNN.get_stencil(u[t,...], x, y, n_lon)
                    train_in[i,18*n_lev:]        = ThreeByThreeNN.get_stencil(v[t,...], x, y, n_lon)

                    train_out[i,:2]  = q[t+1,:,y,x] - q[t,:,y,x]
                    train_out[i,2:4] = u[t+1,:,y,x] - u[t,:,y,x]
                    train_out[i,4:]  = v[t+1,:,y,x] - v[t,:,y,x]

                    i+=1

        print("Training data prepared")

        model = ThreeByThreeNN.build_model()
        model.fit(train_in, train_out, epochs=200, batch_size=128, validation_split=0.2)
        model.save_weights(ThreeByThreeNN.weights_file)

    @staticmethod
    def build_model():
        from keras.models import Sequential
        from keras.layers import Dense

        # Build multilayer perceptron with tanh activation functions
        model = Sequential()
        model.add(Dense(n_input, input_dim=n_input, activation='tanh'))
        for _ in range(BoundariesNN.n_hidden_layers):
            model.add(Dense(BoundariesNN.n_per_hidden_layer, activation='tanh'))
        model.add(Dense(n_output, activation='tanh'))
        model.compile(loss='mean_absolute_error', optimizer='SGD', metrics=['mae'])

    @staticmethod
    def get_stencil(full_array, lon, lat, n_lon):
        stencil = full_array[:,lat-1:lat+2,np.array(range(lon-1,lon+2))%n_lon]
        return stencil.flatten()