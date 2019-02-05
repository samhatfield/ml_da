import numpy as np

class ThreeByThreeNN:
    """
    Class for predicting the interior domain of the QG model using a neural net.
    The local stencil size is 3x3.
    """

    # Name of output files (minus extension)
    out_file = "threebythreenn"

    # Number of hidden layers and nodes per hidden layer
    n_hidden_layers = 2
    n_per_hidden_layer = 40

    # Number of input and output variables to neural net
    # Stencil size * number of layers * number of variables
    n_input = 9*2*3
    # Number of layers * number of variables
    n_output = 2*3

    """
    Train the neural net based on the input training data of q (quasigeostrophic vorticity), u
    (zonal wind) and v (meridional wind).
    """
    @staticmethod
    def train(q, u, v):
        from util import build_model, save_history

        # Get dimensions
        n_lon, n_lat, _, n_time = q.shape
        print(f"{n_lon} longitudes, {n_lat} latitudes, 2 levels, {n_time} timesteps")

        # Compute number of training pairs
        # number of time steps (minus 1) * number of layers
        # * number of latitudes (minus top and bottom) * number of longitudes
        n_train = (n_time-1)*2*(n_lat-2)*n_lon

        print(f"Training with {n_train} training pairs,\
            dimensions: ({ThreeByThreeNN.n_input}, {ThreeByThreeNN.n_output})")

        # Define input and output arrays
        train_in  = np.zeros((n_train,ThreeByThreeNN.n_input))
        train_out = np.zeros((n_train,ThreeByThreeNN.n_output))

        # Prepare training data. Different grid points and time steps are considered as independent
        # training pairs.
        i = 0
        for t in range(n_time-1):
            for x in range(n_lon):
                for y in range(1,n_lat-1):
                    train_in[i,:9*2]     = ThreeByThreeNN.get_stencil(q[...,t], x, y, n_lon)
                    train_in[i,9*2:18*2] = ThreeByThreeNN.get_stencil(u[...,t], x, y, n_lon)
                    train_in[i,18*2:]    = ThreeByThreeNN.get_stencil(v[...,t], x, y, n_lon)
                    train_out[i,:2]  = q[x,y,:,t+1] - q[x,y,:,t]
                    train_out[i,2:4] = u[x,y,:,t+1] - u[x,y,:,t]
                    train_out[i,4:]  = v[x,y,:,t+1] - v[x,y,:,t]
                    i+=1

        # Normalize input
        train_in = ThreeByThreeNN.normalize(train_in)

        print("Training data prepared")

        # Build model for training
        model = build_model(
            ThreeByThreeNN.n_input, ThreeByThreeNN.n_output,
            ThreeByThreeNN.n_hidden_layers, ThreeByThreeNN.n_per_hidden_layer
        )

        # Train!
        history = model.fit(train_in, train_out, epochs=200, batch_size=128, validation_split=0.2)

        # Output weights and diagnostics files
        save_history(f"{ThreeByThreeNN.out_file}_history.txt", history)
        model.save_weights(f"{ThreeByThreeNN.out_file}.hdf")

    """
    Extracts the 3x3 stencil corresponding to the requested longitude and latitude.
    e.g. if you request the 2nd longitude, 1st latitude (index starting from 0)
    ---------------------------    -------
    |a|b|c|d|e|f|g|h|i|j|k|l|m|    |b|c|d|
    ---------------------------    -------
    |n|o|p|q|r|s|t|u|v|w|x|y|z| => |o|p|q|
    ---------------------------    -------
    |a|b|c|d|e|f|g|h|i|j|k|l|m|    |b|c|d|
    ---------------------------    -------
    """
    @staticmethod
    def get_stencil(full_array, lon, lat, n_lon):
        stencil = full_array[np.array(range(lon-1,lon+2))%n_lon,lat-1:lat+2,:]
        return stencil.flatten()

    """
    Normalize the given training data so values are between -1.0 and 1.0.
    """
    @staticmethod
    def normalize(training_data):
        # Maximum and minimum values of q, u, and v based on a long run of the numerical model
        q_max, q_min = 40.0, -37.0
        u_max, u_min = 10.0, -6.0
        v_max, v_min = 2.0, -2.0

        # Normalize the training data
        normalized = training_data[:,:]
        normalized[:,:9*2]     = 2.0*(normalized[:,:9*2]     - q_min)/(q_max - q_min) - 1.0
        normalized[:,9*2:18*2] = 2.0*(normalized[:,9*2:18*2] - u_min)/(u_max - u_min) - 1.0
        normalized[:,18*2:]    = 2.0*(normalized[:,18*2:]    - v_min)/(v_max - v_min) - 1.0
        return normalized