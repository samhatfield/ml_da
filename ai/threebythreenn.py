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

    """
    Train the neural net based on the input training data of q (quasigeostrophic vorticity), u
    (zonal wind) and v (meridional wind).
    """
    @staticmethod
    def train(q, u, v):
        from util import build_model, save_history

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

        # Prepare training data. Different grid points and time steps are considered as independent
        # training pairs.
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

        # Build model for training
        model = build_model(
            n_input, n_output,
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
        stencil = full_array[:,lat-1:lat+2,np.array(range(lon-1,lon+2))%n_lon]
        return stencil.flatten()