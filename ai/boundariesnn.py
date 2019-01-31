import numpy as np

class BoundariesNN:
    """
    Class for predicting the northern and southern boundaries of the QG model using a neural net.
    """

    # Name of output files (minus extension)
    out_file = "boundariesnn"

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
        # 2 (top and bottom) * number of time steps (minus 1) * number of layers
        # * number of longitudes
        n_train = 2*(n_time-1)*2*n_lon

        # Number of input and output variables to neural net
        # Stencil size * number of layers * number of variables
        n_input = 6*2*3
        # Number of layers * number of variables
        n_output = 2*3

        print(f"Training with {n_train} training pairs, dimensions: ({n_input}, {n_output})")

        # Define input and output arrays
        train_in  = np.zeros((n_train,n_input))
        train_out = np.zeros((n_train,n_output))

        # Prepare training data. Different grid points and time steps are considered as independent
        # training pairs. The northern and southern boundaries are also treated equivalently, only
        # the southern boundary is flipped
        i = 0
        for t in range(n_time-1):
            for x in range(n_lon):
                # Form training pairs for top of domain
                train_in[i,:6*2]     = BoundariesNN.get_stencil(q[t,...], x, n_lon)
                train_in[i,6*2:12*2] = BoundariesNN.get_stencil(u[t,...], x, n_lon)
                train_in[i,12*2:]    = BoundariesNN.get_stencil(v[t,...], x, n_lon)
                train_out[i,:2]  = q[t+1,:,0,x] - q[t,:,0,x]
                train_out[i,2:4] = u[t+1,:,0,x] - u[t,:,0,x]
                train_out[i,4:]  = v[t+1,:,0,x] - v[t,:,0,x]
                i+=1

                # Form training pairs for bottom of domain (just reverse the vertical coordinate
                # and call the same function)
                train_in[i,:6*2]     = BoundariesNN.get_stencil(q[t,:,::-1,:], x, n_lon)
                train_in[i,6*2:12*2] = BoundariesNN.get_stencil(u[t,:,::-1,:], x, n_lon)
                train_in[i,12*2:]    = BoundariesNN.get_stencil(v[t,:,::-1,:], x, n_lon)
                train_out[i,:2]  = q[t+1,:,-1,x] - q[t,:,-1,x]
                train_out[i,2:4] = u[t+1,:,-1,x] - u[t,:,-1,x]
                train_out[i,4:]  = v[t+1,:,-1,x] - v[t,:,-1,x]
                i+=1

        print("Training data prepared")

        # Build model for training
        model = build_model(
            n_input, n_output,
            BoundariesNN.n_hidden_layers, BoundariesNN.n_per_hidden_layer
        )

        # Train!
        history = model.fit(train_in, train_out, epochs=200, batch_size=128, validation_split=0.2)

        # Output weights and diagnostics files
        save_history(f"{BoundariesNN.out_file}_history.txt", history)
        model.save_weights(f"{BoundariesNN.out_file}.hdf")

    """
    Extracts the stencil corresponding to the requested longitude.
    e.g. if you request the 2nd longitude (index starting from 0)
    ---------------------------    -------
    |a|b|c|d|e|f|g|h|i|j|k|l|m|    |b|c|d|
    --------------------------- => -------
    |n|o|p|q|r|s|t|u|v|w|x|y|z|    |o|p|q|
    ---------------------------    -------
    """
    @staticmethod
    def get_stencil(full_array, lon, n_lon):
        top = full_array[:,:2,:]
        stencil = top[:,:,np.array(range(lon-1,lon+2))%n_lon]
        return stencil.flatten()