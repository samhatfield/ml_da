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

    # Number of input and output variables to neural net
    # Stencil size * number of layers
    n_input = 6*2
    # Number of layers
    n_output = 2

    def __init__(self):
        from numerical_model.qg_constants import qg_constants as const
        from util import build_model

        # Build model for inference
        self.model = build_model(
            BoundariesNN.n_input, BoundariesNN.n_output,
            BoundariesNN.n_hidden_layers, BoundariesNN.n_per_hidden_layer
        )

        # Try loading weights file
        try:
            self.model.load_weights(f"{BoundariesNN.out_file}.hdf", by_name=False)
        except OSError as e:
            print("Weights file for BoundariesNN doesn't exist\nHave you trained this model yet?")
            raise e

        # Store number of longitudes
        self.n_lon = int(const.nx)

    """
    Compute tendencies of the variables on the northern and southern boundaries.
    """
    def get_tend(self, 𝛙):
        # Prepare input array for neural net
        infer_in = np.zeros((self.n_lon*2,6*2))

        # Loop over all longitudes, extracting the northern and southern boundary variables
        i = 0
        for x in range(self.n_lon):
            infer_in[i,:]     = BoundariesNN.get_stencil(𝛙, x, self.n_lon)
            i+=1

            infer_in[i,:]     = BoundariesNN.get_stencil(𝛙[:,::-1,:], x, self.n_lon)
            i+=1

        # Normalize input
        infer_in = BoundariesNN.normalize_input(infer_in)

        # Predict new tendencies (tendencies include dt term)
        tendencies = self.model.predict(infer_in, batch_size=1)

        # Denormalize output
        tendencies = BoundariesNN.denormalize_output(tendencies)

        # Unpack tendencies
        𝛙_tend = np.zeros((self.n_lon,2,2))
        𝛙_tend[:,:,0] = tendencies[:,0].reshape((self.n_lon,2))
        𝛙_tend[:,:,1] = tendencies[:,1].reshape((self.n_lon,2))

        return 𝛙_tend

    """
    Train the neural net based on the input training data of 𝛙 (streamfunction).
    """
    @staticmethod
    def train():
        from util import build_model, save_history
        from iris import load_cube
        from numpy.random import shuffle

        # Attempt to load processed training data
        print("Attempting to load prepared training data")
        try:
            training_data = np.load(f"{BoundariesNN.out_file}_training_data.npz")

            # Split up training data into input and output
            train_in, train_out  = training_data["train_in"], training_data["train_out"]
        except FileNotFoundError:
            print("Prepared training data not found. Preparing now...")

            # Load training data
            𝛙 = load_cube("training_data.nc", ["psi"])

            # Transpose data so it's lon, lat, lev, time
            𝛙.transpose()

            train_in, train_out = BoundariesNN.prepare_training_data(𝛙.data)

            print("Training data prepared")

        print(f"Training with {train_in.shape[0]} training pairs,\
            dimensions: ({BoundariesNN.n_input}, {BoundariesNN.n_output})")

        # Build model for training
        model = build_model(
            BoundariesNN.n_input, BoundariesNN.n_output,
            BoundariesNN.n_hidden_layers, BoundariesNN.n_per_hidden_layer
        )

        # Shuffle training data
        print("Shuffling training data")
        indices = np.arange(train_in.shape[0], dtype=np.int32)
        shuffle(indices)
        train_in  = train_in[indices,:]
        train_out = train_out[indices,:]

        # Train!
        history = model.fit(train_in, train_out, epochs=20, batch_size=128, validation_split=0.2)

        # Output weights and diagnostics files
        save_history(f"{BoundariesNN.out_file}_history.txt", history)
        model.save_weights(f"{BoundariesNN.out_file}.hdf")

    @staticmethod
    def prepare_training_data(𝛙):
        # Get dimensions
        n_lon, n_lat, _, n_time = 𝛙.shape
        print(f"{n_lon} longitudes, {n_lat} latitudes, 2 levels, {n_time} timesteps")

        # Compute number of training pairs
        # 2 (top and bottom) * number of time steps (minus 1) * number of layers
        # * number of longitudes
        n_train = 2*(n_time-1)*n_lon

        # Define input and output arrays
        train_in  = np.zeros((n_train,BoundariesNN.n_input))
        train_out = np.zeros((n_train,BoundariesNN.n_output))

        # Prepare training data. Different grid points and time steps are considered as independent
        # training pairs. The northern and southern boundaries are also treated equivalently, only
        # the southern boundary is flipped
        i = 0
        for t in range(n_time-1):
            for x in range(n_lon):
                # Form training pairs for top of domain
                train_in[i,:]   = BoundariesNN.get_stencil(𝛙[...,t], x, n_lon)
                train_out[i,:]  = 𝛙[x,0,:,t+1] - 𝛙[x,0,:,t]
                i+=1

                # Form training pairs for bottom of domain (just reverse the vertical coordinate
                # and call the same function)
                train_in[i,:]   = BoundariesNN.get_stencil(𝛙[:,::-1,:,t], x, n_lon)
                train_out[i,:]  = 𝛙[x,-1,:,t+1] - 𝛙[x,-1,:,t]
                i+=1

        # Normalize training data
        train_in  = BoundariesNN.normalize_input(train_in)
        train_out = BoundariesNN.normalize_output(train_out)

        np.savez(f"{BoundariesNN.out_file}_training_data.npz",\
            train_in=train_in, train_out=train_out)
        return train_in, train_out

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
        stencil = top[np.array(range(lon-1,lon+2))%n_lon,:,:]
        return stencil.flatten()

    """
    Normalize the given input training data so values are between -1.0 and 1.0.
    """
    @staticmethod
    def normalize_input(training_data):
        # Maximum and minimum values of 𝛙 based on a long run of the numerical model
        𝛙_max, 𝛙_min = 4.5, -27.0

        # Normalize the training data
        return 2.0*(training_data - 𝛙_min)/(𝛙_max - 𝛙_min) - 1.0

    """
    Normalize the given output training data so values are between -1.0 and 1.0.
    """
    @staticmethod
    def normalize_output(training_data):
        # Maximum and minimum values of tendencies of 𝛙 based on a long run of the numerical model
        𝛙_max, 𝛙_min = 0.06, -0.06

        # Normalize the training data
        return 2.0*(training_data - 𝛙_min)/(𝛙_max - 𝛙_min) - 1.0

    """
    Denormalize the given output.
    """
    @staticmethod
    def denormalize_output(output):
        # Maximum and minimum values of tendencies of 𝛙 based on a long run of the numerical model
        𝛙_max, 𝛙_min = 0.06, -0.06

        # Denormalize the output
        return (𝛙_max - 𝛙_min)*(1.0 + output)/2.0  + 𝛙_min