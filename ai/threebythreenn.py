import numpy as np
from boundariesnn import BoundariesNN

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
    n_input = 9*2
    # Number of layers * number of variables
    n_output = 2

    def __init__(self):
        from numerical_model.qg_constants import qg_constants as const
        from util import build_model

        # Build model for inference of boundary variables
        self.boundariesnn = BoundariesNN()

        # Build model for inference of interior of model domain
        self.three_by_three_model = build_model(
            ThreeByThreeNN.n_input, ThreeByThreeNN.n_output,
            ThreeByThreeNN.n_hidden_layers, ThreeByThreeNN.n_per_hidden_layer
        )

        # Try loading weights file
        try:
            self.three_by_three_model.load_weights(f"{ThreeByThreeNN.out_file}.hdf", by_name=False)
        except OSError as e:
            print("Weights file for ThreeByThreeNN doesn't exist\nHave you trained this model yet?")
            raise e

        # Store number of longitudes and latitudes
        self.n_lon = int(const.nx)
        self.n_lat = int(const.ny)

        # Stores Adams-Bashforth steps
        self.𝛙_tends = np.zeros((3,self.n_lon,self.n_lat,2))
        self.mode = 0

    """
    Advance variables by one time step.
    """
    def step(self, 𝛙):
        self.𝛙_tends = np.roll(self.𝛙_tends, 1, axis=0)

        # Prepare input array for neural net
        infer_in = np.zeros((self.n_lon*(self.n_lat-2),9*2))

        # Loop over all longitudes and latitudes
        i = 0
        for x in range(self.n_lon):
            for y in range(1,self.n_lat-1):
                infer_in[i,:] = ThreeByThreeNN.get_stencil(𝛙, x, y, self.n_lon)
                i+=1

        infer_in = ThreeByThreeNN.normalize_input(infer_in)

        # Predict new tendencies (tendencies include dt term)
        tendencies = self.three_by_three_model.predict(infer_in, batch_size=1)

        # Denormalize output
        tendencies = ThreeByThreeNN.denormalize_output(tendencies)

        # Unpack tendencies
        self.𝛙_tends[0,:,1:-1,0] = tendencies[:,0].reshape((self.n_lon,self.n_lat-2))
        self.𝛙_tends[0,:,1:-1,1] = tendencies[:,1].reshape((self.n_lon,self.n_lat-2))

        # Compute tendencies for boundaries
        𝛙_tend_bound = self.boundariesnn.get_tend(𝛙)
        self.𝛙_tends[0,:,0,:]  = 𝛙_tend_bound[:,0,:]
        self.𝛙_tends[0,:,-1,:] = 𝛙_tend_bound[:,1,:]

        # 3rd order Adams-Bashforth
        if self.mode == 0:
            𝛙_tend = self.𝛙_tends[0,...]
            self.mode = 1
        elif self.mode == 1:
            𝛙_tend = 1.5*self.𝛙_tends[0,...] - 0.5*self.𝛙_tends[1,...]
            self.mode = 2
        else:
            𝛙_tend = (23.0/12.0)*self.𝛙_tends[0,...] - (4.0/3.0)*self.𝛙_tends[1,...] \
                + (5.0/12.0)*self.𝛙_tends[2,...]

        # Step forward using forward Euler
        return 𝛙 + 𝛙_tend

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
            training_data = np.load(f"{ThreeByThreeNN.out_file}_training_data.npz")

            # Split up training data into input and output
            train_in, train_out  = training_data["train_in"], training_data["train_out"]
        except FileNotFoundError:
            print("Prepared training data not found. Preparing now...")

            # Load training data
            𝛙 = load_cube("training_data.nc", ["psi"])

            # Transpose data so it's lon, lat, lev, time
            𝛙.transpose()

            train_in, train_out = ThreeByThreeNN.prepare_training_data(𝛙.data)

            print("Training data prepared")

        print(f"Training with {train_in.shape[0]} training pairs,\
            dimensions: ({ThreeByThreeNN.n_input}, {ThreeByThreeNN.n_output})")

        # Build model for training
        model = build_model(
            ThreeByThreeNN.n_input, ThreeByThreeNN.n_output,
            ThreeByThreeNN.n_hidden_layers, ThreeByThreeNN.n_per_hidden_layer
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
        save_history(f"{ThreeByThreeNN.out_file}_history.txt", history)
        model.save_weights(f"{ThreeByThreeNN.out_file}.hdf")

    @staticmethod
    def prepare_training_data(𝛙):
        # Get dimensions
        n_lon, n_lat, _, n_time = 𝛙.shape
        print(f"{n_lon} longitudes, {n_lat} latitudes, 2 levels, {n_time} timesteps")

        # Compute number of training pairs
        # number of time steps (minus 1) * number of layers
        # * number of latitudes (minus top and bottom) * number of longitudes
        n_train = (n_time-1)*(n_lat-2)*n_lon

        # Define input and output arrays
        train_in  = np.zeros((n_train,ThreeByThreeNN.n_input))
        train_out = np.zeros((n_train,ThreeByThreeNN.n_output))

        # Prepare training data. Different grid points and time steps are considered as independent
        # training pairs.
        i = 0
        for t in range(n_time-1):
            for x in range(n_lon):
                for y in range(1,n_lat-1):
                    train_in[i,:]  = ThreeByThreeNN.get_stencil(𝛙[...,t], x, y, n_lon)
                    train_out[i,:] = 𝛙[x,y,:,t+1] - 𝛙[x,y,:,t]
                    i+=1

        # Normalize input
        train_in  = ThreeByThreeNN.normalize_input(train_in)
        train_out = ThreeByThreeNN.normalize_output(train_out)

        np.savez(f"{ThreeByThreeNN.out_file}_training_data.npz",\
            train_in=train_in, train_out=train_out)
        return train_in, train_out

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