import numpy as np
from boundariesnn import BoundariesNN


class InteriorNN:
    """
    Class for predicting the interior domain of the QG model using a neural net.
    The stencil size is passed as a constructor argument.
    """

    # Number of hidden layers and nodes per hidden layer
    n_hidden_layers = 2
    n_per_hidden_layer = 40

    # Number of output variables to neural net (just the number of layers)
    n_output = 2

    # Fraction of raw training data to use for validation
    val_frac = 0.2

    def __init__(self, stencil):
        from numerical_model.qg_constants import qg_constants as const
        from util import build_model

        # Build model for inference of boundary variables
        self.boundariesnn = BoundariesNN()

        # Set stencil size
        self.stencil = stencil

        # Set output file (minus extension)
        self.out_file = f"interior_{stencil}"

        # Number of layers * stencil size
        self.n_input = 2*stencil**2

        # Build model for inference of interior of model domain
        self.interior_model = build_model(
            self.n_input, InteriorNN.n_output,
            InteriorNN.n_hidden_layers, InteriorNN.n_per_hidden_layer
        )

        # Try loading weights file
        try:
            self.interior_model.load_weights(f"models/{self.out_file}.hdf", by_name=False)
        except OSError as e:
            print(f"File models/{self.out_file}.hdf doesn't exist")
            print("Have you trained this model yet?")
            raise e

        # Store number of longitudes and latitudes
        self.n_lon = int(const.nx)
        self.n_lat = int(const.ny)

        # Compute streamfunction above and below boundaries
        Δy = float(const.deltay)
        self.x_south = np.zeros((2))
        self.x_north = -(self.n_lat+1)*Δy*np.array([float(const.u1), float(const.u2)])

        # Stores Adams-Bashforth steps
        self.𝛙_tends = np.zeros((3,self.n_lon,self.n_lat,2))
        self.mode = 0

    """
    Advance variables by one time step.
    """
    def step(self, 𝛙):
        self.𝛙_tends = np.roll(self.𝛙_tends, 1, axis=0)

        # Pad input for inferring grid points near boundaries
        pad = int((self.stencil-1)/2)
        𝛙_pad = np.zeros((𝛙.shape[0], 𝛙.shape[1]+2*pad, 𝛙.shape[2]))
        𝛙_pad[:,:pad,0]  = self.x_south[0]
        𝛙_pad[:,:pad,1]  = self.x_south[1]
        𝛙_pad[:,-pad:,0] = self.x_north[0]
        𝛙_pad[:,-pad:,1] = self.x_north[1]
        𝛙_pad[:,pad:-pad,:] = 𝛙

        # Prepare input array for neural net
        infer_in = np.zeros((self.n_lon*self.n_lat,2*self.stencil**2))

        # Loop over all longitudes and latitudes
        i = 0
        for x in range(self.n_lon):
            for y in range(self.n_lat):
                infer_in[i,:] = InteriorNN.get_stencil(𝛙_pad, x, y+pad, self.n_lon, self.stencil)
                i+=1

        # Normalize input
        infer_in = InteriorNN.normalize_input(infer_in)

        # Predict new tendencies (tendencies include dt term)
        tendencies = self.interior_model.predict(infer_in, batch_size=1)

        # Denormalize output
        tendencies = InteriorNN.denormalize_output(tendencies)

        # Unpack tendencies
        self.𝛙_tends[0,:,:,0] = tendencies[:,0].reshape((self.n_lon,self.n_lat))
        self.𝛙_tends[0,:,:,1] = tendencies[:,1].reshape((self.n_lon,self.n_lat))

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
    def train(stencil):
        from util import build_model, save_history
        from iris import load_cube

        # Attempt to load processed training data
        print("Attempting to load prepared training data")
        try:
            training_data   = np.load(f"training_data/interior_{stencil}_training_data.npz")
            validation_data = np.load(f"training_data/interior_{stencil}_validation_data.npz")

            # Split up training and validation data into input and output
            train_in, train_out  = training_data["train_in"], training_data["train_out"]
            val_in, val_out      = validation_data["val_in"], validation_data["val_out"]
        except FileNotFoundError:
            print("Prepared training data not found. Preparing now...")

            # Load training data
            𝛙 = load_cube("training_data/training_data.nc", ["psi"])

            # Transpose data so it's lon, lat, lev, time
            𝛙.transpose()

            train_in, train_out, val_in, val_out = InteriorNN.prepare_training_data(𝛙.data, stencil)

            print("Training data prepared")

        print(f"Training with {train_in.shape[0]} training pairs,\
            dimensions: ({2*stencil**2}, {InteriorNN.n_output})")

        # Build model for training
        model = build_model(
            2*stencil**2, InteriorNN.n_output,
            InteriorNN.n_hidden_layers, InteriorNN.n_per_hidden_layer
        )

        # Train!
        history = model.fit(train_in, train_out, epochs=20, batch_size=128,
                            validation_data=(val_in, val_out))

        # Output weights and diagnostics files
        save_history(f"models/interior_{stencil}_history.txt", history)
        model.save_weights(f"models/interior_{stencil}.hdf")

    """
    Prepare training data, including validation split.
    """
    @staticmethod
    def prepare_training_data(𝛙, stencil):
        from numpy.random import shuffle

        # Get dimensions
        n_lon, n_lat, _, n_time = 𝛙.shape
        print(f"{n_lon} longitudes, {n_lat} latitudes, 2 levels, {n_time} timesteps")

        # Compute number of training pairs
        # number of time steps (minus 1) * number of layers
        # * number of latitudes (minus stencil-1 rows from top and bottom) * number of longitudes
        n_train = (n_time-1)*(n_lat-(stencil-1))*n_lon

        # Define input and output arrays
        train_in_all  = np.zeros((n_train,2*stencil**2))
        train_out_all = np.zeros((n_train,InteriorNN.n_output))

        # How many rows to skip, depending on stencil size
        skip = int((stencil-1)/2)

        # Prepare training data. Different grid points and time steps are considered as independent
        # training pairs.
        i = 0
        for t in range(n_time-1):
            for x in range(n_lon):
                for y in range(skip,n_lat-skip):
                    train_in_all[i,:]  = InteriorNN.get_stencil(𝛙[...,t], x, y, n_lon, stencil)
                    train_out_all[i,:] = 𝛙[x,y,:,t+1] - 𝛙[x,y,:,t]
                    i += 1

        # Normalize training data
        train_in_all  = InteriorNN.normalize_input(train_in_all)
        train_out_all = InteriorNN.normalize_output(train_out_all)

        # Shuffle training data and extract validation set
        indices = np.arange(n_train, dtype=np.int32)
        shuffle(indices)
        train_indices = indices[:-int(InteriorNN.val_frac*n_train)]
        val_indices   = indices[-int(InteriorNN.val_frac*n_train):]
        train_in  = train_in_all[train_indices,:]
        train_out = train_out_all[train_indices,:]
        val_in    = train_in_all[val_indices,:]
        val_out   = train_out_all[val_indices,:]

        # Save training and validation data to file
        np.savez(f"training_data/interior_{stencil}_training_data.npz",
                 train_in=train_in, train_out=train_out)
        np.savez(f"training_data/interior_{stencil}_validation_data.npz",
                 val_in=val_in, val_out=val_out)

        return train_in, train_out, val_in, val_out

    """
    Extracts the nxn stencil corresponding to the requested longitude and latitude.
    e.g. if you request the 2nd longitude, 1st latitude (index starting from 0), and the stencil
    size is 3x3
    ---------------------------    -------
    |a|b|c|d|e|f|g|h|i|j|k|l|m|    |b|c|d|
    ---------------------------    -------
    |n|o|p|q|r|s|t|u|v|w|x|y|z| => |o|p|q|
    ---------------------------    -------
    |a|b|c|d|e|f|g|h|i|j|k|l|m|    |b|c|d|
    ---------------------------    -------
    """
    @staticmethod
    def get_stencil(full_array, lon, lat, n_lon, stencil):
        include = int((stencil-1)/2)
        lons = np.array(range(lon-include,lon+include+1))%n_lon
        stencil = full_array[lons,lat-include:lat+include+1,:]
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
