from ai.neural_net import build_model
from numerical_model.params import params
import numpy as np

class Forecaster:
    def __init__(self, n_ens):
        from os.path import dirname

        nx = int(params.nx)

        # Build model and load weights
        self.model = build_model(nx, nx)
        weights_file = dirname(__file__) + "/weights"
        self.model.load_weights(weights_file, by_name=False)

        # Stores Adams-Bashforth steps
        self.tends = np.zeros((3,n_ens,nx))
        self.mode = 0

    def step(self, members):
        self.tends = np.roll(self.tends, 1, axis=0)
        mems_norm = self.normalize(members)
        self.tends[0,:,:] = self.model.predict(mems_norm, batch_size=1)

        # Adams-Bashforth
        if self.mode == 0:
            tend = self.tends[0,:,:]
            self.mode = 1
        elif self.mode == 1:
            tend = 1.5*self.tends[0,:,:] - 0.5*self.tends[1,:,:]
            self.mode = 2
        else:
            tend = (23.0/12.0)*self.tends[0,:,:] - (4.0/3.0)*self.tends[1,:,:] + (5.0/12.0)*self.tends[2,:,:]
        return members + tend

    def normalize(self, state):
        # Normalisation factors for input layer
        max_train = 30.0
        min_train = -20.0

        return 2.0*(state - min_train)/(max_train - min_train) - 1.0
