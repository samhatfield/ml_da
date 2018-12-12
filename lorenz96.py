import numpy as np
from model_params import *

class TwoLayerLorenz96Model:
    """
    Step model state forward by one timestep
    """
    def step(self, previous):    
        # 4th order Runge-Kutta
        k1 = self.ode(previous)
        k2 = self.ode(previous+0.5*δt*k1)
        k3 = self.ode(previous+0.5*δt*k2)
        k4 = self.ode(previous+δt*k3)

        return previous + (δt/6)*(k1 + 2*k2 + 2*k3 + k4)

    """
    ODE for two layer Lorenz '96 system
    """
    def ode(self, state):    
        # Break up state vector into X, Y and Z components
        x = state[:nx]
        y = state[nx:nx+nx*ny]

        return np.concatenate((self.dxdt(x,y), self.dydt(x,y)))

    """
    Individual ODEs for X and Y variables
    """
    def dxdt(self, x, y):    
        sum_y = np.sum(np.split(y, x.shape[0]), axis=1)
        return (np.roll(x, -1) - np.roll(x, 2))*np.roll(x, 1) - x + F - (h*c/b)*sum_y

    def dydt(self, x, y):
        x_rpt = np.repeat(x, y.shape[0]/x.shape[0])
        return (np.roll(y, 1) - np.roll(y, -2))*c*b*np.roll(y, -1) - c*y + (h*c/b)*x_rpt\

class ThreeLayerLorenz96Model(TwoLayerLorenz96Model):
    """
    ODE for three layer Lorenz '96 system
    """
    def ode(self, state):    
        # Break up state vector into X, Y and Z components
        x = state[:nx]
        y = state[nx:nx+nx*ny]
        z = state[nx+nx*ny:]

        return np.concatenate((self.dxdt(x,y), self.dydt(x,y,z), self.dzdt(y,z)))

    """
    Individual ODEs for X, Y and Z variables
    """
    def dydt(self, x, y, z):
        x_rpt = np.repeat(x, y.shape[0]/x.shape[0])
        sum_z = np.sum(np.split(z, y.shape[0]), axis=1)
        return (np.roll(y, 1) - np.roll(y, -2))*c*b*np.roll(y, -1) - c*y + (h*c/b)*x_rpt\
            - (h*e/d)*sum_z
    
    def dzdt(self, y, z):
        y_rpt = np.repeat(y, z.shape[0]/y.shape[0])
        return (np.roll(z, -1) - np.roll(z, 2))*e*d*np.roll(z, 1) - g_z*e*z + (h*e/d)*y_rpt