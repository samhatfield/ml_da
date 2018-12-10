import numpy as np
from model_params import F, h, c, b

"""
Step model state forward by one timestep
"""
def step(previous):
    from model_params import δt
    
    # 4th order Runge-Kutta
    k1 = ode(previous)
    k2 = ode(previous+0.5*δt*k1)
    k3 = ode(previous+0.5*δt*k2)
    k4 = ode(previous+δt*k3)

    return previous + (δt/6)*(k1 + 2*k2 + 2*k3 + k4)

"""
ODE for whole Lorenz '96 system
"""
def ode(state):
    from model_params import nx, ny
    
    x = state[:nx]
    y = state[nx:]

    return np.concatenate((dxdt(x,y), dydt(x,y)))

"""
Individual ODEs for X and Y variables
"""
def dxdt(x, y):    
    sum_y = np.sum(np.split(y, x.shape[0]), axis=1)
    return (np.roll(x, -1) - np.roll(x, 2))*np.roll(x, 1) - x + F - (h*c/b)*sum_y

def dydt(x, y):
    x_rpt = np.repeat(x, y.shape[0]/x.shape[0])
    return (np.roll(y, 1) - np.roll(y, -2))*c*b*np.roll(y, -1) - c*y + (h*c/b)*x_rpt