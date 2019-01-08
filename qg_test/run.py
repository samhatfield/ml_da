from numerical_model.subroutines import prepare_integration
from numerical_model.qg_constants import qg_constants as const
import numpy as np
import matplotlib.pyplot as plt

nx = int(const.nx)
ny = int(const.ny)
deltax0 = float(const.deltax0)
deltay0 = float(const.deltay0)
rsmax = float(const.rsmax)
worog = float(const.worog)

rs = np.zeros((nx,ny))
icentre = nx/4
jcentre = 3*ny/4
for j in range(ny):
    for i in range(nx):
        distx = min([icentre - i, nx - (icentre - i)]) * deltax0
        disty = np.abs(j - jcentre)*deltay0
        rs[i,j] = rsmax*np.exp(-(distx*distx+disty*disty)/(worog*worog))

q = np.zeros((nx,ny,2))
x = np.zeros((nx,ny,2))
x_north = np.zeros((2))
x_south = np.zeros((2))

q, x, x_north, x_south = prepare_integration(q, x, x_north, x_south, rs)

