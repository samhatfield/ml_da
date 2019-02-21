import numpy as np
from numerical_model.qg_constants import qg_constants as const


# Define model orography (Gaussian hill centred on (icentre,jcentre))
def define_orography():
    nx = int(const.nx)
    ny = int(const.ny)
    deltax0 = float(const.deltax0)
    deltay0 = float(const.deltay0)
    orogmax = float(const.rsmax)
    worog = float(const.worog)

    orog = np.zeros((nx,ny))
    icentre = nx/4
    jcentre = 3*ny/4
    for j in range(ny):
        for i in range(nx):
            distx = min([icentre - (i+1), nx - (icentre - (i+1))]) * deltax0
            disty = np.abs((j+1) - jcentre)*deltay0
            orog[i,j] = orogmax*np.exp(-(distx*distx+disty*disty)/(worog*worog))

    return orog


# Invent an initial state for the QG model.
#
# This routine invents an initial state for the QG model. It is used to
# initialise the "truth run". The initial state consists of a horizontally
# uniform wind in each layer, with a vertical shear sufficient to produce
# baroclinic instability. Povided the orography is non-zero and is not
# symmetrically place in the domain, this is sufficient to generate a
# non-trivial flow after a few days of integration.
#
# Two slightly different initial states may be created (according to whether
# or not ctype is set to 'f').
def invent_state(orog):
    from numerical_model.subroutines import calc_pv

    nx = int(const.nx)
    ny = int(const.ny)
    deltax = float(const.deltax)
    deltay = float(const.deltay)
    u1 = float(const.u1)
    u2 = float(const.u2)
    f1 = float(const.f1)
    f2 = float(const.f2)
    bet = float(const.bet)

    # Streamfunction just below and above model domain
    x_south = np.zeros((2))
    x_north = -(ny+1)*deltay*np.array([u1, u2])

    # Define streamfunction in model domain
    x = np.zeros((nx,ny,2))
    for j in range(ny):
        for i in range(nx):
            x[i,j,0] = -(j+1)*deltay*u1
            x[i,j,1] = -(j+1)*deltay*u2

    # Get PV from streamfunction
    pv = calc_pv(x, x_north, x_south, f1, f2, deltax, deltay, bet, orog, nx, ny)

    # Define PV just below and above model domain
    q_south = np.zeros((nx,2))
    q_north = np.zeros((nx,2))
    for i in range(nx):
        q_south[i,0] = 2.0*pv[i,0,0] - pv[i,1,0]
        q_south[i,1] = 2.0*pv[i,0,1] - pv[i,1,1]
        q_north[i,0] = 2.0*pv[i,-1,0] - pv[i,-2,0]
        q_north[i,1] = 2.0*pv[i,-1,1] - pv[i,-2,1]

    return x, x_north, x_south, q_north, q_south
