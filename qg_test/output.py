import numpy as np
from netCDF4 import Dataset
from numerical_model.qg_constants import qg_constants as const

def setup_output(start_date):
    nx, ny = int(const.nx), int(const.ny)
    d1, d2 = float(const.d1), float(const.d2)

    # Define NetCDF dataset to store all output
    dataset = Dataset("output.nc", "w", format="NETCDF4_CLASSIC")

    # Define dimensions
    timedim = dataset.createDimension("time", None)
    idim    = dataset.createDimension("i", nx)
    jdim    = dataset.createDimension("j", ny)
    levdim  = dataset.createDimension("lev", 2)

    # Define dimension variables
    timevar = dataset.createVariable("time", np.int32, ("time",))
    ivar    = dataset.createVariable("i",    np.int32, ("i",))
    jvar    = dataset.createVariable("j",    np.int32, ("j",))
    levvar  = dataset.createVariable("lev",  np.int32, ("lev"))

    # Set time units
    timevar.setncatts({"units": f"minutes since {start_date:%Y-%m-%dT%H:%M:%SZ}"})

    # Define multidimensional variables
    pv  = dataset.createVariable("pv",  np.float32, ("time", "lev", "j", "i"))
    psi = dataset.createVariable("psi", np.float32, ("time", "lev", "j", "i"))
    u   = dataset.createVariable("u",   np.float32, ("time", "lev", "j", "i"))
    v   = dataset.createVariable("v",   np.float32, ("time", "lev", "j", "i"))

    # Assign values to non-unlimited dimensions
    ivar[:]   = np.array([i for i in range(nx)], dtype=np.int32)
    jvar[:]   = np.array([j for j in range(ny)], dtype=np.int32)
    levvar[:] = np.array([d1, d2])

    dataset.close()

def output(start_date, date, time_index, pv, psi, u, v):
    # Append latest data along time dimension
    dataset = Dataset("output.nc", "a", format="NETCDF4_CLASSIC")
    dataset["time"][time_index] = (date - start_date).total_seconds()/60.0
    dataset["pv"][time_index,:,:,:]  = np.transpose(pv[:,:,:])
    dataset["psi"][time_index,:,:,:] = np.transpose(psi[:,:,:])
    dataset["u"][time_index,:,:,:] = np.transpose(u[:,:,:])
    dataset["v"][time_index,:,:,:] = np.transpose(v[:,:,:])
    dataset.close()
