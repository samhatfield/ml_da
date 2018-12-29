import numpy as np
from netCDF4 import Dataset
from numerical_model.params import params

def setup_output(n_ens):
    nx = int(params.nx)

    # Define NetCDF dataset to store all output
    dataset = Dataset("output.nc", "w", format="NETCDF4_CLASSIC")

    # Define dimensions
    timedim = dataset.createDimension("time_step", None)
    memdim  = dataset.createDimension("member", n_ens)
    idim    = dataset.createDimension("i", nx)

    # Define dimension variables
    timevar = dataset.createVariable("time_step", np.int32, ("time_step",))
    memvar  = dataset.createVariable("member", np.int32, ("member",))
    ivar    = dataset.createVariable("i", np.int32, ("i",))

    # Define multidimensional variables
    truth_x = dataset.createVariable("truth_x", np.float32, ("time_step", "i"))
    ens_x   = dataset.createVariable("ensemble_x", np.float32, ("time_step", "member", "i"))

    # Assign values to non-unlimited dimensions
    memvar[:] = np.array([mem for mem in range(n_ens)], dtype=np.int32)
    ivar[:]   = np.array([i for i in range(nx)], dtype=np.int32)

    dataset.close()

def output(time_step, time_index, ensemble, truth):
    # Append latest data along time dimension
    dataset = Dataset("output.nc", "a", format="NETCDF4_CLASSIC")
    dataset["truth_x"][time_index,:]      = truth[:]
    dataset["ensemble_x"][time_index,:,:] = ensemble[:,:]
    dataset.close()
