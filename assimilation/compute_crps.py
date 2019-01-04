from iris import load, analysis
import numpy as np

def get_crps(observation, ensemble):
    print("Computing CRPS")

    # Get number of ensemble members
    num_mem = ensemble.coord("member").shape[0]

    # Unify time coordinates for both observation and ensemble cubes
    #unify_time_units([observation, ensemble])
    if observation.coord('time_step').points.shape[0] > ensemble.coord('time_step').points.shape[0]:
        observation = observation.subset(ensemble.coord('time_step'))
    else:
        ensemble = ensemble.subset(observation.coord('time_step'))

    # Extract raw arrays
    ens_arr = ensemble.data
    obs_arr = observation.data

    # Compute mean innovations
    print("Computing mean innovations")
    mean_innov = np.mean(np.abs(ens_arr - obs_arr), axis=0)

    # Compute mean ensemble member difference across all member pairs
    print("Computing mean pair differences")
    mean_pair_difference = np.zeros(obs_arr.shape)
    for i, mem in enumerate(ensemble.slices_over("member")):
        mem_arr = mem.data
        mean_pair_difference += np.sum(np.abs(mem_arr - ens_arr), axis=0)

    if small_ensemble_correction:
        mean_pair_difference /= 2*num_mem*(num_mem-1)
    else:
        mean_pair_difference /= 2*num_mem**2

    crps = observation.copy(); crps.rename("CRPS")
    crps.data = mean_innov - mean_pair_difference

    return crps

# Calculate small-ensemble-corrected CRPS
small_ensemble_correction = True

# Load cubes
truth, ensemble = tuple(load("output.nc", ["truth_x", "ensemble_x"]))

# Compute space-mean time-mean CRPS
crps = get_crps(truth, ensemble).collapsed(["time_step", "i"], analysis.MEAN)

print(f"CRPS = {crps.data}")
