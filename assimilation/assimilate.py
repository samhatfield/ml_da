import numpy as np
from observation import observe

# Background error covariance inflation factor
ρ = 1.1

def assimilate(ensemble, obs, obs_covar):
    n_ens = ensemble.shape[0]

    # Form ensemble mean
    ens_mean = np.mean(ensemble, axis=0)

    # Form the background ensemble perturbation matrix
    X_f = ρ*(ensemble - ens_mean)

    # Sequentially process observations
    for i, ob in enumerate(obs):
        # Ensemble covariance times transpose of observation matrix
        P_f_H_T = np.matmul(X_f.T, observe(X_f, i))/(n_ens - 1)

        HP_f_H_T = observe(P_f_H_T, i)

        # Kalman gain
        gain = P_f_H_T / (HP_f_H_T + obs_covar[i,i])

        # Update ensemble mean
        ens_mean += gain*(ob - observe(ens_mean, i))

        # Update perturbations
        α = 1.0/(1.0+np.sqrt(obs_covar[i,i]/(HP_f_H_T + obs_covar[i,i])))
        for j in range(n_ens):
            X_f[j,:] += -α*gain*observe(X_f[j,:], i)

    # Form final ensemble
    return ens_mean + X_f
