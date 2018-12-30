import matplotlib.pyplot as plt
import seaborn as sns
from iris import load, analysis, Constraint
import iris.plot as iplt

sns.set_style('whitegrid', {'font.sans-serif': "Helvetica"})

# X variable to plot
x_plot = 0

# Load cubes
truth, ens, obs = tuple(load("output.nc", ["truth_x", "ensemble_x", "obs_x"]))

# Constrain on one X variable
truth = truth.extract(Constraint(i=x_plot))
ens   = ens.extract(Constraint(i=x_plot))
obs   = obs.extract(Constraint(i=x_plot))

# Compute ensemble mean and standard deviation
ens_mean = ens.collapsed(['member'], analysis.MEAN)
ens_std = ens.collapsed(['member'], analysis.STD_DEV)

# Get time coordinate
time = truth.coord('time_step')

# Plot truth, ens mean and ens spread
fig = plt.figure(figsize=(10,3))
truth_h, = iplt.plot(truth, label="truth")
ens_h, = iplt.plot(ens_mean, label="ensemble mean")
obs_h, = iplt.plot(obs, "x", label="observations")
plt.fill_between(
        time.points, (ens_mean-ens_std).data, (ens_mean+ens_std).data,
        facecolor=ens_h.get_color(), alpha=0.5, edgecolor='none', label="ensemble spread"
)

plt.legend()
plt.xlabel("Time steps")
plt.ylabel(f"$X_{x_plot}$")

plt.tight_layout()
plt.show()
