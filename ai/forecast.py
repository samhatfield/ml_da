from argparse import ArgumentParser
from interiornn import InteriorNN
from numerical_model.qg_constants import qg_constants as const
from qg_output import setup_output, output
from datetime import datetime, timedelta
from iris import load_cube, Constraint
import numpy as np

# Parse command line arguments
parser = ArgumentParser(description="Runs a neural net-based forecast")
parser.add_argument("stencil", help="What stencil size to use for the forecast")
args = parser.parse_args()
stencil = int(args.stencil)

print(f"Forecasting with {stencil}x{stencil} stencil")

# Model time step
dt = float(const.dt0)

# Start and end date
start = datetime(2018,4,1)
end   = datetime(2018,4,4)

# Output file name
output_file = "neural_net.nc"

# Construct range of dates for each timestep
simul_len = (end - start).total_seconds()
date_range = [start + timedelta(seconds=i*dt) for i in range(int(simul_len/dt))]

# Get initial condition
𝛙 = load_cube("training_data/training_data.nc",
              Constraint(name="psi", time=lambda t: t == start)).data.T.copy()
dummy = np.zeros(𝛙.shape)

# Set up output NetCDF file and print zeroth time step
setup_output(output_file, start)
output(output_file, start, start, 0, dummy, 𝛙, dummy, dummy)

forecaster = InteriorNN(stencil)

# Main model loop, starting from first time step
for i, date in enumerate(date_range[1:], 1):
    print(f"Integrating {date}")

    # Compute time step
    𝛙 = forecaster.step(𝛙)

    # Output prognostic variables
    output(output_file, start, date, i, dummy, 𝛙, dummy, dummy)
