from threebythreenn import ThreeByThreeNN
from numerical_model.subroutines import prepare_integration
from numerical_model.qg_constants import qg_constants as const
from qg_setup import define_orography, invent_state
from qg_output import setup_output, output
from datetime import datetime, timedelta
from iris import load_cube, Constraint
import numpy as np
from sys import argv

# Model time step
dt = float(const.dt0)

# Start and end date
start = datetime(2018,6,1)
end   = datetime(2018,6,4)

# Output file name
output_file = "neural_net.nc"

# Construct range of dates for each timestep
simul_len = (end - start).total_seconds()
date_range = [start + timedelta(seconds=i*dt) for i in range(int(simul_len/dt))]

# Get initial condition
time_constraint = Constraint(time=lambda t: t > start and t <= end)
𝛙_real = load_cube("training_data.nc", ["psi"])
𝛙_real = 𝛙_real.extract(time_constraint)
𝛙_real.transpose()

𝛙 = 𝛙_real[...,0].data.copy()
dummy = np.zeros(𝛙.shape)

# Set up output NetCDF file and print zeroth time step
setup_output(output_file, start)
output(output_file, start, start, 0, dummy, 𝛙, dummy, dummy)

stencil = int(argv[1])

forecaster = InteriorNN(stencil)

# Main model loop, starting from first time step
for i, date in enumerate(date_range[1:], 1):
    print(f"Integrating {date}")

    # Compute time step
    𝛙 = forecaster.step(𝛙)

    # Output prognostic variables
    output(output_file, start, date, i, dummy, 𝛙, dummy, dummy)
