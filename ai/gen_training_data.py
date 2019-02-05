from numerical_model.subroutines import prepare_integration, propagate
from numerical_model.qg_constants import qg_constants as const
from qg_setup import define_orography, invent_state
from qg_output import setup_output, output
from datetime import datetime, timedelta

# Model time step
dt = float(const.dt0)

# Start and end date
start = datetime(2018,1,1)
end   = datetime(2019,1,1)

# Output file name
output_file = "training_data.nc"

# Construct range of dates for each timestep
simul_len = (end - start).total_seconds()
date_range = [start + timedelta(seconds=i*dt) for i in range(int(simul_len/dt))]

# Define model orography
orog = define_orography()

# Set up up initial stream function and boundary arrays
x, x_north, x_south, q_north, q_south = invent_state(orog)

# Get PV and wind from streamfunction
q, u, v = prepare_integration(x, x_north, x_south, orog)

# Set up output NetCDF file and print zeroth time step
setup_output(output_file, start)
output(output_file, start, start, 0, q, u, v)

# Main model loop, starting from first time step
for i, date in enumerate(date_range[1:], 1):
    print(f"Integrating {date}")

    # Compute time step
    q, _, u, v = propagate(q, q_north, q_south, x_north, x_south, u, v, orog)

    # Output prognostic variables
    output(output_file, start, date, i, q, u, v)