from numerical_model.subroutines import prepare_integration, propagate
from numerical_model.qg_constants import qg_constants as const
from qg_setup import define_orography, invent_state
from qg_output import setup_output, output
from datetime import datetime, timedelta

# Model time step
dt = float(const.dt0)

# Start and end date
start = datetime(2018,1,1)
end   = datetime(2019,4,1)

# Date by which spin up should be completed
spin_up_complete = datetime(2018,4,1)

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

# Main model loop, starting from first time step
i = 0
for date in date_range[1:]:
    print(f"Integrating {date}")

    # Compute time step
    q, 𝛙, u, v = propagate(q, q_north, q_south, x_north, x_south, u, v, orog)

    # Output prognostic variables
    if date >= spin_up_complete:
        # Set up output NetCDF file
        if date == spin_up_complete:
            setup_output(output_file, spin_up_complete)
        output(output_file, spin_up_complete, date, i, q, 𝛙, u, v)
        i+=1