from threebythreenn import ThreeByThreeNN
from numerical_model.subroutines import prepare_integration
from numerical_model.qg_constants import qg_constants as const
from qg_setup import define_orography, invent_state
from qg_output import setup_output, output
from datetime import datetime, timedelta
from iris import load_cubes, Constraint

# Model time step
dt = float(const.dt0)

# Start and end date
start = datetime(2018,1,1)
end   = datetime(2018,1,2)

# Output file name
output_file = "neural_net.nc"

# Construct range of dates for each timestep
simul_len = (end - start).total_seconds()
date_range = [start + timedelta(seconds=i*dt) for i in range(int(simul_len/dt))]

# Get initial condition
time_constraint = Constraint(time=lambda t: t > start and t <= end)
q_real, u_real, v_real = load_cubes("training_data.nc", ["pv", "u", "v"])
q_real = q_real.extract(time_constraint)
u_real = u_real.extract(time_constraint)
v_real = v_real.extract(time_constraint)
q_real.transpose()
u_real.transpose()
v_real.transpose()

q = q_real[...,0].data.copy()
u = u_real[...,0].data.copy()
v = v_real[...,0].data.copy()

# Set up output NetCDF file and print zeroth time step
setup_output(output_file, start)
output(output_file, start, start, 0, q, u, v)

forecaster = ThreeByThreeNN()

# Main model loop, starting from first time step
for i, date in enumerate(date_range[1:], 1):
    print(f"Integrating {date}")

    # Compute time step
    q, u, v = forecaster.step(q, u, v)

    # Output prognostic variables
    output(output_file, start, date, i, q, u, v)
