To configure the code, open higherdim.py. Parameters can be changed manually at the top of the file. To run, use python3 higherdim.py. This will produce a folder labeled by the parameters containing the output files as well as a log of the parameters used. The notebook reads in the data and produces plots of the structure factor, frames from the simulation, cell number, and mu*.

Parameter information:

L - "real" system length
dx - lattice spacing (only supports square, symmetric lattices)
dt - time step, set in terms of dx for numerical stability
k - inverse consumption rate
y_cri - critical concentration of the resource (C_crit from the main text)
p - resource regeneration rate
lambda - cell deposition rate
D - resource diffusion constant
T - simulation run time
nu0 - initial value of the viability parameter