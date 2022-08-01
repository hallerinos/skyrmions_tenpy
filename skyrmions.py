import numpy as np
import matplotlib.pyplot as plt

from DMI_model import MySpinModel
np.set_printoptions(precision=5, suppress=True, linewidth=100)
plt.rcParams['figure.dpi'] = 150

import random
import tenpy
import tenpy.linalg.np_conserved as npc
from tenpy.algorithms import dmrg, tebd
from tenpy.networks.mps import MPS

tenpy.tools.misc.setup_logging(to_stdout="INFO")

Bx = By = 0.0
Bz = -0.0
D = 1.0
Jx = Jy = Jz = -0.5*D

bond_dim = 8

model_params = {
    'J': [Jx, Jy, Jz],
    'B': [Bx, By, Bz],
    'D' : D,
    'bc_x': 'open', 'bc_y': 'ladder',
    'Lx' : 8, 'Ly': 8, 'lattice': 'Triangular', 'conserve': None
}

M = MySpinModel(model_params)

tenpy.show_config()

plt.figure()
ax = plt.gca()
M.lat.plot_coupling(ax, linewidth=3.)
ax.set_aspect('equal')
M.lat.plot_sites(ax)
M.lat.plot_basis(ax, origin=-0.5*(M.lat.basis[0] + M.lat.basis[1]))
M.lat.plot_order(ax, linestyle=':', linewidth=2)
plt.show()
exit()

sites = M.lat.mps_sites()
p_state = ['down']*len(sites)
psi = MPS.from_product_state(sites, p_state)
TEBD_params = {'N_steps': 20, 'trunc_params':{'chi_max': bond_dim}}
eng = tebd.RandomUnitaryEvolution(psi, TEBD_params)
eng.run()
psi.canonical_form() # important if you truncate strongly during the random evolution!!!

dmrg_params = {
    'mixer': False,  # setting this to True helps to escape local minima
    'max_E_err': 1.e-6,
    'trunc_params': {
        'chi_max': bond_dim,
        'svd_min': 1.e-12,
    }
}
eng = dmrg.SingleSiteDMRGEngine(psi, M, dmrg_params) 
E, psi = eng.run() # the main work; modifies psi in place

# the ground state energy was directly returned by dmrg.run()
print("ground state energy = ", E)

# there are other ways to extract the energy from psi:
E1 = M.H_MPO.expectation_value(psi)  # based on the MPO
print(abs(E-E1))