import numpy as np
import matplotlib.pyplot as plt

import tenpy
from tenpy.algorithms import dmrg, tebd
from tenpy.networks.mps import MPS

import h5py
from tenpy.tools import hdf5_io

from DMI_model import MySpinModel

import pandas as pd

tenpy.tools.misc.setup_logging(to_stdout="INFO")

Bx = By = 0.0
Bz = -0.5
D = 1.0
Jx = Jy = Jz = -0.5*D

bond_dim = 32

model_params = {
    'J': [Jx, Jy, Jz],
    'B': [Bx, By, Bz],
    'D' : D,
    'bc_x': 'open', 'bc_y': 'ladder',
    'Lx' : 15, 'Ly': 15, 'lattice': 'Triangular', 'conserve': None
}

M = MySpinModel(model_params)

sites = M.lat.mps_sites()
p_state = ['down']*len(sites)
psi = MPS.from_product_state(sites, p_state)

# generate a random initial state
TEBD_params = {'N_steps': 10, 'trunc_params':{'chi_max': bond_dim}}
eng = tebd.RandomUnitaryEvolution(psi, TEBD_params)
eng.run()
psi.canonical_form()

dmrg_params = {
    'mixer': False,  # setting this to True helps to escape local minima
    'max_E_err': 1.e-6,
    'max_sweeps': 10,
    'trunc_params': {
        'chi_max': bond_dim,
        'svd_min': 1.e-12,
    }
}
eng = dmrg.SingleSiteDMRGEngine(psi, M, dmrg_params) 
E, psi = eng.run()
psi.canonical_form()

exp_Sx = psi.expectation_value("Sx")
exp_Sy = psi.expectation_value("Sy")
exp_Sz = psi.expectation_value("Sz")

abs_exp_Svec = np.sqrt(np.power(exp_Sx,2) + np.power(exp_Sy,2) + np.power(exp_Sz,2))
vmin = np.min(abs_exp_Svec)
vmax = np.max(abs_exp_Svec)

pos = np.asarray([M.lat.position(M.lat.mps2lat_idx(i)) for i in range(psi.L)])
pos_av = np.mean(pos)
pos = pos - pos_av

df = pd.DataFrame()
df['x'] = pos[:,0]
df['y'] = pos[:,0]
df['Sx'] = exp_Sx
df['Sy'] = exp_Sy
df['Sz'] = exp_Sz

df.to_csv('data.csv')

fig, ax = plt.subplots(1,1)
ax.scatter(pos[:,0], pos[:,1], marker='H', s=500, cmap='RdBu_r', c=exp_Sz)
ax.quiver(pos[:,0], pos[:,1], exp_Sx, exp_Sy, units='xy', width=0.07, scale=vmax, pivot='middle', color='white')
ax.set_aspect('equal')

mmx = np.asarray([np.min(pos[:,0]),np.max(pos[:,0])])
mmy = np.asarray([np.min(pos[:,1]),np.max(pos[:,1])])
ax.set_xlim(1.25*mmx)
ax.set_ylim(1.25*mmy)
ax.axis('off')
plt.tight_layout()
plt.savefig("last_sim.jpg")
plt.close()

data = {"psi": psi,  # e.g. an MPS
        "model": M,
        "parameters": model_params}

with h5py.File("file.h5", 'w') as f:
    hdf5_io.save_to_hdf5(f, data)
