import numpy as np
import matplotlib.pyplot as plt

import tenpy
from tenpy.algorithms import dmrg, tebd
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO

import h5py
from tenpy.tools import hdf5_io

from AFM_model import MySpinModel

import pandas as pd
import copy

tenpy.tools.misc.setup_logging(to_stdout="INFO")

B = -0
J1 = -1
J2 = 0.5

bc_MPS, N_sweeps, E_tol, bond_dim = 'infinite', 100, 1e-12, 512
lattice, mkr, sze, Lx, Ly = 'Square', 's', 500, 9, 9
lattice, mkr, sze, Lx, Ly = 'Triangular', 'H', 500, 9, 9

params = {
    'S': 0.5,
    'J1': J1,
    'J2': J2,
    'B': B,
    'bc_y': 'cylinder', 
    'bc_MPS': bc_MPS,
    'Lx' : Lx, 
    'Ly': Ly, 
    'lattice': lattice, 
    'conserve': 'Sz'
    # 'conserve': None
}

M = MySpinModel(params)

sites = M.lat.mps_sites()
p_state = ['up']*len(sites)
p_state[0] = 'down'
p_state[10] = 'down'
p_state[20] = 'down'
p_state[30] = 'down'
p_state[40] = 'down'
p_state[50] = 'down'
p_state[60] = 'down'
p_state[70] = 'down'
psi = MPS.from_product_state(sites, p_state, bc=bc_MPS)

# annihilate_MPO = MPO.from_wavepacket(sites, [1]*len(sites), "Sm", eps=1e-15)
# [annihilate_MPO.apply(psi, {'compression_method':'variational', 'trunc_params': {'chi_max': bond_dim}}) for a in range(8)]

# generate a random initial state
# TEBD_params = {'N_steps': 10, 'trunc_params':{'chi_max': bond_dim}}
# eng = tebd.RandomUnitaryEvolution(psi, TEBD_params)
# eng.run()
# psi.canonical_form()

dmrg_params = {
    # 'mixer': None,  # no subspace expansion
    'mixer': 'DensityMatrixMixer',
    'diag_method': 'lanczos',
    'lanczos_params': {
        # https://tenpy.readthedocs.io/en/latest/reference/tenpy.linalg.lanczos.LanczosGroundState.html#cfg-config-Lanczos
        'N_max': 3,  # fix the number of Lanczos iterations: the number of `matvec` calls
        'N_min': 3,
        'N_cache': 20,  # keep the states during Lanczos in memory
        'reortho': False,
    },
    'max_E_err': E_tol,
    'max_sweeps': N_sweeps,
    'trunc_params': {
        'chi_max': bond_dim,
        'svd_min': 1.e-12,
    }
}
eng = dmrg.SingleSiteDMRGEngine(psi, M, dmrg_params) 
E, psi = eng.run()
psi.canonical_form()

exp_Sz = psi.expectation_value("Sz")
vmin = -0.5
vmax = +0.5

pos = np.asarray([M.lat.position(M.lat.mps2lat_idx(i)) for i in range(psi.L)])
pos_av = np.mean(pos)
pos = pos - pos_av

df = pd.DataFrame()
df['x'] = pos[:,0]
df['y'] = pos[:,1]
df['Sz'] = exp_Sz

df.to_csv('lobs.csv')

fig, ax = plt.subplots(1,1)
ax.scatter(pos[:,0], pos[:,1], marker=mkr, s=sze, cmap='RdBu_r', c=exp_Sz, vmin=-0.5, vmax=0.5)
# ax.quiver(pos[:,0], pos[:,1], exp_Sx, exp_Sy, units='xy', width=0.07, scale=vmax, pivot='middle', color='white')
ax.set_aspect('equal')

mmx = np.asarray([np.min(pos[:,0]),np.max(pos[:,0])])
mmy = np.asarray([np.min(pos[:,1]),np.max(pos[:,1])])
ax.set_xlim(1.25*mmx)
ax.set_ylim(1.25*mmy)
ax.axis('off')
plt.tight_layout()
plt.savefig("snap.jpg", dpi=300)
plt.close()

data = {"psi": psi,
        "model": M,
        "parameters": params}

with h5py.File("save.h5", 'w') as f:
    hdf5_io.save_to_hdf5(f, data)
