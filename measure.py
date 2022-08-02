import numpy as np
import matplotlib.pyplot as plt

import tenpy
from tenpy.algorithms import dmrg, tebd
from tenpy.networks.mps import MPS

import h5py
from tenpy.tools import hdf5_io

from DMI_model import MySpinModel

import pandas as pd
from glob import glob

tenpy.tools.misc.setup_logging(to_stdout="INFO")

fns = glob('*.h5')
print(fns)
for fn in fns:
    fn_part = fn[:-3]
    with h5py.File(fn, 'r') as f:
        data = hdf5_io.load_from_hdf5(f)

    psi = data['psi']
    M = data['model']
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
    df['y'] = pos[:,1]
    df['Sx'] = exp_Sx
    df['Sy'] = exp_Sy
    df['Sz'] = exp_Sz

    df.to_csv(f'{fn_part}.csv')