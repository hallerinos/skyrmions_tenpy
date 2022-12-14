import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

lobs_files = glob('*.csv')
print(lobs_files)
mkrs, szes = ['H', 's', 'H', 'H', 'H'], [160, 1000, 550, 550, 550]
for (fn, mkr, sze) in zip(lobs_files, mkrs, szes):
    df = pd.read_csv(fn)

    abs_exp_Svec = np.sqrt(np.power(df['Sx'],2) + np.power(df['Sy'],2) + np.power(df['Sz'],2))
    vmin = np.min(abs_exp_Svec)
    vmax = np.max(abs_exp_Svec)

    rmax = np.max(df['x']**2+df['y']**2)

    fig, ax = plt.subplots(1,1, figsize=(5,5))
    ax.scatter(df['x'], df['y'], marker=mkr, s=sze, cmap='RdBu_r', c=df['Sz'], vmin=-0.5, vmax=0.5)
    ax.quiver(df['x'], df['y'], df['Sx'], df['Sy'], units='xy', width=0.07, scale=vmax, pivot='middle', color='white')
    ax.set_aspect('equal')

    mmx = np.asarray([np.min(df['x']),np.max(df['x'])])
    mmy = np.asarray([np.min(df['y']),np.max(df['y'])])
    ax.set_xlim(1.25*mmx)
    ax.set_ylim(1.25*mmy)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{fn[:-4]}.jpg', dpi=300)
    plt.close()