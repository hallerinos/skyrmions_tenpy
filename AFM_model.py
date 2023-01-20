"""Nearest-neighbour spin-S models.

Uniform lattice of spin-S sites, coupled by nearest-neighbour interactions.
"""
# Copyright 2018-2021 TeNPy Developers, GNU GPLv3

import numpy as np

from tenpy.networks.site import SpinSite
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.models.lattice import Chain
import tenpy.models.lattice as lat
from tenpy.tools.params import asConfig

__all__ = ['MySpinModel', 'MySpinChain']


class MySpinModel(CouplingMPOModel):
    r"""Spin-S sites coupled by nearest neighbour interactions with antiferromagnetic next-nearest neighbor frustration.
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`MySpinModel` below.

    Options
    -------
    .. cfg:config :: MySpinModel
        :include: CouplingMPOModel

        S : {0.5, 1, 1.5, 2, ...}
            The 2S+1 local states range from m = -S, -S+1, ... +S.
        conserve : 'best' | 'Sz' | 'parity' | None
            What should be conserved. See :class:`~tenpy.networks.Site.SpinSite`.
            For ``'best'``, we check the parameters what can be preserved.
        Jx, Jy, Jz, hx, hy, hz, muJ, D, E  : float | array
            Coupling as defined for the Hamiltonian above.

    """
    def init_sites(self, model_params):
        S = model_params.get('S', 0.5)
        conserve = model_params.get('conserve', None)
        site = SpinSite(S, conserve)
        return site

    def init_terms(self, model_params):
        J1 = model_params.get('J1', -1.0)
        J2 = model_params.get('J2', +0.0)
        B = model_params.get('B', 0.0)

        # (u is always 0 as we have only one site in the unit cell)
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(B, u, "Sz")
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(0.5*J1, u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling(1.0*J1, u1, 'Sz', u2, 'Sz', dx)
        for u1, u2, dx in self.lat.pairs['next_nearest_neighbors']:
            self.add_coupling(0.5*J2, u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling(1.0*J2, u1, 'Sz', u2, 'Sz', dx)
        # done


class MySpinChain(MySpinModel, NearestNeighborModel):
    """The :class:`MySpinModel` on a Chain, suitable for TEBD.

    See the :class:`MySpinModel` for the documentation of parameters.
    """
    default_lattice = Chain
    force_default_lattice = True
