from abc import ABCMeta, abstractmethod

import numpy as np
from hmf._internals._framework import Component, pluggable


# We name the classes here with lowercase in order to keep it simpler for the user.


@pluggable
class Poisson(Component, metaclass=ABCMeta):
    """
    Class to calculate the Poisson parameter for use in Pgg integrals.

    Can be either a scalar (P = poisson), a power law (P = poisson x (M/M_0)^slope), 
    or a logistic function (P = poisson x L / (1 + exp(-k * (log10(M) - log10(M_0))))),
    by using the appropriate child class. Parent class is just an instance of an abstract class
    Further models can be added if necessary.

    Parameters:
    -----------
    mass : array_like
        Array of halo masses.

    """

    def __init__(self, mass, **model_parameters):
        self.mass = mass

        super().__init__(**model_parameters)

    @property
    @abstractmethod
    def poisson_func(self):  # pragma: no cover
        return


class constant(Poisson):
    """Constant Poisson parameter."""

    _defaults = {'poisson': 1.0}

    @property
    def poisson_func(self):
        return self.params['poisson'] * np.ones_like(self.mass)


class power_law(Poisson):
    """Power law mass dependent Poisson parameter."""

    _defaults = {'poisson': 1.0, 'pivot': None, 'slope': None}

    @property
    def poisson_func(self):
        poisson = self.params['poisson']
        M_0 = self.params['pivot']
        slope = self.params['slope']
        if M_0 is None or slope is None:
            raise ValueError(
                "pivot and slope must be provided for 'PowerLaw' poisson type."
            )
        return poisson * (self.mass / (10.0**M_0)) ** slope


class logistic_function(Poisson):
    """Logistic function mass dependent Poisson parameter."""

    _defaults = {'poisson': 1.0, 'L': None, 'k': None, 'log10_M0': None}

    @property
    def poisson_func(self):
        poisson = self.params['poisson']
        L = self.params['L']
        k = self.params['k']
        log10_M0 = self.params['log10_M0']
        if L is None or k is None or log10_M0 is None:
            raise ValueError(
                "L, k, and log10_M0 must be provided for 'logistic_function' poisson type."
            )
        return poisson * L / (1 + np.exp(-k * (np.log10(self.mass) - log10_M0)))



# Add more if needed!
