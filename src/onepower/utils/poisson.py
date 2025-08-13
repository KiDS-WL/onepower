import numpy as np
from abc import ABCMeta, abstractmethod
from functools import cached_property

from hmf._internals._framework import Component, pluggable

# We name the classes here with lowercase in order to keep it simpler for the user.


@pluggable
class Poisson(Component, metaclass=ABCMeta):
    """
    Class to calculate the Poisson parameter for use in Pgg integrals.

    Can be either a scalar (P = poisson) or a power law (P = poisson x (M/M_0)^slope),
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


# Add more if needed!
