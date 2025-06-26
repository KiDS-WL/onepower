from functools import cached_property
import warnings
import numpy as np
from scipy.interpolate import interp1d
from hmf._internals._cache import cached_quantity, parameter, subframework
from hmf._internals._framework import Framework

#from .pk import MatterSpecta, GalaxySpectra, AlignmentSpectra
import .pk as power_class

class HaloModel(Framework):
    """
    The halo model.

    This class generates one or two :class:`~halomodel.pk.<Matter | Galaxy | Alignment>Spectra`,
    extrapolates them to the desired output grid and adds them if requested.

    Parameters
    ----------
    model_1_params, model_2_params : dict
        Parameters for the halo models.

    """
    def __init__(
            self,
            fraction=1.0,
            model_1_params=None,
            model_2_params=None,
        ):
        super().__init__()
        self.fraction = fraction
        self._model_1_params = model_1_params or {}
        self._model_2_params = model_2_params or {}
        
    def select_power(self, val):
        """
        Select a power spectra to use.

        Parameters:
        -----------
        val : str
            The power spectra class to select.

        Returns:
        --------
        object
            The selected power spectra.
        """
        if val is None:
            return val
        return getattr(power_class, val)
        
    @subframework
    def power_1(self):
        """First Halo Model."""
        return self.select_power(self.power_name_1)(**self._model_1_params)

    @subframework
    def power_2(self):
        """Second Halo Model."""
        return self.select_power(self.power_name_2)(**self._model_2_params)

    def extrapolate_power(self, z_ext, k_ext, extrapolate_option):
        # For matter-intrinsic and galaxy-intrinsic, pk_tot will usually be negative (for A_IA > 0)
        # If we're interpolating over log10(pk_tot) negative power is problematic
        # Check to see if it is negative, and take the absolute value
        if np.sum(pk_in) < 0:
            power *= -1
            changed_sign = True
        else:
            changed_sign = False
    
        inter_func_z = interp1d(self.z, power, kind='linear', fill_value=extrapolate_option, bounds_error=False, axis=0)
        pk_tot_ext_z = inter_func_z(z_ext)
    
        inter_func_k = interp1d(np.log10(self.k), pk_tot_ext_z, kind='linear', fill_value='extrapolate', bounds_error=False, axis=1)
        pk_tot_ext = inter_func_k(np.log10(k_ext))
    
        # Introduce the sign convention back for the GI terms
        if changed_sign:
            pk_tot_ext *= -1
    
        return pk_tot_ext

    def add_red_and_blue_power(self, pk_red, pk_blue, name):
        # TODO: Add the cross terms
        # This is not optimised, but it is good to first choose what we want to implement
        # in terms of cross terms.
        if name in ['intrinsic_power', 'galaxy_power', 'galaxy_intrinsic_power']:
            pk_tot = self.fraction[:, np.newaxis]**2.0 * pk_red + (1.0 - self.fraction[:, np.newaxis])**2.0 * pk_blue
        else:
            pk_tot = self.fraction[:, np.newaxis] * pk_red + (1.0 - self.fraction[:, np.newaxis]) * pk_blue

        return pk_tot
