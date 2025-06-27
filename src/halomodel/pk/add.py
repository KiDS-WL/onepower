from functools import cached_property
import warnings
import numpy as np
from scipy.interpolate import interp1d
from hmf._internals._cache import cached_quantity, parameter, subframework
from hmf._internals._framework import Framework

#from .pk import MatterSpecta, GalaxySpectra, AlignmentSpectra
import .pk as power_class

class AddUpsample(Framework):
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
            z=0.0,
            k=0.0,
            fraction_z=None,
            fraction=None,
            power_name_1,
            power_name_2,
            model_1_params=None,
            model_2_params=None,
            requested_spectra={},
        ):
        super().__init__()
        self.z = z
        self.k = k
        self.fraction_z = fraction_z
        self.fraction = fraction
        self.power_name_1 = power_name_1
        self.power_name_2 = power_name_2
        self._model_1_params = model_1_params or {}
        self._model_2_params = model_2_params or {}
        self.requested_spectra = requested_spectra
        
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
        
    @cached_property
    def frac_1(self):
        if self.fraction is None:
            return None
        f = np.interp1d(self.fraction_z, self.fraction, kind='linear', fill_value=extrapolate_option, bounds_error=False, axis=0)
        return f(self.z)
        
    @cached_property
    def frac_2(self):
        if self.fraction is None:
            return None
        return 1.0-self.frac_1
        
    @subframework
    def power_1(self):
        """First Halo Model."""
        return self.select_power(self.power_name_1)(**self._model_1_params)

    @subframework
    def power_2(self):
        """Second Halo Model."""
        return self.select_power(self.power_name_2)(**self._model_2_params)
        
    def select_spectra(self):
        for mode in requested_spectra:
            



    def extrapolate_spectra(self, z_ext, k_ext, extrapolate_option):
    
        inter_func_z = interp1d(z, power, kind='linear', fill_value=extrapolate_option, bounds_error=False, axis=0)
        pk_tot_ext_z = inter_func_z(z_ext)
    
        inter_func_k = interp1d(np.log10(k), pk_tot_ext_z, kind='linear', fill_value='extrapolate', bounds_error=False, axis=1)
        pk_tot_ext = inter_func_k(np.log10(k_ext))

        return pk_tot_ext

    def add_spectra(self, pk_1, pk_2, mode):
        # TODO: Add the cross terms
        # Not valid / implemented for matter-intrinsic, matter-matter
        if mode == 'mm':
            raise ValueError("Cannot add matter-matter power spectra! Use extrapolate option!")
        
        if mode in ['gm', 'mi']:
            pk_tot = self.frac_1[:, np.newaxis] * pk_1 + (1.0 - self.frac_1[:, np.newaxis]) * pk_2
        else:
            pk_tot = self.frac_1[:, np.newaxis]**2.0 * pk_1 + (1.0 - self.frac_1[:, np.newaxis])**2.0 * pk_2

        return pk_tot
