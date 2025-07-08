from functools import cached_property
import warnings
import numpy as np
from scipy.interpolate import interp1d
from hmf._internals._cache import cached_quantity, parameter, subframework
from hmf._internals._framework import Framework

from .pk import Spectra, PowerSpectrumResult

class UpsampledSpectra(Framework):
    """
    The halo model.

    This class generates one or two :class:`~halomodel.pk.Spectra`,
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
            model=None,
            model_1_params={},
            model_2_params=None,
        ):
        super().__init__()
        self.z = z
        self.k = k
        self.fraction_z = fraction_z
        self.fraction = fraction
        self.model = model
        self._model_1_params = model_1_params
        self._model_2_params = model_2_params

    @parameter("model")
    def model(self, val):
        if val is None:
            val = Spectra(**self._model_1_params)
        return val

    @parameter("param")
    def _model_1_params(self, val):
        return val
    
    @parameter("param")
    def _model_2_params(self, val):
        return val
    
    @parameter("param")
    def fraction_z(self, val):
        return val
    
    @parameter("param")
    def fraction(self, val):
        return val
    
    @parameter("param")
    def z(self, val):
        return val
    
    @parameter("param")
    def k(self, val):
        return val
                    
    @cached_property
    def frac_1(self):
        if self.fraction is None:
            # We assume that if no fraction is given, the first model is 100% and the second is 0%
            # This is useful for cases where only one model is used.
            return np.ones_like(self.z)
        f = np.interp1d(self.fraction_z, self.fraction, kind='linear', fill_value='extrapolate', bounds_error=False, axis=0)
        return f(self.z)
        
    @cached_property
    def frac_2(self):
        if self.fraction is None:
            return np.zeros_like(self.z)
        return 1.0-self.frac_1
        
    @cached_property
    def power_1(self):
        """First Halo Model."""
        return self.model

    @cached_property
    def power_2(self):
        """Second Halo Model."""
        # We use the update method so that the second model does not have to recalculate 
        # all the methods if they are the same as the first one.
        if self._model_2_params is None:
            spectra2 = None
        else:
            spectra2 = self.power_1.clone()
            spectra2.update(**self._model_2_params)
        return spectra2
        
    def results(self, requested_spectra):
        for mode in requested_spectra:
            collected_spectra = {}
            for component in ['1h', '2h', 'tot']:    
                p1 = getattr(self.power_1, f'power_spectrum_{mode}')
                p1_component = getattr(p1, f'pk_{component}')
                extrapolated_p1 = self.extrapolate_spectra(
                    self.z, self.k, self.power_1.z_vec, self.power_1.k_vec, p1_component, extrapolate_option='extrapolate'
                )
                if self.power_2 is not None:
                    p2 = getattr(self.power_2, f'power_spectrum_{mode}')
                    p2_component = getattr(p2, f'pk_{component}')
                    extrapolated_p2 = self.extrapolate_spectra(
                        self.z, self.k, self.power_2.z_vec, self.power_2.k_vec, p2_component, extrapolate_option='extrapolate'
                    )
                else:
                    extrapolated_p2 = np.zeros_like(extrapolated_p1)
                added_power = self.add_spectra(extrapolated_p1, extrapolated_p2, mode)
                collected_spectra[f'pk_{component}'] = added_power
            # Create a PowerSpectrumResult object to hold the results
            power_spectrum_result = PowerSpectrumResult(**collected_spectra)
            setattr(self, f'power_spectrum_{mode}', power_spectrum_result)

    def extrapolate_spectra(self, z_ext, k_ext, z_in, k_in, power, extrapolate_option):
        inter_func_z = interp1d(z_in, power, kind='linear', fill_value=extrapolate_option, bounds_error=False, axis=1)
        pk_tot_ext_z = inter_func_z(z_ext)

        inter_func_k = interp1d(np.log10(k_in), pk_tot_ext_z, kind='linear', fill_value='extrapolate', bounds_error=False, axis=2)
        pk_tot_ext = inter_func_k(np.log10(k_ext))
        return pk_tot_ext

    def add_spectra(self, pk_1, pk_2, mode):
        # TODO: Add the cross terms
        # Not valid / implemented for matter-intrinsic, matter-matter
        if mode == 'mm':
            return pk_1
        if mode in ['gm', 'mi']:
            pk_tot = self.frac_1[:, np.newaxis] * pk_1 + (1.0 - self.frac_1[:, np.newaxis]) * pk_2
        else:
            pk_tot = self.frac_1[:, np.newaxis]**2.0 * pk_1 + (1.0 - self.frac_1[:, np.newaxis])**2.0 * pk_2
        return pk_tot
