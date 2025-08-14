import numpy as np
import warnings
from functools import cached_property
from scipy.interpolate import interp1d

from hmf._internals._cache import cached_quantity, parameter, subframework
from hmf._internals._framework import Framework

from .pk import PowerSpectrumResult, Spectra


class UpsampledSpectra(Framework):
    """
    This class generates one or two :class:`~halomodel.pk.Spectra`,
    extrapolates them to the desired output grid and adds them if requested.

    Parameters
    ----------
    z : array_like
        Output redshifts to which to interpolate and optionally extrapolate power spectra.
    k : array_like
        Output k-vector to which to interpolate and optionally extrapolate power spectra.
    fraction_z : array_like
        Redshifts of the red/blue fraction of galaxies in sample.
    fraciton : array_like
        Red/blue fraction of galaxies in sample as a function of redshift.
    model_1_params : dict
        Parameters for the first halo model.
    model_2_params : dict
        Parameters for the second halo model.
    extrapolate_option : str or float
        Extrapolation option to pass to interp1d.
    model : object
        Instance of Spectra(), pre-initialised for saving computing resources.
    """

    def __init__(
        self,
        z=0.0,
        k=0.0,
        fraction_z=None,
        fraction=None,
        model=None,
        model_1_params: dict | None = None,
        model_2_params: dict | None = None,
        extrapolate_option='extrapolate',
    ):
        super().__init__()
        self.z = z
        self.k = k
        self.fraction_z = fraction_z
        self.fraction = fraction
        self.model = model
        self._model_1_params = model_1_params or {}
        self._model_2_params = model_2_params
        self.extrapolate_option = extrapolate_option

    @parameter('param')
    def model(self, val):
        """
        Instance of Spectra(), pre-initialised for saving computing resources.

        :type: object
        """
        return val

    @parameter('param')
    def _model_1_params(self, val):
        """
        Parameters for the first halo model, will not update the model if they are the same as the pre-initialised model.

        :type: dict
        """
        return val

    @parameter('param')
    def _model_2_params(self, val):
        """
        Parameters for the second halo model.

        :type: dict
        """
        return val

    @parameter('param')
    def fraction_z(self, val):
        """
        Redshifts of the red/blue fraction of galaxies in sample.

        :type: array_like
        """
        return val

    @parameter('param')
    def fraction(self, val):
        """
        Red/blue fraction of galaxies in sample as a function of redshift.

        :type: array_like
        """
        return val

    @parameter('param')
    def z(self, val):
        """
        Output redshifts to which to interpolate and optionally extrapolate power spectra.

        :type: array_like
        """
        return val

    @parameter('param')
    def k(self, val):
        """
        Output k-vector to which to interpolate and optionally extrapolate power spectra.

        :type: array_like
        """
        return val

    @parameter('param')
    def extrapolate_option(self, val):
        """
        Extrapolation option to pass to interp1d.

        :type: str or float
        """
        return val

    @cached_property
    def frac_1(self):
        """
        Calculate the fraction for the first model.

        If no fraction is given, it assumes the first model is 100% and the second is 0%.
        This is useful for cases where only one model is used.

        Returns
        -------
        ndarray
            Fraction for the first model.
        """
        if self.fraction is None:
            return np.ones_like(self.z)
        f = interp1d(
            self.fraction_z,
            self.fraction,
            kind='linear',
            fill_value='extrapolate',
            bounds_error=False,
            axis=0,
        )
        return f(self.z)

    @cached_property
    def frac_2(self):
        """
        Calculate the fraction for the second model.

        Returns
        -------
        ndarray
            Fraction for the second model.
        """
        if self.fraction is None:
            return np.zeros_like(self.z)
        return 1.0 - self.frac_1

    @cached_property
    def power_1(self):
        """
        First Halo Model.

        Returns
        -------
        Spectra
            Instance of Spectra for the first halo model.
        """
        if self.model is None:
            return Spectra(**self._model_1_params)
        return self.model

    @cached_property
    def power_2(self):
        """
        Second Halo Model.

        We use the update method so that the second model does not have to recalculate
        all the methods if they are the same as the first one.

        Returns
        -------
        Spectra or None
            Instance of Spectra for the second halo model.
        """
        if self._model_2_params is None:
            spectra2 = None
        else:
            spectra2 = self.power_1.clone()
            spectra2.update(**self._model_2_params)
        return spectra2

    def results(self, requested_spectra, requested_components):
        """
        Calculate and store the results for the requested spectra and components.

        Parameters
        ----------
        requested_spectra : list
            List of spectra modes to calculate.
        requested_components : list
            List of components to calculate for each spectrum.

        Returns
        -------
        PowerSpectrumResults
            PowerSpectrumResults atributes attached to parent class
        """
        for mode in requested_spectra:
            collected_spectra = {}
            for component in requested_components:
                p1 = getattr(self.power_1, f'power_spectrum_{mode}')
                p1_component = getattr(p1, f'pk_{component}')
                extrapolated_p1 = self.extrapolate_spectra(
                    self.z,
                    self.k,
                    self.power_1.z_vec,
                    self.power_1.k_vec,
                    p1_component,
                    extrapolate_option=self.extrapolate_option,
                )
                if self.power_2 is not None:
                    p2 = getattr(self.power_2, f'power_spectrum_{mode}')
                    p2_component = getattr(p2, f'pk_{component}')
                    extrapolated_p2 = self.extrapolate_spectra(
                        self.z,
                        self.k,
                        self.power_2.z_vec,
                        self.power_2.k_vec,
                        p2_component,
                        extrapolate_option=self.extrapolate_option,
                    )
                else:
                    extrapolated_p2 = np.zeros_like(extrapolated_p1)
                added_power = self.add_spectra(extrapolated_p1, extrapolated_p2, mode)
                collected_spectra[f'pk_{component}'] = added_power
            # Create a PowerSpectrumResult object to hold the results
            power_spectrum_result = PowerSpectrumResult(**collected_spectra)
            setattr(self, f'power_spectrum_{mode}', power_spectrum_result)

    def extrapolate_spectra(self, z_ext, k_ext, z_in, k_in, power, extrapolate_option):
        """
        Extrapolate the power spectra to the desired output grid.

        Parameters
        ----------
        z_ext : array_like
            Output redshifts.
        k_ext : array_like
            Output k-vector.
        z_in : array_like
            Input redshifts.
        k_in : array_like
            Input k-vector.
        power : array_like
            Power spectra to extrapolate.
        extrapolate_option : str or float
            Extrapolation option to pass to interp1d.

        Returns
        -------
        ndarray
            Extrapolated power spectra.
        """
        inter_func_z = interp1d(
            z_in,
            power,
            kind='linear',
            fill_value=extrapolate_option,
            bounds_error=False,
            axis=1,
        )
        pk_tot_ext_z = inter_func_z(z_ext)

        inter_func_k = interp1d(
            np.log10(k_in),
            pk_tot_ext_z,
            kind='linear',
            fill_value='extrapolate',
            bounds_error=False,
            axis=2,
        )
        pk_tot_ext = inter_func_k(np.log10(k_ext))
        return pk_tot_ext

    def add_spectra(self, pk_1, pk_2, mode):
        """
        Add the power spectra of the two models.

        TODO: Add the cross terms
        Not valid / implemented for matter-intrinsic, matter-matter

        Parameters
        ----------
        pk_1 : array_like
            Power spectrum of the first model.
        pk_2 : array_like
            Power spectrum of the second model.
        mode : str
            Mode of the power spectrum.

        Returns
        -------
        ndarray
            Combined power spectrum.
        """
        if mode == 'mm':
            return pk_1
        if mode in ['gm', 'mi']:
            pk_tot = (
                self.frac_1[:, np.newaxis] * pk_1
                + (1.0 - self.frac_1[:, np.newaxis]) * pk_2
            )
        else:
            pk_tot = (
                self.frac_1[:, np.newaxis] ** 2.0 * pk_1
                + (1.0 - self.frac_1[:, np.newaxis]) ** 2.0 * pk_2
            )
        return pk_tot
