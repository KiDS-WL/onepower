r"""
A module for computing 3D power spectra using the halo model approach.

See section 2 of https://arxiv.org/pdf/2303.08752.pdf for details.
Brief description on the formalism:

:math:`P_{uv} = P^{\rm 2h}_{uv} + P^{\rm 1h}_{uv}` (1)

:math:`P^{\rm 1h}_{uv} (k) = \int_{0}^{\infty} {\rm d}M W_{u}(M, k) W_{v}(M, k) n(M)` (2)

:math:`P^{\rm 2h}_{uv} (k) = \int_{0}^{\infty} \int_{0}^{\infty} {\rm d}M_{1} {\rm d}M_{2} P_{\rm hh}(M_{1}, M_{2}, k) W_{u}(M_{1}, k) W_{v}(M_{2}, k) n(M_{1}) n(M_{2})` (3)

:math:`W_{\rm x}` are the profile of the fields, :math:`u` and :math:`v`, showing how they fit into haloes.
:math:`n(M)` is the halo mass function, quantifying the number of haloes of each mass, :math:`M`.
Integrals are taken over halo mass.

The halo-halo power spectrum can be written as,

:math:`P_{\rm hh}(M_{1},M_{2},k) = b(M_{1}) b(M_{2}) P^{\rm lin}_{\rm mm}(k) (1 + \beta_{\rm nl}(M_{1},M_{2},k))` (4)

In the vanilla halo model the 2-halo term is usually simplified by assuming that haloes are linearly biased with respect to matter.
This sets beta_nl to zero and effectively decouples the integrals. Here we allow for both options to be calculated.

Equation (3) then becomes:

:math:`P^{\rm 2h}_{uv} (k) = P^{\rm lin}_{\rm mm}(k) * [I_u * I_v + I^{\rm NL}_{uv}]` (5)

where :math:`I_u` and :math:`I_v` are defined as:

:math:`I_{\rm x} = \int_{0}^{\infty} {\rm d}M b(M) W_{\rm x}(M, k) n(M)` (6)

and the integral over beta_nl is

:math:`I^{\rm NL}_{uv} = \int_{0}^{\infty} \int_{0}^{\infty} {\rm d}M_{1} {\rm d}M_{2} b(M_{1}) b(M_{2}) \beta_{\rm nl}(M_{1},M_{2},k) W_{u}(M_{1}, k) W_{v}(M_{2}, k) n(M_{1}) n(M_{2})`  (7)

"""

import numexpr as ne
import numpy as np
import warnings
from functools import cached_property
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from hmf._internals._cache import cached_quantity, parameter
from hmf._internals._framework import Framework, get_mdl
from hmf.density_field.transfer_models import EH_NoBAO as Tk_EH_nowiggle

from .bnl import NonLinearBias
from .hmi import HaloModelIngredients
from .ia import SatelliteAlignment
from .utils import poisson
from . import hod


class PowerSpectrumResult:
    """
    A helper class to attach the power spectrum outputs as atributes
    """

    def __init__(self, pk_1h=None, pk_2h=None, pk_tot=None, galaxy_linear_bias=None):
        self.pk_1h = pk_1h
        self.pk_2h = pk_2h
        self.pk_tot = pk_tot
        self.galaxy_linear_bias = galaxy_linear_bias


class Spectra(HaloModelIngredients):
    """
    Class to compute matter power spectra using the halo model approach.

    Parameters:
    -----------
    matter_power_lin : array_like, optional
        Linear matter power spectrum.
    matter_power_nl : array_like, optional
        Non-linear matter power spectrum.
    response : bool, optional
        Whether to calculate the resulting power spectra in response formalism.
    mb : float, optional
        Gas distribution mass pivot parameter.
    dewiggle : bool, optional
        Whether to dewiggle the power spectrum.
    bnl : bool, optional
        Whether to include non-linear bias.
    beta_nl : array_like, optional
        Non-linear bias parameter.
    one_halo_ktrunc : float, optional
        Truncation wavenumber for the 1-halo term.
    two_halo_ktrunc : float, optional
        Truncation wavenumber for the 2-halo term.
    hod_settings_mm : dict, optional
        Settings for the HOD model.
    hod_model : str, optional
        HOD model to use.
    hod_params : dict, optional
        Parameters for the HOD model.
    hod_settings : dict, optional
        Settings for the HOD model.
    obs_settings : dict, optional
        Settings for the observable.
    pointmass : bool, optional
        Whether to use point mass approximation.
    compute_observable : bool, optional
        Whether to compute observable.
    poisson_model : str, optional
        Poisson model to use.
    poisson_params : dict, optional
        Parameters for the Poisson distribution.
    t_eff : float, optional
        Effective parameter for the Fortuna model.
    fortuna : bool, optional
        Whether to use the Fortuna model.
    one_halo_ktrunc_ia : float, optional
        Truncation wavenumber for the 1-halo IA term.
    two_halo_ktrunc_ia : float, optional
        Truncation wavenumber for the 2-halo IA term.
    align_params : dict, optional
        Parameters for the alignment model.
    hmf_kwargs : dict
        Additional keyword arguments for the HaloModelIngredients.

    Examples
    --------
    Since all parameters have reasonable defaults, the most obvious thing to do is

    >>> power = Spectra()
    >>> ps.power_spectrum_mm.pk_tot

    Many different parameters may be passed, both models and parameters of those models.
    For instance:

    >>> power = Spectra(z=1.0, Mmin=8, hmf_model="SMT")
    >>> ps.power_spectrum_mm.pk_1h

    Once instantiated, changing parameters should be done through the :meth:`update`
    method:

    >>> power.update(hod_settings={nbins: 6})
    >>> ps.power_spectrum_gm.pk_1h

    """

    def __init__(
        self,
        matter_power_lin=None,
        matter_power_nl=None,
        response=False,
        mb=13.87,
        dewiggle=False,
        bnl=False,
        beta_nl=None,
        one_halo_ktrunc=0.1,
        two_halo_ktrunc=2.0,
        pointmass=False,
        compute_observable=False,
        poisson_model=poisson.constant,
        poisson_params=None,
        hod_model=hod.Cacciato,
        hod_params=None,
        hod_settings=None,
        hod_settings_mm=None,
        obs_settings=None,
        t_eff=0.0,
        fortuna=False,
        one_halo_ktrunc_ia=4.0,
        two_halo_ktrunc_ia=6.0,
        align_params=None,
        **hmf_kwargs,
    ):
        # Call super init MUST BE DONE FIRST.
        super().__init__(**hmf_kwargs)

        self.mb = mb
        self.bnl = bnl
        self.beta_nl = beta_nl
        self.dewiggle = dewiggle
        self.matter_power_lin = matter_power_lin
        self.matter_power_nl = matter_power_nl
        self.response = response

        self.one_halo_ktrunc = one_halo_ktrunc
        self.two_halo_ktrunc = two_halo_ktrunc

        # Galaxy spectra specific kwargs:
        self.pointmass = pointmass
        self.compute_observable = compute_observable
        self.poisson_model = poisson_model
        self.poisson_params = poisson_params or {}
        self.hod_settings = hod_settings or {}
        self.hod_params = hod_params or {}
        self.hod_model = hod_model
        self.obs_settings = obs_settings or {}

        self.hod_settings_mm = hod_settings_mm or {}

        # Alignment spectra specific kwargs:
        self.t_eff = t_eff
        self.fortuna = fortuna
        self.one_halo_ktrunc_ia = one_halo_ktrunc_ia
        self.two_halo_ktrunc_ia = two_halo_ktrunc_ia
        self.align_params = align_params or {}

    @parameter('param')
    def mb(self, val):
        r"""
        Gas distribution mass pivot parameter :math:`M_{\rm b}`.

        :type: float
        """
        return val

    @parameter('switch')
    def bnl(self, val):
        """
        Whether to include non-linear bias.

        :type: bool
        """
        return val

    @parameter('param')
    def beta_nl(self, val):
        r"""
        Non-linear bias parameter :math:`\beta_{\rm nl}`.

        :type: array_like
        """
        return val

    @parameter('param')
    def dewiggle(self, val):
        """
        Whether to dewiggle the power spectrum.

        :type: bool
        """
        return val

    @parameter('param')
    def matter_power_lin(self, val):
        """
        Linear matter power spectrum.

        :type: ndarray
        """
        return val

    @parameter('param')
    def matter_power_nl(self, val):
        """
        Non-linear matter power spectrum

        :type: ndarray
        """
        return val

    @parameter('switch')
    def response(self, val):
        """
        Whether to calculate the resulting power spectra in response formalism.

        :type: bool
        """
        return val

    @parameter('param')
    def one_halo_ktrunc(self, val):
        """
        Truncation wavenumber for the 1-halo term.

        :type: float
        """
        return val

    @parameter('param')
    def two_halo_ktrunc(self, val):
        """
        Truncation wavenumber for the 2-halo term.

        :type: float
        """
        return val

    @parameter('param')
    def hod_settings_mm(self, val):
        """
        Settings for the HOD model.

        :type: dict
        """
        return val

    @parameter('param')
    def pointmass(self, val):
        """
        Whether to use point mass approximation.

        :type: bool
        """
        return val

    @parameter('param')
    def compute_observable(self, val):
        """
        Whether to compute observable.

        :type: bool
        """
        return val

    @parameter('param')
    def poisson_params(self, val):
        """
        Parameters for the Poisson distribution.

        :type: dict
        """
        return val

    @parameter('param')
    def hod_params(self, val):
        """
        Parameters for the HOD model.

        :type: dict
        """
        return val

    @parameter('param')
    def hod_settings(self, val):
        """
        Settings for the HOD model.

        :type: dict
        """
        return val

    @parameter('param')
    def obs_settings(self, val):
        """
        Settings for the observable.

        :type: dict
        """
        return val

    @parameter('param')
    def fortuna(self, val):
        """
        Whether to use the Fortuna model.

        :type: bool
        """
        return val

    @parameter('param')
    def t_eff(self, val):
        """
        Effective parameter for the Fortuna model.

        :type: float
        """
        return val

    @parameter('param')
    def one_halo_ktrunc_ia(self, val):
        """
        Truncation wavenumber for the 1-halo IA term.

        :type: float
        """
        return val

    @parameter('param')
    def two_halo_ktrunc_ia(self, val):
        """
        Truncation wavenumber for the 2-halo IA term.

        :type: float
        """
        return val

    @parameter('param')
    def align_params(self, val):
        """
        Parameters for the alignment model.

        :type: dict
        """
        return val

    @parameter('model')
    def hod_model(self, val):
        r"""
        An HOD model to use

        :type: str or `hod.HOD` subclass
        """
        if val is None:
            return val
        return get_mdl(val, 'HaloOccupationDistribution')

    @parameter('model')
    def poisson_model(self, val):
        r"""
        A Poisson parameter model to use

        :type: str or `poisson.Poisson` subclass
        """
        if val is None:
            return val
        return get_mdl(val, 'Poisson')

    @cached_quantity
    def _beta_nl_array(self):
        """
        Return the pre-calculated beta_nl values or calculates it on the fly if beta_nl == None.

        Returns:
        --------
        ndarray
            beta_nl
        """
        if self.bnl and self.beta_nl is None:
            return self.calc_bnl
        return np.ascontiguousarray(self.beta_nl) if self.beta_nl is not None else None

    @cached_quantity
    def _pk_lin(self):
        """
        Return the pre-calculated linear power spectrum or uses one from hmf.
        Additionally it applies the dewiggle method if desired, or if
        HMCode2020 functionality is needed.

        Returns:
        --------
        ndarray
            linear power spectrum
        """
        # P(k) can be returned by hmf, together with the specified k_vec!
        if self.matter_power_lin is None:
            val_interp = interp1d(
                self.kh,
                self.power,
                fill_value='extrapolate',
                bounds_error=False,
                axis=1,
            )
            val = val_interp(self.k_vec)
        else:
            val = self.matter_power_lin

        if self.mead_correction in ['feedback', 'nofeedback'] or self.dewiggle:
            val = self.dewiggle_plin(val)
        if val.shape != (self.z_vec.size, self.k_vec.size):
            raise ValueError(
                'Shape of input power spectra is not equal to redshift and k-vec dimensions!'
            )
        return val

    @cached_quantity
    def _pk_nl(self):
        """
        Returns the pre-calculated non-linear power spectrum or uses one from hmf.

        Returns:
        --------
        ndarray
            non-linear power spectrum
        """
        val = self.matter_power_nl
        if self.fortuna or self.response:
            # P(k) can be returned by hmf!
            if self.matter_power_nl is None:
                val_interp = interp1d(
                    self.kh,
                    self.nonlinear_power,
                    fill_value='extrapolate',
                    bounds_error=False,
                    axis=1,
                )
                val = val_interp(self.k_vec)
            if val.shape != (self.z_vec.size, self.k_vec.size):
                raise ValueError(
                    'Shape of input power spectra is not equal to redshift and k-vec dimensions!'
                )
        return val

    @cached_quantity
    def peff(self):
        """
        Return the mixture of linear and non-linear power spectrum.
        t_eff as a ratio between the two as used in Fortuna et al. IA model in order to have better 1h to 2h transition.
        Only used if fortuna == True.

        Returns:
        --------
        ndarray
            effective power spectrum
        """
        if self.fortuna:
            return (1.0 - self.t_eff) * self._pk_nl + self.t_eff * self._pk_lin
        return None

    @cached_quantity
    def calc_bnl(self):
        """
        Calculate the non-linear bias using the NonLinearBias class.

        Returns:
        --------
        ndarray
            The non-linear bias.
        """
        bnl = NonLinearBias(
            mass=self.mass,
            z_vec=self.z_vec,
            k_vec=self.k_vec,
            h0=self.h0,
            sigma_8=self.sigma_8,
            omega_b=self.omega_b,
            omega_c=self.omega_c,
            omega_lambda=1.0 - self.omega_m,
            n_s=self.n_s,
            w0=self.w0,
        )
        return np.ascontiguousarray(bnl.bnl)

    @cached_quantity
    def I12(self):
        """
        Returns:
        --------
        ndarray
            I12 integrand
        """
        if self.bnl:
            return self.prepare_I12_integrand(
                self.halo_bias,
                self.halo_bias,
                self.dndlnm,
                self.dndlnm,
                self._beta_nl_array,
            )
        else:
            return None

    @cached_quantity
    def I21(self):
        """
        Returns:
        --------
        ndarray
            I21 integrand
        """
        if self.bnl:
            return self.prepare_I21_integrand(
                self.halo_bias,
                self.halo_bias,
                self.dndlnm,
                self.dndlnm,
                self._beta_nl_array,
            )
        else:
            return None

    @cached_quantity
    def I22(self):
        """
        Returns:
        --------
        ndarray
            I22 integrand
        """
        if self.bnl:
            return self.prepare_I22_integrand(
                self.halo_bias,
                self.halo_bias,
                self.dndlnm,
                self.dndlnm,
                self._beta_nl_array,
            )
        else:
            return None

    @cached_quantity
    def hod_mm(self):
        """
        Initialize and return the HOD model for matter-matter power spectrum.
        Used only to link the galaxy-galaxy and galaxy-matter power spectra to
        the baryon feedback model.

        Returns:
        --------
        object
            The HOD model for matter-matter power spectrum.
        """
        hod = self.hod_model
        if self.mead_correction == 'fit' and hod.__name__ == 'Cacciato':
            return hod(
                cosmo=self.cosmo_model,
                mass=self.mass,
                dndlnm=self.dndlnm,
                halo_bias=self.halo_bias,
                z_vec=self.z_vec,
                hod_settings=self.hod_settings_mm,
                **self.hod_params,
            )
        else:
            return None

    @cached_quantity
    def fstar_mm(self):
        """
        Compute the stellar fraction for the matter-matter power spectrum.

        Returns:
        --------
        ndarray
            The stellar fraction.
        """
        if self.hod_mm is not None:
            return self.hod_mm.stellar_fraction
        else:
            return np.zeros((1, self.z_vec.size, self.mass.size))

    def dewiggle_plin(self, plin):
        """
        Dewiggle the linear power spectrum.

        Parameters:
        -----------
        plin : array_like
            Linear power spectrum.

        Returns:
        --------
        ndarray
            The dewiggled power spectrum.
        """
        sigma = self.sigmaV(self.k_vec, plin)
        pk_wig = self.get_Pk_wiggle(
            self.k_vec,
            plin,
            self.n_s,
        )
        plin_dw = (
            plin
            - (
                1.0
                - np.exp(-((self.k_vec[np.newaxis, :] * sigma[:, np.newaxis]) ** 2.0))
            )
            * pk_wig
        )
        return plin_dw

    def compute_matter_profile(self, mass, mean_density0, u_dm, fnu):
        r"""
        Compute the matter halo profile with a correction for neutrino mass fraction.

        Feedback can be included through u_dm.

        We lower the amplitude of :math:`W(M, k, z)` in the one-halo term by the factor :math:`1-f_{\nu}`,
        where :math:`f_{\nu}=\Omega_{\nu}/\Omega_{\mathrm{m}}` is the neutrino mass fraction, to account for the fact that
        we assume that hot neutrinos cannot cluster in haloes and therefore
        do not contribute power to the one-halo term.

        Therefore :math:`W(M, k \rightarrow 0, z)=(1-f_{\nu})M/\bar{\rho}` and has units of volume.
        This is the same as Mead et al. 2021

        Parameters:
        -----------
        mass : array_like
            Halo mass.
        mean_density0 : array_like
            Mean density at redshift zero.
        u_dm : array_like
            Dark matter profile.
        fnu : array_like
            Neutrino mass fraction.

        Returns:
        --------
        ndarray
            The matter halo profile.
        """
        Wm_0 = mass / mean_density0
        # Given the astropy definiton of Om, we do not need to correct for neutrino fraction as the Om is already without!
        return Wm_0 * u_dm  # * (1.0 - fnu)

    @cached_quantity
    def matter_profile(self):
        """
        Compute the matter profile grid in z, k, and M.

        Returns:
        --------
        ndarray
            The matter profile grid.
        """
        profile = self.compute_matter_profile(
            self.mass[np.newaxis, np.newaxis, np.newaxis, :],
            self.mean_density0[np.newaxis, :, np.newaxis, np.newaxis],
            self.u_dm[np.newaxis, :, :, :],
            0.0,
        )
        return profile

    @cached_quantity
    def matter_profile_2h(self):
        """
        Compute the matter profile grid in z, k, and M.
        This is without the neutrino subtraction, as
        the hot neutrinos do not cluster in haloes

        Returns:
        --------
        ndarray
            The matter profile grid.
        """
        profile = self.compute_matter_profile(
            self.mass[np.newaxis, np.newaxis, np.newaxis, :],
            self.mean_density0[np.newaxis, :, np.newaxis, np.newaxis],
            self.u_dm[np.newaxis, :, :, :],
            0.0,
        )
        return profile

    def compute_matter_profile_with_feedback(self, mass, mean_density0, u_dm, z, fnu):
        r"""
        Compute the matter profile including feedback as modelled by hmcode2020 (eq 25 of 2009.01858).

        :math:`W(M, k) = [\Omega_{\rm c}/\Omega_{\rm m} + f_{\rm g}(M)] W(M, k) + f_{\star} M / \bar{\rho}`

        The parameter :math:`0 < f_{\star} < \Omega_{\rm b}/\Omega_{\rm m}` can be thought of as an effective halo stellar mass fraction.

        Table 4 and eq 26 of 2009.01858 defines as:

        :math:`f_{\star}(z) = f_{\star, 0} 10^{(z f_{\star, z})}`

        This profile does not have :math:`1-f_{\nu}` correction as that is already accounted for in  dm_to_matter_frac

        Parameters:
        -----------
        mass : array_like
            Halo mass.
        mean_density0 : array_like
            Mean density at redshift zero.
        u_dm : array_like
            Dark matter profile.
        z : array_like
            Redshift.
        fnu : array_like
            Neutrino mass fraction.

        Returns:
        --------
        ndarray
            The matter profile with feedback.
        """
        fstar = self.fs(z)
        dm_to_matter_frac = self.omega_c / self.omega_m
        f_gas = self.fg(mass, z, fstar)

        Wm_0 = mass / mean_density0
        # Wm = (dm_to_matter_frac + f_gas) * Wm_0 * u_dm * (1.0 - fnu) + fstar * Wm_0
        # Given the astropy definiton of Om, we do not need to correct for neutrino fraction as the Om is already without!
        Wm = (dm_to_matter_frac + f_gas) * Wm_0 * u_dm + fstar * Wm_0
        return Wm

    @cached_quantity
    def matter_profile_with_feedback(self):
        """
        Compute the matter profile grid with feedback.

        Returns:
        --------
        ndarray
            The matter profile grid with feedback.
        """
        profile = self.compute_matter_profile_with_feedback(
            self.mass[np.newaxis, np.newaxis, np.newaxis, :],
            self.mean_density0[np.newaxis, :, np.newaxis, np.newaxis],
            self.u_dm[np.newaxis, :, :, :],
            self.z_vec[np.newaxis, :, np.newaxis, np.newaxis],
            0.0,
        )
        return profile

    def compute_matter_profile_with_feedback_stellar_fraction_from_obs(
        self, mass, mean_density0, u_dm, z, fnu, mb, fstar
    ):
        r"""
        Compute the matter profile using stellar fraction from observations.

        Using :math:`f_{\star}` from HOD/CSMF/CLF that also provides for point mass estimate when used in the
        GGL power spectra

        This profile does not have :math:`1-f_{\nu}` correction as that is already accounted for in  dm_to_matter_frac

        Parameters:
        -----------
        mass : array_like
            Halo mass.
        mean_density0 : array_like
            Mean density at redshift zero.
        u_dm : array_like
            Dark matter profile.
        z : array_like
            Redshift.
        fnu : array_like
            Neutrino mass fraction.
        mb : float
            Mass parameter.
        fstar : array_like
            Stellar fraction.

        Returns:
        --------
        ndarray
            The matter profile with feedback from observations.
        """
        dm_to_matter_frac = self.omega_c / self.omega_m
        Wm_0 = mass / mean_density0
        f_gas_fit = self.fg_fit(mass, mb, fstar)

        # Wm = (dm_to_matter_frac + f_gas_fit) * Wm_0 * u_dm * (1.0 - fnu) + fstar * Wm_0
        # Given the astropy definiton of Om, we do not need to correct for neutrino fraction as the Om is already without!
        Wm = (dm_to_matter_frac + f_gas_fit) * Wm_0 * u_dm + fstar * Wm_0
        return Wm

    def matter_profile_with_feedback_stellar_fraction_from_obs(self, fstar):
        """
        Compute the matter profile grid using stellar fraction from observations.

        Parameters:
        -----------
        fstar : array_like
            Stellar fraction.

        Returns:
        --------
        ndarray
            The matter profile with feedback from observations.
        """
        profile = self.compute_matter_profile_with_feedback_stellar_fraction_from_obs(
            self.mass[np.newaxis, np.newaxis, np.newaxis, :],
            self.mean_density0[np.newaxis, :, np.newaxis, np.newaxis],
            self.u_dm[np.newaxis, :, :, :],
            self.z_vec[np.newaxis, :, np.newaxis, np.newaxis],
            0.0,
            self.mb,
            fstar[:, :, np.newaxis, :],
        )
        return profile

    @cached_quantity
    def one_halo_truncation(self):
        """
        1-halo term truncation at large scales (small k)

        Parameters:
        -----------
        k_trunc : float, optional
            Truncation wavenumber.

        Returns:
        --------
        ndarray
            The truncation factor.
        """
        if self.one_halo_ktrunc is None:
            return np.ones_like(self.k_vec)
        k_frac = self.k_vec / self.one_halo_ktrunc
        return (k_frac**4.0) / (1.0 + k_frac**4.0)

    @cached_quantity
    def two_halo_truncation(self):
        """
        2-halo term truncation at larger k-values (large k).

        Parameters:
        -----------
        k_trunc : float, optional
            Truncation wavenumber.

        Returns:
        --------
        ndarray
            The truncation factor.
        """
        if self.two_halo_ktrunc is None:
            return np.ones_like(self.k_vec)
        k_d = 0.05699
        nd = 2.853
        k_frac = self.k_vec / k_d
        return 1.0 - 0.05 * (k_frac**nd) / (1.0 + k_frac**nd)

    @cached_quantity
    def one_halo_truncation_ia(self):
        """
        1-halo term truncation for IA.

        Parameters:
        -----------
        k_trunc : float, optional
            Truncation wavenumber.

        Returns:
        --------
        ndarray
            The truncation factor.
        """
        if self.one_halo_ktrunc_ia is None:
            return np.ones_like(self.k_vec)
        return 1.0 - np.exp(-((self.k_vec / self.one_halo_ktrunc_ia) ** 2.0))

    @cached_quantity
    def two_halo_truncation_ia(self):
        """
        2-halo term truncation for IA.

        Parameters:
        -----------
        k_trunc : float, optional
            Truncation wavenumber.

        Returns:
        --------
        ndarray
            The truncation factor.
        """
        if self.two_halo_ktrunc_ia is None:
            return np.ones_like(self.k_vec)
        return np.exp(-((self.k_vec / self.two_halo_ktrunc_ia) ** 2.0))

    def one_halo_truncation_mead(self, sigma8_in):
        """
        1-halo term truncation in Mead et al. 2021, eq 17 and table 2.

        Parameters:
        -----------
        sigma8_in : array_like
            Input sigma_8 values.

        Returns:
        --------
        ndarray
            The truncation factor.
        """
        sigma8_z = sigma8_in[np.newaxis, :, np.newaxis]
        # One-halo term damping wavenumber
        k_star = 0.05618 * sigma8_z ** (-1.013)  # h/Mpc
        k_frac = self.k_vec[np.newaxis, np.newaxis, :] / k_star
        return (k_frac**4.0) / (1.0 + k_frac**4.0)

    def two_halo_truncation_mead(self, sigma8_in):
        """
        2-halo term truncation in Mead et al. 2021, eq 16.

        As long as nd > 0, the multiplicative term in square brackets is
        unity for k << kd and (1 - f) for k >> kd.
        This damping is used instead of the regular 2-halo term integrals

        Parameters:
        -----------
        sigma8_in : array_like
            Input sigma_8 values.

        Returns:
        --------
        ndarray
            The truncation factor.
        """
        sigma8_z = sigma8_in[np.newaxis, :, np.newaxis]
        f = 0.2696 * sigma8_z ** (0.9403)
        k_d = 0.05699 * sigma8_z ** (-1.089)
        nd = 2.853
        k_frac = self.k_vec[np.newaxis, np.newaxis, :] / k_d
        return 1.0 - f * (k_frac**nd) / (1.0 + k_frac**nd)

    def transition_smoothing(self, neff, p_1h, p_2h):
        r"""
        Smooth the transition between 1 and 2 halo terms, eq 23 and table 2 of Mead et al. 2021.

        :math:`\alpha = 1` would correspond to a standard transition.

        :math:`\alpha < 1` smooths the transition while :math:`\alpha > 1` sharpens it.

        :math:`\Delta^{2}(k) = k^{3}/(2 \pi^{2}) P(k)`

        :math:`\Delta^{2}_{\rm hmcode}(k,z) = \left({[\Delta^{2}_{\rm 2h}(k,z)]^{\alpha} +[\Delta^{2}_{\rm 1h}(k,z)]^{\alpha}} \right)^{1/{\alpha}}`

        Parameters:
        -----------
        neff : array_like
            Effective spectral index.
        p_1h : array_like
            1-halo term power spectrum.
        p_2h : array_like
            2-halo term power spectrum.

        Returns:
        --------
        ndarray
            The smoothed power spectrum.
        """
        delta_prefac = (self.k_vec[np.newaxis, np.newaxis, :] ** 3.0) / (
            2.0 * np.pi**2.0
        )
        alpha = 1.875 * (1.603 ** neff[np.newaxis, :, np.newaxis])
        Delta_1h = delta_prefac * p_1h
        Delta_2h = delta_prefac * p_2h
        Delta_hmcode = (Delta_1h**alpha + Delta_2h**alpha) ** (1.0 / alpha)
        return Delta_hmcode / delta_prefac

    def compute_1h_term(self, profile_u, profile_v, mass, dndlnm):
        r"""
        Compute the 1-halo term for two fields u and v, e.g. matter, galaxy, intrinsic alignment

        :math:`P^{\rm 1h}_{uv}(k)= \int W_{u}(k,z,M) W_{v}(k,z,M) n(M) {\rm d}M`

        If the fields are the same and they correspond to discrete tracers (e.g. satellite galaxies):

        :math:`P^{\rm 1h}_{uv}(k)= 1/n_{x}^2 \int \langle N_{x}(M)[N_{x}(M)-1]\rangle U_{x}(k,z,M)^{2} n(M) {\rm d}M + 1/n_{x}`

        :math:`n_{x} = \int N_{x}(M) n(M) {\rm d}M`

        The shot noise term is removed as we do our measurements in real space where it only shows up
        at zero lag which is not measured.
        See eq 22 of Asgari, Mead, Heymans 2023 review paper.

        But for satellite galaxis we use:

        :math:`\langle N_{\rm sat}(N_{\rm sat}-1)\rangle = \mathcal{P} \langle N_{\rm sat}\rangle ^ {2}`:

        :math:`P^{\rm 1h}_{\rm ss}(k)= 1/n_{\rm s}^{2} \int \mathcal{P} \langle N_{\rm sat}\rangle ^{2} U_{\rm s}(k,z,M)^{2} n(M) {\rm d}M`

        and write :math:`W_u = W_v = \langle N_{\rm sat}\rangle U_{\rm s}(k,z,M) \sqrt{\mathcal{P}}/n_{\rm s}`

        for matter halo profile is: :math:`W_{\rm m} = (M/\rho_{\rm m}) U_{\rm m}(z,k,M)`

        for galaxies: :math:`W_{\rm g} = (N_{\rm g}(M)/n_{\rm g}) U_{\rm g}(z,k,M)`

        Parameters:
        -----------
        profile_u : array_like
            Profile of field u.
        profile_v : array_like
            Profile of field v.
        mass : array_like
            Halo mass.
        dndlnm : array_like
            Halo mass function.

        Returns:
        --------
        ndarray
            The 1-halo term.
        """
        integrand = ne.evaluate('profile_u * profile_v * dndlnm / mass')
        return simpson(integrand, x=mass, axis=-1)

    def compute_A_term(self, mass, b_dm, dndlnm, mean_density0):
        r"""
        Integral over the missing haloes.

        This term is used to compensate for low mass haloes that are missing from the integral in the matter 2-halo term.
        Equation A.5 of Mead and Verde 2021, 2011.08858

        :math:`A(M_{\rm min}) = 1 - [1/\bar{\rho} \int_{M_{\rm min}}^{\infty} {\rm d}M M b(M) n(M)]`

        Here all missing mass is assumed to be in halos of minimum mass :math:`M_{\rm min} = {\rm min}(M)`

        This equation arises from

        :math:`\int_{0}^{\infty} M b(M) n(M) {\rm d}M = \bar{\rho}`.

        and

        :math:`\int_{0}^{\infty} M n(M) dM = \bar{\rho}`.

        This :math:`\bar{\rho}` is the mean matter density at that redshift.

        Parameters:
        -----------
        mass : array_like
            Halo mass.
        b_dm : array_like
            Dark matter bias.
        dndlnm : array_like
            Halo mass function.
        mean_density0 : array_like
            Mean density at redshift zero.

        Returns:
        --------
        ndarray
            The integral over the missing haloes.
        """
        integrand_m1 = ne.evaluate('b_dm * dndlnm * (1.0 / mean_density0)')
        A = 1.0 - simpson(integrand_m1, x=mass)
        if (A < 0.0).any():  # pragma: no cover
            warnings.warn(
                'Warning: Mass function/bias correction is negative!', RuntimeWarning
            )
        return A

    @cached_quantity
    def missing_mass_integral(self):
        """
        Compute the integral over the missing mass.

        Returns:
        --------
        ndarray
            The integral over the missing mass.
        """
        missing_mass = self.compute_A_term(
            self.mass[np.newaxis, np.newaxis, np.newaxis, :],
            self.halo_bias[np.newaxis, :, np.newaxis, :],
            self.dndlnm[np.newaxis, :, np.newaxis, :],
            self.mean_density0[np.newaxis, :, np.newaxis, np.newaxis],
        )
        return missing_mass

    @cached_quantity
    def A_term(self):
        """
        Return the integral over the missing mass.

        Returns:
        --------
        ndarray
            The integral over the missing mass.
        """
        return self.missing_mass_integral

    @cached_quantity
    def Im_term(self):
        r"""
        Compute the integral for the matter term in the 2-halo power spectrum, eq 35 of Asgari, Mead, Heymans 2023.

        2-halo term integral for matter,

        :math:`I_{\rm m} = \int_{0}^{infty} {\rm d}M b(M) W_{\rm m}(M,k) n(M) = \int_{0}^{\infty} {\rm d}M b(M) M/{\bar{rho} U_{rm m}(M,k) n(M)`

        Returns:
        --------
        ndarray
            The integral for the matter term.
        """
        I_m_term = self.compute_Im_term(
            self.mass[np.newaxis, np.newaxis, np.newaxis, :],
            self.u_dm[np.newaxis, :, :, :],
            self.halo_bias[np.newaxis, :, np.newaxis, :],
            self.dndlnm[np.newaxis, :, np.newaxis, :],
            self.mean_density0[np.newaxis, :, np.newaxis, np.newaxis],
        )
        return I_m_term + self.A_term

    def compute_Im_term(self, mass, u_dm, b_dm, dndlnm, mean_density0):
        """
        Compute the integral for the matter term in the 2-halo power spectrum.

        Parameters:
        -----------
        mass : array_like
            Halo mass.
        u_dm : array_like
            Dark matter profile.
        b_dm : array_like
            Dark matter bias.
        dndlnm : array_like
            Halo mass function.
        mean_density0 : array_like
            Mean density at redshift zero.

        Returns:
        --------
        ndarray
            The integral for the matter term.
        """
        integrand_m = ne.evaluate('b_dm * dndlnm * u_dm * (1.0 / mean_density0)')
        return simpson(integrand_m, x=mass)

    def prepare_I22_integrand(self, b_1, b_2, dndlnm_1, dndlnm_2, B_NL_k_z):
        """
        Prepare the integrand for the I22 term.

        Parameters:
        -----------
        b_1 : array_like
            Bias for the first halo.
        b_2 : array_like
            Bias for the second halo.
        dndlnm_1 : array_like
            Halo mass function for the first halo.
        dndlnm_2 : array_like
            Halo mass function for the second halo.
        B_NL_k_z : array_like
            Non-linear bias term.

        Returns:
        --------
        ndarray
            Integrand for the I22 term.
        """
        inv_mass = 1.0 / self.mass
        b_1e = np.ascontiguousarray(b_1[:, np.newaxis, :, np.newaxis])
        b_2e = np.ascontiguousarray(b_2[:, np.newaxis, np.newaxis, :])
        dndlnm_1e = np.ascontiguousarray(dndlnm_1[:, np.newaxis, :, np.newaxis])
        dndlnm_2e = np.ascontiguousarray(dndlnm_2[:, np.newaxis, np.newaxis, :])
        inv_mass_1e = np.ascontiguousarray(
            inv_mass[np.newaxis, np.newaxis, :, np.newaxis]
        )
        inv_mass_2e = np.ascontiguousarray(
            inv_mass[np.newaxis, np.newaxis, np.newaxis, :]
        )

        integrand_22 = ne.evaluate(
            'B_NL_k_z * b_1e * b_2e * dndlnm_1e * dndlnm_2e * inv_mass_1e * inv_mass_2e'
        )
        return integrand_22

    def prepare_I12_integrand(self, b_1, b_2, dndlnm_1, dndlnm_2, B_NL_k_z):
        """
        Prepare the integrand for the I12 term.

        Parameters:
        -----------
        b_1 : array_like
            Bias for the first halo.
        b_2 : array_like
            Bias for the second halo.
        dndlnm_1 : array_like
            Halo mass function for the first halo.
        dndlnm_2 : array_like
            Halo mass function for the second halo.
        B_NL_k_z : array_like
            Non-linear bias term.

        Returns:
        --------
        ndarray
            Integrand for the I12 term.
        """
        B_NL_k_z_e = np.ascontiguousarray(B_NL_k_z[:, :, :, 0])
        b_2e = np.ascontiguousarray(b_2[:, np.newaxis, :])
        dndlnm_2e = np.ascontiguousarray(dndlnm_2[:, np.newaxis, :])
        inv_mass_2e = np.ascontiguousarray(1.0 / self.mass[np.newaxis, np.newaxis, :])

        integrand_12 = ne.evaluate('B_NL_k_z_e * b_2e * dndlnm_2e * inv_mass_2e')
        return integrand_12

    def prepare_I21_integrand(self, b_1, b_2, dndlnm_1, dndlnm_2, B_NL_k_z):
        """
        Prepare the integrand for the I21 term.

        Parameters:
        -----------
        b_1 : array_like
            Bias for the first halo.
        b_2 : array_like
            Bias for the second halo.
        dndlnm_1 : array_like
            Halo mass function for the first halo.
        dndlnm_2 : array_like
            Halo mass function for the second halo.
        B_NL_k_z : array_like
            Non-linear bias term.

        Returns:
        --------
        ndarray
            Integrand for the I21 term.
        """
        B_NL_k_z_e = np.ascontiguousarray(B_NL_k_z[:, :, 0, :])
        b_1e = np.ascontiguousarray(b_1[:, np.newaxis, :])
        dndlnm_1e = np.ascontiguousarray(dndlnm_1[:, np.newaxis, :])
        inv_mass_1e = np.ascontiguousarray(1.0 / self.mass[np.newaxis, np.newaxis, :])

        integrand_21 = ne.evaluate('B_NL_k_z_e * b_1e * dndlnm_1e * inv_mass_1e')
        return integrand_21

    def I_NL(
        self,
        W_1,
        W_2,
        b_1,
        b_2,
        dndlnm_1,
        dndlnm_2,
        A,
        rho_mean,
        B_NL_k_z,
        integrand_12_part,
        integrand_21_part,
        integrand_22_part,
    ):
        """
        Calculate the integral over beta_nl using equations A.7 to A.10 from Mead and Verde 2021.

        Parameters:
        -----------
        W_1 : array_like
            Profile for the first halo.
        W_2 : array_like
            Profile for the second halo.
        b_1 : array_like
            Bias for the first halo.
        b_2 : array_like
            Bias for the second halo.
        dndlnm_1 : array_like
            Halo mass function for the first halo.
        dndlnm_2 : array_like
            Halo mass function for the second halo.
        A : array_like
            Integral over the missing mass.
        rho_mean : array_like
            Mean density.
        B_NL_k_z : array_like
            Non-linear bias term.
        integrand_12_part : array_like
            Part of the integrand for the I12 term.
        integrand_21_part : array_like
            Part of the integrand for the I21 term.
        integrand_22_part : array_like
            Part of the integrand for the I22 term.

        Returns:
        --------
        ndarray
            The integral over beta_nl.
        """
        # Reshape W_1 and W_2 for broadcasting
        W_1e = np.ascontiguousarray(W_1[:, :, :, :, np.newaxis])
        W_2e = np.ascontiguousarray(W_2[:, :, :, np.newaxis, :])

        # Calculate integrand_22 using broadcasting
        integrand_22 = ne.evaluate('integrand_22_part * W_1e * W_2e')

        # Perform trapezoidal integration
        I_22 = self.trapezoidal_integrator(
            self.trapezoidal_integrator(integrand_22, x=self.mass, axis=-1),
            x=self.mass,
            axis=-1,
        )

        # Calculate I_11 using broadcasting
        inv_mass0 = np.ascontiguousarray(1.0 / self.mass[0])
        inv_mass0_sq = np.ascontiguousarray(inv_mass0 * inv_mass0)

        # Precompute reusable arrays
        A_sq = np.ascontiguousarray(A * A)
        rho_sq = np.ascontiguousarray(rho_mean[:, None] * rho_mean[:, None])

        # Pre-slice commonly used parts
        W1_0 = np.ascontiguousarray(W_1[:, :, :, 0])
        W2_0 = np.ascontiguousarray(W_2[:, :, :, 0])
        rho_col = np.ascontiguousarray(rho_mean[:, None])
        B_NL = np.ascontiguousarray(B_NL_k_z[:, :, 0, 0])

        I_11 = ne.evaluate('B_NL * A_sq * W1_0 * W2_0 * rho_sq * inv_mass0_sq')

        # Calculate I_12 using broadcasting
        integrand_12 = ne.evaluate('integrand_12_part * W_2')
        integral_12 = self.trapezoidal_integrator(integrand_12, x=self.mass, axis=-1)
        I_12 = ne.evaluate('A * W1_0 * integral_12 * rho_col * inv_mass0')

        # Calculate I_21 using broadcasting
        integrand_21 = ne.evaluate('integrand_21_part * W_1')
        integral_21 = self.trapezoidal_integrator(integrand_21, x=self.mass, axis=-1)
        I_21 = ne.evaluate('A * W2_0 * integral_21 * rho_col * inv_mass0')

        # Combine all terms
        return ne.evaluate('I_11 + I_12 + I_21 + I_22')

    @cached_quantity
    def _trapezoidal_weights(self):
        """Compute trapezoidal integration weights for a given grid."""
        x = self.mass
        w = np.empty_like(x)
        w[1:-1] = (x[2:] - x[:-2]) / 2
        w[0] = (x[1] - x[0]) / 2
        w[-1] = (x[-1] - x[-2]) / 2
        return w

    def trapezoidal_integrator(self, integrand, x, axis):
        """Trapezoidal integrator using fixed weights and tensor dot product as mass does not change."""
        w_mass = self._trapezoidal_weights
        integral = np.tensordot(integrand, w_mass, axes=(axis, 0))
        return integral

    def fg(self, mass, z_vec, fstar, beta=2):
        r"""
        Compute the gas fraction, eq 24 of Mead et al. 2021.

        :math:`f_{\rm g}(M) = [\Omega_{\rm b}/\Omega_{\rm m} - f_{\star}] (M/M_{\rm b})^{\beta}/ (1 + (M/M_{\rm b})^{\beta})`

        where :math:`f_{\rm g}` is the halo gas fraction, the pre-factor in parenthesis is the
        available gas reservoir, while :math:`M_{\rm b}` > 0` and :math:`\beta > 0` are fitted parameters.
        Haloes of :math:`M >> M_{\rm b}` are unaffected while those of :math:`M < M_{\rm b}` have
        lost more than half of their gas

        theta_agn = log10_TAGN - 7.8
        table 4 of 2009.01858, units of M_sun/h

        Parameters:
        -----------
        mass : array_like
            Halo mass.
        z_vec : array_like
            Redshift.
        fstar : array_like
            Stellar fraction.
        beta : float, optional
            Slope parameter.

        Returns:
        --------
        ndarray
            The gas fraction.
        """
        theta_agn = self.log10T_AGN - 7.8
        mb = np.power(10.0, 13.87 + 1.81 * theta_agn) * np.power(
            10.0, z_vec * (0.195 * theta_agn - 0.108)
        )
        baryon_to_matter_fraction = self.omega_b / self.omega_m
        return (
            (baryon_to_matter_fraction - fstar)
            * (mass / mb) ** beta
            / (1.0 + (mass / mb) ** beta)
        )

    def fg_fit(self, mass, mb, fstar, beta=2):
        r"""
        Compute the gas fraction for a general baryonic feedback model, eq 24 of Mead et al. 2021.

        :math:`f_{\rm g}(M) = [\Omega_{\rm b}/\Omega_{\rm m} - f_{\star}] (M/M_{\rm b})^{\beta}/ (1 + (M/M_{\rm b})^{\beta})`

        where :math:`f_{\rm g}` is the halo gas fraction, the pre-factor in parenthesis is the
        available gas reservoir, while :math:`M_{\rm b}` > 0` and :math:`\beta > 0` are fitted parameters.
        Haloes of :math:`M >> M_{\rm b}` are unaffected while those of :math:`M < M_{\rm b}` have
        lost more than half of their gas


        Parameters:
        -----------
        mass : array_like
            Halo mass.
        mb : float
            Mass parameter.
        fstar : array_like
            Stellar fraction.
        beta : float, optional
            Slope parameter.

        Returns:
        --------
        ndarray
            The gas fraction.
        """
        baryon_to_matter_fraction = self.omega_b / self.omega_m
        return (
            (baryon_to_matter_fraction - fstar)
            * (mass / mb) ** beta
            / (1.0 + (mass / mb) ** beta)
        )

    def fs(self, z_vec):
        r"""
        Compute the stellar fraction from table 4 and eq 26 of Mead et al. 2021.

        :math:`f_{\star}(z) = f_{\star, 0} 10^{(z f_{\star,z})}`

        Parameters:
        -----------
        z_vec : array_like
            Redshift.

        Returns:
        --------
        ndarray
            The stellar fraction.
        """
        theta_agn = self.log10T_AGN - 7.8
        fstar_0 = (2.01 - 0.3 * theta_agn) * 0.01
        fstar_z = 0.409 + 0.0224 * theta_agn
        return fstar_0 * np.power(10.0, z_vec * fstar_z)

    @cached_quantity
    def poisson_func(self):
        """
        Calculates the Poisson parameter for use in Pgg integrals.

        Can be either a scalar (P = poisson) or a power law (P = poisson x (M/M_0)^slope).
        Further models can be added to this function if necessary.

        Parameters:
        -----------
        mass : array_like
            Halo mass array.
        model_parameters : dict
            Keyword arguments for different options.

        Returns:
        --------
        ndarray
            The Poisson parameter.
        """
        func = self.poisson_model(self.mass, **self.poisson_params)
        return func.poisson_func

    @cached_quantity
    def power_spectrum_lin(
        self,
    ):
        """
        Return the linear power spectrum.

        Returns:
        --------
        PowerSpectrumResult object
            The total power spectrums.
            Each element of PowerSpectrumResult is a 3D array with shape (1, n_z, n_k).
        """
        return PowerSpectrumResult(pk_tot=self._pk_lin[np.newaxis, :, :])

    @cached_quantity
    def power_spectrum_mm(
        self,
    ):
        """
        Compute the matter-matter power spectrum.

        Returns:
        --------
        PowerSpectrumResult object
            The 1-halo term, 2-halo term, total power spectrum, and galaxy linear bias.
            Each element of PowerSpectrumResult is a 3D array with shape (1, n_z, n_k).
        """
        if self.mead_correction == 'feedback':
            matter_profile_1h = self.matter_profile_with_feedback
        elif self.mead_correction == 'fit':
            matter_profile_1h = (
                self.matter_profile_with_feedback_stellar_fraction_from_obs(
                    self.fstar_mm
                )
            )
        else:
            matter_profile_1h = self.matter_profile
        if self.bnl:
            I_NL = self.I_NL(
                self.matter_profile_2h,
                self.matter_profile_2h,
                self.halo_bias,
                self.halo_bias,
                self.dndlnm,
                self.dndlnm,
                self.A_term,
                self.mean_density0,
                self._beta_nl_array,
                self.I12,
                self.I21,
                self.I22,
            )
            pk_2h = (
                self._pk_lin * self.Im_term * self.Im_term + self._pk_lin * I_NL
            )  # * self.two_halo_truncation[np.newaxis, np.newaxis, :]
            pk_1h = (
                self.compute_1h_term(
                    matter_profile_1h,
                    matter_profile_1h,
                    self.mass[np.newaxis, np.newaxis, np.newaxis, :],
                    self.dndlnm[np.newaxis, :, np.newaxis, :],
                )
                * self.one_halo_truncation
            )
            pk_tot = pk_1h + pk_2h
        else:
            if self.mead_correction in ['feedback', 'nofeedback']:
                pk_2h = self._pk_lin[np.newaxis, :, :] * self.two_halo_truncation_mead(
                    self.sigma8_z
                )
                pk_1h = self.compute_1h_term(
                    matter_profile_1h,
                    matter_profile_1h,
                    self.mass[np.newaxis, np.newaxis, np.newaxis, :],
                    self.dndlnm[np.newaxis, :, np.newaxis, :],
                ) * self.one_halo_truncation_mead(self.sigma8_z)
                pk_tot = self.transition_smoothing(self.neff, pk_1h, pk_2h)
            else:
                pk_2h = (
                    self._pk_lin
                    * self.Im_term
                    * self.Im_term
                    * self.two_halo_truncation[np.newaxis, np.newaxis, :]
                )
                pk_1h = (
                    self.compute_1h_term(
                        matter_profile_1h,
                        matter_profile_1h,
                        self.mass[np.newaxis, np.newaxis, np.newaxis, :],
                        self.dndlnm[np.newaxis, :, np.newaxis, :],
                    )
                    * self.one_halo_truncation
                )
                pk_tot = pk_1h + pk_2h

        return PowerSpectrumResult(
            pk_1h=pk_1h, pk_2h=pk_2h, pk_tot=pk_tot, galaxy_linear_bias=None
        )

    # --------------------------------  Galaxy spectra specific funtions ------------------------------

    @cached_quantity
    def hod(self):
        """
        Initialize and return the HOD model.

        Returns:
        --------
        object
            The HOD model.
        """
        return self.hod_model(
            cosmo=self.cosmo_model,
            mass=self.mass,
            dndlnm=self.dndlnm,
            halo_bias=self.halo_bias,
            z_vec=self.z_vec,
            hod_settings=self.hod_settings,
            **self.hod_params,
        )

    @cached_quantity
    def fstar(self):
        """
        Compute the stellar fraction.

        Returns:
        --------
        ndarray
            The stellar fraction.
        """
        return self.hod.stellar_fraction

    @cached_quantity
    def mass_avg(self):
        """
        Compute the average mass.

        Returns:
        --------
        ndarray
            The average mass.
        """
        return self.hod.avg_halo_mass_cen / self.hod.number_density

    @cached_quantity
    def obs(self):
        """
        Initialize and return the observable function.

        Returns:
        --------
        object
            The observable function.
        """
        hod = self.hod_model
        if hod.__name__ == 'Cacciato' and self.compute_observable:
            return hod(
                cosmo=self.cosmo_model,
                mass=self.mass,
                dndlnm=self.dndlnm,
                halo_bias=self.halo_bias,
                z_vec=self.z_vec,
                hod_settings=self.obs_settings,
                **self.hod_params,
            )
        else:
            return None

    @cached_quantity
    def obs_func(self):
        """
        Return the observable function in correct units.

        Returns:
        --------
        ndarray
            The observable function.
        """
        if self.obs is None:
            return None
        return np.log(10.0) * np.squeeze(self.obs.obs, axis=2) * self.obs.obs_func

    @cached_quantity
    def obs_func_cen(self):
        """
        Return the observable function for central galaxies in correct units.

        Returns:
        --------
        ndarray
            The observable function for central galaxies.
        """
        if self.obs is None:
            return None
        return np.log(10.0) * np.squeeze(self.obs.obs, axis=2) * self.obs.obs_func_cen

    @cached_quantity
    def obs_func_sat(self):
        """
        Return the observable function for satellite galaxies in correct units.

        Returns:
        --------
        ndarray
            The observable function for satellite galaxies.
        """
        if self.obs is None:
            return None
        return np.log(10.0) * np.squeeze(self.obs.obs, axis=2) * self.obs.obs_func_sat

    @cached_quantity
    def obs_func_obs(self):
        """
        Return the observable function x-axis.

        Returns:
        --------
        ndarray
            The observable function x-axis.
        """
        if self.obs is None:
            return None
        return np.squeeze(self.obs.obs, axis=2)

    @cached_quantity
    def obs_func_z(self):
        """
        Return the redshift for the observable function.

        Returns:
        --------
        ndarray
            The redshift for the observable function.
        """
        if self.obs is None:
            return None
        return self.obs.z

    @cached_quantity
    def central_galaxy_profile(self):
        """
        Compute the galaxy profile for a sample of central galaxies.

        Returns:
        --------
        ndarray
            The galaxy profile for central galaxies.
        """
        return (
            self.hod.f_c[:, :, np.newaxis, np.newaxis]
            * self.hod.hod_cen[:, :, np.newaxis, :]
            * np.ones_like(self.u_sat[np.newaxis, :, :, :])
            / self.hod.number_density_cen[:, :, np.newaxis, np.newaxis]
        )

    @cached_quantity
    def satellite_galaxy_profile(self):
        """
        Compute the galaxy profile for a sample of satellite galaxies.

        Returns:
        --------
        ndarray
            The galaxy profile for satellite galaxies.
        """
        return (
            self.hod.f_s[:, :, np.newaxis, np.newaxis]
            * self.hod.hod_sat[:, :, np.newaxis, :]
            * self.u_sat[np.newaxis, :, :, :]
            / self.hod.number_density_sat[:, :, np.newaxis, np.newaxis]
        )

    def compute_Ig_term(self, profile, mass, dndlnm, b_m):
        """
        Compute the integral for the galaxy term in the 2-halo power spectrum.

        Parameters:
        -----------
        profile : array_like
            Galaxy profile.
        mass : array_like
            Halo mass.
        dndlnm : array_like
            Halo mass function.
        b_m : array_like
            Matter bias.

        Returns:
        --------
        ndarray
            The integral for the galaxy term.
        """
        integrand = ne.evaluate('profile * b_m * dndlnm / mass')
        return simpson(integrand, x=mass, axis=-1)

    @cached_quantity
    def Ic_term(self):
        """
        Compute the integral for the central galaxy term in the 2-halo power spectrum.

        Returns:
        --------
        ndarray
            The integral for the central galaxy term.
        """
        term = self.compute_Ig_term(
            self.central_galaxy_profile,
            self.mass[np.newaxis, np.newaxis, np.newaxis, :],
            self.dndlnm[np.newaxis, :, np.newaxis, :],
            self.halo_bias[np.newaxis, :, np.newaxis, :],
        )
        return term

    @cached_quantity
    def Is_term(self):
        """
        Compute the integral for the satellite galaxy term in the 2-halo power spectrum.

        Returns:
        --------
        ndarray
            The integral for the satellite galaxy term.
        """
        term = self.compute_Ig_term(
            self.satellite_galaxy_profile,
            self.mass[np.newaxis, np.newaxis, np.newaxis, :],
            self.dndlnm[np.newaxis, :, np.newaxis, :],
            self.halo_bias[np.newaxis, :, np.newaxis, :],
        )
        return term

    @cached_quantity
    def power_spectrum_gg(self):
        """
        Compute the galaxy-galaxy power spectrum.

        Returns:
        --------
        PowerSpectrumResult object
            The 1-halo term, 2-halo term, total power spectrum, and galaxy linear bias.
            Each element of PowerSpectrumResult is a 3D array with shape (n_obs_bins, n_z, n_k).
        """
        if self.bnl:
            I_NL = self.I_NL(
                self.central_galaxy_profile + self.satellite_galaxy_profile,
                self.central_galaxy_profile + self.satellite_galaxy_profile,
                self.halo_bias,
                self.halo_bias,
                self.dndlnm,
                self.dndlnm,
                self.A_term,
                self.mean_density0,
                self._beta_nl_array,
                self.I12,
                self.I21,
                self.I22,
            )
            pk_cc_2h = self._pk_lin * self.Ic_term * self.Ic_term
            pk_ss_2h = self._pk_lin * self.Is_term * self.Is_term
            pk_cs_2h = self._pk_lin * self.Ic_term * self.Is_term
        else:
            pk_cc_2h = (
                self._pk_lin
                * self.Ic_term
                * self.Ic_term
                * self.two_halo_truncation[np.newaxis, np.newaxis, :]
            )
            pk_ss_2h = (
                self._pk_lin
                * self.Is_term
                * self.Is_term
                * self.two_halo_truncation[np.newaxis, np.newaxis, :]
            )
            pk_cs_2h = (
                self._pk_lin
                * self.Ic_term
                * self.Is_term
                * self.two_halo_truncation[np.newaxis, np.newaxis, :]
            )

        pk_cs_1h = (
            self.compute_1h_term(
                self.central_galaxy_profile,
                self.satellite_galaxy_profile,
                self.mass[np.newaxis, np.newaxis, np.newaxis, :],
                self.dndlnm[np.newaxis, :, np.newaxis, :],
            )
            * self.one_halo_truncation
        )
        pk_ss_1h = (
            self.compute_1h_term(
                self.satellite_galaxy_profile * self.poisson_func,
                self.satellite_galaxy_profile,
                self.mass[np.newaxis, np.newaxis, np.newaxis, :],
                self.dndlnm[np.newaxis, :, np.newaxis, :],
            )
            * self.one_halo_truncation
        )

        pk_1h = 2.0 * pk_cs_1h + pk_ss_1h
        pk_2h = pk_cc_2h + pk_ss_2h + 2.0 * pk_cs_2h
        if self.bnl:
            pk_2h += self._pk_lin * I_NL

        pk_tot = pk_1h + pk_2h
        galaxy_linear_bias = np.sqrt(
            self.Ic_term * self.Ic_term
            + self.Is_term * self.Is_term
            + 2.0 * self.Ic_term * self.Is_term
        )

        if self.response:
            pk_1h *= self._pk_nl / self.power_spectrum_mm.pk_tot
            pk_2h *= self._pk_nl / self.power_spectrum_mm.pk_tot
            pk_tot *= self._pk_nl / self.power_spectrum_mm.pk_tot

        return PowerSpectrumResult(
            pk_1h=pk_1h,
            pk_2h=pk_2h,
            pk_tot=pk_tot,
            galaxy_linear_bias=galaxy_linear_bias,
        )

    @cached_quantity
    def power_spectrum_gm(self):
        """
        Compute the galaxy-matter power spectrum.

        Returns:
        --------
        PowerSpectrumResult object
            The 1-halo term, 2-halo term, total power spectrum, and galaxy linear bias.
            Each element of PowerSpectrumResult is a 3D array with shape (n_obs_bins, n_z, n_k).
        """
        if self.mead_correction == 'feedback':
            matter_profile_1h = self.matter_profile_with_feedback
        elif self.mead_correction == 'fit' or self.pointmass:
            matter_profile_1h = (
                self.matter_profile_with_feedback_stellar_fraction_from_obs(self.fstar)
            )
        else:
            matter_profile_1h = self.matter_profile

        if self.bnl:
            I_NL = self.I_NL(
                self.central_galaxy_profile + self.satellite_galaxy_profile,
                self.matter_profile_2h,
                self.halo_bias,
                self.halo_bias,
                self.dndlnm,
                self.dndlnm,
                self.A_term,
                self.mean_density0,
                self._beta_nl_array,
                self.I12,
                self.I21,
                self.I22,
            )
            pk_cm_2h = self._pk_lin * self.Ic_term * self.Im_term
            pk_sm_2h = self._pk_lin * self.Is_term * self.Im_term
        else:
            pk_cm_2h = (
                self._pk_lin
                * self.Ic_term
                * self.Im_term
                * self.two_halo_truncation[np.newaxis, np.newaxis, :]
            )
            pk_sm_2h = (
                self._pk_lin
                * self.Is_term
                * self.Im_term
                * self.two_halo_truncation[np.newaxis, np.newaxis, :]
            )
        pk_cm_1h = (
            self.compute_1h_term(
                self.central_galaxy_profile,
                matter_profile_1h,
                self.mass[np.newaxis, np.newaxis, np.newaxis, :],
                self.dndlnm[np.newaxis, :, np.newaxis, :],
            )
            * self.one_halo_truncation
        )
        pk_sm_1h = (
            self.compute_1h_term(
                self.satellite_galaxy_profile,
                matter_profile_1h,
                self.mass[np.newaxis, np.newaxis, np.newaxis, :],
                self.dndlnm[np.newaxis, :, np.newaxis, :],
            )
            * self.one_halo_truncation
        )

        pk_1h = pk_cm_1h + pk_sm_1h
        pk_2h = pk_cm_2h + pk_sm_2h
        if self.bnl:
            pk_2h += self._pk_lin * I_NL

        pk_tot = pk_1h + pk_2h
        galaxy_linear_bias = np.sqrt(
            self.Ic_term * self.Im_term + self.Is_term * self.Im_term
        )

        if self.response:
            pk_1h *= self._pk_nl / self.power_spectrum_mm.pk_tot
            pk_2h *= self._pk_nl / self.power_spectrum_mm.pk_tot
            pk_tot *= self._pk_nl / self.power_spectrum_mm.pk_tot

        return PowerSpectrumResult(
            pk_1h=pk_1h,
            pk_2h=pk_2h,
            pk_tot=pk_tot,
            galaxy_linear_bias=galaxy_linear_bias,
        )

    # --------------------------------  Alignment spectra specific funtions ------------------------------

    @cached_quantity
    def alignment_class(self):
        """
        Set up the aligment amplitudes and radial dependence.

        Returns:
        --------
        object
            The SatelliteAligment class
        """
        self.align_params.update(
            {
                'z_vec': self.z_vec,
                'mass_in': self.mass,
                'c_in': self.conc_cen,
                'r_s_in': self.r_s_cen,
                'rvir_in': self.rvir_cen[0],
                'method': 'fftlog',
            }
        )
        return SatelliteAlignment(**self.align_params)

    @cached_quantity
    def beta_cen(self):
        """
        Beta parameter for central galaxies.
        """
        return self.alignment_class.beta_cen

    @cached_quantity
    def beta_sat(self):
        """
        Beta parameter for satellite galaxies.
        """
        return self.alignment_class.beta_sat

    @cached_quantity
    def mpivot_cen(self):
        """
        Pivot mass for central galaxies.
        """
        return self.alignment_class.mpivot_cen

    @cached_quantity
    def mpivot_sat(self):
        """
        Pivot mass for satellite galaxies.
        """
        return self.alignment_class.mpivot_sat

    @cached_quantity
    def alignment_gi(self):
        """
        2h alignment amplitude.
        """
        return self.alignment_class.alignment_gi

    @cached_quantity
    def alignment_amplitude_2h(self):
        """
        The alignment amplitude for GI.
        """
        return -1.0 * (
            self.C1[:, :, 0]
            * self.mean_density0[:, np.newaxis]
            / self.growth_factor[:, np.newaxis]
        )

    @cached_quantity
    def alignment_amplitude_2h_II(self):
        """
        The alignment amplitude for II.
        """
        return (
            self.C1[:, :, 0]
            * self.mean_density0[:, np.newaxis]
            / self.growth_factor[:, np.newaxis]
        ) ** 2.0

    @cached_quantity
    def C1(self):
        """
        Linear alignment coefficient C1 multiplited with the 2h amplitude.
        """
        return 5e-14 * self.alignment_gi[:, np.newaxis, np.newaxis]

    @cached_quantity
    def wkm_sat(self):
        """
        Compute the radial alignment profile of satellite galaxies.

        Returns:
        --------
        ndarray
            The radial alignment profile of satellite galaxies.
        """
        return np.ascontiguousarray(
            self.alignment_class.upsampled_wkm(self.k_vec, self.mass).transpose(
                0, 2, 1
            )[np.newaxis, :, :, :]
        )

    def compute_central_galaxy_alignment_profile(
        self, growth_factor, f_c, C1, mass, beta=None, mpivot=None, mass_avg=None
    ):
        """
        Compute the central galaxy alignment profile.

        Parameters:
        -----------
        growth_factor : array_like
            Growth factor.
        f_c : array_like
            Fraction of central galaxies.
        C1 : array_like
            Amplitude of the alignment.
        mass : array_like
            Halo mass.
        beta : float, optional
            Beta parameter.
        mpivot : float, optional
            Pivot mass.
        mass_avg : array_like, optional
            Average mass.

        Returns:
        --------
        ndarray
            The central galaxy alignment profile.
        """
        if beta is not None and mpivot is not None and mass_avg is not None:
            additional_term = (mass_avg / mpivot) ** beta
        else:
            additional_term = 1.0
        return (
            f_c * (C1 / growth_factor) * mass * additional_term
        )  # * scale_factor**2.0

    def compute_satellite_galaxy_alignment_profile(
        self, Nsat, numdenssat, f_s, wkm_sat, beta=None, mpivot=None, mass_avg=None
    ):
        """
        Compute the satellite galaxy alignment profile.

        Parameters:
        -----------
        Nsat : array_like
            Number of satellite galaxies.
        numdenssat : array_like
            Number density of satellite galaxies.
        f_s : array_like
            Fraction of satellite galaxies.
        wkm_sat : array_like
            wkm for satellite galaxies.
        beta : float, optional
            Beta parameter.
        mpivot : float, optional
            Pivot mass.
        mass_avg : array_like, optional
            Average mass.

        Returns:
        --------
        ndarray
            The satellite galaxy alignment profile.
        """
        if beta is not None and mpivot is not None and mass_avg is not None:
            additional_term = (mass_avg / mpivot) ** beta
        else:
            additional_term = 1.0
        return f_s * Nsat * wkm_sat / numdenssat * additional_term

    @cached_quantity
    def central_alignment_profile(self):
        r"""
        Prepare the grid in z, k and mass for the central alignment

        :math:`\frac{f_{\rm cen}}{n_{\rm cen}} N_{\rm cen} \hat{\gamma}(k,M)`

        where :math:`\hat{\gamma}(k,M)` is the Fourier transform of the density weighted shear, i.e. the radial dependent power law
        times the NFW profile, here computed by the module wkm, while gamma_1h is only the luminosity dependence factor.

        Returns:
        --------
        ndarray
            The central alignment profile.
        """
        profile = self.compute_central_galaxy_alignment_profile(
            self.growth_factor[np.newaxis, :, np.newaxis, np.newaxis],
            self.hod.f_c[:, :, np.newaxis, np.newaxis],
            self.C1[np.newaxis, :, :, :],
            self.mass[np.newaxis, np.newaxis, np.newaxis, :],
            self.beta_cen,
            self.mpivot_cen,
            self.mass_avg[:, :, np.newaxis, np.newaxis],
        )
        return profile

    @cached_quantity
    def satellite_alignment_profile(self):
        r"""
        Prepare the grid in z, k and mass for the satellite alignment

        :math:`\frac{f_{\rm sat}}{n_{\rm sat}} N_{\rm sat} \hat{\gamma}(k,M)`

        where :math:`\hat{\gamma}(k,M)` is the Fourier transform of the density weighted shear, i.e. the radial dependent power law
        times the NFW profile, here computed by the module wkm, while gamma_1h is only the luminosity dependence factor.

        Returns:
        --------
        ndarray
            The satellite alignment profile.
        """
        profile = self.compute_satellite_galaxy_alignment_profile(
            self.hod.hod_sat[:, :, np.newaxis, :],
            self.hod.number_density_sat[:, :, np.newaxis, np.newaxis],
            self.hod.f_s[:, :, np.newaxis, np.newaxis],
            self.wkm_sat,
            self.beta_sat,
            self.mpivot_sat,
            self.mass_avg[:, :, np.newaxis, np.newaxis],
        )
        return profile

    @cached_quantity
    def Ic_align_term(self):
        """
        Compute the integral for the central galaxy alignment term in the 2-halo power spectrum.

        Returns:
        --------
        ndarray
            The integral for the central galaxy alignment term.
        """
        I_g_align = self.compute_Ig_term(
            self.central_alignment_profile,
            self.mass[np.newaxis, np.newaxis, np.newaxis, :],
            self.dndlnm[np.newaxis, :, np.newaxis, :],
            self.halo_bias[np.newaxis, :, np.newaxis, :],
        )
        return (
            I_g_align
            + self.A_term
            * self.central_alignment_profile[:, :, :, 0]
            * self.mean_density0[np.newaxis, :, np.newaxis]
            / self.mass[0]
        )

    @cached_quantity
    def Is_align_term(self):
        """
        Compute the integral for the satellite galaxy alignment term in the 2-halo power spectrum.

        Returns:
        --------
        ndarray
            The integral for the satellite galaxy alignment term.
        """
        I_g_align = self.compute_Ig_term(
            self.satellite_alignment_profile,
            self.mass[np.newaxis, np.newaxis, np.newaxis, :],
            self.dndlnm[np.newaxis, :, np.newaxis, :],
            self.halo_bias[np.newaxis, :, np.newaxis, :],
        )
        return (
            I_g_align
            + self.A_term
            * self.satellite_alignment_profile[:, :, :, 0]
            * self.mean_density0[np.newaxis, :, np.newaxis]
            / self.mass[0]
        )

    @cached_quantity
    def power_spectrum_mi(self):
        """
        Compute the matter-intrinsic alignment power spectrum.

        Returns:
        --------
        PowerSpectrumResult object
            The 1-halo term, 2-halo term, total power spectrum, and galaxy linear bias.
            Each element of PowerSpectrumResult is a 3D array with shape (n_obs_bins, n_z, n_k).
        """
        if self.mead_correction == 'feedback':
            matter_profile_1h = self.matter_profile_with_feedback
        elif self.mead_correction == 'fit' or self.pointmass:
            matter_profile_1h = (
                self.matter_profile_with_feedback_stellar_fraction_from_obs(self.fstar)
            )
        else:
            matter_profile_1h = self.matter_profile

        if self.bnl:
            I_NL = self.I_NL(
                self.central_alignment_profile + self.satellite_alignment_profile,
                self.matter_profile_2h,
                self.halo_bias,
                self.halo_bias,
                self.dndlnm,
                self.dndlnm,
                self.A_term,
                self.mean_density0,
                self._beta_nl_array,
                self.I12,
                self.I21,
                self.I22,
            )
            pk_sm_2h = (-1.0) * self._pk_lin * self.Is_align_term * self.Im_term
            pk_cm_2h = (-1.0) * self._pk_lin * self.Ic_align_term * self.Im_term
        elif self.fortuna:
            pk_cm_2h = (
                self.hod.f_c[:, :, np.newaxis]
                * self.peff
                * self.alignment_amplitude_2h
                * self.two_halo_truncation_ia[np.newaxis, np.newaxis, :]
            )
        else:
            pk_sm_2h = (
                (-1.0)
                * self._pk_lin
                * self.Is_align_term
                * self.Im_term
                * self.two_halo_truncation_ia[np.newaxis, np.newaxis, :]
            )
            pk_cm_2h = (
                (-1.0)
                * self._pk_lin
                * self.Ic_align_term
                * self.Im_term
                * self.two_halo_truncation_ia[np.newaxis, np.newaxis, :]
            )

        pk_sm_1h = (
            (-1.0)
            * self.compute_1h_term(
                matter_profile_1h,
                self.satellite_alignment_profile,
                self.mass[np.newaxis, np.newaxis, np.newaxis, :],
                self.dndlnm[np.newaxis, :, np.newaxis, :],
            )
            * self.one_halo_truncation_ia
        )
        # pk_cm_1h = (-1.0) * self.compute_1h_term(matter_profile_1h, self.central_alignment_profile, self.mass[np.newaxis, np.newaxis, np.newaxis, :], self.dndlnm[np.newaxis, :, np.newaxis, :]) * self.one_halo_truncation_ia

        if self.bnl:
            pk_1h = pk_sm_1h
            pk_2h = pk_cm_2h + pk_sm_2h - self._pk_lin * I_NL
            pk_tot = pk_sm_1h + pk_cm_2h + pk_sm_2h - self._pk_lin * I_NL
        elif self.fortuna:
            pk_1h = pk_sm_1h
            pk_2h = pk_cm_2h
            pk_tot = pk_sm_1h + pk_cm_2h
        else:
            pk_1h = pk_sm_1h
            pk_2h = pk_cm_2h + pk_sm_2h
            pk_tot = pk_sm_1h + pk_cm_2h + pk_sm_2h

        if self.response:
            pk_1h *= self._pk_nl / self.power_spectrum_mm.pk_tot
            pk_2h *= self._pk_nl / self.power_spectrum_mm.pk_tot
            pk_tot *= self._pk_nl / self.power_spectrum_mm.pk_tot

        return PowerSpectrumResult(
            pk_1h=pk_1h, pk_2h=pk_2h, pk_tot=pk_tot, galaxy_linear_bias=None
        )

    @cached_quantity
    def power_spectrum_ii(self):
        """
        Compute the intrinsic-intrinsic alignment power spectrum.

        Returns:
        --------
        PowerSpectrumResult object
            The 1-halo term, 2-halo term, total power spectrum, and galaxy linear bias.
            Each element of PowerSpectrumResult is a 3D array with shape (n_obs_bins, n_z, n_k).
        """
        # Needs Poisson parameter as well!
        if self.bnl:
            I_NL_ss = self.I_NL(
                self.satellite_alignment_profile,
                self.satellite_alignment_profile,
                self.halo_bias,
                self.halo_bias,
                self.dndlnm,
                self.dndlnm,
                self.A_term,
                self.mean_density0,
                self._beta_nl_array,
                self.I12,
                self.I21,
                self.I22,
            )
            I_NL_cc = self.I_NL(
                self.central_alignment_profile,
                self.central_alignment_profile,
                self.halo_bias,
                self.halo_bias,
                self.dndlnm,
                self.dndlnm,
                self.A_term,
                self.mean_density0,
                self._beta_nl_array,
                self.I12,
                self.I21,
                self.I22,
            )
            I_NL_cs = self.I_NL(
                self.central_alignment_profile,
                self.satellite_alignment_profile,
                self.halo_bias,
                self.halo_bias,
                self.dndlnm,
                self.dndlnm,
                self.A_term,
                self.mean_density0,
                self._beta_nl_array,
                self.I12,
                self.I21,
                self.I22,
            )
            pk_ss_2h = (
                self._pk_lin * self.Is_align_term * self.Is_align_term
                + self._pk_lin * I_NL_ss
            )
            pk_cc_2h = (
                self._pk_lin * self.Ic_align_term * self.Ic_align_term
                + self._pk_lin * I_NL_cc
            )
            pk_cs_2h = (
                self._pk_lin * self.Ic_align_term * self.Is_align_term
                + self._pk_lin * I_NL_cs
            )
        elif self.fortuna:
            pk_cc_2h = (
                self.hod.f_c[:, :, np.newaxis] ** 2.0
                * self.peff
                * self.alignment_amplitude_2h_II
                * self.two_halo_truncation_ia[np.newaxis, np.newaxis, :]
            )
        else:
            pk_ss_2h = (
                self._pk_lin
                * self.Is_align_term
                * self.Is_align_term
                * self.two_halo_truncation_ia[np.newaxis, np.newaxis, :]
            )
            pk_cc_2h = (
                self._pk_lin
                * self.Ic_align_term
                * self.Ic_align_term
                * self.two_halo_truncation_ia[np.newaxis, np.newaxis, :]
            )
            pk_cs_2h = (
                self._pk_lin
                * self.Ic_align_term
                * self.Is_align_term
                * self.two_halo_truncation_ia[np.newaxis, np.newaxis, :]
            )

        pk_ss_1h = (
            self.compute_1h_term(
                self.satellite_alignment_profile,
                self.satellite_alignment_profile,
                self.mass[np.newaxis, np.newaxis, np.newaxis, :],
                self.dndlnm[np.newaxis, :, np.newaxis, :],
            )
            * self.one_halo_truncation_ia
        )
        # pk_cs_1h = self.compute_1h_term(self.central_alignment_profile, self.satellite_alignment_profile, self.mass[np.newaxis, np.newaxis, np.newaxis, :], self.dndlnm[np.newaxis, :, np.newaxis, :]) * self.one_halo_truncation_ia

        if self.fortuna:
            pk_1h = pk_ss_1h
            pk_2h = pk_cc_2h
            pk_tot = pk_ss_1h + pk_cc_2h
        else:
            pk_1h = pk_ss_1h
            pk_2h = pk_cc_2h + pk_ss_2h + pk_cs_2h
            pk_tot = pk_ss_1h + pk_cc_2h + pk_ss_2h + pk_cs_2h

        if self.response:
            pk_1h *= self._pk_nl / self.power_spectrum_mm.pk_tot
            pk_2h *= self._pk_nl / self.power_spectrum_mm.pk_tot
            pk_tot *= self._pk_nl / self.power_spectrum_mm.pk_tot

        return PowerSpectrumResult(
            pk_1h=pk_1h, pk_2h=pk_2h, pk_tot=pk_tot, galaxy_linear_bias=None
        )

    @cached_quantity
    def power_spectrum_gi(self):
        """
        Compute the galaxy-intrinsic alignment power spectrum.

        Returns:
        --------
        PowerSpectrumResult object
            The 1-halo term, 2-halo term, total power spectrum, and galaxy linear bias.
            Each element of PowerSpectrumResult is a 3D array with shape (n_obs_bins, n_z, n_k).
        """
        if self.bnl:
            I_NL_cc = self.I_NL(
                self.central_alignment_profile,
                self.central_galaxy_profile,
                self.halo_bias,
                self.halo_bias,
                self.dndlnm,
                self.dndlnm,
                self.A_term,
                self.mean_density0,
                self._beta_nl_array,
                self.I12,
                self.I21,
                self.I22,
            )
            I_NL_cs = self.I_NL(
                self.satellite_alignment_profile,
                self.central_galaxy_profile,
                self.halo_bias,
                self.halo_bias,
                self.dndlnm,
                self.dndlnm,
                self.A_term,
                self.mean_density0,
                self._beta_nl_array,
                self.I12,
                self.I21,
                self.I22,
            )
            pk_cc_2h = (
                self._pk_lin * self.Ic_term * self.Ic_align_term
                + self._pk_lin * I_NL_cc
            )
            pk_cs_2h = (
                self._pk_lin * self.Ic_term * self.Is_align_term
                + self._pk_lin * I_NL_cs
            )
        elif self.fortuna:
            pk_cc_2h = (
                -1.0
                * self.peff
                * self.Ic_term
                * self.alignment_amplitude_2h[:,]
                * self.two_halo_truncation_ia[np.newaxis, np.newaxis, :]
            )
        else:
            pk_cc_2h = (
                self._pk_lin
                * self.Ic_term
                * self.Ic_align_term
                * self.two_halo_truncation_ia[np.newaxis, np.newaxis, :]
            )
            pk_cs_2h = (
                self._pk_lin
                * self.Ic_term
                * self.Is_align_term
                * self.two_halo_truncation_ia[np.newaxis, np.newaxis, :]
            )

        pk_cs_1h = (
            self.compute_1h_term(
                self.central_galaxy_profile,
                self.satellite_alignment_profile,
                self.mass[np.newaxis, np.newaxis, np.newaxis, :],
                self.dndlnm[np.newaxis, :, np.newaxis, :],
            )
            * self.one_halo_truncation_ia
        )

        if self.fortuna:
            pk_1h = pk_cs_1h
            pk_2h = pk_cc_2h
            pk_tot = pk_cs_1h + pk_cc_2h
        else:
            pk_1h = pk_cs_1h
            pk_2h = pk_cs_2h + pk_cc_2h
            pk_tot = pk_cs_1h + pk_cs_2h + pk_cc_2h

        if self.response:
            pk_1h *= self._pk_nl / self.power_spectrum_mm.pk_tot
            pk_2h *= self._pk_nl / self.power_spectrum_mm.pk_tot
            pk_tot *= self._pk_nl / self.power_spectrum_mm.pk_tot

        return PowerSpectrumResult(
            pk_1h=pk_1h, pk_2h=pk_2h, pk_tot=pk_tot, galaxy_linear_bias=None
        )

    # Helper functions borrowed from Alex Mead, no need to reinvent the wheel.
    def sigmaV(self, k, power):
        """
        Calculate the dispersion from the power spectrum.

        Parameters:
        -----------
        k : array_like
            Wavenumber array.
        power : array_like
            Power spectrum.

        Returns:
        --------
        ndarray
            The dispersion.
        """
        # In the limit where r -> 0
        dlnk = np.log(k[1] / k[0])
        # we multiply by k because our steps are in logk.
        integ = power * k
        sigma = (0.5 / np.pi**2.0) * simpson(integ, dx=dlnk, axis=-1)
        return np.sqrt(sigma / 3.0)

    @cached_quantity
    def Tk_EH_nowiggle_lnt(self):
        return Tk_EH_nowiggle(self.cosmo_model).lnt

    def get_Pk_wiggle(self, k, Pk_lin, ns, sigma_dlnk=0.25):
        """
        Extract the wiggle from the linear power spectrum.

        TODO: Should get to work for uneven log(k) spacing
        NOTE: https://stackoverflow.com/questions/24143320/gaussian-sum-filter-for-irregular-spaced-points

        Parameters:
        -----------
        k : array_like
            Wavenumber array.
        Pk_lin : array_like
            Linear power spectrum.
        h : float
            Hubble parameter.
        ombh2 : float
            Baryon density parameter times h^2.
        ommh2 : float
            Matter density parameter times h^2.
        ns : float
            Spectral index.
        T_CMB : float, optional
            Temperature of the CMB.
        sigma_dlnk : float, optional
            Smoothing scale in log(k).

        Returns:
        --------
        ndarray
            The wiggle component of the power spectrum.
        """
        if not np.isclose(np.all(np.diff(k) - np.diff(k)[0]), 0.0):
            raise ValueError('Dewiggle only works with linearly-spaced k array')

        dlnk = np.log(k[1] / k[0])
        sigma = sigma_dlnk / dlnk

        Pk_nowiggle = (k**ns) * np.exp(self.Tk_EH_nowiggle_lnt(np.log(k))) ** 2.0
        Pk_ratio = Pk_lin / Pk_nowiggle
        Pk_ratio = gaussian_filter1d(Pk_ratio, sigma)
        Pk_smooth = Pk_ratio * Pk_nowiggle
        Pk_wiggle = Pk_lin - Pk_smooth
        return Pk_wiggle
