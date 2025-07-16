"""
A module for computing Halo Occupation Distribution (HOD) models.
This module provides classes and functions to calculate properties of galaxies within dark matter halos,
using various HOD models and conditional observable functions (COFs).
"""

from functools import cached_property
import numpy as np
from scipy.integrate import simpson
from scipy.special import erf
from scipy.interpolate import interp1d
from hmf._internals._cache import cached_quantity, parameter
from hmf._internals._framework import Framework

_defaults = {
    'observables_file': None,
    'obs_min': np.array([8.0]),
    'obs_max': np.array([12.0]),
    'zmin': np.array([0.0]),
    'zmax': np.array([0.2]),
    'nz': 15,
    'nobs': 300,
    'observable_h_unit': '1/h^2',
}

def load_data(file_name):
    """
    Load data from a file.

    Parameters:
    -----------
    file_name : str
        Path to the file to load data from.

    Returns:
    --------
    tuple
        Redshift data, minimum observable data, and maximum observable data.
    """
    z_data, obs_min, obs_max = np.loadtxt(
        file_name, usecols=(0, 1, 2), unpack=True, dtype=float
    )
    return z_data, obs_min, obs_max


class HOD(Framework):
    """
    Base class for Halo Occupation Distribution (HOD) models.
    This class provides the framework for computing various properties of galaxies within dark matter halos.

    Parameters:
    -----------
    mass : array_like
        Array of halo masses.
    dndlnm : array_like
        Halo mass function.
    halo_bias : array_like
        Halo bias.
    z_vec : array_like
        Array of redshifts.
    hod_settings : dict
        Dictionary of HOD settings.
    """
    def __init__(
            self,
            mass=None,
            dndlnm=None,
            halo_bias=None,
            z_vec=None,
            hod_settings: dict =_defaults
        ):
        self.mass = mass
        self.z_vec = z_vec
        self.hod_settings = hod_settings
        self.dndlnm = dndlnm
        self.halo_bias = halo_bias
        
    @parameter("param")
    def z_vec(self, val):
        """
        Array of redshifts.

        :type: array_like
        """
        return val

    @parameter("param")
    def mass(self, val):
        """
        Array of halo masses.

        :type: array_like
        """
        if val is None:
            raise ValueError("Mass needs to be specified!")
        # With newaxis we make sure the HOD shape is (nb, nz, nmass)
        return val[np.newaxis, np.newaxis, :]

    @parameter("param")
    def dndlnm(self, val):
        """
        Halo mass function.

        :type: array_like
        """
        if val is None:
            raise ValueError("Halo mass function needs to be specified!")
        return val

    @parameter("param")
    def halo_bias(self, val):
        """
        Halo bias.

        :type: array_like
        """
        if val is None:
            raise ValueError("Halo bias function needs to be specified!")
        return val
        
    @parameter("param")
    def hod_settings(self, val):
        """
        Dictionary of HOD settings.

        :type: dict
        """
        return val
        
    @cached_quantity
    def obs(self):
        """
        Returns the array of observables at which to calculate the CSMF/CLF and integrate over to get the HOD

        Returns:
        --------
        array_like
            array of observables
        """
        obs = np.array([[np.logspace(self.log_obs_min[nb, jz], self.log_obs_max[nb, jz], self.nobs) for jz in range(self.nz)] for nb in range(self.nbins)])
        # With newaxis we make sure the COF shape is (nb, nz, nmass, nobs)
        return obs[:, :, np.newaxis, :]
        
    @cached_quantity
    def dndlnm_int(self):
        """
        Returns the interpolated halo mass function at HOD specific redshifts.

        Returns:
        --------
        array_like
            interpolated halo mass function at self.z
        """
        dndlnm_fnc = interp1d(
            self.z_vec, self.dndlnm, kind='linear', fill_value='extrapolate',
            bounds_error=False, axis=0
        )
        return dndlnm_fnc(self.z)

    @cached_quantity
    def halo_bias_int(self):
        """
        Returns the interpolated halo bias function at HOD specific redshifts.

        Returns:
        --------
        array_like
            interpolated halo bias function at self.z
        """
        halo_bias_fnc = interp1d(
            self.z_vec, self.halo_bias, kind='linear', fill_value='extrapolate',
            bounds_error=False, axis=0
        )
        return halo_bias_fnc(self.z)
        
    @cached_quantity
    def data(self):
        """
        Returns the z_bins, obs_min and obs_max quantities from tabulated observables file.

        Returns:
        --------
        tupple or None
        """
        if self.hod_settings['observables_file'] is not None:
            z_bins, obs_min, obs_max = load_data(self.hod_settings['observables_file'])
            return z_bins, obs_min, obs_max
        else:
            return None

    @cached_quantity
    def nobs(self):
        """
        Sets the number of observables in the observable array.
        
        Returns:
        --------
        array_like
            number of obserables in the observable array
        """
        return self.hod_settings['nobs']

    @cached_quantity
    def nbins(self):
        """
        Returns the number of HOD bins.

        Returns:
        --------
        int
            number of HOD bins
        """
        if self.hod_settings['observables_file'] is not None:
            return 1
        else:
            return len(self.hod_settings['obs_min'])

    @cached_quantity
    def nz(self):
        """
        Sets the number of redshift bins.

        Returns:
        --------
        int
            number of redshift bins
        """
        if self.hod_settings['observables_file'] is not None:
            return len(self.data[0])
        else:
            return self.hod_settings['nz']
        
    @cached_quantity
    def z(self):
        """
        Sets the HOD specific redshift array.

        Returns:
        --------
        array_like
            HOD specific redshifts
        """
        if self.hod_settings['observables_file'] is not None:
            return self.data[0][np.newaxis, :]
        else:
            zmin = self.hod_settings['zmin']
            zmax = self.hod_settings['zmax']
            return np.array([np.linspace(zmin_i, zmax_i, self.nz) for zmin_i, zmax_i in zip(zmin, zmax)])
        
    @cached_quantity
    def log_obs_min(self):
        """
        Sets the min observable limits.

        Returns:
        --------
        array_like
            min observable limits
        """
        if self.hod_settings['observables_file'] is not None:
            return np.log10(self.data[1])[np.newaxis, :]
        else:
            obs_min = self.hod_settings['obs_min']
            return np.array([np.repeat(obs_min_i, self.nz) for obs_min_i in obs_min])

    @cached_quantity
    def log_obs_max(self):
        """
        Sets the max observable limits.

        Returns:
        --------
        array_like
            max observable limits
        """
        if self.hod_settings['observables_file'] is not None:
            return np.log10(self.data[2])[np.newaxis, :]
        else:
            obs_max = self.hod_settings['obs_max']
            return np.array([np.repeat(obs_max_i, self.nz) for obs_max_i in obs_max])

    def _mass_integral(self, hod):
        """
        Compute the mass integral for a given HOD.

        Parameters:
        -----------
        hod : array_like
            Halo Occupation Distribution.

        Returns:
        --------
        array_like
            Integral of the HOD weighted by the halo mass function.
        """
        integrand = hod * self.dndlnm_int / self.mass
        return simpson(integrand, self.mass, axis=-1)

    def _mean_mass_integral(self, hod):
        """
        Compute the mean mass integral for a given HOD.

        Parameters:
        -----------
        hod : array_like
            Halo Occupation Distribution.

        Returns:
        --------
        array_like
            Integral of the HOD weighted by the halo mass function and halo mass.
        """
        integrand = hod * self.dndlnm_int
        return simpson(integrand, self.mass, axis=-1)

    def _bias_integral(self, hod):
        """
        Compute the bias integral for a given HOD.

        Parameters:
        -----------
        hod : array_like
            Halo Occupation Distribution.

        Returns:
        --------
        array_like
            Integral of the HOD weighted by the halo bias and halo mass function.
        """
        bg_integrand = hod * self.halo_bias_int * self.dndlnm_int / self.mass
        return simpson(bg_integrand, self.mass, axis=-1) / self.ntot

    def _interpolate(self, data, fill_value='extrapolate', axis=-1):
        """
        Helper function to interpolate data along a given axis in an array.

        Parameters:
        -----------
        data : array_like
            Data to interpolate.
        fill_value : str, optional
            Value to use for points outside the data range.
        axis : int, optional
            Axis along which to interpolate.

        Returns:
        --------
        array_like
            Interpolated data.
        """
        n_int = [interp1d(self.z[i], data[i], fill_value=fill_value, bounds_error=False, axis=axis) for i in range(self.z.shape[0])]
        return np.array([f(self.z_vec) for f in n_int])

    @cached_quantity
    def _ncen(self):
        r"""
        Total number density of central galaxies with the given HOD.
        This is an integral over the HOD and the halo mass function to remove the halo mass dependence.
        
        :math:`N_{\rm x} = \int ⟨N_{rm x}|M⟩ n(M) {\rm d}M`
        """
        return self._mass_integral(self._compute_hod_cen)

    @cached_quantity
    def _nsat(self):
        r"""
        Total number density of satellite galaxies with the given HOD.
        This is an integral over the HOD and the halo mass function to remove the halo mass dependence.
       
        :math:`N_{\rm x} = \int ⟨N_{rm x}|M⟩ n(M) {\rm d}M`
        """
        return self._mass_integral(self._compute_hod_sat)

    @cached_quantity
    def _ntot(self):
        r"""
        Total number density of galaxies with the given HOD.
        This is an integral over the HOD and the halo mass function to remove the halo mass dependence.
        
        :math:`N_{\rm x} = \int ⟨N_{rm x}|M⟩ n(M) {\rm d}M`
        """
        return self._mass_integral(self._compute_hod)

    @cached_quantity
    def _mass_avg_cen(self):
        r"""
        The mean halo mass for the given population of galaxies, e.g. central and satellite galaxies.
        
        :math:`M_{\rm mean} = \int ⟨N_{_rm x}|M⟩ M n(M) {\rm d}M`
        """
        return self._mean_mass_integral(self._compute_hod_cen)

    @cached_quantity
    def _mass_avg_sat(self):
        r"""
        The mean halo mass for the given population of galaxies, e.g. central and satellite galaxies.
        
        :math:`M_{\rm mean} = \int ⟨N_{_rm x}|M⟩ M n(M) {\rm d}M`
        """
        return self._mean_mass_integral(self._compute_hod_sat)

    @cached_quantity
    def _mass_avg_tot(self):
        r"""
        The mean halo mass for the given population of galaxies, e.g. central and satellite galaxies.
        
        :math:`M_{\rm mean} = \int ⟨N_{_rm x}|M⟩ M n(M) {\rm d}M`
        """
        return self._mean_mass_integral(self._compute_hod)

    @cached_quantity
    def _bg_cen(self):
        r"""
        Mean linear halo bias for the given population of galaxies.

        :math:`b_{\rm lin, x} = \int ⟨N_{\rm x}|M⟩ b_h(M) n(M) {\rm d}M`
        """
        return self._bias_integral(self._compute_hod_cen)

    @cached_quantity
    def _bg_sat(self):
        r"""
        Mean linear halo bias for the given population of galaxies.
        
        :math:`b_{\rm lin, x} = \int ⟨N_{\rm x}|M⟩ b_h(M) n(M) {\rm d}M`
        """
        return self._bias_integral(self._compute_hod_sat)

    @cached_quantity
    def _bg_tot(self):
        r"""
        Mean linear halo bias for the given population of galaxies.
        
        :math:`b_{\rm lin, x} = \int ⟨N_{\rm x}|M⟩ b_h(M) n(M) {\rm d}M`
        """
        return self._bias_integral(self._compute_hod)

    @cached_quantity
    def number_density_cen(self):
        """
        Compute the number density of central galaxies.

        Returns:
        --------
        array_like
            number density of central galaxies
        """
        return self._interpolate(self._ncen)

    @cached_quantity
    def number_density_sat(self):
        """
        Compute the number density of satellite galaxies.

        Returns:
        --------
        array_like
            number density of satellite galaxies
        """
        return self._interpolate(self._nsat)

    @cached_quantity
    def number_density(self):
        """
        Compute the number density of galaxies.
        
        Returns:
        --------
        array_like 
            number density of all galaxies
        """
        return self._interpolate(self._ntot)

    @cached_quantity
    def f_c(self):
        """
        Fraction of central galaxies.

        Returns:
        --------
        array_like
            fraction of central galaxies
        """
        f_c = self._ncen / self._ntot
        return self._interpolate(f_c, fill_value=0.0)

    @cached_quantity
    def f_s(self):
        """
        Fraction of satellite galaxies.

        Returns:
        --------
        array_like
            fraction of satellite galaxies
        """
        f_s = self._nsat / self._ntot
        return self._interpolate(f_s, fill_value=0.0)

    @cached_quantity
    def avg_halo_mass_cen(self):
        """
        Compute the average halo mass for central galaxies.

        Returns:
        --------
        array_like
            average halo mass for central galaxies
        """
        return self._interpolate(self._mass_avg_cen, fill_value=0.0)

    @cached_quantity
    def avg_halo_mass_sat(self):
        """
        Compute the average halo mass for satellite galaxies.

        Returns:
        --------
        array_like
            average halo mass for satellite galaxies
        """
        return self._interpolate(self._mass_avg_sat, fill_value=0.0)

    @cached_quantity
    def avg_halo_mass(self):
        """
        Compute the average halo mass for galaxies.

        Returns:
        --------
        array_like
            average halo mass for all galaxies
        """
        return self._interpolate(self._mass_avg_tot, fill_value=0.0)

    @cached_quantity
    def galaxy_linear_bias_cen(self):
        """
        Compute the galaxy linear bias for central galaxies.

        Returns:
        --------
        array_like
            galaxy linear bias for central galaxies
        """
        return self._interpolate(self._bg_cen)

    @cached_quantity
    def galaxy_linear_bias_sat(self):
        """
        Compute the galaxy linear bias for satellite galaxies.

        Returns:
        --------
        array_like
            galaxy linear bias for satellite galaxies
        """
        return self._interpolate(self._bg_sat)

    @cached_quantity
    def galaxy_linear_bias(self):
        """
        Compute the galaxy linear bias for galaxies.

        Returns:
        --------
        array_like
            galaxy linear bias for all galaxies
        """
        return self._interpolate(self._bg_tot)

    @cached_quantity
    def hod_cen(self):
        """
        Compute the HOD for central galaxies.

        Returns:
        --------
        array_like
            HOD of central galaxies
        """
        return self._interpolate(self._compute_hod_cen, axis=0)

    @cached_quantity
    def hod_sat(self):
        """
        Compute the HOD for satellite galaxies.

        Returns:
        --------
        array_like
            HOD of satellite galaxies
        """
        return self._interpolate(self._compute_hod_sat, axis=0)

    @cached_quantity
    def hod(self):
        """
        Compute the HOD for galaxies.

        Returns:
        --------
        array_like
            HOD of all galaxies
        """
        return self._interpolate(self._compute_hod, axis=0)

    @cached_quantity
    def stellar_fraction_cen(self):
        """
        Compute the stellar fraction for central galaxies.

        Returns:
        --------
        array_like
            stellar fraction of central galaxies
        """
        if self._compute_stellar_fraction_cen is None:
            return np.zeros((self.nbins, self.z_vec.size, self.mass.shape[-1]))
        else:
            return self._interpolate(self._compute_stellar_fraction_cen, axis=0)

    @cached_quantity
    def stellar_fraction_sat(self):
        """
        Compute the stellar fraction for satellite galaxies.

        Returns:
        --------
        array_like
            stellar fraction of satellite galaxies
        """
        if self._compute_stellar_fraction_sat is None:
            return np.zeros((self.nbins, self.z_vec.size, self.mass.shape[-1]))
        else:
            return self._interpolate(self._compute_stellar_fraction_sat, axis=0)

    @cached_quantity
    def stellar_fraction(self):
        """
        Compute the stellar fraction for galaxies.

        Returns:
        --------
        array_like
            stellar fraction of all galaxies
        """
        if self._compute_stellar_fraction is None:
            return np.zeros((self.nbins, self.z_vec.size, self.mass.shape[-1]))
        else:
            return self._interpolate(self._compute_stellar_fraction, axis=0)

    @property
    def _compute_hod_cen(self):
        """
        Compute the HOD for central galaxies.
        """
        raise NotImplementedError

    @property
    def _compute_hod_sat(self):
        """
        Compute the HOD for satellite galaxies.
        """
        raise NotImplementedError

    @property
    def _compute_hod(self):
        """
        Compute the total HOD by summing central and satellite HODs.
        """
        return self._compute_hod_cen + self._compute_hod_sat

    @property
    def _compute_stellar_fraction_cen(self):
        return None

    @property
    def _compute_stellar_fraction_sat(self):
        return None

    @property
    def _compute_stellar_fraction(self):
        return None

class Cacciato(HOD):
    r"""
    CSMF/CLF model from Cacciato et al. (2013) [1]_.
    
    The conditional observable functions (COFs) tell us how many galaxies with the observed property O, 
    exist in haloes of mass M: :math:`\Phi(O|M)`.

    Integrating over the observable will give us the total number of galaxies in haloes of a given mass, 
    the so-called Halo Occupation Distribution (HOD).

    The observable can be galaxy stellar mass or galaxy luminosity or possibly other properties of galaxies.
    Note that the general mathematical form of the COFs might not hold for other observables.

    COF is different for central and satellite galaxies. The total COF can be written as the sum of the two:

    :math:`\Phi(O|M) = \Phi_{\rm c}(O|M) + \Phi_{\rm s}(O|M)`
    
    The halo mass dependence comes in through pivot observable values denoted by :math:`\star`, e.g. :math:`O_{\star, {\rm c}}`, :math:`O_{\star, {\rm s}}`
    
    Parameters:
    -----------
    log10_obs_norm_c : float
        Log10 of the normalization for central galaxies.
    log10_m_ch : float
        Log10 of the characteristic mass.
    g1 : float
        Low mass slope parameter for central galaxies.
    g2 : float
        High mass slope parameter for central galaxies.
    sigma_log10_O_c : float
        Scatter in log10 of the observable for central galaxies.
    norm_s : float
        Normalization for satellite galaxies.
    pivot : float
        Pivot mass for the normalization of the stellar mass function.
    alpha_s : float
        Slope parameter for satellite galaxies.
    beta_s : float
        Exponent parameter for satellite galaxies.
    b0 : float
        Parameter for the conditional stellar mass function.
    b1 : float
        Parameter for the conditional stellar mass function.
    b2 : float
        Parameter for the conditional stellar mass function.
    A_cen : float, optional
        Decorated HOD assembly bias parameter for central galaxies.
    A_sat : float, optional
        Decorated HOD assembly bias parameter for satellite galaxies.
    hod_kwargs : dict
        Additional keyword arguments for the HOD class.

    References
    ----------
    .. [1]  Cacciato, M. et al., "Cosmological constraints from a combination of galaxy clustering and lensing - III. Application to SDSS data",
            https://academic.oup.com/mnras/article/430/2/767/2891826.
    """
    def __init__(
            self,
            log10_obs_norm_c=9.95,
            log10_m_ch=11.24,
            g1=3.18,
            g2=0.245,
            sigma_log10_O_c=0.157,
            norm_s=0.562,
            pivot=12.0,
            alpha_s=-1.18,
            beta_s=2,
            b0=-1.17,
            b1=1.53,
            b2=-0.217,
            A_cen=None,
            A_sat=None,
            **hod_kwargs
        ):

        # Call super init MUST BE DONE FIRST.
        super().__init__(**hod_kwargs)
        # Set all given parameters.

        # centrals
        # all observable masses in units of log10(M_sun h^-2)
        self.log10_m_ch = log10_m_ch  # log10_m_ch
        self.g1 = g1  # gamma_1
        self.g2 = g2  # gamma_2
        self.log10_obs_norm_c = log10_obs_norm_c  # O_0, O_norm_c
        self.sigma_log10_O_c = sigma_log10_O_c  # sigma_log10_O_c
        # satellites
        self.norm_s = norm_s  # extra normalisation factor for satellites
        self.pivot = pivot  # pivot mass for the normalisation of the stellar mass function: ϕ∗s
        self.alpha_s = alpha_s  # goes into the conditional stellar mass function COF_sat(M*|M)
        self.beta_s = beta_s  # goes into the conditional stellar mass function COF_sat(M*|M)
        # log10[ϕ∗s(M)] = b0 + b1(log10 m_p)+ b2(log10 m_p)^2, m_p = M/pivot
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        # Decorated HOD assembly bias parameters
        self.A_cen = A_cen
        self.A_sat = A_sat

    @parameter("param")
    def log10_m_ch(self, val):
        """
        Log10 of the characteristic mass.

        :type: float
        """
        return val
        
    @parameter("param")
    def g1(self, val):
        """
        Low mass slope parameter for central galaxies.

        :type: float
        """
        return val
        
    @parameter("param")
    def g2(self, val):
        """
        High mass slope parameter for central galaxies.

        :type: float
        """
        return val
        
    @parameter("param")
    def log10_obs_norm_c(self, val):
        """
        Log10 of the normalization for central galaxies.

        :type: float
        """
        return val
        
    @parameter("param")
    def sigma_log10_O_c(self, val):
        """
        Scatter in log10 of the observable for central galaxies.

        :type: float
        """
        return val
    
    @parameter("param")
    def norm_s(self, val):
        """
        Normalization for satellite galaxies.

        :type: float
        """
        return val
        
    @parameter("param")
    def pivot(self, val):
        """
        Pivot mass for the normalization of the stellar mass function.

        :type: float
        """
        return val
        
    @parameter("param")
    def alpha_s(self, val):
        """
        Slope parameter for satellite galaxies.

        :type: float
        """
        return val
        
    @parameter("param")
    def beta_s(self, val):
        """
        Exponent parameter for satellite galaxies.

        :type: float
        """
        return val
        
    @parameter("param")
    def b0(self, val):
        """
        Parameter for the conditional stellar mass function.

        :type: float
        """
        return val
        
    @parameter("param")
    def b1(self, val):
        """
        Parameter for the conditional stellar mass function.

        :type: float
        """
        return val
        
    @parameter("param")
    def b2(self, val):
        """
        Parameter for the conditional stellar mass function.

        :type: float
        """
        return val
        
    @parameter("param")
    def A_cen(self, val):
        """
        Decorated HOD assembly bias parameter for central galaxies.

        :type: float
        """
        return val
        
    @parameter("param")
    def A_sat(self, val):
        """
        Decorated HOD assembly bias parameter for satellite galaxies.

        :type: float
        """
        return val
        
    @cached_quantity
    def M_char(self):
        """
        Return the 10.**log10_m_ch

        Returns:
        --------
        array_like
            10.**log10_m_ch
        """
        return 10.0**self.log10_m_ch

    @cached_quantity
    def Obs_norm_c(self):
        """
        Return the 10.**log10_obs_norm_c

        Returns:
        --------
        array_like
            10.**log10_obs_norm_c
        """
        return 10.0**self.log10_obs_norm_c

    @cached_quantity
    def COF_cen(self):
        r"""
        COF for Central galaxies (eq 17 of D23: 2210.03110):

        :math:`\Phi_{\rm c}(O|M) = 1/[\sqrt{(2\pi)} \ln(10) \sigma_{\rm c} O] \exp[-\log(O/O_{\star, {\rm c}})^{2}/ (2 \sigma_{\rm c}^{2})]`
        
        Note  :math:`\Phi_{\rm c}(O|M)` is unitless.

        Returns:
        --------
        array_like
            COF of central galaxies
        """
        mean_obs_c = self.cal_mean_obs_c[:, :, :, np.newaxis]  # O∗c
        COF_c = (1.0 / (np.sqrt(2.0 * np.pi) * np.log(10.0) * self.sigma_log10_O_c * self.obs) *
                 np.exp(-(np.log10(self.obs / mean_obs_c))**2 / (2.0 * self.sigma_log10_O_c**2)))
        return COF_c

    @cached_quantity
    def COF_sat(self):
        r"""
        COF for satellite galaxies (eq 18 of D23: 2210.03110):

        :math:`\Phi_{\rm s}(O|M) = \phi_{\star, {\rm s}}/O_{\star, {\rm s}} (O/O_{\star, {\rm s}})^{\alpha_{\rm s}} \exp [-(O/O_{\star, {\rm s}})^{2}]`,
         
        :math:`O_{\star, {\rm s}}` is :math:`O_{\star, {\rm s}}(M) = 0.56 O_{\star, {\rm c}}(M)`
        Note :math:`\Phi_{\rm s}(O|M)` is unitless.

        Returns:
        --------
        array_like
            COF of satellite galaxies
        """
        obs_s_star = self.norm_s * self.cal_mean_obs_c[:, :, :, np.newaxis]
        obs_tilde = self.obs / obs_s_star
        phi_star_val = self.phi_star_s[:, :, :, np.newaxis]
        COF_s = (phi_star_val / obs_s_star) * (obs_tilde**self.alpha_s) * np.exp(-obs_tilde**self.beta_s)
        return COF_s

    @cached_quantity
    def COF(self):
        """
        Total COF, sum of central and satellite COFs.

        Returns:
        --------
        array_like
            COF of all galaxies
        """
        return self.COF_cen + self.COF_sat

    @cached_quantity
    def obs_func_cen(self):
        r"""
        The observable function (SMF/LF) for central galaxies.
        Defined as: :math:`\Phi_{\rm c}(O|M)`, :math:`\Phi_{\rm s}(O|M)` integrated over the halo mass weighted
        with the Halo Mass Function (HMF) to give:  :math:`\Phi_{\rm c}(O)`, :math:`\Phi_{\rm c}(O)`

        :math:`\Phi_{\rm x}(O) = \int \Phi_{\rm x}(O|M) n(M) {\rm d}M`,

        dndlnm is basically n(M) x mass, it is the output of hmf
        The differential mass function in terms of natural log of m,
        len=len(m) [units \(h^3 Mpc^{-3}\)]
        
        dn(m)/ dln m eq1 of 1306.6721
        
        obs_func unit is h^3 Mpc^{-3} dex^-1

        Returns:
        --------
        array_like
            observable function of central galaxies
        """
        integrand = self.COF_cen * self.dndlnm_int[:, :, :, np.newaxis] / self.mass[:, :, :, np.newaxis]
        obs_function = simpson(integrand, self.mass[:, :, :, np.newaxis], axis=-2)
        return obs_function

    @cached_quantity
    def obs_func_sat(self):
        r"""
        The observable function (SMF/LF) for satellite galaxies.
        Defined as: :math:`\Phi_{\rm c}(O|M)`, :math:`\Phi_{\rm s}(O|M)` integrated over the halo mass weighted
        with the Halo Mass Function (HMF) to give:  :math:`\Phi_{\rm c}(O)`, :math:`\Phi_{\rm c}(O)`

        :math:`\Phi_{\rm x}(O) = \int \Phi_{\rm x}(O|M) n(M) {\rm d}M`,

        dndlnm is basically n(M) x mass, it is the output of hmf
        The differential mass function in terms of natural log of m,
        len=len(m) [units \(h^3 Mpc^{-3}\)]
        
        dn(m)/ dln m eq1 of 1306.6721
        
        obs_func unit is h^3 Mpc^{-3} dex^-1

        Returns:
        --------
        array_like
            observable function of satellite galaxies
        """
        integrand = self.COF_sat * self.dndlnm_int[:, :, :, np.newaxis] / self.mass[:, :, :, np.newaxis]
        obs_function = simpson(integrand, self.mass[:, :, :, np.newaxis], axis=-2)
        return obs_function

    @cached_quantity
    def obs_func(self):
        r"""
        The observable function (SMF/LF).
        Defined as: :math:`\Phi_{\rm c}(O|M)`, :math:`\Phi_{\rm s}(O|M)` integrated over the halo mass weighted
        with the Halo Mass Function (HMF) to give:  :math:`\Phi_{\rm c}(O)`, :math:`\Phi_{\rm c}(O)`

        :math:`\Phi_{\rm x}(O) = \int \Phi_{\rm x}(O|M) n(M) {\rm d}M`,

        dndlnm is basically n(M) x mass, it is the output of hmf
        The differential mass function in terms of natural log of m,
        len=len(m) [units \(h^3 Mpc^{-3}\)]
        
        dn(m)/ dln m eq1 of 1306.6721
        
        obs_func unit is h^3 Mpc^{-3} dex^-1

        Returns:
        --------
        array_like
            observable function of all galaxies
        """
        integrand = self.COF * self.dndlnm_int[:, :, :, np.newaxis] / self.mass[:, :, :, np.newaxis]
        obs_function = simpson(integrand, self.mass[:, :, :, np.newaxis], axis=-2)
        return obs_function

    @cached_quantity
    def cal_mean_obs_c(self):
        r"""
        Stellar to halo mass relation (observable to halo mass relation).
        Eqs 19 of D23: 2210.03110

        :math:`O_{\star, {\rm c}}(M)` = O_0 (M/M_{1})^{\gamma_{1}} / [1 + (M/M_{1})]^{(\gamma_{1} - \gamma_{2})}`

        To get the values for the satellite call this * hod_par.norm_s

        :math:`O_{\star, {\rm s}}(M)` = 0.56 O_{\star, {\rm c}}(M)`

        Here  :math:`M_1` is a characteristic mass scale, and :math:`O_0` is the normalization.

        (observable can be galaxy luminosity or stellar mass)
        returns the observable given halo mass.

        Returns:
        --------
        array_like
            observable given halo mass - SMHM relation
        """
        mean_obs_c = (self.Obs_norm_c * (self.mass / self.M_char)**self.g1 /
                      (1.0 + (self.mass / self.M_char))**(self.g1 - self.g2))
        return mean_obs_c

    @cached_quantity
    def phi_star_s(self):
        r"""
        Normalisation of satellite COF function.
        Eqs 21 and 22 of D23: 2210.03110

        :math:`log[\phi_{\star, {\rm s}}(M)] = b_0 + b_1(\log M_{13})` , :math:`M_{13} = M/({\rm pivot})`

        Returns:
        --------
        array_like
            normalisation of satellite COF function
        """
        logM_pivot = np.log10(self.mass) - self.pivot
        log_phi_s = self.b0 + self.b1 * logM_pivot + self.b2 * (logM_pivot**2.0)
        return 10.0**log_phi_s

    @property
    def _compute_hod_cen(self):
        r"""
        eq 23 of D23: 2210.03110

        :math:`\langle N_{\rm x}|M \rangle = \int_{O_{\rm low}}^{O_{\rm high}} \Phi_{\rm x}(O|M) {\rm d}O`
        """
        N_cen = simpson(self.COF_cen, self.obs)
        if self.A_cen is not None:
            delta_pop_c = self.A_cen * np.fmin(N_cen, 1.0 - N_cen)
            N_cen = N_cen + delta_pop_c
        return N_cen

    @property
    def _compute_hod_sat(self):
        """
        eq 23 of D23: 2210.03110

        :math:`\langle N_{\rm x}|M \rangle = \int_{O_{\rm low}}^{O_{\rm high}} \Phi_{\rm x}(O|M) {\rm d}O`
        """
        N_sat = simpson(self.COF_sat, self.obs)
        if self.A_sat is not None:
            delta_pop_s = self.A_sat * N_sat
            N_sat = N_sat + delta_pop_s
        return N_sat

    @cached_quantity
    def _compute_stellar_fraction_cen(self):
        """
        The mean value of the observable for the given galaxy population for a given halo mass.
        O is weighted by the number of galaxies with the property O for each halo mass: :math:`\Phi_{\rm x}(O|M)`

        :math:`f_{\star} = \int_{O_{\rm low}}^{O_{\rm high}} \Phi_{\rm x}(O|M) O {\rm d}O`
        """
        return simpson(self.COF_cen * self.obs, self.obs) / self.mass

    @cached_quantity
    def _compute_stellar_fraction_sat(self):
        """
        The mean value of the observable for the given galaxy population for a given halo mass.
        O is weighted by the number of galaxies with the property O for each halo mass: :math:`\Phi_{\rm x}(O|M)`

        :math:`f_{\star} = \int_{O_{\rm low}}^{O_{\rm high}} \Phi_{\rm x}(O|M) O {\rm d}O`
        """
        return simpson(self.COF_sat * self.obs, self.obs) / self.mass

    @cached_quantity
    def _compute_stellar_fraction(self):
        """
        The mean value of the observable for the given galaxy population for a given halo mass.
        O is weighted by the number of galaxies with the property O for each halo mass: :math:`\Phi_{\rm x}(O|M)`

        :math:`f_{\star} = \int_{O_{\rm low}}^{O_{\rm high}} \Phi_{\rm x}(O|M) O {\rm d}O`
        """
        return simpson(self.COF * self.obs, self.obs) / self.mass

class Simple(HOD):
    """
    Simple HOD model
    
    Parameters:
    -----------
    log10_Mmin : array_like
        Log10 of the minimum mass for central galaxies.
    log10_Msat : array_like
        Log10 of the minimum mass for satellite galaxies.
    alpha : array_like
        Slope parameter for satellite galaxies.
    A_cen : float, optional
        Decorated HOD assembly bias parameter for central galaxies.
    A_sat : float, optional
        Decorated HOD assembly bias parameter for satellite galaxies.
    hod_kwargs : dict
        Additional keyword arguments for the HOD class.
    """
    def __init__(
            self,
            log10_Mmin=12.0,
            log10_Msat=13.0,
            alpha=1.0,
            A_cen=None,
            A_sat=None,
            **hod_kwargs
        ):
        # Call super init MUST BE DONE FIRST.
        super().__init__(**hod_kwargs)
        self.log10_Mmin = log10_Mmin
        self.log10_Msat = log10_Msat
        self.alpha = alpha
        # Decorated HOD assembly bias parameters
        self.A_cen = A_cen
        self.A_sat = A_sat
        
    @parameter("param")
    def log10_Mmin(self, val):
        """
        Log10 of the minimum mass for central galaxies.

        :type: array_like
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)
        
    @parameter("param")
    def log10_Msat(self, val):
        """
        Log10 of the minimum mass for satellite galaxies.

        :type: array_like
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)
        
    @parameter("param")
    def alpha(self, val):
        """
        Slope parameter for satellite galaxies.

        :type: array_like
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)[:, np.newaxis, np.newaxis]
        
    @parameter("param")
    def A_cen(self, val):
        """
        Decorated HOD assembly bias parameter for central galaxies.

        :type: float
        """
        return val
        
    @parameter("param")
    def A_sat(self, val):
        """
        Decorated HOD assembly bias parameter for satellite galaxies.

        :type: float
        """
        return val
        
    @cached_quantity
    def Mmin(self):
        """
        Return the Mmin parameter in the correct shape and units.

        Returns:
        --------
        array_like
            Mmin
        """
        return 10.0**self.log10_Mmin[:, np.newaxis, np.newaxis]
        
    @cached_quantity
    def Msat(self):
        """
        Return the Msat parameter in the correct shape and units.

        Returns:
        --------
        array_like
            Msat
        """
        return 10.0**self.log10_Msat[:, np.newaxis, np.newaxis]

    @cached_quantity
    def _compute_hod_cen(self):
        """
        Compute the HOD for central galaxies.
        """
        N_cen = np.heaviside(self.mass - self.Mmin, 1.0)
        if self.A_cen is not None:
            delta_pop_c = self.A_cen * np.fmin(N_cen, 1.0 - N_cen)
            N_cen = N_cen + delta_pop_c
        return np.tile(N_cen, (self.nz, 1))

    @cached_quantity
    def _compute_hod_sat(self):
        """
        Compute the HOD for satellite galaxies.
        """
        N_sat = self.compute_hod_cen * (self.mass / self.Msat)**self.alpha
        if self.A_sat is not None:
            delta_pop_s = self.A_sat * N_sat
            N_sat = N_sat + delta_pop_s
        return np.tile(N_sat, (self.nz, 1))
        

class Zehavi(HOD):
    """
    HOD model from Zehavi et al. (2004) [1]_.

    Same as Zheng model in the limit that sigma=0 and M0=0
    Mean number of central galaxies is only ever 0 or 1 in this HOD
    
    Parameters:
    -----------
    log10_Mmin : array_like
        Log10 of the minimum mass for central galaxies.
    log10_Msat : array_like
        Log10 of the minimum mass for satellite galaxies.
    alpha : array_like
        Slope parameter for satellite galaxies.
    A_cen : float, optional
        Decorated HOD assembly bias parameter for central galaxies.
    A_sat : float, optional
        Decorated HOD assembly bias parameter for satellite galaxies.
    hod_kwargs : dict
        Additional keyword arguments for the HOD class.

    References
    ----------
    .. [1]  Zheng, Z. et al., "Galaxy Evolution from Halo Occupation Distribution Modeling of DEEP2 and SDSS Galaxy Clustering",
            https://iopscience.iop.org/article/10.1086/521074.
    """
    def __init__(
            self,
            log10_Mmin=12.0,
            log10_Msat=13.0,
            alpha=1.0,
            A_cen=None,
            A_sat=None,
            **hod_kwargs
        ):
        # Call super init MUST BE DONE FIRST.
        super().__init__(**hod_kwargs)
        self.log10_Mmin = log10_Mmin
        self.log10_Msat = log10_Msat
        self.alpha = alpha
        # Decorated HOD assembly bias parameters
        self.A_cen = A_cen
        self.A_sat = A_sat
        
    @parameter("param")
    def log10_Mmin(self, val):
        """
        Log10 of the minimum mass for central galaxies.

        :type: array_like
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)
        
    @parameter("param")
    def log10_Msat(self, val):
        """
        Log10 of the minimum mass for satellite galaxies.

        :type: array_like
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)
        
    @parameter("param")
    def alpha(self, val):
        """
        Slope parameter for satellite galaxies.

        :type: array_like
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)[:, np.newaxis, np.newaxis]
        
    @parameter("param")
    def A_cen(self, val):
        """
        Decorated HOD assembly bias parameter for central galaxies.

        :type: float
        """
        return val
        
    @parameter("param")
    def A_sat(self, val):
        """
        Decorated HOD assembly bias parameter for satellite galaxies.

        :type: float
        """
        return val
        
    @cached_quantity
    def Mmin(self):
        """
        Return the Mmin parameter in the correct shape and units.

        Returns:
        --------
        array_like
            Mmin
        """
        return 10.0**self.log10_Mmin[:, np.newaxis, np.newaxis]
        
    @cached_quantity
    def Msat(self):
        """
        Return the Msat parameter in the correct shape and units.

        Returns:
        --------
        array_like
            Msat
        """
        return 10.0**self.log10_Msat[:, np.newaxis, np.newaxis]

    @cached_quantity
    def _compute_hod_cen(self):
        """
        Compute the HOD for central galaxies.
        """
        N_cen = np.heaviside(self.mass - self.Mmin, 1.0)
        if self.A_cen is not None:
            delta_pop_c = self.A_cen * np.fmin(N_cen, 1.0 - N_cen)
            N_cen = N_cen + delta_pop_c
        return np.tile(N_cen, (self.nz, 1))

    @cached_quantity
    def _compute_hod_sat(self):
        """
        Compute the HOD for satellite galaxies.
        """
        N_sat = (self.mass / self.Msat)**self.alpha
        if self.A_sat is not None:
            delta_pop_s = self.A_sat * N_sat
            N_sat = N_sat + delta_pop_s
        return np.tile(N_sat, (self.nz, 1))


class Zheng(HOD):
    """
    HOD model from Zheng et al. (2005) [1]_.
    
    Parameters:
    -----------
    log10_Mmin : array_like
        Log10 of the minimum mass for central galaxies.
    log10_M0 : array_like
        Log10 of the cutoff mass for satellite galaxies.
    log10_M1 : array_like
        Log10 of the normalization mass for satellite galaxies.
    sigma : array_like
        Scatter in the central galaxy occupation.
    alpha : array_like
        Slope parameter for satellite galaxies.
    A_cen : float, optional
        Decorated HOD assembly bias parameter for central galaxies.
    A_sat : float, optional
        Decorated HOD assembly bias parameter for satellite galaxies.
    hod_kwargs : dict
        Additional keyword arguments for the HOD class.

    References
    ----------
    .. [1]  Zheng, Z. et al., "Theoretical Models of the Halo Occupation Distribution: Separating Central and Satellite Galaxies",
            https://iopscience.iop.org/article/10.1086/466510.
    """
    def __init__(
            self,
            log10_Mmin=12.0,
            log10_M0=12.0,
            log10_M1=13.0,
            sigma=0.15,
            alpha=1.0,
            A_cen=None,
            A_sat=None,
            **hod_kwargs
        ):
        # Call super init MUST BE DONE FIRST.
        super().__init__(**hod_kwargs)
        self.log10_Mmin = log10_Mmin
        self.log10_M0 = log10_M0
        self.log10_M1 = log10_M1
        self.sigma = sigma
        self.alpha = alpha
        # Decorated HOD assembly bias parameters
        self.A_cen = A_cen
        self.A_sat = A_sat

    @parameter("param")
    def log10_Mmin(self, val):
        """
        Log10 of the minimum mass for central galaxies.

        :type: array_like
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)
        
    @parameter("param")
    def log10_M0(self, val):
        """
        Log10 of the cutoff mass for satellite galaxies.

        :type: array_like
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)
    
    @parameter("param")
    def log10_M1(self, val):
        """
        Log10 of the normalization mass for satellite galaxies.

        :type: array_like
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)
        
    @parameter("param")
    def alpha(self, val):
        """
        Slope parameter for satellite galaxies.

        :type: array_like
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)[:, np.newaxis, np.newaxis]
        
    @parameter("param")
    def sigma(self, val):
        """
        Scatter in the central galaxy occupation.

        :type: array_like
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)[:, np.newaxis, np.newaxis]
        
    @parameter("param")
    def A_cen(self, val):
        """
        Decorated HOD assembly bias parameter for central galaxies.

        :type: float
        """
        return val
        
    @parameter("param")
    def A_sat(self, val):
        """
        Decorated HOD assembly bias parameter for satellite galaxies.

        :type: float
        """
        return val
        
    @cached_quantity
    def Mmin(self):
        """
        Return the Mmin parameter in the correct shape and units.

        Returns:
        --------
        array_like
            Mmin
        """
        return 10.0**self.log10_Mmin[:, np.newaxis, np.newaxis]
        
    @cached_quantity
    def M0(self):
        """
        Return the M0 parameter in the correct shape and units.

        Returns:
        --------
        array_like
            M0
        """
        return 10.0**self.log10_M0[:, np.newaxis, np.newaxis]
        
    @cached_quantity
    def M1(self):
        """
        Return the M1 parameter in the correct shape and units.ž

        Returns:
        --------
        array_like
            M1
        """
        return 10.0**self.log10_M1[:, np.newaxis, np.newaxis]

    @cached_quantity
    def _compute_hod_cen(self):
        """
        Compute the HOD for central galaxies.
        """
        N_cen = np.where(self.sigma == 0.0, np.heaviside(self.mass - self.Mmin, 1.0), 0.5 * (1.0 + erf(np.log10(self.mass / self.Mmin) / self.sigma)))
        if self.A_cen is not None:
            delta_pop_c = self.A_cen * np.fmin(N_cen, 1.0 - N_cen)
            N_cen = N_cen + delta_pop_c
        return np.tile(N_cen, (self.nz, 1))

    @cached_quantity
    def _compute_hod_sat(self):
        """
        Compute the HOD for satellite galaxies.
        """
        N_sat = (np.heaviside(self.mass - self.M0, 1.0) * (self.mass - self.M0) / self.M1)**self.alpha
        if self.A_sat is not None:
            delta_pop_s = self.A_sat * N_sat
            N_sat = N_sat + delta_pop_s
        return np.tile(N_sat, (self.nz, 1))


class Zhai(HOD):
    """
    HOD model from Zhai et al. (2017) [1]_.
    
    Parameters:
    -----------
    log10_Mmin : array_like
        Log10 of the minimum mass for central galaxies.
    log10_Msat : array_like
        Log10 of the minimum mass for satellite galaxies.
    log10_Mcut : array_like
        Log10 of the cutoff mass for satellite galaxies.
    sigma : array_like
        Scatter in the central galaxy occupation.
    alpha : array_like
        Slope parameter for satellite galaxies.
    A_cen : float, optional
        Decorated HOD assembly bias parameter for central galaxies.
    A_sat : float, optional
        Decorated HOD assembly bias parameter for satellite galaxies.
    hod_kwargs : dict
        Additional keyword arguments for the HOD model.

    References
    ----------
    .. [1]  Zhai, Z. et al., "The clustering of luminous red galaxies at z ~ 0.7 from eBOSS and BOSS data",
            https://iopscience.iop.org/article/10.3847/1538-4357aa8eee.
    """
    def __init__(
            self,
            log10_Mmin=13.68,
            log10_Msat=14.87,
            log10_Mcut=12.32,
            sigma=0.82,
            alpha=0.41,
            A_cen=None,
            A_sat=None,
            **hod_kwargs
        ):
        # Call super init MUST BE DONE FIRST.
        super().__init__(**hod_kwargs)
        self.log10_Mmin = log10_Mmin
        self.log10_Msat = log10_Msat
        self.log10_Mcut = log10_Mcut
        self.sigma = sigma
        self.alpha = alpha
        # Decorated HOD assembly bias parameters
        self.A_cen = A_cen
        self.A_sat = A_sat
        
    @parameter("param")
    def log10_Mmin(self, val):
        """
        Log10 of the minimum mass for central galaxies.

        :type: array_like
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)
        
    @parameter("param")
    def log10_Msat(self, val):
        """
        Log10 of the minimum mass for satellite galaxies.

        :type: array_like
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)
    
    @parameter("param")
    def log10_Mcut(self, val):
        """
        Log10 of the cutoff mass for satellite galaxies.

        :type: array_like
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)
        
    @parameter("param")
    def alpha(self, val):
        """
        Slope parameter for satellite galaxies.

        :type: array_like
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)[:, np.newaxis, np.newaxis]
        
    @parameter("param")
    def sigma(self, val):
        """
        Scatter in the central galaxy occupation.

        :type: array_like
        """
        if not hasattr(val, "__len__"):
            val = [val]
        return np.array(val)[:, np.newaxis, np.newaxis]
        
    @parameter("param")
    def A_cen(self, val):
        """
        Decorated HOD assembly bias parameter for central galaxies.

        :type: float
        """
        return val
        
    @parameter("param")
    def A_sat(self, val):
        """
        Decorated HOD assembly bias parameter for satellite galaxies.
        
        :type: float
        """
        return val
        
    @cached_quantity
    def Mmin(self):
        """
        Return the Mmin parameter in the correct shape and units.

        Returns:
        --------
        array_like
            Mmin
        """
        return 10.0**self.log10_Mmin[:, np.newaxis, np.newaxis]
        
    @cached_quantity
    def Msat(self):
        """
        Return the Msat parameter in the correct shape and units.

        Returns:
        --------
        array_like
            Msat
        """
        return 10.0**self.log10_Msat[:, np.newaxis, np.newaxis]
        
    @cached_quantity
    def Mcut(self):
        """
        Return the Mcut parameter in the correct shape and units.

        Returns:
        --------
        array_like
            Mcut
        """
        return 10.0**self.log10_Mcut[:, np.newaxis, np.newaxis]

    @cached_quantity
    def _compute_hod_cen(self):
        """
        Compute the HOD for central galaxies.
        """
        N_cen = np.where(self.sigma == 0.0, np.heaviside(self.mass - self.Mmin, 1.0), 0.5 * (1.0 + erf(np.log10(self.mass / self.Mmin) / self.sigma)))
        if self.A_cen is not None:
            delta_pop_c = self.A_cen * np.fmin(N_cen, 1.0 - N_cen)
            N_cen = N_cen + delta_pop_c
        return np.tile(N_cen, (self.nz, 1))

    @cached_quantity
    def _compute_hod_sat(self):
        """
        Compute the HOD for satellite galaxies.
        """
        # Paper has a Nc(M) multiplication, but I think the central condition covers this
        N_sat = ((self.mass / self.Msat)**self.alpha) * np.exp(-self.Mcut / self.mass)
        if self.A_sat is not None:
            delta_pop_s = self.A_sat * N_sat
            N_sat = N_sat + delta_pop_s
        return np.tile(N_sat, (self.nz, 1))
