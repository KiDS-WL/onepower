"""
A module for computing intrinsic alignment properties.
This module provides classes and functions to calculate various properties
related to the intrinsic alignment of central and satellite galaxies within dark matter halos.
"""

import numpy as np
from astropy.io import fits
from functools import cached_property
from hankel import HankelTransform
from scipy.fft import fht, fhtoffset
from scipy.integrate import simpson
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.special import binom, gamma

from hmf._internals._cache import cached_quantity, parameter
from hmf._internals._framework import Framework


class AlignmentAmplitudes(Framework):
    """
    Class to IA amplitudes given options and dependencies on either luminosity, stellar mass, or halo mass.

    Parameters:
    -----------
    z_vec : array-like
        Redshift vector.
    central_ia_depends_on : str
        Dependency for central intrinsic alignment. Options: 'constant', 'luminosity', 'halo_mass'.
    satellite_ia_depends_on : str
        Dependency for satellite intrinsic alignment. Options: 'constant', 'luminosity', 'halo_mass'.
    gamma_2h_amplitude : float
        Amplitude for 2-halo term.
    beta_cen : float
        Beta parameter for central galaxies.
    beta_two : float, optional
        Secondary beta parameter.
    gamma_1h_amplitude : float
        Amplitude for 1-halo term.
    gamma_1h_slope : float
        Slope for 1-halo term.
    beta_sat : float
        Beta parameter for satellite galaxies.
    mpivot_cen : float, optional
        Pivot mass for central galaxies.
    mpivot_sat : float, optional
        Pivot mass for satellite galaxies.
    lpivot_cen : float, optional
        Pivot luminosity for central galaxies.
    lpivot_sat : float, optional
        Pivot luminosity for satellite galaxies.
    z_loglum_file_centrals : str, optional
        File path for central galaxy luminosity data.
    z_loglum_file_satellites : str, optional
        File path for satellite galaxy luminosity data.
    """

    def __init__(
        self,
        z_vec=None,
        central_ia_depends_on='halo_mass',
        satellite_ia_depends_on='halo_mass',
        gamma_2h_amplitude=5.33,
        beta_cen=0.44,
        beta_two=None,
        gamma_1h_amplitude=0.0015,
        gamma_1h_slope=-2.0,
        beta_sat=0.44,
        mpivot_cen=13.5,
        mpivot_sat=13.5,
        lpivot_cen=None,
        lpivot_sat=None,
        z_loglum_file_centrals=None,
        z_loglum_file_satellites=None,
    ):
        self.z_vec = z_vec
        self.central_ia_depends_on = central_ia_depends_on
        self.satellite_ia_depends_on = satellite_ia_depends_on

        self.gamma_2h_amplitude = gamma_2h_amplitude
        self.beta_cen = beta_cen
        self.beta_two = beta_two
        self.gamma_1h_slope = gamma_1h_slope
        self.gamma_1h_amplitude = gamma_1h_amplitude
        self.beta_sat = beta_sat

        self.mpivot_cen = mpivot_cen
        self.mpivot_sat = mpivot_sat
        self.lpivot_cen = lpivot_cen
        self.lpivot_sat = lpivot_sat

        self.z_loglum_file_centrals = z_loglum_file_centrals
        self.z_loglum_file_satellites = z_loglum_file_satellites

    @parameter('param')
    def z_vec(self, val):
        """
        Redshift vector.

        :type: array-like
        """
        return val

    @parameter('param')
    def central_ia_depends_on(self, val):
        """
        Validate the central intrinsic alignment dependencies.

        Raises:
        -------
        ValueError : If an invalid option is provided for central_ia_depends_on or satellite_ia_depends_on.
        """
        valid_options = ['constant', 'luminosity', 'halo_mass']
        if val not in valid_options:
            raise ValueError(
                f'Choose one of the following options for central_IA_depends_on: {valid_options}'
            )
        return val

    @parameter('param')
    def satellite_ia_depends_on(self, val):
        """
        Validate the satellite intrinsic alignment dependencies.

        Raises:
        -------
        ValueError : If an invalid option is provided for central_ia_depends_on or satellite_ia_depends_on.
        """
        valid_options = ['constant', 'luminosity', 'halo_mass']
        if val not in valid_options:
            raise ValueError(
                f'Choose one of the following options for satellite_IA_depends_on: {valid_options}'
            )
        return val

    @parameter('param')
    def gamma_2h_amplitude(self, val):
        """
        Amplitude for 2-halo term.

        :type: float
        """
        return val

    @parameter('param')
    def gamma_1h_slope(self, val):
        """
        Slope for 1-halo term.

        :type: float
        """
        return val

    @parameter('param')
    def gamma_1h_amplitude(self, val):
        """
        Amplitude for 1-halo term.

        :type: float
        """
        return val

    @parameter('param')
    def beta_cen(self, val):
        """
        Beta parameter for central galaxies.

        :type: float
        """
        return val

    @parameter('param')
    def beta_sat(self, val):
        """
        beta_sat : float
            Beta parameter for satellite galaxies.
        """
        return val

    @parameter('param')
    def mpivot_cen(self, val):
        """
        Pivot mass for central galaxies.

        :type: float
        """
        return 10.0**val if val is not None else None

    @parameter('param')
    def mpivot_sat(self, val):
        """
        Pivot mass for satellite galaxies.

        :type: float
        """
        return 10.0**val if val is not None else None

    @parameter('param')
    def lpivot_cen(self, val):
        """
        Pivot luminosity for central galaxies.

        :type: float
        """
        return 10.0**val if val is not None else None

    @parameter('param')
    def lpivot_sat(self, val):
        """
        Pivot luminosity for satellite galaxies.

        :type: float
        """
        return 10.0**val if val is not None else None

    @parameter('param')
    def z_loglum_file_centrals(self, val):
        """
        File path for central galaxy luminosity data.

        :type: str
        """
        return val

    @parameter('param')
    def z_loglum_file_satellites(self, val):
        """
        File path for satellite galaxy luminosity data.

        :type: str
        """
        return val

    @cached_quantity
    def lum_centrals(self):
        """
        Returns the luminosity array for centrals.
        """
        return self._initialize_luminosity_array('centrals')[0]

    @cached_quantity
    def lum_pdf_z_centrals(self):
        """
        Returns the luminosity array for satellites.
        """
        return self._initialize_luminosity_array('centrals')[1]

    @cached_quantity
    def lum_satellites(self):
        """
        Returns the luminosity PDF array for centrals.
        """
        return self._initialize_luminosity_array('satellites')[0]

    @cached_quantity
    def lum_pdf_z_satellites(self):
        """
        Returns the luminosity PDF array for satellites.
        """
        return self._initialize_luminosity_array('satellites')[1]

    def _initialize_luminosity_array(self, galaxy_type):
        """
        Initialize and return the luminosity array based on galaxy type.

        Parameters:
        -----------
        galaxy_type : str
            Type of galaxy, either 'centrals' or 'satellites'.

        Returns:
        --------
        numpy.ndarray : lum
            Luminosity array.
        """
        depends_on = (
            self.central_ia_depends_on
            if galaxy_type == 'centrals'
            else self.satellite_ia_depends_on
        )
        z_loglum_file = (
            self.z_loglum_file_centrals
            if galaxy_type == 'centrals'
            else self.z_loglum_file_satellites
        )
        if z_loglum_file is None:
            raise ValueError(
                f'You have not provided a luminosity file for {galaxy_type}. Please include z_loglum_file_{galaxy_type}.'
            )

        nlbins = 10000
        lum, lum_pdf_z = self.compute_luminosity_pdf(z_loglum_file, nlbins)

        return lum, lum_pdf_z

    def mean_l_l0_to_beta(self, xlum, pdf, l0, beta):
        """
        Compute the mean luminosity scaling.

        Parameters:
        -----------
        xlum : array-like
            Luminosity values.
        pdf : array-like
            Probability density function values.
        l0 : float
            Pivot luminosity.
        beta : float
            Beta parameter.

        Returns:
        --------
        float : Mean luminosity scaling.
        """
        return simpson(pdf * (xlum / l0) ** beta, x=xlum)

    def broken_powerlaw(self, xlum, pdf, gamma_2h_lum, l0, beta, beta_low):
        """
        Compute the broken power law.

        Parameters:
        -----------
        xlum : array-like
            Luminosity values.
        pdf : array-like
            Probability density function values.
        gamma_2h_lum : float
            Amplitude for 2-halo term.
        l0 : float
            Pivot luminosity.
        beta : float
            Beta parameter.
        beta_low : float
            Secondary beta parameter.

        Returns:
        --------
        float : Integral of the broken power law.
        """
        alignment_ampl = np.where(
            xlum > l0,
            gamma_2h_lum * (xlum / l0) ** beta,
            gamma_2h_lum * (xlum / l0) ** beta_low,
        )
        return simpson(pdf * alignment_ampl, x=xlum)

    def compute_luminosity_pdf(self, z_loglum_file, nlbins):
        """
        Compute the luminosity PDF.

        Parameters:
        -----------
        z_loglum_file : str
            File path for galaxy luminosity data.
        nlbins : int
            Number of bins for the histogram.

        Returns:
        --------
        tuple : (bincen, pdf)
            Bin centers and probability density function values.
        """
        galfile = fits.open(z_loglum_file)[1].data
        z_gal = np.array(galfile['z'])
        loglum_gal = np.array(galfile['loglum'])

        z_bins = self.z_vec
        dz = 0.5 * (z_bins[1] - z_bins[0])
        z_edges = np.append(z_bins - dz, z_bins[-1] + dz)

        bincen = np.zeros([self.z_vec.size, nlbins])
        pdf = np.zeros([self.z_vec.size, nlbins])

        for i in range(self.z_vec.size):
            mask_z = (z_gal >= z_edges[i]) & (z_gal < z_edges[i + 1])
            loglum_bin = loglum_gal[mask_z]
            if loglum_bin.size:
                lum = 10.0**loglum_bin
                pdf_tmp, _lum_bins = np.histogram(lum, bins=nlbins, density=True)
                _dbin = (_lum_bins[-1] - _lum_bins[0]) / (1.0 * nlbins)
                bincen[i] = _lum_bins[:-1] + 0.5 * _dbin
                pdf[i] = pdf_tmp

        return bincen, pdf

    @cached_quantity
    def alignment_gi(self):
        """
        Compute the alignment_gi based on the central galaxies' properties.
        """
        if self.central_ia_depends_on == 'constant':
            return self.gamma_2h_amplitude * np.ones_like(self.z_vec)
        elif self.central_ia_depends_on == 'luminosity':
            if self.lpivot_cen is None:
                raise ValueError(
                    'You have chosen central luminosity scaling without providing a pivot luminosity parameter. Include lpivot_cen.'
                )
            if self.beta_two is not None:
                mean_lscaling = np.array(
                    [
                        self.broken_powerlaw(
                            self.lum_centrals[i],
                            self.lum_pdf_z_centrals[i],
                            self.gamma_2h_amplitude,
                            self.lpivot_cen,
                            self.beta_cen,
                            self.beta_two,
                        )
                        for i in range(self.z_vec.size)
                    ]
                )
            else:
                mean_lscaling = self.gamma_2h_amplitude * self.mean_l_l0_to_beta(
                    self.lum_centrals,
                    self.lum_pdf_z_centrals,
                    self.lpivot_cen,
                    self.beta_cen,
                )
            return mean_lscaling
        elif self.central_ia_depends_on == 'halo_mass':
            if self.mpivot_cen is None:
                raise ValueError(
                    'You have chosen central halo-mass scaling without providing a pivot mass parameter. Include mpivot_cen.'
                )
            if self.beta_two is not None:
                raise ValueError(
                    'A double power law model for the halo mass dependence of centrals has not been implemented.'
                )
            return self.gamma_2h_amplitude * np.ones_like(self.z_vec)

    @cached_quantity
    def gamma_1h_amp(self):
        """
        Compute the gamma_1h_amplitude based on the satellite galaxies' properties.
        """
        if self.satellite_ia_depends_on == 'constant':
            return self.gamma_1h_amplitude * np.ones_like(self.z_vec)
        elif self.satellite_ia_depends_on == 'luminosity':
            if self.lpivot_sat is None:
                raise ValueError(
                    'You have chosen satellite luminosity scaling without providing a pivot luminosity parameter. Include lpivot_sat.'
                )
            mean_lscaling = self.mean_l_l0_to_beta(
                self.lum_satellites,
                self.lum_pdf_z_satellites,
                self.lpivot_sat,
                self.beta_sat,
            )
            return self.gamma_1h_amplitude * mean_lscaling
        elif self.satellite_ia_depends_on == 'halo_mass':
            if self.mpivot_sat is None:
                raise ValueError(
                    'You have chosen satellite halo-mass scaling without providing a pivot mass parameter. Include mpivot_sat.'
                )
            return self.gamma_1h_amplitude * np.ones_like(self.z_vec)


class SatelliteAlignment(AlignmentAmplitudes):
    """
    A class to compute the alignment properties of satellite galaxies within dark matter halos.
    This includes calculating the Hankel transform, radial profiles, and other related quantities.

    Parameters:
    -----------
    mass_in : array_like
        Array of halo masses.
    c_in : array_like
        Concentration parameter.
    r_s_in : array_like
        Scale radius.
    rvir_in : array_like
        Virial radius.
    n_hankel : int, optional
        Number of steps in the Hankel transform integration.
    nmass : int, optional
        Number of mass bins.
    nk : int, optional
        Number of k bins.
    ell_max : int, optional
        Maximum multipole moment.
    truncate : bool, optional
        Whether to truncate the NFW profile at the virial radius.
    method : str, optional
        Which method to perform Fourier/Hankel transform to use
    amplitude_kwargs : dict
        Extra parameters passed to the AlignmentAmplitudes class.
    """

    def __init__(
        self,
        mass_in=None,
        c_in=None,
        r_s_in=None,
        rvir_in=None,
        n_hankel=350,
        nmass=5,
        nk=10,
        ell_max=6,
        truncate=False,
        method='fftlog',
        **amplitude_kwargs,
    ):
        # Call super init MUST BE DONE FIRST.
        super().__init__(**amplitude_kwargs)

        self.method = method
        self.nmass = nmass
        self.ell_max = ell_max
        self.truncate = truncate
        self.nk = nk
        self.n_hankel = n_hankel
        self.mass_in = mass_in
        self.c_in = c_in
        self.r_s_in = r_s_in
        self.rvir_in = rvir_in

        # These are for now hardcoded choices
        self.theta_k = np.pi / 2.0
        self.phi_k = 0.0

    @parameter('param')
    def mass_in(self, val):
        """
        Array of halo masses.

        :type: array_like
        """
        return val

    @parameter('param')
    def c_in(self, val):
        """
        Concentration parameter.

        :type: array_like
        """
        return val

    @parameter('param')
    def r_s_in(self, val):
        """
        Scale radius.

        :type: array_like
        """
        return val

    @parameter('param')
    def rvir_in(self, val):
        """
        Virial radius.

        :type: array_like
        """
        return val

    @parameter('param')
    def n_hankel(self, val):
        """
        Number of steps in the Hankel transform integration.

        :type: int
        """
        return val

    @parameter('param')
    def nmass(self, val):
        """
        Number of mass bins.

        :type: int
        """
        return int(val)

    @parameter('param')
    def nk(self, val):
        """
        Number of k bins.

        :type: int
        """
        return val

    @parameter('param')
    def ell_max(self, val):
        """
        Maximum multipole moment.

        :type: int
        """
        if val > 11:
            raise ValueError(
                'Please reduce ell_max < 11 or update ia_radial_interface.py'
            )
        return val

    @parameter('param')
    def truncate(self, val):
        """
        Whether to truncate the NFW profile at the virial radius.

        :type: bool
        """
        return val

    @parameter('param')
    def method(self, val):
        """
        Which method to perform Fourier/Hankel transform to use

        :type: str
        """
        valid_methods = ['hankel', 'fftlog']
        if val not in valid_methods:
            raise ValueError(
                f'The valid methods to evaluate the fourier tranform of IA shear field are: {valid_methods}. Requested method: {val}!'
            )
        return val

    @cached_quantity
    def ell_values(self):
        """
        Sets the ell values array.

        Returns:
        --------
        array_like
            ell_values
        """
        # CCL and Fortuna use ell_max=6. SB10 uses ell_max = 2.
        # Higher/lower increases/decreases accuracy but slows/speeds the code
        return np.arange(0, self.ell_max + 1, 2)

    @cached_quantity
    def k_vec(self):
        """
        Sets the k vector array.

        Returns:
        --------
        array_like
            k_vec
        """
        nk = 100 if self.method == 'fftlog' else self.nk
        return np.logspace(np.log10(1e-3), np.log10(1e3), nk)

    def _downsample(self, quantity):
        """
        Downsample the halo parameters to reduce computational complexity.

        Parameter:
        -----------
        quantity : array_like
            An array with a quantity that needs to be downsamples (c, rs, rvir or mass)

        Returns:
        --------
        array
            Downsampled arrays of either halo masses, concentration parameters, scale radii, and virial radii.
        """
        if self.method != 'hankel':
            return quantity

        if self.mass_in.size == self.nmass:
            return quantity

        if self.mass_in.size < self.nmass:
            raise ValueError(
                'The halo mass resolution is too low for the radial IA calculation. '
                'Please increase nmass when you run halo_model_ingredients.py'
            )

        downsample_factor = self.mass_in.size // self.nmass

        if isinstance(quantity, np.ndarray) and quantity.ndim == 1:
            downsampled_quantity = quantity[::downsample_factor]
            if downsampled_quantity[-1] != quantity[-1]:
                downsampled_quantity = np.append(downsampled_quantity, quantity[-1])
        else:
            downsampled_quantity = quantity[:, ::downsample_factor]
            if np.all(downsampled_quantity[:, -1] != quantity[:, -1]):
                downsampled_quantity = np.concatenate(
                    (downsampled_quantity, np.atleast_2d(quantity[:, -1]).T), axis=1
                )
        return downsampled_quantity

    @cached_quantity
    def mass(self):
        """
        Downsampled mass_in if method == hankel, mass_in otherwise.

        Returns:
        --------
        array_like
            mass
        """
        return self._downsample(self.mass_in)

    @cached_quantity
    def c(self):
        """
        Downsampled c_in if method == hankel, c_in otherwise.

        Returns:
        --------
        array_like
            c
        """
        return self._downsample(self.c_in)

    @cached_quantity
    def r_s(self):
        """
        Downsampled r_s_in if method == hankel, r_s_in otherwise.

        Returns:
        --------
        array_like
            r_s
        """
        return self._downsample(self.r_s_in)

    @cached_quantity
    def rvir(self):
        """
        Downsampled rvir_in if method == hankel, rvir_in otherwise.

        Returns:
        --------
        array_like
            rvir
        """
        return self._downsample(self.rvir_in)

    @cached_quantity
    def hankel(self):
        """
        Initialize Hankel transform.

        We've used hankel.get_h to set h, N is then h=pi/N, finding best_h = 0.05, best_N=62
        If you want perfect agreement with CCL use: N=50000, h=0.00006 (VERY SLOW!!)

        Ideally we need to find a way to just evaluate this part outside of the class
        so it can work nicely with CosmoSIS setup function

        Returns:
        --------
        list
            A list of HankelTransform objects for each multipole moment.
        """
        if self.method == 'hankel':
            return [
                HankelTransform(ell + 0.5, self.n_hankel, np.pi / self.n_hankel)
                for ell in self.ell_values
            ]
        else:
            return None

    def I_x(self, a, b):
        """
        Compute the integral of (1 - x^2)^(a/2) * x^b from -1 to 1.

        Parameters:
        -----------
        a : float
            Exponent for the (1 - x^2) term.
        b : float
            Exponent for the x term.

        Returns:
        --------
        float
            The value of the integral.
        """
        return (
            (1.0 + (-1.0) ** b)
            * gamma(a / 2.0 + 1.0)
            * gamma((b + 1.0) / 2.0)
            / (2.0 * gamma(a / 2.0 + b / 2.0 + 3.0 / 2.0))
        )

    def calculate_f_ell(self, ell, gamma_b):
        """
        Computes the angular part of the satellite intrinsic shear field.

        Eq. (C8) in `Fortuna et al. 2021 <https://arxiv.org/abs/2003.02700>`

        Parameters:
        -----------
        ell : int
            Multipole moment.
        gamma_b : float
            Slope parameter.

        Returns:
        --------
        complex
            The value of the angular part of the satellite intrinsic shear field.
        """
        phase = np.cos(2.0 * self.phi_k) + 1j * np.sin(2.0 * self.phi_k)

        # Follow CCL by hard-coding for most common cases (b=0, b=-2) to gain speed
        # (in CCL gain is ~1.3sec - gain here depends on how many times this is called).
        if self.theta_k == np.pi / 2.0 and gamma_b in [0, -2]:
            pre_calc_f_ell = {
                0: np.array(
                    [
                        0,
                        0,
                        2.77582637,
                        0,
                        -0.19276603,
                        0,
                        0.04743899,
                        0,
                        -0.01779024,
                        0,
                        0.00832446,
                        0,
                        -0.00447308,
                        0,
                    ]
                ),
                -2: np.array(
                    [
                        0,
                        0,
                        4.71238898,
                        0,
                        -2.61799389,
                        0,
                        2.06167032,
                        0,
                        -1.76714666,
                        0,
                        1.57488973,
                        0,
                        -1.43581368,
                        0,
                    ]
                ),
            }
            return pre_calc_f_ell.get(gamma_b)[ell] * phase

        else:
            # If either of the above expressions are met the return statement is executed and the function ends.
            # Otherwise, the function continues to calculate the general case.
            gj = np.array(
                [
                    0,
                    0,
                    np.pi / 2,
                    0,
                    np.pi / 2,
                    0,
                    15 * np.pi / 32,
                    0,
                    7 * np.pi / 16,
                    0,
                    105 * np.pi / 256,
                    0,
                ]
            )
            sum1 = sum(
                binom(ell, m)
                * binom(0.5 * (ell + m - 1.0), ell)
                * sum(
                    binom(m, j)
                    * gj[j]
                    * np.sin(self.theta_k) ** j
                    * np.cos(self.theta_k) ** (m - j)
                    * self.I_x(j + gamma_b, m - j)
                    for j in range(m + 1)
                )
                for m in range(ell + 1)
            )
            return 2.0**ell * sum1 * phase

    @cached_quantity
    def wkm_f_ell(self):
        """
        Integral of the angular part in eq B8 (SB10) using the Legendre polynomials assuming theta_e=theta, phi_e=phi (perfect radial alignment).

        Note CCL only calculates the real parts of w(k|m)f_ell and doesn't take the absolute value....
        which means you'll get negative values for wkm in CCL: they take the absolute value later.

        Returns:
        --------
        ndarray
            The absolute value of the sum of the radial and angular parts.
        """
        uell = self.compute_uell_gamma_r_hankel
        nz, nm, nk = uell.shape[1], uell.shape[2], uell.shape[3]
        sum_ell = np.zeros([nz, nm, nk], dtype=complex)

        for ell in self.ell_values:
            angular = self.calculate_f_ell(ell, self.gamma_1h_slope)
            radial = (1j) ** ell * (2.0 * ell + 1.0) * uell[ell // 2, :, :, :]
            sum_ell += radial * angular

        return np.abs(sum_ell)

    def gamma_r_nfw_profile(self, r, rs, rvir, a, b, rcore=0.06, truncate=True):
        """
        Compute the radial profile of the NFW (Navarro-Frenk-White) profile with a power-law correction.

        Parameters:
        -----------
        r : float
            Radial distance.
        rs : float
            Scale radius.
        rvir : float
            Virial radius.
        a : float
            Amplitude of the power-law correction.
        b : float
            Slope of the power-law correction.
        rcore : float, optional
            Core radius.
        truncate : bool, optional
            Whether to truncate the profile at the virial radius.

        Returns:
        --------
        float
            The value of the radial profile.
        """
        gamma = a * (r / rvir) ** b
        gamma = np.where(r < rcore, a * (rcore / rvir) ** b, gamma)
        gamma = np.clip(gamma, None, 0.3)

        nfw = 1.0 / ((r / rs) * (1.0 + (r / rs)) ** 2.0)
        if truncate:
            nfw = np.where(r >= rvir, 0.0, nfw)
        return gamma * nfw

    @cached_quantity
    def compute_uell_gamma_r_hankel(self):
        """
        Computes a 4D array containing u_ell as a function of l, z, m, and k. THIS FUNCTION IS THE SLOWEST PART!

        h_transf = HankelTransform(ell+0.5,N_hankel,pi/N_hankel)
        Note even though ell is not used in this function, h_transf depends on ell
        We initialize the class in setup as it only depends on predefined ell values

        Note: I experimented coding the use of Simpson integration for where the Bessel function is flat
        and then switching to the Hankel transform for where the Bessel function oscillates.
        This is more accurate than using the Hankel transform for all k values with lower accuracy
        settings, but it's slower than using the Hankel transform for all k values.
        It's also difficult to decide how to define the transition between the two methods.
        Given that low-k accuracy is unimportant for IA, I've decided to use the Hankel transform for all k values.

        Returns:
        --------
        ndarray
            A 4D array of u_ell values.
        """
        mnfw = (
            4.0
            * np.pi
            * self.r_s**3.0
            * (np.log(1.0 + self.c) - self.c / (1.0 + self.c))
        )
        uk_l = np.zeros(
            [self.ell_values.size, self.z_vec.size, self.mass.size, self.k_vec.size]
        )

        for i, ell in enumerate(self.ell_values):
            if self.method == 'hankel':
                for jz in range(self.z_vec.size):
                    for im in range(self.mass.size):
                        nfw_f = lambda x: self.gamma_r_nfw_profile(
                            x,
                            self.r_s[jz, im],
                            self.rvir[im],
                            self.gamma_1h_amp[jz],
                            self.gamma_1h_slope,
                            truncate=self.truncate,
                        ) * np.sqrt((x * np.pi) / 2.0)
                        uk_l[i, jz, im, :] = self.hankel[i].transform(
                            nfw_f, self.k_vec
                        )[0] / (self.k_vec**0.5 * mnfw[jz, im])

            if self.method == 'fftlog':
                mu = ell + 0.5
                # Precision settings that seem to be working fine
                bias = -0.5 * (i + 2.0)
                low_r = -8
                high_r = 8
                r = np.logspace(low_r, high_r, 512)
                dlnr = np.log(r[1] / r[0])
                offset = fhtoffset(dlnr, mu=mu, initial=-2 * np.log(10.0), bias=bias)
                k = np.exp(offset) / r[::-1]

                nfw_f = (
                    self.gamma_r_nfw_profile(
                        r[np.newaxis, np.newaxis, :],
                        self.r_s[:, :, np.newaxis],
                        self.rvir[np.newaxis, :, np.newaxis],
                        self.gamma_1h_amp[:, np.newaxis, np.newaxis],
                        self.gamma_1h_slope,
                        truncate=self.truncate,
                    )
                    * r**1.5
                    * np.sqrt(np.pi / 2.0)
                )
                ft = fht(nfw_f, dlnr, mu=mu, offset=offset, bias=bias) / (
                    k**1.5 * mnfw[:, :, np.newaxis]
                )
                uk_l[i, :, :, :] = interp1d(
                    k,
                    ft,
                    axis=-1,
                    kind='linear',
                    fill_value='extrapolate',
                    bounds_error=False,
                )(self.k_vec)

        return uk_l

    @cached_quantity
    def wkm(self):
        """
        Return the computed wkm_f_ell values along with the redshift, mass, and k vectors.

        Returns:
        --------
        tuple
            Computed values of wkm_f_ell, array of redshifts, array of halo masses, and array of k values.
        """
        return self.wkm_f_ell, self.z_vec, self.mass, self.k_vec

    def upsampled_wkm(self, k_vec_out, mass_out):
        """
        Interpolates the wkm profiles and upsamples back to the original grid.

        Parameters:
        -----------
        k_vec_out : array_like
            Output array of k values.
        mass_out : array_like
            Output array of halo masses.

        Returns:
        --------
        ndarray
            Upsampled array of wkm values.
        """
        wkm_in = self.wkm_f_ell
        if self.method == 'hankel':
            wkm_out = np.empty([self.z_vec.size, mass_out.size, k_vec_out.size])
            for jz in range(self.z_vec.size):
                # Create the interpolator
                lg_w_interp2d = RegularGridInterpolator(
                    (np.log10(self.k_vec).T, np.log10(self.mass).T),
                    np.log10(wkm_in[jz, :, :] / self.k_vec**2).T,
                    bounds_error=False,
                    fill_value=None,
                )

                # Prepare the grid for interpolation
                lgkk, lgmm = np.meshgrid(
                    np.log10(k_vec_out), np.log10(mass_out), sparse=True
                )

                # Interpolate the values
                lg_wkm_interpolated = lg_w_interp2d((lgkk.T, lgmm.T)).T

                # Convert back to original scale
                wkm_out[jz, :, :] = 10.0 ** (lg_wkm_interpolated) * k_vec_out**2.0
            return wkm_out

        if self.method == 'fftlog':
            # Need to only upsample the k vector!
            return interp1d(
                self.k_vec,
                wkm_in,
                axis=-1,
                kind='linear',
                fill_value='extrapolate',
                bounds_error=False,
            )(k_vec_out)
