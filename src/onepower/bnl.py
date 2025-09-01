"""
A module for computing non-linear halo bias.
This module provides classes and functions to calculate non-linear halo bias using the Dark Emulator.
In future we might want to include other prescription of calculating the said quantity, namely the old Tinker05 and
an analytic prescription from Flamingo sims.
"""

import numpy as np
from collections import OrderedDict
from dark_emulator import darkemu
from functools import cached_property
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.optimize import curve_fit

from hmf._internals._cache import cached_quantity, parameter
from hmf._internals._framework import Framework

from scipy.ndimage import map_coordinates


class NonLinearBias(Framework):
    """
    A class to compute the non-linear bias using the Dark Emulator.

    Parameters:
    -----------
    mass : array_like
        Array of halo masses.
    z_vec : array_like
        Array of redshifts.
    k_vec : array_like
        Array of wavenumbers.
    h0 : float
        Hubble parameter.
    sigma_8 : float
        Amplitude of matter fluctuations on 8 Mpc scales.
    A_s : float, optional
        Amplitude of the primordial power spectrum.
    omega_b : float
        Baryon density parameter.
    omega_c : float
        Cold dark matter density parameter.
    omega_lambda : float
        Dark energy density parameter.
    n_s : float
        Spectral index.
    w0 : float
        Dark energy equation of state parameter.
    z_dep : bool, optional
        If redshift dependence is to be evaluated in Bnl
    """

    def __init__(
        self,
        mass=None,
        z_vec=None,
        k_vec=None,
        h0=0.7,
        sigma_8=0.8,
        A_s=None,
        omega_b=0.05,
        omega_c=0.25,
        omega_lambda=0.7,
        n_s=1.0,
        w0=-1.0,
        z_dep=False,
    ):
        self.mass = mass
        self.z_vec = z_vec
        self.k_vec = k_vec
        self.h0 = h0
        self.sigma_8 = sigma_8
        self.A_s = A_s

        self.omega_b = omega_b
        self.omega_c = omega_c
        self.omega_lambda = omega_lambda
        self.n_s = n_s
        self.w0 = w0

        self.z_dep = z_dep

    @parameter('param')
    def mass(self, val):
        """
        Array of halo masses.

        :type: array_like
        """
        return val

    @parameter('param')
    def z_vec(self, val):
        """
        Array of redshifts.

        :type: array_like
        """
        return val

    @parameter('param')
    def k_vec(self, val):
        """
        Array of wavenumbers.

        :type: array_like
        """
        return val

    @parameter('param')
    def sigma_8(self, val):
        """
        Amplitude of matter fluctuations on 8 Mpc scales.

        :type: float
        """
        return val

    @parameter('param')
    def A_s(self, val):
        """
        Amplitude of the primordial power spectrum.

        :type: float
        """
        return val

    @parameter('param')
    def h0(self, val):
        """
        Hubble parameter.

        :type: float
        """
        return val

    @parameter('param')
    def omega_b(self, val):
        """
        Baryon density parameter.

        :type: float
        """
        return val

    @parameter('param')
    def omega_c(self, val):
        """
        Cold dark matter density parameter.

        :type: float
        """
        return val

    @parameter('param')
    def omega_lambda(self, val):
        """
        Dark energy density parameter.

        :type: float
        """
        return val

    @parameter('param')
    def n_s(self, val):
        """
        Spectral index.

        :type: float
        """
        return val

    @parameter('param')
    def w0(self, val):
        """
        Dark energy equation of state parameter.

        :type: float
        """
        return val

    @parameter('param')
    def z_dep(self, val):
        """
        If redshift dependence is to be evaluated in Bnl

        :type: bool
        """
        return val

    @cached_quantity
    def ombh2(self):
        """
        Return the physical baryon density parameter.

        Returns:
        --------
        array_like
            ombh2
        """
        return self.omega_b * self.h0**2.0

    @cached_quantity
    def omch2(self):
        """
        Return the physical cold dark matter density parameter.

        Returns:
        --------
        array_like
            omch2
        """
        return self.omega_c * self.h0**2.0

    @cached_quantity
    def emulator(self):
        """
        Initialize the Dark Emulator with the given cosmological parameters.

        Returns:
        --------
        darkemu.base_class
            An instance of the Dark Emulator.
        """
        emu = darkemu.base_class()
        if self.A_s is None and self.sigma_8 is not None:
            A_s_init = 2.1e-9

            cparam = self.test_cosmo(
                np.array(
                    [
                        self.ombh2,
                        self.omch2,
                        self.omega_lambda - 0.00064,
                        np.log(A_s_init * 1e10),
                        self.n_s,
                        self.w0,
                    ]
                )
            )
            emu.set_cosmology(cparam)

            sigma_8_init = emu.get_sigma8()
            scaling = self.sigma_8**2.0 / sigma_8_init**2.0
            A_s = A_s_init * scaling
            lnA = np.log(A_s * 1e10)
        elif self.A_s is not None:
            # We preffer A_s for DQ emulator!
            lnA = np.log(self.A_s * 1e10)
        else:
            raise ValueError('One of A_s or sigma_8 need to be specified!')

        cparam = self.test_cosmo(
            np.array(
                [self.ombh2, self.omch2, self.omega_lambda, lnA, self.n_s, self.w0]
            )
        )
        emu.set_cosmology(cparam)
        return emu

    @cached_quantity
    def bnl(self):
        """
        Compute the non-linear bias interpolation function.

        Returns:
        --------
        ndarray
            Interpolated non-linear bias values.
        """
        beta_interp_tmp = self.create_bnl_interpolation_function

        x = np.log10(self.mass)
        y = np.log10(self.mass)
        z = np.log10(self.k_vec)

        n, m, p = len(x), len(y), len(z)

        # Precompute all combinations of values (reordered for direct fill)
        Z = np.repeat(z, n * m)  # p varies fastest (outer axis first)
        X = np.tile(np.repeat(x, m), p)
        Y = np.tile(y, n * p)
        values = np.column_stack((X, Y, Z))

        # Precompute flat indices for direct assignment
        kk = np.repeat(np.arange(p), n * m)
        ii = np.tile(np.repeat(np.arange(n), m), p)
        jj = np.tile(np.arange(m), n * p)

        # Allocate and fill beta_interp efficiently
        if self.z_dep:
            beta_interp = np.zeros((len(self.z_vec), p, n, m))
            for i, _zi in enumerate(self.z_vec):
                beta_interp[i, kk, ii, jj] = beta_interp_tmp[i](values)
            return beta_interp
        else:
            beta_interp = np.zeros((p, n, m))
            beta_interp[kk, ii, jj] = beta_interp_tmp(values)
            return beta_interp[np.newaxis, :, :, :]

    def low_k_truncation(self, k, k_trunc):
        """
        Apply low-k truncation to the non-linear bias.

        Parameters:
        -----------
        k : array_like
            Wavenumber.
        k_trunc : float
            Truncation wavenumber.

        Returns:
        --------
        ndarray
            Truncation array.
        """
        return 1.0 / (1.0 + np.exp(-(10.0 * (np.log10(k) - np.log10(k_trunc)))))

    def high_k_truncation(self, k, k_trunc):
        """
        Apply high-k truncation to the non-linear bias.

        Parameters:
        -----------
        k : array_like
            Wavenumber.
        k_trunc : float
            Truncation wavenumber.

        Returns:
        --------
        ndarray
            Truncation array.
        """
        return 1.0 / (1.0 + np.exp(10.0 * (np.log10(k) - np.log10(k_trunc))))

    @property
    def minimum_halo_mass(self):
        """
        Compute the minimum halo mass for the set of cosmological parameters.

        Returns:
        --------
        tuple
            Minimum halo mass and corresponding wavenumber.
        """
        np_min = 200.0  # Minimum number of halo particles
        npart = 2048.0  # Cube root of number of simulation particles
        Lbox_HR = 1000.0  # Box size for high-resolution simulations [Mpc/h]
        Lbox_LR = 2000.0  # Box size for low-resolution simulations [Mpc/h]

        Om_m = self.emulator.cosmo.get_Omega0()
        rhom = 2.77536627e11 * Om_m

        Mbox_HR = rhom * Lbox_HR**3.0
        mmin = Mbox_HR * np_min / npart**3.0

        vmin = Lbox_HR**3.0 * np_min / npart**3.0
        rmin = ((3.0 * vmin) / (4.0 * np.pi)) ** (1.0 / 3.0)

        return mmin, 2.0 * np.pi / rmin

    def rvir(self, mass):
        """
        Compute the virial radius for a given halo mass.

        Parameters:
        -----------
        mass : array_like
            Halo mass.

        Returns:
        --------
        ndarray
            Virial radius.
        """
        Om_m = self.emulator.cosmo.get_Omega0()
        rhom = 2.77536627e11 * Om_m
        return ((3.0 * mass) / (4.0 * np.pi * 200 * rhom)) ** (1.0 / 3.0)

    def hl_envelopes_idx(self, data, dmin=1, dmax=1):
        """
        Extract high and low envelope indices from a 1D data signal.

        Parameters:
        -----------
        data : array_like
            Data signal from which to extract high and low envelopes.
        dmin : int, optional
            Size of chunks for local minima.
        dmax : int, optional
            Size of chunks for local maxima.

        Returns:
        --------
        tuple
            Indices of high and low envelopes of the input signal.
        """
        # Find local minima indices
        lmin = (np.diff(np.sign(np.diff(data))) > 0).nonzero()[0] + 1
        # Find local maxima indices
        lmax = (np.diff(np.sign(np.diff(data))) < 0).nonzero()[0] + 1

        # Global min of dmin-chunks of local minima
        lmin = lmin[
            [i + np.argmin(data[lmin[i : i + dmin]]) for i in range(0, len(lmin), dmin)]
        ]
        # Global max of dmax-chunks of local maxima
        lmax = lmax[
            [i + np.argmax(data[lmax[i : i + dmax]]) for i in range(0, len(lmax), dmax)]
        ]

        return lmin, lmax

    def compute_bnl_darkquest(self, z, log10M1, log10M2, k, kmax):
        """
        Compute the non-linear bias using the Dark Emulator.

        Parameters:
        -----------
        z : float
            Redshift.
        log10M1 : array_like
            Log10 of the first halo mass.
        log10M2 : array_like
            Log10 of the second halo mass.
        k : array_like
            Wavenumber.
        kmax : float
            Maximum wavenumber.

        Returns:
        --------
        ndarray
            Non-linear bias values.
        """
        M1 = 10.0**log10M1
        M2 = 10.0**log10M2

        # Large 'linear' scale for linear halo bias [h/Mpc]
        klin = np.array([0.05])

        # Calculate beta_NL by looping over mass arrays
        beta_func = np.zeros((len(M1), len(M2), len(k)))

        # Linear power
        Pk_lin = self.emulator.get_pklin_from_z(k, z)
        Pk_klin = self.emulator.get_pklin_from_z(klin, z)

        # Calculate b01 for all M1
        b01 = np.zeros(len(M1))
        # b02 = np.zeros(len(M2))
        for iM, M0 in enumerate(M1):
            b01[iM] = np.sqrt(self.emulator.get_phh_mass(klin, M0, M0, z) / Pk_klin)

        for iM1, M01 in enumerate(M1):
            for iM2, M02 in enumerate(M2):
                if iM2 < iM1:
                    # Use symmetry to not double calculate
                    beta_func[iM1, iM2, :] = beta_func[iM2, iM1, :]
                else:
                    # Linear halo bias
                    b1 = b01[iM1]
                    b2 = b01[iM2]

                    # Halo-halo power spectrum
                    Pk_hh = self.emulator.get_phh_mass(k, M01, M02, z)

                    # rmax = max(self.rvir(M01), self.rvir(M02))
                    # kmax = 2.0*np.pi/rmax

                    # Create beta_NL
                    shot_noise = lambda x, a: a
                    popt, popc = curve_fit(
                        shot_noise,
                        k[(k > 100) & (k < 200)],
                        Pk_hh[(k > 100) & (k < 200)],
                    )
                    Pk_hh = Pk_hh - np.ones_like(k) * shot_noise(k, *popt)

                    beta_func[iM1, iM2, :] = Pk_hh / (b1 * b2 * Pk_lin) - 1.0

                    Pk_hh0 = self.emulator.get_phh_mass(klin, M01, M02, z)
                    Pk_hh0 = Pk_hh0 - np.ones_like(klin) * shot_noise(klin, *popt)
                    db = Pk_hh0 / (b1 * b2 * Pk_klin) - 1.0

                    lmin, lmax = self.hl_envelopes_idx(
                        np.abs(beta_func[iM1, iM2, :] + 1.0)
                    )
                    beta_func_interp = interp1d(
                        k[lmax],
                        np.abs(beta_func[iM1, iM2, lmax] + 1.0),
                        kind='quadratic',
                        bounds_error=False,
                        fill_value='extrapolate',
                    )
                    beta_func[iM1, iM2, :] = (
                        beta_func_interp(k) - 1.0
                    )  # * low_k_truncation(k, klin)
                    db = beta_func_interp(klin) - 1.0

                    # beta_func[iM1, iM2, :] = ((beta_func[iM1, iM2, :] + 1.0) * high_k_truncation(k, 30.0)/(db + 1.0) - 1.0) * low_k_truncation(k, klin)
                    # beta_func[iM1, iM2, :] = ((beta_func[iM1, iM2, :] + 1.0)/(db + 1.0) - 1.0) #* low_k_truncation(k, klin) * high_k_truncation(k, 30.0)#/(1.0+z))
                    beta_func[iM1, iM2, :] = (
                        (beta_func[iM1, iM2, :] - db)
                        * self.low_k_truncation(k, klin)
                        * self.high_k_truncation(k, 3.0 * kmax)
                    )

        return beta_func

    @cached_quantity
    def create_bnl_interpolation_function(self):
        """
        Create an interpolation function for the non-linear bias.

        Returns:
        --------
        RegularGridInterpolator
            Interpolation function for the non-linear bias.
        """
        lenM = 5
        lenk = 1000
        zc = self.z_vec.copy()

        Mmin, kmax = self.minimum_halo_mass
        M_up = np.log10(10.0**14.0)
        # M_lo = np.log10((10.0**12.0))
        M_lo = np.log10(Mmin)

        M = np.logspace(M_lo, M_up, lenM)
        k = np.logspace(-3.0, np.log10(200), lenk)

        if not self.z_dep:
            beta_func = self.compute_bnl_darkquest(
                0.01, np.log10(M), np.log10(M), k, kmax
            )
            beta_nl_interp_i = RegularGridInterpolator(
                [np.log10(M), np.log10(M), np.log10(k)],
                beta_func,
                fill_value=None,
                bounds_error=False,
                method='nearest',
            )

        if self.z_dep:
            beta_nl_interp_i = np.empty(len(self.z_vec), dtype=object)
            for i, zi in enumerate(zc):
                beta_func = self.compute_bnl_darkquest(
                    zi, np.log10(M), np.log10(M), k, kmax
                )
                beta_nl_interp_i[i] = RegularGridInterpolator(
                    [np.log10(M), np.log10(M), np.log10(k)],
                    beta_func,
                    fill_value=None,
                    bounds_error=False,
                    method='nearest',
                )
        return beta_nl_interp_i

    def test_cosmo(self, cparam_in):
        """
        Adjust cosmological parameters to be within the range of the Dark Emulator.

        Parameters:
        -----------
        cparam_in : array_like
            Input cosmological parameters.

        Returns:
        --------
        ndarray
            Adjusted cosmological parameters.
        """
        cparam_range = OrderedDict(
            [
                ['omegab', [0.0211375, 0.0233625]],
                ['omegac', [0.10782, 0.13178]],
                ['Omagede', [0.54752, 0.82128]],
                ['ln(10^10As)', [2.4752, 3.7128]],
                ['ns', [0.916275, 1.012725]],
                ['w', [-1.2, -0.8]],
            ]
        )

        cparam_in = cparam_in.reshape(1, 6)
        cparam_out = np.copy(cparam_in)

        for i, (_key, edges) in enumerate(cparam_range.items()):
            if cparam_in[0, i] < edges[0]:
                cparam_out[0, i] = edges[0]
            if cparam_in[0, i] > edges[1]:
                cparam_out[0, i] = edges[1]

        return cparam_out
