from functools import cached_property
import numpy as np
from scipy.integrate import simpson
from scipy.special import binom
from scipy.interpolate import RegularGridInterpolator
from hankel import HankelTransform

"""
A module for computing satellite alignment properties.
This module provides classes and functions to calculate various properties 
related to the alignment of satellite galaxies within dark matter halos.
"""

class SatelliteAlignment:
    """
    A class to compute the alignment properties of satellite galaxies within dark matter halos.
    This includes calculating the Hankel transform, radial profiles, and other related quantities.

    Parameters:
    - mass: Array of halo masses.
    - z_vec: Array of redshifts.
    - c: Concentration parameter.
    - r_s: Scale radius.
    - rvir: Virial radius.
    - gamma_1h_slope: Slope of the power law describing the satellite alignment.
    - gamma_1h_amplitude: Amplitude of the satellite alignment.
    - n_hankel: Number of steps in the Hankel transform integration.
    - nmass: Number of mass bins.
    - nk: Number of k bins.
    - ell_max: Maximum multipole moment.
    - truncate: Whether to truncate the NFW profile at the virial radius.
    """
    def __init__(
            self,
            mass = None,
            z_vec = None,
            c = None,
            r_s = None,
            rvir = None,
            gamma_1h_slope = None,
            gamma_1h_amplitude = None,
            n_hankel = 350,
            nmass = 5,
            nk = 10,
            ell_max = 6,
            truncate = False
        ):
        
        self.k_vec = np.logspace(np.log10(1e-3), np.log10(1e3), nk)
        self.nk = nk
        self.nmass = nmass
        self.n_hankel = n_hankel
        self.ell_max = ell_max
        
        # Slope of the power law that describes the satellite alignment
        self.gamma_1h_slope = gamma_1h_slope
        
        # This already contains the luminosity dependence if there
        self.gamma_1h_amplitude = gamma_1h_amplitude
        
        # Also load the redshift dimension
        self.z_vec = z_vec
        nz = self.z_vec.size
        
        self.mass_in = mass
        self.c_in = c
        self.r_s_in = r_s
        self.rvir_in = rvir
        
        if self.mass_in.size < self.nmass:
            raise ValueError(
                "The halo mass resolution is too low for the radial IA calculation. "
                "Please increase nmass when you run halo_model_ingredients.py"
            )
        self.mass, self.c, self.r_s, self.rvir = self.downsample_halo_parameters(
            self.mass_in.size, self.nmass, self.mass_in, self.c_in, self.r_s_in, self.rvir_in
        )
        
        # CCL and Fortuna use ell_max=6. SB10 uses ell_max = 2.
        # Higher/lower increases/decreases accuracy but slows/speeds the code
        if self.ell_max > 11:
            raise ValueError("Please reduce ell_max < 11 or update ia_radial_interface.py")
        self.ell_values = np.arange(0, self.ell_max + 1, 2)
        
        self.truncate = truncate
        # These are for now hardcoded choices
        self.theta_k = np.pi / 2.0
        self.phi_k = 0.0
        
        # Initilise the hankel transform
        self.hankel = self.h_transform
        
    def downsample_halo_parameters(
            self,
            nmass,
            nmass_in,
            mass_in,
            c_in,
            r_s_in,
            rvir_in
        ):
        """
        Downsample the halo parameters to reduce computational complexity.

        Parameters:
        - nmass: Number of mass bins desired.
        - nmass_in: Number of mass bins in input.
        - mass_in: Input array of halo masses.
        - c_in: Input array of concentration parameters.
        - r_s_in: Input array of scale radii.
        - rvir_in: Input array of virial radii.

        Returns:
        - Downsampled arrays of halo masses, concentration parameters, scale radii, and virial radii.
        """
        if nmass == nmass_in:
            return mass_in, c_in, r_s_in, rvir_in

        downsample_factor = nmass // nmass_in
        mass = mass_in[::downsample_factor]
        c = c_in[:, ::downsample_factor]
        r_s = r_s_in[:, ::downsample_factor]
        rvir = rvir_in[::downsample_factor]

        # We make sure that the highest mass is included to avoid extrapolation issues
        if mass[-1] != mass_in[-1]:
            mass = np.append(mass, mass_in[-1])
            c = np.concatenate((c, np.atleast_2d(c_in[:, -1]).T), axis=1)
            r_s = np.concatenate((r_s, np.atleast_2d(r_s_in[:, -1]).T), axis=1)
            rvir = np.append(rvir, rvir_in[-1])

        return mass, c, r_s, rvir
        
    @cached_property
    def h_transform(self):
        """
        Initialize Hankel transform
        HankelTransform(nu, # The order of the bessel function
                       N,  # Number of steps in the integration
                       h   # Proxy for "size" of steps in integration)
        We've used hankel.get_h to set h, N is then h=pi/N, finding best_h = 0.05, best_N=62
        If you want perfect agreement with CCL use: N=50000, h=0.00006 (VERY SLOW!!)
        
        Ideally we need to find a way to just evaluate this part outside of the class
        so it can work nicely with CosmoSIS setup function
        
        Returns:
        - A list of HankelTransform objects for each multipole moment.
        """
        self.h_hankel = np.pi / self.n_hankel
        return [
            HankelTransform(ell + 0.5, self.n_hankel, self.h_hankel)
                for ell in self.ell_values
        ]
        
    def I_x(self, a, b):
        """
        Compute the integral of (1 - x^2)^(a/2) * x^b from -1 to 1.

        Parameters:
        - a: Exponent for the (1 - x^2) term.
        - b: Exponent for the x term.

        Returns:
        - The value of the integral.
        """
        eps = 1e-10
        x = np.linspace(-1.0 + eps, 1.0 - eps, 500)
        return simpson((1.0 - x**2.0)**(a / 2.0) * x**b, x)

    def calculate_f_ell(self, l, gamma_b):
        """
        Computes the angular part of the satellite intrinsic shear field,
        Eq. (C8) in `Fortuna et al. 2021 <https://arxiv.org/abs/2003.02700>`
        
        Parameters:
        - l: Multipole moment.
        - gamma_b: Slope parameter.

        Returns:
        - The value of the angular part of the satellite intrinsic shear field.
        """
        phase = np.cos(2.0 * self.phi_k) + 1j * np.sin(2.0 * self.phi_k)

        # Follow CCL by hard-coding for most common cases (b=0, b=-2) to gain speed
        # (in CCL gain is ~1.3sec - gain here depends on how many times this is called).
        if self.theta_k == np.pi / 2.0 and gamma_b in [0, -2]:
            pre_calc_f_ell = {
                0: np.array([0, 0, 2.77582637, 0, -0.19276603, 0, 0.04743899, 0, -0.01779024, 0, 0.00832446, 0, -0.00447308, 0]),
                -2: np.array([0, 0, 4.71238898, 0, -2.61799389, 0, 2.06167032, 0, -1.76714666, 0, 1.57488973, 0, -1.43581368, 0])
            }
            return pre_calc_f_ell.get(gamma_b)[l] * phase

        else:
            # If either of the above expressions are met the return statement is executed and the function ends.
            # Otherwise, the function continues to calculate the general case.
            gj = np.array([0, 0, np.pi / 2, 0, np.pi / 2, 0, 15 * np.pi / 32, 0, 7 * np.pi / 16, 0, 105 * np.pi / 256, 0])
            sum1 = sum(
                    binom(l, m) * binom(0.5 * (l + m - 1.0), l) *
                    sum(
                        binom(m, j) * gj[j] * np.sin(self.theta_k)**j * np.cos(self.theta_k)**(m - j) * self.I_x(j + gamma_b, m - j)
                        for j in range(m + 1)
                    )
                    for m in range(l + 1)
                )
            return 2.0**l * sum1 * phase

    @cached_property
    def wkm_f_ell(self):
        """
        Integral of the angular part in eq B8 (SB10) using the Legendre polynomials
        assuming theta_e=theta, phi_e=phi (perfect radial alignment)
    
        Note CCL only calculates the real parts of w(k|m)f_ell and doesn't take the absolute value....
        which means you'll get negative values for wkm in CCL: they take the absolute value later.
        
        Returns:
        - The absolute value of the sum of the radial and angular parts.
        """
        uell = self.compute_uell_gamma_r_hankel
        nz, nm, nk = uell.shape[1], uell.shape[2], uell.shape[3]
        sum_ell = np.zeros([nz, nm, nk], dtype=complex)
    
        for ell in self.ell_values:
            angular = self.calculate_f_ell(ell, self.gamma_1h_slope)
            radial = (1j)**ell * (2.0 * ell + 1.0) * uell[ell // 2, :, :, :]
            sum_ell += radial * angular
    
        return np.abs(sum_ell)

    def gamma_r_nfw_profile(self, r, rs, rvir, a, b, rcore=0.06, truncate=True):
        """
        Compute the radial profile of the NFW (Navarro-Frenk-White) profile with a power-law correction.

        Parameters:
        - r: Radial distance.
        - rs: Scale radius.
        - rvir: Virial radius.
        - a: Amplitude of the power-law correction.
        - b: Slope of the power-law correction.
        - rcore: Core radius.
        - truncate: Whether to truncate the profile at the virial radius.

        Returns:
        - The value of the radial profile.
        """
        gamma = a * (r / rvir)**b
        gamma = np.where(r < rcore, a * (rcore / rvir)**b, gamma)
        gamma = np.clip(gamma, None, 0.3)
        
        nfw = 1.0 / ((r / rs) * (1.0 + (r / rs))**2.0)
        if truncate:
            nfw = np.where(r >= rvir, 0.0, nfw)
        return gamma * nfw

    @cached_property
    def compute_uell_gamma_r_hankel(self):
        """
        THIS FUNCTION IS THE SLOWEST PART!
        
        Computes a 4D array containing u_ell as a function of l, z, m, and k.
  
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
        - A 4D array of u_ell values.
        """
        mnfw = 4.0 * np.pi * self.r_s**3.0 * (np.log(1.0 + self.c) - self.c / (1.0 + self.c))
        uk_l = np.zeros([self.ell_values.size, self.z_vec.size, self.mass.size, self.k_vec.size])

        for i, ell in enumerate(self.ell_values):
            for jz in range(self.z_vec.size):
                for im in range(self.mass.size):
                    nfw_f = lambda x: self.gamma_r_nfw_profile(x, self.r_s[jz, im], self.rvir[im], self.gamma_1h_amplitude[jz], self.gamma_1h_slope, truncate=self.truncate) * np.sqrt((x * np.pi) / 2.0)
                    uk_l[i, jz, im, :] = self.hankel[i].transform(nfw_f, self.k_vec)[0] / (self.k_vec**0.5 * mnfw[jz, im])
        return uk_l

    def wkm(self):
        """
        Return the computed wkm_f_ell values along with the redshift, mass, and k vectors.

        Returns:
        - wkm_f_ell: Computed values of wkm_f_ell.
        - z_vec: Array of redshifts.
        - mass: Array of halo masses.
        - k_vec: Array of k values.
        """
        return self.wkm_f_ell, self.z_vec, self.mass, self.k_vec
    
    def upsampled_wkm(self, k_vec_out, mass_out):
        """
        Interpolates the wkm profiles and upsamples back to the original grid.

        Parameters:
        - k_vec_out: Output array of k values.
        - mass_out: Output array of halo masses.

        Returns:
        - Upsampled array of wkm values.
        """
        wkm_out = np.empty([self.z_vec.size, mass_out.size, k_vec_out.size])
        for jz in range(self.z_vec.size):
            # Create the interpolator
            lg_w_interp2d = RegularGridInterpolator(
                (np.log10(self.k_vec).T, np.log10(self.mass).T),
                np.log10(self.wkm_f_ell[jz, :, :] / self.k_vec**2).T,
                bounds_error=False,
                fill_value=None
            )
            
            # Prepare the grid for interpolation
            lgkk, lgmm = np.meshgrid(np.log10(k_vec_out), np.log10(mass_out), sparse=True)
            
            # Interpolate the values
            lg_wkm_interpolated = lg_w_interp2d((lgkk.T, lgmm.T)).T
            
            # Convert back to original scale
            wkm_out[jz, :, :] = 10.0**(lg_wkm_interpolated) * k_vec_out**2.0
        return wkm_out
