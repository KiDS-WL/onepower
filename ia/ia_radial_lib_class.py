from cosmosis.datablock import option_section
import numpy as np
from scipy.integrate import simpson
from scipy.special import binom
from hankel import HankelTransform
from functools import cached_property
from scipy.interpolate import RegularGridInterpolator

class SatelliteAlignment:
    def __init__(
            self,
            mass = None,
            k_vec = None,
            z_vec = None,
            c = None,
            r_s = None,
            rvir = None,
            gamma_1h_slope = None,
            gamma_1h_amplitude = None,
            n_hankel = 350,
            nmass = 5,
            ell_max = 6,
            truncate = False
        ):
        
        self.k_vec_d = k_vec
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

        self.mass = mass
        self.nmass_halo = self.mass.size
        self.c = c
        self.r_s = r_s
        self.rvir = rvir
        
        if self.nmass_halo < self.nmass:
            raise ValueError(
                "The halo mass resolution is too low for the radial IA calculation. "
                "Please increase nmass when you run halo_model_ingredients.py"
            )
        self.mass_d, self.c_d, self.r_s_d, self.rvir_d = self.downsample_halo_parameters(
            self.nmass_halo, self.nmass, self.mass, self.c, self.r_s, self.rvir
        )
        
        # CCL and Fortuna use ell_max=6.  SB10 uses ell_max = 2.
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
            nmass_halo,
            nmass_setup,
            mass_halo,
            c_halo,
            r_s_halo,
            rvir_halo
        ):
        if nmass_halo == nmass_setup:
            return mass_halo, c_halo, r_s_halo, rvir_halo

        downsample_factor = nmass_halo // nmass_setup
        mass = mass_halo[::downsample_factor]
        c = c_halo[:, ::downsample_factor]
        r_s = r_s_halo[:, ::downsample_factor]
        rvir = rvir_halo[::downsample_factor]

        # We make sure that the highest mass is included to avoid extrapolation issues
        if mass[-1] != mass_halo[-1]:
            mass = np.append(mass, mass_halo[-1])
            c = np.concatenate((c, np.atleast_2d(c_halo[:, -1]).T), axis=1)
            r_s = np.concatenate((r_s, np.atleast_2d(r_s_halo[:, -1]).T), axis=1)
            rvir = np.append(rvir, rvir_halo[-1])

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
        """
        self.h_hankel = np.pi / self.n_hankel
        return [
            HankelTransform(ell + 0.5, self.n_hankel, self.h_hankel)
                for ell in self.ell_values
        ]
        
    def I_x(self, a, b):
        eps = 1e-10
        x = np.linspace(-1.0 + eps, 1.0 - eps, 500)
        return simpson((1.0 - x**2.0)**(a / 2.0) * x**b, x)

    def calculate_f_ell(self, l, gamma_b):
        """
        Computes the angular part of the satellite intrinsic shear field,
        Eq. (C8) in `Fortuna et al. 2021 <https://arxiv.org/abs/2003.02700>`
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
        uell[l, z, m, k]
  
        h_transf = HankelTransform(ell+0.5,N_hankel,pi/N_hankel)
        Note even though ell is not used in this function, h_transf depends on ell
        We initialize the class in setup as it only depends on predefined ell values
    
        Note: I experimented coding the use of Simpson integration for where the Bessel function is flat
        and then switching to the Hankel transform for where the Bessel function oscillates.
        This is more accurate than using the Hankel transform for all k values with lower accuracy
        settings, but it's slower than using the Hankel transform for all k values.
        It's also difficult to decide how to define the transition between the two methods.
        Given that low-k accuracy is unimportant for IA, I've decided to use the Hankel transform for all k values.
        """
        mnfw = 4.0 * np.pi * self.r_s_d**3.0 * (np.log(1.0 + self.c_d) - self.c_d / (1.0 + self.c_d))
        uk_l = np.zeros([self.ell_values.size, self.z_vec.size, self.mass_d.size, self.k_vec_d.size])

        for i, ell in enumerate(self.ell_values):
            for jz in range(self.z_vec.size):
                for im in range(self.mass_d.size):
                    nfw_f = lambda x: self.gamma_r_nfw_profile(x, self.r_s_d[jz, im], self.rvir_d[im], self.gamma_1h_amplitude[jz], self.gamma_1h_slope, truncate=self.truncate) * np.sqrt((x * np.pi) / 2.0)
                    uk_l[i, jz, im, :] = self.hankel[i].transform(nfw_f, self.k_vec_d)[0] / (self.k_vec_d**0.5 * mnfw[jz, im])
        return uk_l

    def wkm(self):
        #wkm_out = self.wkm_f_ell
        return self.wkm_f_ell, self.z_vec, self.mass_d, self.k_vec_d
    
    def upsampled_wkm(k_vec, mass, z_vec):
        """
        Interpolates the wkm profiles and upsamples back to original grid
        """
        wkm_out = np.empty([z_vec.size, mass.size, k_vec.size])
        for jz in range(z_vec.size):
            lg_w_interp2d = RegularGridInterpolator(
                (np.log10(self.k_vec_d).T, np.log10(self.mass_d).T),
                np.log10(self.wkm_f_ell / self.k_vec_d**2).T,
                bounds_error=False,
                fill_value=None
            )
            lgkk, lgmm = np.meshgrid(np.log10(k_vec), np.log10(mass), sparse=True)
            lg_wkm_interpolated = lg_w_interp2d((lgkk.T, lgmm.T)).T
            wkm_out[jz] = 10.0**(lg_wkm_interpolated) * k_vec**2.0
        return wkm_out
