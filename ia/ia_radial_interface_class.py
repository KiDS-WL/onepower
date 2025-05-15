from cosmosis.datablock import option_section
import numpy as np
from scipy.integrate import simpson
from scipy.special import legendre, binom
from hankel import HankelTransform

class SatelliteAlignment:
    def __init__(self):
        pass

    def I_x(self, a, b):
        eps = 1e-10
        x = np.linspace(-1.0 + eps, 1.0 - eps, 500)
        return simpson((1.0 - x**2.0)**(a / 2.0) * x**b, x)

    def legendre_coefficients(self, l, m):
        return legendre(l)[m]

    def calculate_f_ell(self, theta_k, phi_k, l, gamma_b):
        phase = np.cos(2.0 * phi_k) + 1j * np.sin(2.0 * phi_k)

        if theta_k == np.pi / 2.0:
            pre_calc_f_ell = {
                0: np.array([0, 0, 2.77582637, 0, -0.19276603, 0, 0.04743899, 0, -0.01779024, 0, 0.00832446, 0, -0.00447308, 0]),
                -2: np.array([0, 0, 4.71238898, 0, -2.61799389, 0, 2.06167032, 0, -1.76714666, 0, 1.57488973, 0, -1.43581368, 0])
            }
            return pre_calc_f_ell.get(gamma_b)[l] * phase

        gj = np.array([0, 0, np.pi / 2, 0, np.pi / 2, 0, 15 * np.pi / 32, 0, 7 * np.pi / 16, 0, 105 * np.pi / 256, 0])
        sum1 = 0.0
        for m in range(l + 1):
            sum2 = 0.0
            for j in range(m + 1):
                sum2 += binom(m, j) * gj[j] * np.sin(theta_k)**j * np.cos(theta_k)**(m - j) * self.I_x(j + gamma_b, m - j)
            sum1 += binom(l, m) * binom(0.5 * (l + m - 1.0), l) * sum2
        return 2.0**l * sum1 * phase

    def nfw_profile(self, r, rs):
        x = r / rs
        return 1.0 / (x * (1.0 + x)**2.0)

    def mass_nfw(self, r_s, c):
        return 4.0 * np.pi * r_s**3.0 * (np.log(1.0 + c) - c / (1.0 + c))

    def nfw_profile_trunc(self, r, rs, rvir):
        return np.where(r >= rvir, 0.0, self.nfw_profile(r, rs))

    def gamma_r_nfw_profile(self, r, rs, rvir, a, b, rcore=0.06, truncate=True):
        gamma = a * (r / rvir)**b
        gamma = np.where(r < rcore, a * (rcore / rvir)**b, gamma)
        gamma = np.clip(gamma, None, 0.3)
        nfw = self.nfw_profile_trunc(r, rs, rvir) if truncate else self.nfw_profile(r, rs)
        return gamma * nfw

    def compute_uell_gamma_r_hankel(self, gamma_1h_amplitude, gamma_b, k, c, z, r_s, rvir, mass, ell_max, h_transf, truncate=False):
        ell_values = np.arange(0, ell_max + 1, 2)
        mnfw = self.mass_nfw(r_s, c)
        uk_l = np.zeros([ell_values.size, z.size, mass.size, k.size])

        for i, ell in enumerate(ell_values):
            for jz in range(z.size):
                for im in range(mass.size):
                    nfw_f = lambda x: self.gamma_r_nfw_profile(x, r_s[jz, im], rvir[im], gamma_1h_amplitude[jz], gamma_b, truncate=truncate) * np.sqrt((x * np.pi) / 2.0)
                    uk_l[i, jz, im, :] = h_transf[i].transform(nfw_f, k)[0] / (k**0.5 * mnfw[jz, im])

        return uk_l

    def wkm_f_ell(self, uell, theta_k, phi_k, ell_max, gamma_b):
        nz, nm, nk = uell.shape[1], uell.shape[2], uell.shape[3]
        sum_ell = np.zeros([nz, nm, nk], dtype=complex)

        for ell in range(0, ell_max + 1, 2):
            angular = self.calculate_f_ell(theta_k, phi_k, ell, gamma_b)
            c_, d_ = np.real(angular), np.imag(angular)
            radial = (1j)**ell * (2.0 * ell + 1.0) * uell[ell // 2, :, :, :]
            a_, b_ = np.real(radial), np.imag(radial)
            sum_ell += (a_ * c_ - b_ * d_) + 1j * (a_ * d_ + b_ * c_)

        return np.sqrt(np.real(sum_ell)**2 + np.imag(sum_ell)**2)

    def downsample_halo_parameters(self, nmass_halo, nmass_setup, mass_halo, c_halo, r_s_halo, rvir_halo):
        if nmass_halo == nmass_setup:
            return mass_halo, c_halo, r_s_halo, rvir_halo

        downsample_factor = nmass_halo // nmass_setup
        mass = mass_halo[::downsample_factor]
        c = c_halo[:, ::downsample_factor]
        r_s = r_s_halo[:, ::downsample_factor]
        rvir = rvir_halo[::downsample_factor]

        if mass[-1] != mass_halo[-1]:
            mass = np.append(mass, mass_halo[-1])
            c = np.concatenate((c, np.atleast_2d(c_halo[:, -1]).T), axis=1)
            r_s = np.concatenate((r_s, np.atleast_2d(r_s_halo[:, -1]).T), axis=1)
            rvir = np.append(rvir, rvir_halo[-1])

        return mass, c, r_s, rvir

    def calculate_wkm(self, block, config):
        k_setup, nmass_setup, suffix, h_transform, ell_max = config

        gamma_1h_slope = block[f'intrinsic_alignment_parameters{suffix}', 'gamma_1h_radial_slope']
        gamma_1h_amplitude = block[f'ia_small_scale_alignment{suffix}', 'alignment_1h']
        z = block['concentration_m', 'z']
        nz = z.size

        mass_halo = block['concentration_m', 'm_h']
        nmass_halo = mass_halo.size
        c_halo = block['concentration_m', 'c']
        r_s_halo = block['nfw_scale_radius_m', 'rs']
        rvir_halo = block['virial_radius', 'rvir_m']

        if nmass_halo < nmass_setup:
            raise ValueError(
                "The halo mass resolution is too low for the radial IA calculation. "
                "Please increase nmass when you run halo_model_ingredients.py"
            )
        mass, c, r_s, rvir = self.downsample_halo_parameters(
            nmass_halo, nmass_setup, mass_halo, c_halo, r_s_halo, rvir_halo
        )

        k = k_setup
        uell = self.compute_uell_gamma_r_hankel(
            gamma_1h_amplitude, gamma_1h_slope, k, c, z, r_s, rvir, mass, ell_max,
            h_transform, truncate=False
        )
        theta_k = np.pi / 2.0
        phi_k = 0.0
        wkm = self.wkm_f_ell(uell, theta_k, phi_k, ell_max, gamma_1h_slope)

        return wkm, z, mass, k, suffix

def setup(options):
    nmass = options.get_int(option_section, 'nmass', default=5)
    kmin = options.get_double(option_section, 'kmin', default=1e-3)
    kmax = options.get_double(option_section, 'kmax', default=1e3)
    nk = options.get_int(option_section, 'nk', default=10)
    k_vec = np.logspace(np.log10(kmin), np.log10(kmax), nk)

    name = options.get_string(option_section, 'output_suffix', default='').lower()
    suffix = f'_{name}' if name else ''

    ell_max = options.get_int(option_section, 'ell_max', default=6)
    if ell_max > 11:
        raise ValueError("Please reduce ell_max < 11 or update ia_radial_interface.py")

    n_hankel = options.get_int(option_section, 'N_hankel', default=350)
    h_hankel = np.pi / n_hankel
    h_transform = [
        HankelTransform(ell + 0.5, n_hankel, h_hankel)
        for ell in range(0, ell_max + 1, 2)
    ]

    return k_vec, nmass, suffix, h_transform, ell_max

def execute(block, config):
    satellite_alignment = SatelliteAlignment()
    wkm, z, mass, k, suffix = satellite_alignment.calculate_wkm(block, config)

    nz = z.size
    for jz in range(nz):
        block.put_grid(
            'wkm', f'mass_{jz}{suffix}', mass, f'k_h_{jz}{suffix}', k,
            f'w_km_{jz}{suffix}', wkm[jz, :, :]
        )
    block.put_double_array_1d('wkm', f'z{suffix}', z)

    return 0

def cleanup(config):
    pass
