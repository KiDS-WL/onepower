import numpy as np
from cosmosis.datablock import option_section
from hankel import HankelTransform
from ia_radial_lib import compute_uell_gamma_r_hankel, wkm_f_ell


def downsample_halo_parameters(nmass_halo, nmass_setup, mass_halo, c_halo,
                               r_s_halo, rvir_halo):
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

def setup(options):
    # Set up the resolution for redshift, mass, and k grid for calculating w(k,z|m)
    # function which is slow: the lower the resolution the better
    # and we'll interpolate over this later to get the
    # right resolution for the power spectrum calculation

    # The set of default parameters here are fast and reasonably accurate
    # Needs more testing to be sure what the optimal defaults are though
    # TODO:  I have not yet tested the resolution in the z-dimension

    nmass = options.get_int(option_section, 'nmass', default=5)
    kmin = options.get_double(option_section, 'kmin', default=1e-3)
    kmax = options.get_double(option_section, 'kmax', default=1e3)
    nk = options.get_int(option_section, 'nk', default=10)
    k_vec = np.logspace(np.log10(kmin), np.log10(kmax), nk)

    # Are we calculating the alignment for say red or blue galaxies?
    name = options.get_string(option_section, 'output_suffix', default='').lower()
    suffix = f'_{name}' if name else ''

    # CCL and Fortuna use ell_max=6.  SB10 uses ell_max = 2.
    # Higher/lower increases/decreases accuracy but slows/speeds the code
    ell_max = options.get_int(option_section, 'ell_max', default=6)
    if ell_max > 11:
        raise ValueError("Please reduce ell_max < 11 or update ia_radial_interface.py")

    # Initialize Hankel transform
    # HankelTransform(nu, # The order of the bessel function
    #                N,  # Number of steps in the integration
    #                h   # Proxy for "size" of steps in integration)
    # We've used hankel.get_h to set h, N is then h=pi/N, finding best_h = 0.05, best_N=62
    # If you want perfect agreement with CCL use: N=50000, h=0.00006 (VERY SLOW!!)

    n_hankel = options.get_int(option_section, 'N_hankel', default=350)
    h_hankel = np.pi / n_hankel
    h_transform = [
        HankelTransform(ell + 0.5, n_hankel, h_hankel)
        for ell in range(0, ell_max + 1, 2)
    ]

    return k_vec, nmass, suffix, h_transform, ell_max

def execute(block, config):
    k_setup, nmass_setup, suffix, h_transform, ell_max = config

    # Load slope of the power law that describes the satellite alignment
    gamma_1h_slope = block[f'intrinsic_alignment_parameters{suffix}',
                           'gamma_1h_radial_slope']
    # This already contains the luminosity dependence if there
    gamma_1h_amplitude = block[f'ia_small_scale_alignment{suffix}',
                               'alignment_1h']
    # Also load the redshift dimension
    z = block['concentration_m', 'z']
    nz = z.size

    # Now I want to load the high resolution halo parameters calculated with the halo model module
    # and then downsample them to a lower resolution grid for the radial IA calculation
    # When downsampling we note that this doesn't need to be perfect, our final resolution does not need to
    # perfectly match the user input value - just as close as possible

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
    mass, c, r_s, rvir = downsample_halo_parameters(
        nmass_halo, nmass_setup, mass_halo, c_halo, r_s_halo, rvir_halo
    )

    k = k_setup
    # uell[l,z,m,k]
    # AD: THIS FUNCTION IS THE SLOWEST PART!
    uell = compute_uell_gamma_r_hankel(
        gamma_1h_amplitude, gamma_1h_slope, k, c, z, r_s, rvir, mass, ell_max,
        h_transform, truncate=False
    )
    theta_k = np.pi / 2.0
    phi_k = 0.0
    wkm = wkm_f_ell(uell, theta_k, phi_k, ell_max, gamma_1h_slope)

    for jz in range(nz):
        block.put_grid(
            'wkm', f'mass_{jz}{suffix}', mass, f'k_h_{jz}{suffix}', k,
            f'w_km_{jz}{suffix}', wkm[jz, :, :]
        )
    block.put_double_array_1d('wkm', f'z{suffix}', z)

    return 0

def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
