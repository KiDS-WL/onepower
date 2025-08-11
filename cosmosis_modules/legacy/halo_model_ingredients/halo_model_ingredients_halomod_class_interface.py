import halo_model_utility as hmu
import halomod.concentration as concentration_classes
import halomod.profiles as profile_classes
import numpy as np
import time
import warnings
from astropy.cosmology import Flatw0waCDM
from cosmosis.datablock import names, option_section
from halomod.concentration import interp_concentration, make_colossus_cm
from halomod.halo_model import DMHaloModel

import hmf
from onepower.hmi import HaloModelIngredients

# Silencing a warning from hmf for which the nonlinear mass is still correctly calculated
warnings.filterwarnings("ignore", message="Nonlinear mass outside mass range")

# Cosmological parameters section name in block
cosmo_params = names.cosmological_parameters

def setup(options):

    config = {}
    # Log10 Minimum, Maximum and number of log10 mass bins for halo masses: M_halo
    # Units are in log10(M_sun h^-1)
    config['log_mass_min'] = options[option_section, 'log_mass_min']
    config['log_mass_max'] = options[option_section, 'log_mass_max']
    nmass = options[option_section, 'nmass']
    config['dlog10m'] = ( config['log_mass_max'] - config['log_mass_min']) / nmass

    # Minimum and Maximum redshift and number of redshift bins for calculating the ingredients
    zmin = options[option_section, 'zmin']
    zmax = options[option_section, 'zmax']
    nz = options[option_section, 'nz']
    config['z_vec'] = np.linspace(zmin, zmax, nz)
    # If this is smaller than nz then downsample for concentration to speed it up.
    # The concentration function is slow!
    # nz_conc = options.get_int(option_section, 'nz_conc', default=5)

    # Model choices
    config['nk'] = options[option_section, 'nk']
    config['profile'] = options.get_string(option_section, 'profile', default='NFW')
    config['profile_value_name'] = options.get_string(option_section, 'profile_value_name', default='profile_parameters')
    config['hmf_model'] = options.get_string(option_section, 'hmf_model')
    config['mdef_model'] = options.get_string(option_section, 'mdef_model')
    config['overdensity'] = options[option_section, 'overdensity']
    config['cm_model'] = options.get_string(option_section, 'cm_model')
    config['delta_c'] = options[option_section, 'delta_c']
    config['bias_model'] = options.get_string(option_section, 'bias_model')

    # Option to set similar corrections to HMcode2020
    use_mead = options.get_string(option_section, 'use_mead2020_corrections', default='None')

    # Mapping of use_mead values to mead_correction values
    mead_correction_map = {
        'mead2020': 'nofeedback',
        'mead2020_feedback': 'feedback',
        # Add more mappings here if needed
    }

    # Determine the mead_correction based on the mapping
    config['mead_correction'] = mead_correction_map.get(use_mead, None)

    config['lnk_min'] = -18.0
    config['lnk_max'] = 18.0
    config['dlnk'] = 0.001

    return config

def execute(block, config):


    # TODO: will the inputs depend on the profile model?
    norm_cen = block[config['profile_value_name'], 'norm_cen']
    norm_sat = block[config['profile_value_name'], 'norm_sat']
    eta_cen = block[config['profile_value_name'], 'eta_cen']
    eta_sat = block[config['profile_value_name'], 'eta_sat']

    # Get the k range from the linear matter power spectrum section.
    # But use the nk that was given as input
    k_vec_original = block['matter_power_lin', 'k_h']
    k_vec = np.logspace(np.log10(k_vec_original[0]), np.log10(k_vec_original[-1]), num=config['nk'])

    # Power spectrum transfer function used to update the transfer function in hmf
    transfer_k = block['matter_power_transfer_func', 'k_h']
    transfer_func = block['matter_power_transfer_func', 't_k']
    growth_z = block['growth_parameters', 'z']
    growth_func = block['growth_parameters', 'd_z']

    z_vec = config['z_vec']

    hmf = HaloModelIngredients(
        k_vec = k_vec,
        z_vec = z_vec,
        lnk_min = config['lnk_min'],
        lnk_max = config['lnk_max'],
        dlnk = config['dlnk'],
        Mmin = config['log_mass_min'],
        Mmax = config['log_mass_max'],
        dlog10m = config['dlog10m'],
        mdef_model = config['mdef_model'],
        hmf_model = config['hmf_model'],
        bias_model = config['bias_model'],
        halo_profile_model_dm = config['profile'],
        halo_concentration_model_dm = config['cm_model'],
        halo_profile_model_sat = config['profile'],
        halo_concentration_model_sat = config['cm_model'],
        transfer_model = 'FromArray',
        transfer_params = {'k': transfer_k, 'T': transfer_func},
        growth_model = 'FromArray',
        growth_params = {'z': growth_z, 'd': growth_func},
        h0 = block[cosmo_params, 'h0'],
        omega_c = block[cosmo_params, 'omega_c'],
        omega_b = block[cosmo_params, 'omega_b'],
        omega_m = block[cosmo_params, 'omega_m'],
        w0 = block[cosmo_params, 'w'],
        wa = block[cosmo_params, 'wa'],
        n_s = block[cosmo_params, 'n_s'],
        tcmb = block.get_double(cosmo_params, 'TCMB', default=2.7255),
        m_nu = block[cosmo_params, 'mnu'],
        sigma_8 = block[cosmo_params, 'sigma_8'],
        log10T_AGN = block['halo_model_parameters', 'logT_AGN'],
        norm_cen = norm_cen,
        norm_sat = norm_sat,
        eta_cen = eta_cen,
        eta_sat = eta_sat,
        overdensity = config['overdensity'],
        delta_c = config['delta_c'],
        mead_correction = config['mead_correction']
    )

    mass = hmf.mass

    u_dm_cen = hmf.u_dm
    u_dm_sat = hmf.u_sat
    mean_density0 = hmf.mean_density0
    mean_density_z = hmf.mean_density_z
    rho_crit = hmf.mean_density0 / block[cosmo_params, 'omega_m']
    rho_halo = hmf.rho_halo

    dndlnm = hmf.dndlnm
    halo_bias = hmf.halo_bias
    nu = hmf.nu
    neff = hmf.neff
    sigma8_z = hmf.sigma8_z
    fnu = hmf.fnu

    conc_cen = hmf.conc_cen
    conc_sat = hmf.conc_sat
    r_s_cen = hmf.r_s_cen
    r_s_sat = hmf.r_s_sat

    rvir_cen = hmf.rvir_cen
    rvir_sat = hmf.rvir_sat

    # TODO: Clean these up. Put more of them into the same folder
    block.put_grid('concentration_m', 'z', z_vec, 'm_h', mass, 'c', conc_cen)
    block.put_grid('concentration_sat', 'z', z_vec, 'm_h', mass, 'c', conc_sat)
    block.put_grid('nfw_scale_radius_m', 'z', z_vec, 'm_h', mass, 'rs', r_s_cen)
    block.put_grid('nfw_scale_radius_sat', 'z', z_vec, 'm_h', mass, 'rs', r_s_sat)

    block.put_double_array_1d('virial_radius', 'm_h', mass)
    # rvir doesn't change with z, hence no z-dimension
    block.put_double_array_1d('virial_radius', 'rvir_m', rvir_cen[0])
    block.put_double_array_1d('virial_radius', 'rvir_sat', rvir_sat[0])

    block.put_double_array_1d('fourier_nfw_profile', 'z', z_vec)
    block.put_double_array_1d('fourier_nfw_profile', 'm_h', mass)
    block.put_double_array_1d('fourier_nfw_profile', 'k_h', k_vec)
    block.put_double_array_nd('fourier_nfw_profile', 'ukm', u_dm_cen)
    block.put_double_array_nd('fourier_nfw_profile', 'uksat', u_dm_sat)


    # Density
    block['density', 'mean_density0'] = mean_density0
    block['density', 'rho_crit'] = rho_crit
    block.put_double_array_1d('density', 'mean_density_z', mean_density_z)
    block.put_double_array_1d('density', 'rho_halo', rho_halo)
    block.put_double_array_1d('density', 'z', z_vec)

    # Halo mass function
    block.put_grid('hmf', 'z', z_vec, 'm_h', mass, 'dndlnmh', dndlnm)
    block.put_grid('hmf', 'z', z_vec, 'm_h', mass, 'nu', nu)
    block.put_double_array_1d('hmf', 'neff', neff)
    block.put_double_array_1d('hmf', 'sigma8_z', sigma8_z)

    # Linear halo bias
    block.put_grid('halobias', 'z', z_vec, 'm_h', mass, 'b_hb', halo_bias)

    # Fraction of neutrinos to total matter, f_nu = Ω_nu /Ω_m
    block[cosmo_params, 'fnu'] = fnu

    return 0

def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
