from cosmosis.datablock import names, option_section
import numpy as np

from darkmatter_lib import radvir_from_mass, scale_radius, compute_u_dm

# We have a collection of commonly used pre-defined block section names.
# If none of the names here is relevant for your calculation you can use any
# string you want instead.

cosmo = names.cosmological_parameters


# --------------------------------------------------------------------------------#

def setup(options):
    # This function is called once per processor per chain.
    # It is a chance to read any fixed options from the configuration file,
    # load any data, or do any calculations that are fixed once.

    """
    log_mass_min = options[option_section, 'log_mass_min']
    log_mass_max = options[option_section, 'log_mass_max']
    nmass = options[option_section, 'nmass']
    # log-spaced mass in units of M_sun/h
    dlog10m = (log_mass_max-log_mass_min)/nmass
    mass = 10.0 ** np.arange(log_mass_min, log_mass_max, dlog10m) #To be consistent with hmf!
    
    zmin = options[option_section, 'zmin']
    zmax = options[option_section, 'zmax']
    nz = options[option_section, 'nz']
    nk = options[option_section, 'nk']
    z_vec = np.linspace(zmin, zmax, nz)
    print(z_vec)
    #model_cm = options[option_section, 'model']
    #mdef = options[option_section, 'mdef_model']
    #overdensity = options[option_section, 'overdensity']
    """
    nk = options[option_section, 'nk']
    if options.has_value(option_section, 'profile'):
        print('\'profile\' option not yet implemented, using the default NFW profile.')
        #profile = options[option_section, 'profile']
        profile = None
    else:
        profile = None
        
    check_mead = options.has_value('hmf_and_halo_bias', 'use_mead2020_corrections')
    if check_mead:
        use_mead = options['hmf_and_halo_bias', 'use_mead2020_corrections']
        if use_mead == 'mead2020':
            mead_correction = 'nofeedback'
        elif use_mead == 'mead2020_feedback':
            mead_correction = 'feedback'
        elif use_mead == 'fit':
            mead_correction = 'fit'
    else:
        mead_correction = None

    return nk, profile, mead_correction#z_vec, nz, nk, mass, nmass#, model_cm, mdef, overdensity


def execute(block, config):
    # This function is called every time you have a new sample of cosmological and other parameters.
    # It is the main workhorse of the code. The block contains the parameters and results of any
    # earlier modules, and the config is what we loaded earlier.

    #z, nz, nk, mass, nmass = config
    nk, profile, mead_correction = config

    z = block['hmf', 'z']
    rho_m = block['density', 'mean_density0']
    rho_mz = block['density', 'mean_density_z']
    mass = block['hmf', 'm_h']
    nu = block['hmf', 'nu']
    #print('rho_m = ', rho_m)

    # compute the virial radius and the scale radius associated with a halo of mass M
    rho_halo = block['density', 'rho_halo']#overdensity * rho_m # array 1d (size of rhom)

    norm_cen = block['nfw_halo', 'norm_cen']
    norm_sat = block['nfw_halo', 'norm_sat']
    eta_cen = block['nfw_halo', 'eta_cen']
    eta_sat = block['nfw_halo', 'eta_sat']

    if mead_correction == 'nofeedback':
        norm_cen = 1.0#(5.196/3.85)#0.85*1.299
        sigma_var = block['hmf', 'sigma_var']
        eta_cen = (0.1281 * sigma_var[:,np.newaxis]**(-0.3644))
    if mead_correction == 'feedback':
        theta_agn = block['halo_model_parameters', 'logT_AGN'] - 7.8
        norm_cen = (((3.44 - 0.496*theta_agn) * 10.0**(z*(-0.0671 - 0.0371*theta_agn))) / 4.0)[:,np.newaxis]
        eta_cen = (0.15 * (1.0+z)**0.5)[:,np.newaxis]
    
    conc_cen = norm_cen * block['concentration', 'c']
    conc_sat = norm_sat * block['concentration', 'c']
    rvir_cen = radvir_from_mass(mass, rho_halo)
    rvir_sat = radvir_from_mass(mass, rho_halo)
    
    r_s_cen = scale_radius(rvir_cen, conc_cen) * nu**eta_cen
    r_s_sat = scale_radius(rvir_sat, conc_sat) * nu**eta_sat

    # compute the Fourier-transform of the NFW profile (normalised to the mass of the halo)
    #k = np.logspace(-2,1,200) # AD: inherit the range from Plin? That would avoid intepolations ...
    k_vec_original = block['matter_power_lin', 'k_h']
    k = np.logspace(np.log10(k_vec_original[0]), np.log10(k_vec_original[-1]), num=nk)
    u_dm_cen = compute_u_dm(k, r_s_cen, conc_cen, mass)
    u_dm_sat = compute_u_dm(k, r_s_sat, conc_sat, mass)

    block.put_grid('concentration_dm', 'z', z, 'm_h', mass, 'c', conc_cen)
    block.put_grid('concentration_sat', 'z', z, 'm_h', mass, 'c', conc_sat)
    block.put_grid('nfw_scale_radius_dm', 'z', z, 'm_h', mass, 'rs', r_s_cen)
    block.put_grid('nfw_scale_radius_sat', 'z', z, 'm_h', mass, 'rs', r_s_sat)
    #block.put_grid('virial_radius', 'z', z, 'm_h', mass, 'rvir', rvir)
    #print(rvir[0].shape)
    block.put_double_array_1d('virial_radius', 'm_h', mass)
    block.put_double_array_1d('virial_radius', 'rvir_dm', rvir_cen[0])
    block.put_double_array_1d('virial_radius', 'rvir_sat', rvir_sat[0])


    block.put_double_array_1d('fourier_nfw_profile', 'z', z)
    block.put_double_array_1d('fourier_nfw_profile', 'm_h', mass)
    block.put_double_array_1d('fourier_nfw_profile', 'k_h', k)
    block.put_double_array_nd('fourier_nfw_profile', 'ukm', u_dm_cen)
    block.put_double_array_nd('fourier_nfw_profile', 'uksat', u_dm_sat)
    
    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
