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

# Silencing a warning from hmf for which the nonlinear mass is still correctly calculated
warnings.filterwarnings("ignore", message="Nonlinear mass outside mass range")

# Cosmological parameters section name in block
cosmo_params = names.cosmological_parameters

def setup(options):
    # Log10 Minimum, Maximum and number of log10 mass bins for halo masses: M_halo
    # Units are in log10(M_sun h^-1)
    log_mass_min = options[option_section, 'log_mass_min']
    log_mass_max = options[option_section, 'log_mass_max']
    nmass = options[option_section, 'nmass']
    dlog10m = (log_mass_max - log_mass_min) / nmass

    # Minimum and Maximum redshift and number of redshift bins for calculating the ingredients
    zmin = options[option_section, 'zmin']
    zmax = options[option_section, 'zmax']
    nz = options[option_section, 'nz']
    z_vec = np.linspace(zmin, zmax, nz)
    # If this is smaller than nz then downsample for concentration to speed it up.
    # The concentration function is slow!
    # nz_conc = options.get_int(option_section, 'nz_conc', default=5)

    # Model choices
    nk = options[option_section, 'nk']
    profile = options.get_string(option_section, 'profile', default='NFW')
    profile_value_name = options.get_string(option_section, 'profile_value_name', default='profile_parameters')
    hmf_model = options.get_string(option_section, 'hmf_model')
    mdef_model = options.get_string(option_section, 'mdef_model')
    overdensity = options[option_section, 'overdensity']
    cm_model = options.get_string(option_section, 'cm_model')
    delta_c = options[option_section, 'delta_c']
    bias_model = options.get_string(option_section, 'bias_model')
    mdef_params = {} if mdef_model == 'SOVirial' else {'overdensity': overdensity}

    # Option to set similar corrections to HMcode2020
    use_mead = options.get_string(option_section, 'use_mead2020_corrections', default='None')

    # Mapping of use_mead values to mead_correction values
    mead_correction_map = {
        'mead2020': 'nofeedback',
        'mead2020_feedback': 'feedback',
        # Add more mappings here if needed
    }

    # Determine the mead_correction based on the mapping
    mead_correction = mead_correction_map.get(use_mead, None)

    # If Mead correction is applied, set the ingredients to match Mead et al. (2021)
    if mead_correction is not None:
        hmf_model = 'ST'
        bias_model = 'ST99'
        mdef_model = 'SOVirial'
        mdef_params = {}
        cm_func = 'Duffy08'  # Dummy cm model, correct one calculated in execute
        disable_mass_conversion = True
    if mead_correction is None:
        try:
            cm_func = interp_concentration(getattr(concentration_classes, cm_model))
        except:
            cm_func = interp_concentration(make_colossus_cm(cm_model))
        disable_mass_conversion = False

    # Initialize cosmology model
    initialise_cosmo = Flatw0waCDM(H0=100.0, Ob0=0.044, Om0=0.3, Tcmb0=2.7255, w0=-1., wa=0.)

    # Growth Factor from hmf
    # gf._GrowthFactor.supported_cosmos = (FlatLambdaCDM, Flatw0waCDM, LambdaCDM)

    # Halo Mass function from hmf
    DM_hmf = DMHaloModel(
        z=0.0,
        cosmo_model=initialise_cosmo,
        sigma_8=0.8,
        n=0.96,
        transfer_model='CAMB',
        growth_model='CambGrowth',
        lnk_min=-18.0,
        lnk_max=18.0,
        dlnk=0.001,
        Mmin=log_mass_min,
        Mmax=log_mass_max,
        dlog10m=dlog10m,
        hmf_model=hmf_model,
        mdef_model=mdef_model,
        mdef_params=mdef_params,
        delta_c=delta_c,
        disable_mass_conversion=disable_mass_conversion,
        bias_model=bias_model,
        halo_profile_model=profile,
        halo_concentration_model=cm_func
    )

    # Array of halo masses
    mass = DM_hmf.m
    DM_hmf.ERROR_ON_BAD_MDEF = False

    # Configuration dictionary
    config = {
        'z_vec': z_vec,
        'nz': nz,
        'mass': mass,
        'DM_hmf': DM_hmf,
        'mdef_model': mdef_model,
        'overdensity': overdensity,
        'delta_c': delta_c,
        'mead_correction': mead_correction,
        'nk': nk,
        'profile_value_name': profile_value_name
    }

    return config

def execute(block, config):
    # Read in the config as returned by setup
    z_vec = config['z_vec']
    nz = config['nz']
    mass = config['mass']
    DM_hmf = config['DM_hmf']
    mdef_model = config['mdef_model']
    overdensity = config['overdensity']
    delta_c = config['delta_c']
    mead_correction = config['mead_correction']
    nk = config['nk']
    profile_value_name = config['profile_value_name']

    # Astropy cosmology requires the CMB temperature as an input.
    # If it exists in the values file read it from there otherwise set to its default value
    tcmb = block.get_double(cosmo_params, 'TCMB', default=2.7255)

    # Update the cosmological parameters
    this_cosmo_run = Flatw0waCDM(
        H0=block[cosmo_params, 'hubble'],
        Ob0=block[cosmo_params, 'omega_b'],
        Om0=block[cosmo_params, 'omega_m'],
        m_nu=[0, 0, block[cosmo_params, 'mnu']],
        Tcmb0=tcmb,
        w0=block[cosmo_params, 'w'],
        wa=block[cosmo_params, 'wa']
    )
    ns = block[cosmo_params, 'n_s']
    sigma_8 = block[cosmo_params, 'sigma_8']

    # TODO: will the inputs depend on the profile model?
    norm_cen = block[profile_value_name, 'norm_cen']
    norm_sat = block[profile_value_name, 'norm_sat']
    eta_cen = block[profile_value_name, 'eta_cen']
    eta_sat = block[profile_value_name, 'eta_sat']

    # Initialize arrays
    nmass_hmf = len(mass)
    dndlnmh = np.empty([nz, nmass_hmf])
    nu = np.empty([nz, nmass_hmf])
    b_nu = np.empty([nz, nmass_hmf])
    rho_halo = np.empty([nz])
    neff = np.empty([nz])
    sigma8_z = np.empty([nz])
    f_nu = np.empty([nz])
    mean_density_z = np.empty([nz])
    halo_overdensity_mean = np.empty([nz])
    u_dm_cen = np.empty([nz, nk, nmass_hmf])
    u_dm_sat = np.empty([nz, nk, nmass_hmf])
    conc_cen = np.empty([nz, nmass_hmf])
    conc_sat = np.empty([nz, nmass_hmf])
    r_s_cen = np.empty([nz, nmass_hmf])
    r_s_sat = np.empty([nz, nmass_hmf])
    rvir_cen = np.empty([nz, nmass_hmf])
    rvir_sat = np.empty([nz, nmass_hmf])

    if mead_correction:
        growth = hmu.get_growth_interpolator(this_cosmo_run)
        # growth_LCDM = hmu.get_growth_interpolator(LCDMcosmo)

    # Get the k range from the linear matter power spectrum section.
    # But use the nk that was given as input
    k_vec_original = block['matter_power_lin', 'k_h']
    k = np.logspace(np.log10(k_vec_original[0]), np.log10(k_vec_original[-1]), num=nk)

    # Power spectrum transfer function used to update the transfer function in hmf
    transfer_k = block['matter_power_transfer_func', 'k_h']
    transfer_func = block['matter_power_transfer_func', 't_k']
    growth_z = block['growth_parameters', 'z']
    growth_func = block['growth_parameters', 'd_z']

    DM_hmf.update(
        cosmo_model=this_cosmo_run,
        sigma_8=sigma_8,
        n=ns,
        transfer_model='FromArray',
        transfer_params={'k': transfer_k, 'T': transfer_func},
        halo_profile_params={'cosmo': this_cosmo_run},
        growth_model='FromArray',
        growth_params={'z': growth_z, 'd': growth_func}
    )

    # Loop over a series of redshift values defined by z_vec = np.linspace(zmin, zmax, nz)
    for jz, z_iter in enumerate(z_vec):
        if mead_correction is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                # This disables the warning from hmf. hmf is just telling us what we know
                # hmf's internal way of calculating the overdensity and the collapse threshold are fixed.
                # When we use the mead correction we want to define the haloes using the virial definition.
                # To avoid conflicts we manually pass the overdensity and the collapse threshold,
                # but for that we need to set the mass definition to be "mean",
                # so that it is compared to the mean density of the Universe rather than critical density.
                # hmf warns us that the value is not a native definition for the given halo mass function,
                # but will interpolate between the known ones (this is happening when one uses Tinker hmf for instance).
                a = this_cosmo_run.scale_factor(z_iter)
                g = growth(a)
                G = hmu.get_accumulated_growth(a, growth)
                delta_c_z = hmu.dc_Mead(a, this_cosmo_run.Om(z_iter) + this_cosmo_run.Onu(z_iter),
                                        this_cosmo_run.Onu0 / (this_cosmo_run.Om0 + this_cosmo_run.Onu0), g, G)
                halo_overdensity_mead = hmu.Dv_Mead(a, this_cosmo_run.Om(z_iter) + this_cosmo_run.Onu(z_iter),
                                                    this_cosmo_run.Onu0 / (this_cosmo_run.Om0 + this_cosmo_run.Onu0), g, G)
                DM_hmf.update(
                    z=z_iter,
                    delta_c=delta_c_z,
                    mdef_model=hmu.SOVirial_Mead,
                    mdef_params={'overdensity': halo_overdensity_mead}
                )

                eta_cen = 0.1281 * DM_hmf.sigma8_z**(-0.3644)
                if mead_correction == 'nofeedback':
                    norm_cen = 5.196  # /3.85#1.0#(5.196/3.85) #0.85*1.299
                elif mead_correction == 'feedback':
                    theta_agn = block['halo_model_parameters', 'logT_AGN'] - 7.8
                    norm_cen = (5.196 / 4.0) * ((3.44 - 0.496 * theta_agn) * np.power(10.0, z_iter * (-0.0671 - 0.0371 * theta_agn)))

                zf = hmu.get_halo_collapse_redshifts(mass, z_iter, delta_c_z, growth, this_cosmo_run, DM_hmf)
                conc_cen[jz, :] = norm_cen * (1.0 + zf) / (1.0 + z_iter)
                conc_sat[jz, :] = norm_sat * (1.0 + zf) / (1.0 + z_iter)

                DM_hmf.update(halo_profile_params={'eta_bloat': eta_cen})
                nfw_cen = DM_hmf.halo_profile.u(k, DM_hmf.m, c=conc_cen[jz, :])
                u_dm_cen[jz, :, :] = nfw_cen / np.expand_dims(nfw_cen[0, :], 0)
                r_s_cen[jz, :] = DM_hmf.halo_profile._rs_from_m(DM_hmf.m)
                rvir_cen[jz, :] = DM_hmf.halo_profile.halo_mass_to_radius(DM_hmf.m)

                DM_hmf.update(halo_profile_params={'eta_bloat': eta_sat})
                nfw_sat = DM_hmf.halo_profile.u(k, DM_hmf.m, c=conc_sat[jz, :])
                u_dm_sat[jz, :, :] = nfw_sat / np.expand_dims(nfw_sat[0, :], 0)
                r_s_sat[jz, :] = DM_hmf.halo_profile._rs_from_m(DM_hmf.m)
                rvir_sat[jz, :] = DM_hmf.halo_profile.halo_mass_to_radius(DM_hmf.m)

        if mead_correction is None:
            delta_c_z = (
                (3.0 / 20.0) * (12.0 * np.pi) ** (2.0 / 3.0) * (1.0 + 0.0123 * np.log10(this_cosmo_run.Om(z_iter)))
                if mdef_model == 'SOVirial' else
                delta_c
            )

            DM_hmf.update(
                z=z_iter,
                delta_c=delta_c_z,
                halo_profile_params={'eta_bloat': eta_cen},
                halo_concentration_params={'norm': norm_cen}
            )

            conc_cen[jz, :] = DM_hmf.cmz_relation
            nfw_cen = DM_hmf.halo_profile.u(k, DM_hmf.m)
            u_dm_cen[jz, :, :] = nfw_cen / np.expand_dims(nfw_cen[0, :], 0)
            r_s_cen[jz, :] = DM_hmf.halo_profile._rs_from_m(DM_hmf.m)
            rvir_cen[jz, :] = DM_hmf.halo_profile.halo_mass_to_radius(DM_hmf.m)

            DM_hmf.update(halo_profile_params={'eta_bloat': eta_sat},
                          halo_concentration_params={'norm': norm_sat})
            conc_sat[jz, :] = DM_hmf.cmz_relation
            nfw_sat = DM_hmf.halo_profile.u(k, DM_hmf.m)
            u_dm_sat[jz, :, :] = nfw_sat / np.expand_dims(nfw_sat[0, :], 0)
            r_s_sat[jz, :] = DM_hmf.halo_profile._rs_from_m(DM_hmf.m)
            rvir_sat[jz, :] = DM_hmf.halo_profile.halo_mass_to_radius(DM_hmf.m)

        halo_overdensity_mean[jz] = DM_hmf.halo_overdensity_mean
        nu[jz, :] = DM_hmf.nu**0.5
        dndlnmh[jz, :] = DM_hmf.dndlnm
        mean_density0 = DM_hmf.mean_density0
        mean_density_z[jz] = DM_hmf.mean_density
        rho_halo[jz] = halo_overdensity_mean[jz] * DM_hmf.mean_density0
        b_nu[jz, :] = DM_hmf.halo_bias

        neff[jz] = DM_hmf.n_eff_at_collapse
        # Rnl = DM_hmf.filter.mass_to_radius(DM_hmf.mass_nonlinear, DM_hmf.mean_density0)
        # neff[jz] = -3.0 - 2.0*DM_hmf.normalised_filter.dlnss_dlnm(Rnl)

        # Only used for mead_corrections
        sigma8_z[jz] = DM_hmf.sigma8_z
        # pk_cold = DM_hmf.power * hmu.Tk_cold_ratio(DM_hmf.k, g, block[cosmo_params, 'ommh2'], block[cosmo_params, 'h0'], this_cosmo_run.Onu0/this_cosmo_run.Om0, this_cosmo_run.Neff, T_CMB=tcmb)**2.0
        # sigma8_z[jz] = hmu.sigmaR_cc(pk_cold, DM_hmf.k, 8.0)

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
    block.put_double_array_1d('fourier_nfw_profile', 'k_h', k)
    block.put_double_array_nd('fourier_nfw_profile', 'ukm', u_dm_cen)
    block.put_double_array_nd('fourier_nfw_profile', 'uksat', u_dm_sat)

    ###################################################################################################################

    # Density
    block['density', 'mean_density0'] = mean_density0
    block['density', 'rho_crit'] = mean_density0 / this_cosmo_run.Om0
    block.put_double_array_1d('density', 'mean_density_z', mean_density_z)
    block.put_double_array_1d('density', 'rho_halo', rho_halo)
    block.put_double_array_1d('density', 'z', z_vec)

    # Halo mass function
    block.put_grid('hmf', 'z', z_vec, 'm_h', mass, 'dndlnmh', dndlnmh)
    block.put_grid('hmf', 'z', z_vec, 'm_h', mass, 'nu', nu)
    block.put_double_array_1d('hmf', 'neff', neff)
    block.put_double_array_1d('hmf', 'sigma8_z', sigma8_z)

    # Linear halo bias
    block.put_grid('halobias', 'z', z_vec, 'm_h', mass, 'b_hb', b_nu)

    # Fraction of neutrinos to total matter, f_nu = Ω_nu /Ω_m
    f_nu = this_cosmo_run.Onu0 / this_cosmo_run.Om0
    block[cosmo_params, 'fnu'] = f_nu

    config['DM_hmf'] = DM_hmf

    return 0

def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
