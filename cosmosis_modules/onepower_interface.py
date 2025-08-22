import numbers
import numpy as np
from cosmosis.datablock import names, option_section
from scipy.interpolate import interp1d

from onepower.add import UpsampledSpectra
from onepower.bnl import NonLinearBias
from onepower.pk import Spectra

cosmo_params = names.cosmological_parameters

parameters_models = {
    'Zheng': ['log10_Mmin', 'sigma', 'log10_M0', 'log10_M1', 'alpha'],
    'Zhai': ['log10_Mmin', 'sigma', 'log10_Msat', 'log10_Mcut', 'alpha'],
    'Cacciato': [
        'log10_obs_norm_c',
        'log10_m_ch',
        'g1',
        'g2',
        'sigma_log10_O_c',
        'norm_s',
        'pivot',
        'alpha_s',
        'beta_s',
        'b0',
        'b1',
        'b2',
    ],
}

poisson_parameters = {
    'constant': ['poisson'],
    'power_law': ['poisson', 'pivot', 'slope'],
}

# Mapping of use_mead values to mead_correction values
mead_correction_map = {
    'mead2020': 'nofeedback',
    'mead2020_feedback': 'feedback',
    'fit_feedback': 'fit',
    # Add more mappings here if needed
}


def get_string_or_none(cosmosis_block, section, name, default):
    """
    A helper function to return a number or None explicitly from config files
    or return None if no value is present.
    """
    if cosmosis_block.has_value(section, name):
        test_param = cosmosis_block.get(section, name)
        if isinstance(test_param, numbers.Number):
            param = cosmosis_block.get_double(section, name, default)
        if isinstance(test_param, str):
            str_in = cosmosis_block.get_string(section, name)
            if str_in == 'None':
                param = None
    else:
        try:
            param = cosmosis_block.get_double(section, name, default)
        except ValueError:
            param = None

    if not isinstance(param, (numbers.Number | type(None))):
        raise ValueError(
            f'Parameter {name} is not an instance of a number or NoneType!'
        )

    return param


def interpolate_in_z(input_grid, z_in, z_out, axis=0):
    """
    Interpolation in redshift
    Default redshift axis is the first one.
    """
    f_interp = interp1d(z_in, input_grid, axis=axis)
    return f_interp(z_out)


def log_linear_interpolation_k(power_in, k_in, k_out, axis=1, kind='linear'):
    """
    log-linear interpolation for power spectra. This works well for extrapolating to higher k.
    Ideally we want to have a different routine for interpolation (spline) and extrapolation (log-linear)
    """
    power_interp = interp1d(
        np.log(k_in), np.log(power_in), axis=axis, kind=kind, fill_value='extrapolate'
    )
    return np.exp(power_interp(np.log(k_out)))


def get_linear_power_spectrum(block, z_vec):
    """
    Reads in linear matter power spectrum and downsamples
    """
    k_vec = block['matter_power_lin', 'k_h']
    z_pl = block['matter_power_lin', 'z']
    matter_power_lin = block['matter_power_lin', 'p_k']
    return k_vec, interpolate_in_z(matter_power_lin, z_pl, z_vec)


def get_nonlinear_power_spectrum(block, z_vec):
    """
    Reads in the non-linear matter power specturm and downsamples
    """
    k_nl = block['matter_power_nl', 'k_h']
    z_nl = block['matter_power_nl', 'z']
    matter_power_nl = block['matter_power_nl', 'p_k']
    return k_nl, interpolate_in_z(matter_power_nl, z_nl, z_vec)


def setup_hod_settings(options, suffix):
    """Setup HOD settings based on options."""
    hod_settings = {}
    if options.has_value(option_section, f'observables_file_{suffix}'):
        hod_settings['observables_file'] = options.get_string(
            option_section, f'observables_file_{suffix}'
        )
    else:
        hod_settings['observables_file'] = None
        hod_settings['obs_min'] = np.asarray(
            [options[option_section, f'log10_obs_min_{suffix}']]
        ).flatten()
        hod_settings['obs_max'] = np.asarray(
            [options[option_section, f'log10_obs_max_{suffix}']]
        ).flatten()
        hod_settings['zmin'] = np.asarray(
            [options[option_section, f'zmin_{suffix}']]
        ).flatten()
        hod_settings['zmax'] = np.asarray(
            [options[option_section, f'zmax_{suffix}']]
        ).flatten()
        hod_settings['nz'] = options[option_section, f'nz_{suffix}']
    hod_settings['nobs'] = options[option_section, f'nobs_{suffix}']
    hod_settings['observable_h_unit'] = options.get_string(
        option_section, 'observable_h_unit', default='1/h^2'
    ).lower()
    return hod_settings


def setup_hod_settings_mm(hod_settings):
    """Setup matter-matter HOD settings based on options."""
    hod_settings_mm = hod_settings.copy()
    if hod_settings_mm['observables_file'] is None:
        hod_settings_mm['obs_min'] = np.array([hod_settings['obs_min'].min()])
        hod_settings_mm['obs_max'] = np.array([hod_settings['obs_max'].max()])
        hod_settings_mm['zmin'] = np.array([hod_settings['zmin'].min()])
        hod_settings_mm['zmax'] = np.array([hod_settings['zmax'].max()])
        hod_settings_mm['nz'] = 15
    hod_settings_mm['nobs'] = 100
    return hod_settings_mm


def setup_hmf_config(options):
    """Setup function to parse options and return configuration for hmf."""
    config_hmf = {
        'log_mass_min': options.get_double(option_section, 'log_mass_min', default=9.0),
        'log_mass_max': options.get_double(
            option_section, 'log_mass_max', default=16.0
        ),
        'dlog10m': (
            options.get_double(option_section, 'log_mass_max', default=16.0)
            - options.get_double(option_section, 'log_mass_min', default=9.0)
        )
        / options.get_int(option_section, 'nmass', default=200),
        'z_vec': np.linspace(
            options.get_double(option_section, 'zmin_hmf'),
            options.get_double(option_section, 'zmax_hmf'),
            options.get_int(option_section, 'nz_hmf', default=15),
        ),
        'nk': options.get_int(option_section, 'nk', default=100),
        'profile_cen': options.get_string(option_section, 'profile_cen', default='NFW'),
        'profile_sat': options.get_string(option_section, 'profile_sat', default='NFW'),
        'profile_value_name': options.get_string(
            option_section, 'profile_value_name', default='profile_parameters'
        ),
        'hmf_model': options.get_string(
            option_section, 'hmf_model', default='Tinker10'
        ),
        'mdef_model': options.get_string(
            option_section, 'mdef_model', default='SOMean'
        ),
        'overdensity': options.get_double(option_section, 'overdensity', default=200.0),
        'cm_model_cen': options.get_string(
            option_section, 'cm_model_cen', default='Duffy08'
        ),
        'cm_model_sat': options.get_string(
            option_section, 'cm_model_sat', default='Duffy08'
        ),
        'delta_c': options.get_double(option_section, 'delta_c', default=1.686),
        'bias_model': options.get_string(
            option_section, 'bias_model', default='Tinker10'
        ),
        'lnk_min': np.log(1e-8),
        'lnk_max': np.log(1e8),
        'dlnk': 0.05,  # 0.001
    }
    return config_hmf


def setup_pipeline_parameters(options):
    """Setup pipeline parameters."""
    p_mm = options.get_bool(option_section, 'p_mm', default=False)
    p_gg = options.get_bool(option_section, 'p_gg', default=False)
    p_gm = options.get_bool(option_section, 'p_gm', default=False)
    p_gI = options.get_bool(option_section, 'p_gI', default=False)
    p_mI = options.get_bool(option_section, 'p_mI', default=False)
    p_II = options.get_bool(option_section, 'p_II', default=False)
    # If True, calculate the response of the halo model for the requested power spectra compared to matter power
    # multiplies this to input non-linear matter power spectra.
    response = options.get_bool(option_section, 'response', default=False)
    # If true, use the IA formalism of Fortuna et al. 2021: Truncated NLA at high k + 1-halo term
    fortuna = options.get_bool(option_section, 'fortuna', default=False)
    # If True, uses beta_nl
    bnl = options.get_bool(option_section, 'bnl', default=False)
    split_ia = options.get_bool(option_section, 'split_ia', default=False)

    matter = p_mm
    galaxy = p_gg or p_gm
    alignment = p_gI or p_mI or p_II

    # Fortuna introduces a truncation of the 1-halo term at large scales to avoid the halo exclusion problem
    # and a truncation of the NLA 2-halo term at small scales to avoid double-counting of the 1-halo term
    # The user can change these values.
    one_halo_ktrunc_ia = get_string_or_none(
        options, option_section, 'one_halo_ktrunc_ia', default=4.0
    )  # h/Mpc or None
    two_halo_ktrunc_ia = get_string_or_none(
        options, option_section, 'two_halo_ktrunc_ia', default=6.0
    )  # h/Mpc or None
    # General truncation of non-IA terms:
    one_halo_ktrunc = get_string_or_none(
        options, option_section, 'one_halo_ktrunc', default=0.1
    )  # h/Mpc or None
    two_halo_ktrunc = get_string_or_none(
        options, option_section, 'two_halo_ktrunc', default=2.0
    )  # h/Mpc or None

    # Additional parameters
    mead_correction = mead_correction_map.get(
        options.get_string(option_section, 'mead2020_correction', default='None'), None
    )
    dewiggle = options.get_bool(option_section, 'dewiggle', default=False)
    point_mass = options.get_bool(option_section, 'point_mass', default=False)
    poisson_type = options.get_string(option_section, 'poisson_type', default='')

    return (
        p_mm,
        p_gg,
        p_gm,
        p_gI,
        p_mI,
        p_II,
        response,
        fortuna,
        matter,
        galaxy,
        bnl,
        alignment,
        split_ia,
        one_halo_ktrunc,
        two_halo_ktrunc,
        one_halo_ktrunc_ia,
        two_halo_ktrunc_ia,
        mead_correction,
        dewiggle,
        point_mass,
        poisson_type,
    )


def setup_hod(options, alignment, split_ia):
    hod_section_name = options.get_string(option_section, 'hod_section_name')
    hod_values_name = options.get_string(
        option_section, 'values_name', default='hod_parameters'
    ).lower()

    population_name = options.get_string(
        option_section, 'output_suffix', default=''
    ).lower()
    pop_name = f'_{population_name}' if population_name else ''

    if alignment:
        central_IA = options.get_string(
            option_section, 'central_IA_depends_on', default='halo_mass'
        )
        satellite_IA = options.get_string(
            option_section, 'satellite_IA_depends_on', default='halo_mass'
        )
        if split_ia:
            hod_section_name_ia_1 = options.get_string(
                option_section, 'hod_section_name_ia_1'
            )
            population_name_ia_1 = options.get_string(
                option_section, 'output_suffix_ia_1', default=''
            ).lower()
            pop_name_ia_1 = f'_{population_name_ia_1}' if population_name_ia_1 else ''

            hod_section_name_ia_2 = options.get_string(
                option_section, 'hod_section_name_ia_2'
            )
            population_name_ia_2 = options.get_string(
                option_section, 'output_suffix_ia_2', default=''
            ).lower()
            pop_name_ia_2 = f'_{population_name_ia_2}' if population_name_ia_2 else ''
        else:
            hod_section_name_ia_1 = options.get_string(
                option_section, 'hod_section_name_ia'
            )
            population_name_ia_1 = options.get_string(
                option_section, 'output_suffix_ia', default=''
            ).lower()
            pop_name_ia_1 = f'_{population_name_ia_1}' if population_name_ia_1 else ''
            hod_section_name_ia_2 = None
            pop_name_ia_2 = None
    else:
        hod_section_name_ia_1 = None
        pop_name_ia_1 = ''
        hod_section_name_ia_2 = None
        pop_name_ia_2 = ''
        central_IA = None
        satellite_IA = None

    hod_model = options.get_string(option_section, 'hod_model', default='Cacciato')
    hod_settings = setup_hod_settings(options, 'hod')
    hod_settings_mm = setup_hod_settings_mm(hod_settings)
    nbins = len(hod_settings['obs_min'])
    if alignment:
        if split_ia:
            hod_settings_ia_1 = setup_hod_settings(options, 'ia_1')
            hod_settings_ia_2 = setup_hod_settings(options, 'ia_2')
        else:
            hod_settings_ia_1 = setup_hod_settings(options, 'ia')
            hod_settings_ia_2 = {}
    else:
        hod_settings_ia_1 = {}
        hod_settings_ia_2 = {}
    if options.get_bool(option_section, 'save_observable', default=True):
        obs_settings = setup_hod_settings(options, 'smf')
        obs_settings['save_observable'] = options.get_bool(
            option_section, 'save_observable', default=True
        )
        obs_settings['observable_section_name'] = options.get_string(
            option_section, 'observable_section_name', default='stellar_mass_function'
        ).lower()
    else:
        obs_settings = {}

    return (
        hod_section_name,
        hod_values_name,
        pop_name,
        hod_model,
        hod_settings,
        hod_settings_mm,
        obs_settings,
        nbins,
        hod_settings_ia_1,
        hod_settings_ia_2,
        hod_section_name_ia_1,
        hod_section_name_ia_2,
        pop_name_ia_1,
        pop_name_ia_2,
        central_IA,
        satellite_IA,
    )


def save_matter_results(block, power, z_vec, k_vec):
    """Save matter results to the block."""
    mass = power.mass
    mean_density0 = power.mean_density0
    mean_density_z = power.mean_density_z
    rho_crit = power.mean_density0 / block[cosmo_params, 'omega_m']
    rho_halo = power.rho_halo

    dndlnm = power.dndlnm
    halo_bias = power.halo_bias
    nu = power.nu
    neff = power.neff
    sigma8_z = power.sigma8_z
    fnu = power.fnu

    u_dm_cen = power.u_dm
    u_dm_sat = power.u_sat

    conc_cen = power.conc_cen
    conc_sat = power.conc_sat
    r_s_cen = power.r_s_cen
    r_s_sat = power.r_s_sat

    rvir_cen = power.rvir_cen
    rvir_sat = power.rvir_sat

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


def save_hod_results(block, power, z_vec, hod_section_name, hod_settings):
    """Save HOD results to the block."""
    mass = power.mass
    N_cen = power.hod.hod_cen
    N_sat = power.hod.hod_sat
    N_tot = power.hod.hod
    numdens_cen = power.hod.number_density_cen
    numdens_sat = power.hod.number_density_sat
    numdens_tot = power.hod.number_density
    fraction_c = power.hod.f_c
    fraction_s = power.hod.f_s
    mass_avg = power.mass_avg
    f_star = power.fstar

    hod_bins = N_cen.shape[0]
    block.put_int(hod_section_name, 'nbins', hod_bins)
    for nb in range(hod_bins):
        suffix = f'_{nb + 1}' if hod_bins != 1 else ''
        block.put_grid(
            hod_section_name,
            f'z{suffix}',
            z_vec,
            f'mass{suffix}',
            mass,
            f'N_sat{suffix}',
            N_sat[nb],
        )
        block.put_grid(
            hod_section_name,
            f'z{suffix}',
            z_vec,
            f'mass{suffix}',
            mass,
            f'N_cen{suffix}',
            N_cen[nb],
        )
        block.put_grid(
            hod_section_name,
            f'z{suffix}',
            z_vec,
            f'mass{suffix}',
            mass,
            f'N_tot{suffix}',
            N_tot[nb],
        )
        block.put_grid(
            hod_section_name,
            f'z{suffix}',
            z_vec,
            f'mass{suffix}',
            mass,
            f'f_star{suffix}',
            f_star[nb],
        )
        block.put_double_array_1d(
            hod_section_name, f'number_density_cen{suffix}', numdens_cen[nb]
        )
        block.put_double_array_1d(
            hod_section_name, f'number_density_sat{suffix}', numdens_sat[nb]
        )
        block.put_double_array_1d(
            hod_section_name, f'number_density_tot{suffix}', numdens_tot[nb]
        )
        block.put_double_array_1d(
            hod_section_name, f'central_fraction{suffix}', fraction_c[nb]
        )
        block.put_double_array_1d(
            hod_section_name, f'satellite_fraction{suffix}', fraction_s[nb]
        )
        block.put_double_array_1d(
            hod_section_name, f'average_halo_mass{suffix}', mass_avg[nb]
        )
    return hod_bins


def save_obs_results(block, power, observable_section_name, obs_settings):
    mass = power.mass
    obs_func = power.obs_func
    obs_func_c = power.obs_func_cen
    obs_func_s = power.obs_func_sat
    obs_z = power.obs_func_z
    obs_range = power.obs_func_obs
    obs_bins = obs_range.shape[0]

    block.put(observable_section_name, 'obs_func_definition', 'obs_func * obs * ln(10)')
    for nb in range(obs_bins):
        suffix_obs = f'_{nb + 1}'
        if np.all(
            np.array(
                [
                    obs_settings['obs_min'].size,
                    obs_settings['obs_max'].size,
                    obs_settings['zmin'].size,
                    obs_settings['zmax'].size,
                    obs_settings['nz'],
                ]
            )
            == 1
        ):
            block.put_double_array_1d(
                observable_section_name, f'obs_val{suffix_obs}', np.squeeze(obs_range)
            )
            block.put_double_array_1d(
                observable_section_name, f'obs_func{suffix_obs}', np.squeeze(obs_func)
            )
            block.put_double_array_1d(
                observable_section_name,
                f'obs_func_c{suffix_obs}',
                np.squeeze(obs_func_c),
            )
            block.put_double_array_1d(
                observable_section_name,
                f'obs_func_s{suffix_obs}',
                np.squeeze(obs_func_s),
            )
        else:
            block.put_grid(
                observable_section_name,
                f'z_bin{suffix_obs}',
                obs_z[nb],
                f'obs_val{suffix_obs}',
                obs_range[nb, 0, :],
                f'obs_func{suffix_obs}',
                obs_func[nb],
            )
            block.put_grid(
                observable_section_name,
                f'z_bin{suffix_obs}',
                obs_z[nb],
                f'obs_val{suffix_obs}',
                obs_range[nb, 0, :],
                f'obs_func_c{suffix_obs}',
                obs_func_c[nb],
            )
            block.put_grid(
                observable_section_name,
                f'z_bin{suffix_obs}',
                obs_z[nb],
                f'obs_val{suffix_obs}',
                obs_range[nb, 0, :],
                f'obs_func_s{suffix_obs}',
                obs_func_s[nb],
            )


def save_pk_to_grid(block, z_vec, k_vec, base_name, suffix, pk_1h, pk_2h, pk_tot):
    """Save P(k) to the block."""
    section_name = f'{base_name}{suffix}'
    block.put_grid(section_name, 'z', z_vec, 'k_h', k_vec, 'p_k_1h', pk_1h)
    block.put_grid(section_name, 'z', z_vec, 'k_h', k_vec, 'p_k_2h', pk_2h)
    block.put_grid(section_name, 'z', z_vec, 'k_h', k_vec, 'p_k', pk_tot)


def setup(options):
    (
        p_mm,
        p_gg,
        p_gm,
        p_gI,
        p_mI,
        p_II,
        response,
        fortuna,
        matter,
        galaxy,
        bnl,
        alignment,
        split_ia,
        one_halo_ktrunc,
        two_halo_ktrunc,
        one_halo_ktrunc_ia,
        two_halo_ktrunc_ia,
        mead_correction,
        dewiggle,
        point_mass,
        poisson_type,
    ) = setup_pipeline_parameters(options)

    (
        hod_section_name,
        hod_values_name,
        pop_name,
        hod_model,
        hod_settings,
        hod_settings_mm,
        obs_settings,
        nbins,
        hod_settings_ia_1,
        hod_settings_ia_2,
        hod_section_name_ia_1,
        hod_section_name_ia_2,
        pop_name_ia_1,
        pop_name_ia_2,
        central_IA,
        satellite_IA,
    ) = setup_hod(options, alignment, split_ia)

    # hmf config
    config_hmf = setup_hmf_config(options)

    if bnl:
        cached_bnl = {
            'num_calls': 0,
            'cached_bnl': None,
            'update_bnl': options[option_section, 'update_bnl'],
        }
    else:
        cached_bnl = None

    config_hmf['power'] = Spectra()

    return (
        p_mm,
        p_gg,
        p_gm,
        p_gI,
        p_mI,
        p_II,
        response,
        fortuna,
        matter,
        galaxy,
        bnl,
        alignment,
        split_ia,
        one_halo_ktrunc,
        two_halo_ktrunc,
        one_halo_ktrunc_ia,
        two_halo_ktrunc_ia,
        hod_section_name,
        mead_correction,
        dewiggle,
        point_mass,
        poisson_type,
        pop_name,
        hod_model,
        hod_settings,
        hod_settings_mm,
        obs_settings,
        hod_values_name,
        config_hmf,
        cached_bnl,
        central_IA,
        satellite_IA,
        nbins,
        hod_settings_ia_1,
        hod_settings_ia_2,
        hod_section_name_ia_1,
        hod_section_name_ia_2,
        pop_name_ia_1,
        pop_name_ia_2,
    )


def execute(block, config):
    """Execute function to compute power spectra based on configuration."""
    (
        p_mm,
        p_gg,
        p_gm,
        p_gI,
        p_mI,
        p_II,
        response,
        fortuna,
        matter,
        galaxy,
        bnl,
        alignment,
        split_ia,
        one_halo_ktrunc,
        two_halo_ktrunc,
        one_halo_ktrunc_ia,
        two_halo_ktrunc_ia,
        hod_section_name,
        mead_correction,
        dewiggle,
        point_mass,
        poisson_type,
        pop_name,
        hod_model,
        hod_settings,
        hod_settings_mm,
        obs_settings,
        hod_values_name,
        config_hmf,
        cached_bnl,
        central_IA,
        satellite_IA,
        nbins,
        hod_settings_ia_1,
        hod_settings_ia_2,
        hod_section_name_ia_1,
        hod_section_name_ia_2,
        pop_name_ia_1,
        pop_name_ia_2,
    ) = config

    hod_params = {}

    # Power spectrum transfer function used to update the transfer function in hmf
    transfer_k = block['matter_power_transfer_func', 'k_h']
    transfer_func = block['matter_power_transfer_func', 't_k']
    growth_z = block['growth_parameters', 'z']
    growth_func = block['growth_parameters', 'd_z']

    z_vec = config_hmf['z_vec']

    # Load the linear power spectrum and growth factor
    k_vec_original, plin_original = get_linear_power_spectrum(block, z_vec)
    k_vec = np.logspace(
        np.log10(k_vec_original[0]), np.log10(k_vec_original[-1]), num=config_hmf['nk']
    )  # , endpoint=False)
    z_vec_original = block.get_double_array_1d('matter_power_lin', 'z')

    # If hmf and input k_vec to match!
    # config_hmf['lnk_min'] = np.log(k_vec_original[0])
    # config_hmf['lnk_max'] = np.log(k_vec_original[-1])
    # config_hmf['dlnk'] = (config_hmf['lnk_max'] - config_hmf['lnk_min']) / config_hmf['nk']

    plin = log_linear_interpolation_k(plin_original, k_vec_original, k_vec)

    power_kwargs = {
        'matter_power_lin': plin,
        'mead_correction': mead_correction,
        'dewiggle': dewiggle,
        'k_vec': k_vec,
        'z_vec': config_hmf['z_vec'],
        'lnk_min': config_hmf['lnk_min'],
        'lnk_max': config_hmf['lnk_max'],
        'dlnk': config_hmf['dlnk'],
        'Mmin': config_hmf['log_mass_min'],
        'Mmax': config_hmf['log_mass_max'],
        'dlog10m': config_hmf['dlog10m'],
        'mdef_model': config_hmf['mdef_model'],
        'hmf_model': config_hmf['hmf_model'],
        'bias_model': config_hmf['bias_model'],
        'halo_profile_model_dm': config_hmf['profile_cen'],
        'halo_concentration_model_dm': config_hmf['cm_model_cen'],
        'halo_profile_model_sat': config_hmf['profile_sat'],
        'halo_concentration_model_sat': config_hmf['cm_model_sat'],
        'transfer_model': 'FromArray',
        'transfer_params': {'k': transfer_k, 'T': transfer_func},
        'growth_model': 'FromArray',
        'growth_params': {'z': growth_z, 'd': growth_func},
        'norm_cen': block[config_hmf['profile_value_name'], 'norm_cen'],
        'norm_sat': block[config_hmf['profile_value_name'], 'norm_sat'],
        'eta_cen': block[config_hmf['profile_value_name'], 'eta_cen'],
        'eta_sat': block[config_hmf['profile_value_name'], 'eta_sat'],
        'overdensity': config_hmf['overdensity'],
        'delta_c': config_hmf['delta_c'],
        'one_halo_ktrunc': one_halo_ktrunc,
        'two_halo_ktrunc': two_halo_ktrunc,
        'omega_c': block[cosmo_params, 'omega_c'],
        'omega_b': block[cosmo_params, 'omega_b'],
        'h0': block[cosmo_params, 'h0'],
        'n_s': block[cosmo_params, 'n_s'],
        'sigma_8': block[cosmo_params, 'sigma_8'],
        'm_nu': block[cosmo_params, 'mnu'],
        'w0': block[cosmo_params, 'w'],
        'wa': block[cosmo_params, 'wa'],
        'tcmb': block.get_double(cosmo_params, 'TCMB', default=2.7255),
        'log10T_AGN': block['halo_model_parameters', 'logT_AGN'],
        'mb': 10.0 ** block['halo_model_parameters', 'm_b'],
    }

    if bnl:
        num_calls = cached_bnl['num_calls']
        update_bnl = cached_bnl['update_bnl']

        if num_calls % update_bnl == 0:
            bnl = NonLinearBias(
                mass=10
                ** np.arange(
                    config_hmf['log_mass_min'],
                    config_hmf['log_mass_max'],
                    config_hmf['dlog10m'],
                ),
                z_vec=z_vec,
                k_vec=k_vec,
                h0=block[cosmo_params, 'h0'],
                A_s=block[cosmo_params, 'A_s'],
                omega_b=block[cosmo_params, 'omega_b'],
                omega_c=block[cosmo_params, 'omega_c'],
                omega_lambda=1.0 - block[cosmo_params, 'omega_m'],
                n_s=block[cosmo_params, 'n_s'],
                w0=block[cosmo_params, 'w'],
                z_dep=False,
            )

            beta_interp = bnl.bnl
            cached_bnl['cached_bnl'] = beta_interp
        else:
            beta_interp = cached_bnl['cached_bnl']

        cached_bnl['num_calls'] = num_calls + 1

        power_kwargs.update(
            {
                'bnl': bnl,
                'beta_nl': beta_interp,
            }
        )

    if response or fortuna:
        k_nl, p_nl = get_nonlinear_power_spectrum(block, z_vec)
        pk_mm_in = log_linear_interpolation_k(p_nl, k_nl, k_vec)
        power_kwargs.update({'matter_power_nl': pk_mm_in, 'response': response})
    else:
        pk_mm_in = None

    if galaxy or alignment:
        poisson_params = {}
        for param in poisson_parameters[poisson_type]:
            if not block.has_value('pk_parameters', param):
                raise Exception(
                    f'Error: parameter {param} is needed for the requested poisson model: {poisson_type}'
                )
            poisson_params[param] = get_string_or_none(
                block, 'pk_parameters', param, default=None
            )

        hod_params['A_cen'] = (
            block[hod_values_name, 'A_cen']
            if block.has_value(hod_values_name, 'A_cen')
            else None
        )
        hod_params['A_sat'] = (
            block[hod_values_name, 'A_sat']
            if block.has_value(hod_values_name, 'A_sat')
            else None
        )
        hod_parameters = parameters_models[hod_model]
        # Dinamically load required HOD parameters givent the model and number of bins!
        for param in hod_parameters:
            if hod_model == 'Cacciato':
                param_bin = param
                if not block.has_value(hod_values_name, param_bin):
                    raise Exception(
                        f'Error: parameter {param} is needed for the requested hod model: {hod_model}'
                    )
                hod_params[param] = block[hod_values_name, param_bin]
            else:
                param_list = []
                for nb in range(nbins):
                    suffix = f'_{nb + 1}' if nbins != 1 else ''
                    param_bin = f'{param}{suffix}'
                    if not block.has_value(hod_values_name, param_bin):
                        raise Exception(
                            f'Error: parameter {param} is needed for the requested hod model: {hod_model}'
                        )
                    param_list.append(block[hod_values_name, param_bin])
                hod_params[param] = np.array(param_list)

        power_kwargs.update(
            {
                'poisson_model': poisson_type,
                'poisson_params': poisson_params,
                'pointmass': point_mass,
                'hod_model': hod_model,
                'hod_params': hod_params,
                'hod_settings': hod_settings,
                'obs_settings': obs_settings,
                'compute_observable': obs_settings['save_observable'],
                'hod_settings_mm': hod_settings_mm,
            }
        )

    # TO-DO: for aligments at least we need to split the calculation in red/blue and add here!

    if alignment:
        align_params = {}
        if central_IA == 'halo_mass':
            align_params.update(
                {
                    'beta_sat': block[
                        f'intrinsic_alignment_parameters{pop_name_ia_1}', 'beta_sat'
                    ],
                    'mpivot_sat': block[
                        f'intrinsic_alignment_parameters{pop_name_ia_1}', 'M_pivot'
                    ],
                }
            )
        else:
            align_params.update(
                {
                    'beta_sat': None,
                    'mpivot_sat': None,
                }
            )
        if satellite_IA == 'halo_mass':
            align_params.update(
                {
                    'beta_cen': block[
                        f'intrinsic_alignment_parameters{pop_name_ia_1}', 'beta'
                    ],
                    'mpivot_cen': block[
                        f'intrinsic_alignment_parameters{pop_name_ia_1}', 'M_pivot'
                    ],
                }
            )
        else:
            align_params.update(
                {
                    'beta_cen': None,
                    'mpivot_cen': None,
                }
            )
        align_params.update(
            {
                'nmass': 5,
                'n_hankel': 350,
                'nk': 10,
                'ell_max': 6,
                'gamma_1h_slope': block[
                    f'intrinsic_alignment_parameters{pop_name_ia_1}',
                    'gamma_1h_radial_slope',
                ],
                'gamma_1h_amplitude': block[
                    f'intrinsic_alignment_parameters{pop_name_ia_1}',
                    'gamma_1h_amplitude',
                ],
                'gamma_2h_amplitude': block[
                    f'intrinsic_alignment_parameters{pop_name_ia_1}',
                    'gamma_2h_amplitude',
                ],
            }
        )
        power_kwargs.update(
            {
                'align_params': align_params,
                'one_halo_ktrunc_ia': one_halo_ktrunc_ia,
                'two_halo_ktrunc_ia': two_halo_ktrunc_ia,
                'fortuna': fortuna,
                't_eff': block.get_double(
                    'pk_parameters', 'linear_fraction_fortuna', default=0.0
                ),
            }
        )
        if split_ia:
            align_params_2 = align_params.copy()
            if central_IA == 'halo_mass':
                align_params_2.update(
                    {
                        'beta_sat': block[
                            f'intrinsic_alignment_parameters{pop_name_ia_2}', 'beta_sat'
                        ],
                        'mpivot_sat': block[
                            f'intrinsic_alignment_parameters{pop_name_ia_2}', 'M_pivot'
                        ],
                    }
                )
            else:
                align_params_2.update(
                    {
                        'beta_sat': None,
                        'mpivot_sat': None,
                    }
                )
            if satellite_IA == 'halo_mass':
                align_params_2.update(
                    {
                        'beta_cen': block[
                            f'intrinsic_alignment_parameters{pop_name_ia_2}', 'beta'
                        ],
                        'mpivot_cen': block[
                            f'intrinsic_alignment_parameters{pop_name_ia_2}', 'M_pivot'
                        ],
                    }
                )
            else:
                align_params_2.update(
                    {
                        'beta_cen': None,
                        'mpivot_cen': None,
                    }
                )
            align_params_2.update(
                {
                    'gamma_1h_slope': block[
                        f'intrinsic_alignment_parameters{pop_name_ia_2}',
                        'gamma_1h_radial_slope',
                    ],
                    'gamma_1h_amplitude': block[
                        f'intrinsic_alignment_parameters{pop_name_ia_2}',
                        'gamma_1h_amplitude',
                    ],
                    'gamma_2h_amplitude': block[
                        f'intrinsic_alignment_parameters{pop_name_ia_2}',
                        'gamma_2h_amplitude',
                    ],
                }
            )

    hod = not (matter and not galaxy)
    # power = Spectra(**power_kwargs)
    power = config_hmf['power']
    power.update(**power_kwargs)

    results = UpsampledSpectra(
        z=z_vec_original,
        k=k_vec_original,
        fraction_z=None,
        fraction=None,
        model=power,
        model_1_params=power_kwargs,
        model_2_params=None,
    )

    if matter:
        save_matter_results(block, power, z_vec, k_vec)

    if hod:
        hod_bins = save_hod_results(block, power, z_vec, hod_section_name, hod_settings)
        if power.obs_func is not None and obs_settings['save_observable']:
            save_obs_results(
                block, power, obs_settings['observable_section_name'], obs_settings
            )

    z_out = z_vec_original
    k_out = k_vec_original
    if p_mm or response:
        results.results(['mm'], ['1h', '2h', 'tot'])
        pk_mm_1h = results.power_spectrum_mm.pk_1h
        pk_mm_2h = results.power_spectrum_mm.pk_2h
        pk_mm = results.power_spectrum_mm.pk_tot
        if response:
            # Here we save the computed Pmm to datablock as matter_power_hm,
            # but not replacing the Pnl with it, as in the response
            # method, the Pnl stays the same as one from CAMB
            block.put_grid(
                'matter_power_hm', 'z', z_out, 'k_h', k_out, 'p_k_1h', pk_mm_1h[0]
            )
            block.put_grid(
                'matter_power_hm', 'z', z_out, 'k_h', k_out, 'p_k_2h', pk_mm_2h[0]
            )
            block.put_grid('matter_power_hm', 'z', z_out, 'k_h', k_out, 'p_k', pk_mm[0])
        else:
            block.put_grid(
                'matter_power_nl', 'z', z_out, 'k_h', k_out, 'p_k_1h', pk_mm_1h[0]
            )
            block.put_grid(
                'matter_power_nl', 'z', z_out, 'k_h', k_out, 'p_k_2h', pk_mm_2h[0]
            )
            block.put_grid('matter_power_nl', 'z', z_out, 'k_h', k_out, 'p_k', pk_mm[0])
    else:
        pk_mm = None

    if p_gg:
        results.results(['gg'], ['1h', '2h', 'tot'])
        pk_gg_1h = results.power_spectrum_gg.pk_1h
        pk_gg_2h = results.power_spectrum_gg.pk_2h
        pk_gg = results.power_spectrum_gg.pk_tot
        bg_linear = power.power_spectrum_gg.galaxy_linear_bias
        for nb in range(hod_bins):
            suffix = f'_{nb + 1}' if hod_bins != 1 else ''
            save_pk_to_grid(
                block,
                z_out,
                k_out,
                'galaxy_power',
                suffix,
                pk_gg_1h[nb],
                pk_gg_2h[nb],
                pk_gg[nb],
            )
            block.put_grid(
                f'galaxy_linear_bias{suffix}',
                'z',
                z_vec,
                'k_h',
                k_vec,
                'bg_linear',
                bg_linear[nb],
            )

    if p_gm:
        results.results(['gm'], ['1h', '2h', 'tot'])
        pk_gm_1h = results.power_spectrum_gm.pk_1h
        pk_gm_2h = results.power_spectrum_gm.pk_2h
        pk_gm = results.power_spectrum_gm.pk_tot
        bgm_linear = power.power_spectrum_gm.galaxy_linear_bias
        for nb in range(hod_bins):
            suffix = f'_{nb + 1}' if hod_bins != 1 else ''
            save_pk_to_grid(
                block,
                z_out,
                k_out,
                'matter_galaxy_power',
                suffix,
                pk_gm_1h[nb],
                pk_gm_2h[nb],
                pk_gm[nb],
            )
            block.put_grid(
                f'galaxy_matter_linear_bias{suffix}',
                'z',
                z_vec,
                'k_h',
                k_vec,
                'bgm_linear',
                bgm_linear[nb],
            )

    if alignment:
        power.update(hod_settings=hod_settings_ia_1)
        hod_bins = save_hod_results(
            block, power, z_vec, hod_section_name_ia_1, hod_settings_ia_1
        )

        if split_ia:
            model_2_params = {
                'hod_settings': hod_settings_ia_2,
                'align_params': align_params_2,
            }
            results = UpsampledSpectra(
                z=z_vec_original,
                k=k_vec_original,
                fraction_z=z_vec_original,
                fraction=np.ones_like(z_vec_original),
                model=power,
                model_1_params=power_kwargs,
                model_2_params=model_2_params,
            )

        if p_II:
            results.results(['ii'], ['1h', '2h', 'tot'])
            pk_II_1h = results.power_spectrum_ii.pk_1h
            pk_II_2h = results.power_spectrum_ii.pk_2h
            pk_II = results.power_spectrum_ii.pk_tot
            for nb in range(hod_bins):
                suffix = f'_{nb + 1}' if hod_bins != 1 else ''
                save_pk_to_grid(
                    block,
                    z_out,
                    k_out,
                    'intrinsic_power',
                    suffix,
                    pk_II_1h[nb],
                    pk_II_2h[nb],
                    pk_II[nb],
                )

        if p_gI:
            results.results(['gi'], ['1h', '2h', 'tot'])
            pk_gI_1h = results.power_spectrum_gi.pk_1h
            pk_gI_2h = results.power_spectrum_gi.pk_2h
            pk_gI = results.power_spectrum_gi.pk_tot
            for nb in range(hod_bins):
                suffix = f'_{nb + 1}' if hod_bins != 1 else ''
                save_pk_to_grid(
                    block,
                    z_out,
                    k_out,
                    'galaxy_intrinsic_power',
                    suffix,
                    pk_gI_1h[nb],
                    pk_gI_2h[nb],
                    pk_gI[nb],
                )

        if p_mI:
            results.results(['mi'], ['1h', '2h', 'tot'])
            pk_mI_1h = results.power_spectrum_mi.pk_1h
            pk_mI_2h = results.power_spectrum_mi.pk_2h
            pk_mI = results.power_spectrum_mi.pk_tot
            for nb in range(hod_bins):
                suffix = f'_{nb + 1}' if hod_bins != 1 else ''
                save_pk_to_grid(
                    block,
                    z_out,
                    k_out,
                    'matter_intrinsic_power',
                    suffix,
                    pk_mI_1h[nb],
                    pk_mI_2h[nb],
                    pk_mI[nb],
                )

    config_hmf['power'] = power
    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
