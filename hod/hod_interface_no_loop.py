"""
IMPORTANT: in the code all observable units are in what the data is specified to be through observable_h_unit.
Usually M_star is reported in units of M_sun/h^2.
The observable_h_unit should be set to what the inputs are (from the data and from the values file).
"""

from cosmosis.datablock import names, option_section
import numpy as np
from scipy.interpolate import interp1d
import hod_lib_class_no_loop as hods

cosmo_params = names.cosmological_parameters


def load_hods(block, section_name, suffix):
    """
    Loads and interpolates the hod quantities to match the
    calculation of power spectra
    """
    Ncen = block[section_name, f'n_cen{suffix}']
    Nsat = block[section_name, f'n_sat{suffix}']
    numdencen = block[section_name, f'number_density_cen{suffix}']
    numdensat = block[section_name, f'number_density_sat{suffix}']
    f_c = block[section_name, f'central_fraction{suffix}']
    f_s = block[section_name, f'satellite_fraction{suffix}']
    mass_avg = block[section_name, f'average_halo_mass{suffix}']

    #If we're using an unconditional HOD, we need to define the stellar fraction with zeros
    try:
        fstar = block[section_name, f'f_star{suffix}']
    except:
        fstar = np.zeros((len(z_hod), len(m_hod)))

    return Ncen, Nsat, numdencen, numdensat, f_c, f_s, mass_avg, fstar



# Used for reading data from a text file.
def load_data(file_name):
    z_data, min_magnitude, max_magnitude = np.loadtxt(
        file_name, usecols=(0, 1, 2), unpack=True, dtype=float
    )
    if min_magnitude[0] > max_magnitude[0]:
        raise ValueError(
            'Minimum magnitude must be more negative than the maximum magnitude.'
        )
    return z_data, min_magnitude, max_magnitude

parameters_models = {
    'Zheng': ['log10_Mmin', 'sigma', 'log10_M0', 'log10_M1', 'alpha'],
    'Zhai': ['log10_Mmin', 'sigma', 'log10_Msat', 'log10_Mcut', 'alpha'],
    'Cacciato': [
        'log10_obs_norm_c', 'log10_m_ch', 'g1', 'g2', 'sigma_log10_O_c',
        'norm_s', 'pivot', 'alpha_s', 'beta_s', 'b0', 'b1', 'b2'
    ]
}

def setup(options):
    """Setup configuration for HOD and observable calculations."""
    hod_section_name = options.get_string(option_section, 'hod_section_name').lower()
    values_name = options.get_string(option_section, 'values_name', default='hod_parameters').lower()
    observable_section_name = options.get_string(
        option_section, 'observable_section_name', default='stellar_mass_function'
    ).lower()
    hod_model = options.get_string(option_section, 'hod_model', default='Cacciato')
    nobs = options.get_int(option_section, 'nobs', default=200)
    galaxy_bias_option = options.get_bool(option_section, 'do_galaxy_linear_bias', default=False)
    save_observable = options.get_bool(option_section, 'save_observable', default=True)
    observable_mode = options.get_string(option_section, 'observable_mode', default='obs_z')
    z_median = options.get_double(option_section, 'z_median', default=0.1)
    observable_h_unit = options.get_string(option_section, 'observable_h_unit', default='1/h^2').lower()
    valid_units = ['1/h', '1/h^2']

    if observable_h_unit not in valid_units:
        raise ValueError(f'Currently supported h factors in observable are {valid_units}')

    if hod_model != 'Cacciato' and save_observable:
        raise ValueError(
            'Observable function cannot be calculated for non-conditional observable function based HODs'
        )

    if options.has_value(option_section, 'observables_file'):
        observables_z = True
        file_name = options.get_string(option_section, 'observables_file')
        z_bins, obs_min, obs_max = load_data(file_name)
        nz = len(z_bins)
        log_obs_min = np.log10(obs_min)[np.newaxis, :]
        log_obs_max = np.log10(obs_max)[np.newaxis, :]
        z_bins = z_bins[np.newaxis, :]
        nbins = 1

    elif options.has_value(option_section, 'mass_lim') and options.has_value(option_section, 'mass_lim_low'):
        observables_z = True
        file_name = options.get_string(option_section, 'mass_lim')
        file_name_low = options.get_string(option_section, 'mass_lim_low')
        z_bins = np.linspace(options[option_section, 'zmin'], options[option_section, 'zmax'], options[option_section, 'nz'])
        with open(file_name, 'rb') as dill_file:
            fit_func_inv = pickle.load(dill_file)
        with open(file_name_low, 'rb') as dill_file:
            fit_func_low = pickle.load(dill_file)
        obs_min = fit_func_inv(z_bins)
        obs_max = fit_func_low(z_bins)
        nz = options[option_section, 'nz']
        log_obs_min = np.log10(obs_min)[np.newaxis, :]
        log_obs_max = np.log10(obs_max)[np.newaxis, :]
        z_bins = z_bins[np.newaxis, :]
        nbins = 1

    else:
        observables_z = False
        obs_min = np.asarray([options[option_section, 'log10_obs_min']]).flatten()
        obs_max = np.asarray([options[option_section, 'log10_obs_max']]).flatten()
        zmin = np.asarray([options[option_section, 'zmin']]).flatten()
        zmax = np.asarray([options[option_section, 'zmax']]).flatten()
        nz = options[option_section, 'nz']
        if not np.all(np.array([len(obs_min), len(obs_max), len(zmin), len(zmax)]) == len(obs_min)):
            raise ValueError('obs_min, obs_max, zmin, and zmax need to be of the same length.')
        nbins = len(obs_min)
        z_bins = np.array([np.linspace(zmin_i, zmax_i, nz) for zmin_i, zmax_i in zip(zmin, zmax)])
        log_obs_min = np.array([np.repeat(obs_min_i, nz) for obs_min_i in obs_min])
        log_obs_max = np.array([np.repeat(obs_max_i, nz) for obs_max_i in obs_max])

    obs_simps = np.array([[np.logspace(log_obs_min[nb, jz], log_obs_max[nb, jz], nobs) for jz in range(nz)] for nb in range(nbins)])

    return {
        'obs_simps': obs_simps,
        'nbins': nbins,
        'nz': nz,
        'nobs': nobs,
        'z_bins': z_bins,
        'z_median': z_median,
        'galaxy_bias_option': galaxy_bias_option,
        'save_observable': save_observable,
        'observable_mode': observable_mode,
        'hod_section_name': hod_section_name,
        'values_name': values_name,
        'observables_z': observables_z,
        'observable_section_name': observable_section_name,
        'observable_h_unit': observable_h_unit,
        'valid_units': valid_units,
        'hod_model': hod_model
    }

def execute(block, config):
    """Execute the HOD and observable calculations."""
    obs_simps = config['obs_simps']
    nbins = config['nbins']
    nz = config['nz']
    nobs = config['nobs']
    z_bins = config['z_bins']
    z_median = config['z_median']
    galaxy_bias_option = config['galaxy_bias_option']
    save_observable = config['save_observable']
    observable_mode = config['observable_mode']
    hod_section_name = config['hod_section_name']
    values_name = config['values_name']
    observables_z = config['observables_z']
    observable_section_name = config['observable_section_name']
    observable_h_unit = config['observable_h_unit']
    valid_units = config['valid_units']
    hod_model = config['hod_model']
    
    if save_observable:
        block.put(observable_section_name, 'obs_func_definition', 'obs_func * obs * ln(10)')
        block.put(observable_section_name, 'observable_mode', observable_mode)
    block.put_int(hod_section_name, 'nbins', nbins)
    block.put_bool(hod_section_name, 'observable_z', observables_z)

    hod_kwargs = {}
    hod_parameters = parameters_models[hod_model]

    dndlnM_grid = block['hmf', 'dndlnmh']
    mass = block['hmf', 'm_h']
    z_dn = block['hmf', 'z']

    # set interpolator for the halo mass function
    f_int_dndlnM = interp1d(
        z_dn, dndlnM_grid, kind='linear', fill_value='extrapolate',
        bounds_error=False, axis=0
    )

    hod_kwargs['A_cen'] = block[values_name, 'A_cen'] if block.has_value(values_name, 'A_cen') else None
    hod_kwargs['A_sat'] = block[values_name, 'A_sat'] if block.has_value(values_name, 'A_sat') else None
        
    # Dinamically load required HOD parameters givent the model and number of bins!
    for param in hod_parameters:
        if hod_model == 'Cacciato':
            param_bin = param
            if not block.has_value(values_name, param_bin):
                raise Exception(f'Error: parameter {param} is needed for the requested hod model: {hod_model}')
            hod_kwargs[param] = block[values_name, param_bin]
        else:
            param_list = []
            for nb in range(nbins):
                suffix = f'_{nb+1}' if nbins != 1 else ''
                param_bin = f'{param}{suffix}'
                if not block.has_value(values_name, param_bin):
                    raise Exception(f'Error: parameter {param} is needed for the requested hod model: {hod_model}')
                param_list.append(block[values_name, param_bin])
            hod_kwargs[param] = np.array(param_list)

    dndlnM = f_int_dndlnM(z_bins)
    hod_kwargs['mass'] = mass
    hod_kwargs['obs'] = obs_simps
    hod_kwargs['dndlnm'] = dndlnM
    hod_kwargs['nz'] = nz

    COF_class = getattr(hods, hod_model)(**hod_kwargs)

    N_tot = COF_class.compute_hod
    N_cen = COF_class.compute_hod_cen
    N_sat = COF_class.compute_hod_sat
    
    if (N_sat < 0).any() or (N_cen < 0).any():
        raise ValueError('Some HOD values are negative. Increase nobs for a more stable integral.')

    if hod_model == 'Cacciato':
        f_star = COF_class.compute_stellar_fraction
        if observable_h_unit == valid_units[1]:
            f_star = f_star * block['cosmological_parameters', 'h0']
        for nb in range(nbins):
            suffix = f'_{nb+1}'
            block.put_grid(hod_section_name, f'z{suffix}', z_bins[nb], f'mass{suffix}', mass, f'f_star{suffix}', f_star[nb])

    numdens_cen = COF_class.compute_number_density_cen
    numdens_sat = COF_class.compute_number_density_sat
    numdens_tot = COF_class.compute_number_density
    fraction_cen = numdens_cen / numdens_tot
    fraction_sat = numdens_sat / numdens_tot
    mass_avg = COF_class.compute_avg_halo_mass_cen / numdens_cen
    
    for nb in range(nbins):
        suffix = f'_{nb+1}'
        block.put_grid(hod_section_name, f'z{suffix}', z_bins[nb], f'mass{suffix}', mass, f'N_sat{suffix}', N_sat[nb])
        block.put_grid(hod_section_name, f'z{suffix}', z_bins[nb], f'mass{suffix}', mass, f'N_cen{suffix}', N_cen[nb])
        block.put_grid(hod_section_name, f'z{suffix}', z_bins[nb], f'mass{suffix}', mass, f'N_tot{suffix}', N_tot[nb])
        block.put_double_array_1d(hod_section_name, f'number_density_cen{suffix}', numdens_cen[nb])
        block.put_double_array_1d(hod_section_name, f'number_density_sat{suffix}', numdens_sat[nb])
        block.put_double_array_1d(hod_section_name, f'number_density_tot{suffix}', numdens_tot[nb])
        block.put_double_array_1d(hod_section_name, f'central_fraction{suffix}', fraction_cen[nb])
        block.put_double_array_1d(hod_section_name, f'satellite_fraction{suffix}', fraction_sat[nb])
        block.put_double_array_1d(hod_section_name, f'average_halo_mass{suffix}', mass_avg[nb])

    if galaxy_bias_option:
        z_hbf = block['halobias', 'z']
        halobias_hbf = block['halobias', 'b_hb']
        f_interp_halobias = interp1d(
            z_hbf, halobias_hbf, kind='linear', fill_value='extrapolate',
            bounds_error=False, axis=0
        )
        hbias = f_interp_halobias(z_bins)

        galaxybias_cen = COF_class.compute_galaxy_linear_bias_cen(hbias) / numdens_tot
        galaxybias_sat = COF_class.compute_galaxy_linear_bias_sat(hbias) / numdens_tot
        galaxybias_tot = COF_class.compute_galaxy_linear_bias(hbias) / numdens_tot
        
        for nb in range(nbins):
            suffix = f'_{nb+1}'
            block.put_double_array_1d(hod_section_name, f'galaxy_bias_centrals{suffix}', galaxybias_cen[nb])
            block.put_double_array_1d(hod_section_name, f'galaxy_bias_satellites{suffix}', galaxybias_sat[nb])
            block.put_double_array_1d(hod_section_name, f'b{suffix}', galaxybias_tot[nb])

    if save_observable and observable_mode == 'obs_z' and hod_model == 'Cacciato':
    
        obs_func = COF_class.obs_func
        obs_func_c = COF_class.obs_func_cen
        obs_func_s = COF_class.obs_func_sat

        for nb in range(nbins):
            suffix_obs = f'_{nb+1}'
            block.put_grid(observable_section_name, f'z_bin{suffix_obs}', z_bins[nb], f'obs_val{suffix_obs}', obs_simps[nb, 0, :], f'obs_func{suffix_obs}', np.log(10.0) * obs_func[nb, 0, :] * obs_simps[nb])
            block.put_grid(observable_section_name, f'z_bin{suffix_obs}', z_bins[nb], f'obs_val{suffix_obs}', obs_simps[nb, 0, :], f'obs_func_c{suffix_obs}', np.log(10.0) * obs_func_c[nb, 0, :] * obs_simps[nb])
            block.put_grid(observable_section_name, f'z_bin{suffix_obs}', z_bins[nb], f'obs_val{suffix_obs}', obs_simps[nb, 0, :], f'obs_func_s{suffix_obs}', np.log(10.0) * obs_func_s[nb, 0, :] * obs_simps[nb])

    if hod_model == 'Cacciato':
        # Calculating the full stellar mass fraction and if desired the observable function for one bin case
        nl_obs = 100
        nl_z = 15
        
        z_bins_one = np.array([np.linspace(z_bins.min(), z_bins.max(), nl_z)])
        f_mass_z_one = interp1d(
            z_dn, dndlnM_grid, kind='linear', fill_value='extrapolate',
            bounds_error=False, axis=0
        )
        dn_dlnM_one = f_mass_z_one(z_bins_one)

        obs_range = np.array([
            np.logspace(np.log10(obs_simps.min()), np.log10(obs_simps.max()), nl_obs)
            for _ in range(nl_z)
        ])

        hod_kwargs['obs'] = obs_range[np.newaxis, :]
        hod_kwargs['dndlnm'] = dn_dlnM_one
        hod_kwargs['nz'] = nl_z
        COF_class_onebin = getattr(hods, hod_model)(**hod_kwargs)

        f_star_mm = COF_class_onebin.compute_stellar_fraction
        if observable_h_unit == valid_units[1]:
            f_star_mm = f_star_mm * block['cosmological_parameters', 'h0']
            
        block.put_grid(hod_section_name, 'z_extended', z_bins_one[0], 'mass_extended', mass, 'f_star_extended', f_star_mm[0])

        if save_observable and observable_mode == 'obs_onebin':
            obs_func = COF_class_onebin.obs_func
            obs_func_c = COF_class_onebin.obs_func_cen
            obs_func_s = COF_class_onebin.obs_func_sat

            block.put_grid(observable_section_name, 'z_bin_1', z_bins_one[0], 'obs_val_1', obs_range[0], 'obs_func_1', np.log(10.0) * obs_func[0] * obs_range[0])
            block.put_grid(observable_section_name, 'z_bin_1', z_bins_one[0], 'obs_val_1', obs_range[0], 'obs_func_c_1', np.log(10.0) * obs_func_c[0] * obs_range[0])
            block.put_grid(observable_section_name, 'z_bin_1', z_bins_one[0], 'obs_val_1', obs_range[0], 'obs_func_s_1', np.log(10.0) * obs_func_s[0] * obs_range[0])

        if save_observable and observable_mode == 'obs_zmed':
            f_mass_z_dn = interp1d(
                z_dn, dndlnM_grid, kind='linear', fill_value='extrapolate',
                bounds_error=False, axis=0
            )
            dn_dlnM_zmedian = f_mass_z_dn(z_median)

            obs_range = np.logspace(np.log10(obs_simps.min()), np.log10(obs_simps.max()), nobs)

            hod_kwargs['obs'] = obs_range[np.newaxis, np.newaxis, :]
            hod_kwargs['dndlnm'] = dn_dlnM_zmedian[np.newaxis, np.newaxis, :]
            hod_kwargs['nz'] = 1
            COF_class_zmed = getattr(hods, hod_model)(**hod_kwargs)

            obs_func = COF_class_zmed.obs_func
            obs_func_c = COF_class_zmed.obs_func_cen
            obs_func_s = COF_class_zmed.obs_func_sat

            block.put_double_array_1d(observable_section_name, 'obs_val_med', obs_range)
            block.put_double_array_1d(observable_section_name, 'obs_func_med', np.log(10.0) * obs_func[0, 0, :] * obs_range)
            block.put_double_array_1d(observable_section_name, 'obs_func_med_c', np.log(10.0) * obs_func_c[0, 0, :] * obs_range)
            block.put_double_array_1d(observable_section_name, 'obs_func_med_s', np.log(10.0) * obs_func_s[0, 0, :] * obs_range)

            mean_obs_cen = COF_class_zmed.cal_mean_obs_c[0, 0, :]

            block.put_double_array_1d(observable_section_name, 'halo_mass_med', mass)
            block.put_double_array_1d(observable_section_name, 'mean_obs_halo_mass_relation', mean_obs_cen)

    return 0

def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
