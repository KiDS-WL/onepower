"""
This module combines tomographic / stellar mass bins of the individually calculated observables.
It produces the theoretical prediction for the observable for the full survey.
The number of bins and the mass range can be different to what is calculated in the hod_interface.py module.
"""

from cosmosis.datablock import option_section
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps

def load_and_interpolate_obs(block, obs_section, suffix_in, extrapolate_option=0.0):
    """
    Loads the observable, e.g. stellar mass, the observable function, e.g. stellar mass function,
    and the redshift bins for the observable. Interpolates the observable function for the obs values
    that are given.
    """
    # Load observable values from observable section name, suffix_in is either med for median
    # or a number showing the observable-redshift bin index
    obs_in = block[obs_section, f'obs_val_{suffix_in}']
    obs_func_in = block[obs_section, f'obs_func_{suffix_in}']

    # If there are any observable-redshift bins in the observable section:
    # If there are no bins z_bin_{suffix_in} does not exist
    if block.has_value(obs_section, f'z_bin_{suffix_in}'):
        z_obs = block[obs_section, f'z_bin_{suffix_in}']
        obs_func_interp = interp1d(
            obs_in, obs_func_in, kind='linear',
            fill_value=extrapolate_option, bounds_error=False, axis=1
        )
    else:
        z_obs = None
        obs_func_interp = interp1d(
            obs_in, obs_func_in, kind='linear',
            fill_value=extrapolate_option, bounds_error=False
        )
    # obs_func = obs_func_interp(obs)

    return z_obs, obs_func_interp

def load_redshift(block, redshift_section, bin_num, z, extrapolate_option=0.0):
    """
    Loads the redshift distribution in the redshift section.
    Note: This should match the redshift distribution of the observable sample.
    Then interpolates the redshift distribution for z.
    This is only used if we are not in med (median) mode.
    """
    z_in = block[redshift_section, 'z']
    nz_in = block[redshift_section, f'bin_{bin_num}']
    nz_interp = interp1d(z_in, nz_in, kind='linear', fill_value=extrapolate_option, bounds_error=False)
    nz = nz_interp(z)

    return nz

def setup(options):
    config = {}

    # Input and output section names
    config['input_section_name'] = options.get_string(option_section, 'input_section_name', default='stellar_mass_function')
    config['output_section_name'] = options.get_string(option_section, 'output_section_name', default='obs_out')

    # Check if suffixes exists in the extrapolate_obs section of pipeline.ini
    if options.has_value(option_section, 'suffixes'):
        config['suffixes'] = np.asarray([options[option_section, 'suffixes']]).flatten()
        config['nbins'] = len(config['suffixes'])
        config['sample'] = options.get_string(option_section, 'sample')
    else:
        config['nbins'] = 1
        config['sample'] = None
        config['suffixes'] = ['med']

    obs_dist_file = options.get_string(option_section, 'obs_dist_file', default='')
    config['weighted_binning'] = options.get_bool(option_section, 'weighted_binning', default=False)
    config['log10_obs_min'] = np.asarray([options[option_section, 'log10_obs_min']]).flatten()
    config['log10_obs_max'] = np.asarray([options[option_section, 'log10_obs_max']]).flatten()
    config['n_obs'] = np.asarray([options[option_section, 'n_obs']]).flatten()
    config['edges'] = options.get_bool(option_section, 'edges', default=False)

    # Check if the length of log10_obs_min, log10_obs_max, n_obs match
    if not np.all(np.array([len(config['log10_obs_min']), len(config['log10_obs_max']), len(config['n_obs'])]) == len(config['suffixes'])):
        raise ValueError('log10_obs_min, log10_obs_max, n_obs need to be of same length as the number of suffixes provided or equal to one.')

    # Observable array this is not in log10
    config['obs_arr'] = []
    for i in range(config['nbins']):
        if config['edges']:
            # log10_obs_min and log10_obs_max are in log10 M_sun/h^2 units for stellar masses
            bins = np.linspace(config['log10_obs_min'][i], config['log10_obs_max'][i], config['n_obs'][i] + 1, endpoint=True)
            center = (bins[1:] + bins[:-1]) / 2.0
            config['obs_arr'].append(10.0**center)
        else:
            # If edges is False then we assume that log10_obs_min and log10_obs_max are the center of the bins
            config['obs_arr'].append(np.logspace(config['log10_obs_min'][i], config['log10_obs_max'][i], config['n_obs'][i]))

    if config['weighted_binning']:
        if config['edges']:
            if obs_dist_file:
                # Read in number of galaxies within a narrow observable bin from file
                config['obs_dist'] = np.loadtxt(obs_dist_file, comments='#')
            else:
                config['obs_arr_fine'] = np.linspace(config['log10_obs_min'].min(), config['log10_obs_max'].max(), 10000, endpoint=True)
        else:
            raise ValueError('Please provide edge values for observables to do weighted binning.')

    return config

def execute(block, config):
    input_section_name = config['input_section_name']
    output_section_name = config['output_section_name']
    obs_arr = config['obs_arr']
    suffixes = config['suffixes']
    nbins = config['nbins']

    # TODO: find the binned value of obs_func_binned = \sum_O_{min}^O_{max} Phi(O_i) * N(O_i) / \sum_O_{min}^O_{max} N(O_i)
    # N(O_i) is the number of galaxies with obs = O_i in a fine bin around O_i
    # This should be closer to the estimated obs_func.
    # Number of bins for the observable this is given via len(suffixes)
    for i in range(nbins):
        # Reads in and produce the interpolator for obs_func. z_obs is read if it exists.
        z_obs, obs_func_interp = load_and_interpolate_obs(block, input_section_name, suffixes[i])

        if config['weighted_binning']:
            try:
                obs_values, obs_dist = config['obs_dist'][:, 0], config['obs_dist'][:, 1]
                obs_func_fine = obs_func_interp(10 ** obs_values) if z_obs is None else obs_func_interp(10 ** obs_values)[0]
                obs_edges = np.linspace(config['log10_obs_min'][i], config['log10_obs_max'][i], config['n_obs'][i] + 1, endpoint=True)

                obs_func_binned = np.array([
                    np.sum(obs_func_fine[cond_bin] * obs_dist[cond_bin]) / np.sum(obs_dist[cond_bin])
                    for cond_bin in [(obs_values > obs_edges[j]) & (obs_values <= obs_edges[j + 1]) for j in range(len(obs_arr[i]))]
                ])
                obs_func = obs_func_binned

            except:
                obs_func_fine = obs_func_interp(10 ** config['obs_arr_fine']) if z_obs is None else obs_func_interp(10 ** config['obs_arr_fine'])[0]
                obs_edges = np.linspace(config['log10_obs_min'][i], config['log10_obs_max'][i], config['n_obs'][i] + 1, endpoint=True)

                obs_func_integrated = np.array([
                    np.sum(obs_func_fine[cond_bin]) / np.sum(cond_bin)
                    for cond_bin in [(config['obs_arr_fine'] > obs_edges[j]) & (config['obs_arr_fine'] <= obs_edges[j + 1]) for j in range(config['n_obs'][i])]
                ])
                obs_func = obs_func_integrated
        else:
            obs_func = obs_func_interp(obs_arr[i])

        if z_obs is not None:
            nz = load_redshift(block, config['sample'], i + 1, z_obs)
            obs_func = simps(nz[:, np.newaxis] * obs_func, z_obs, axis=0)

        block.put_double_array_1d(output_section_name, f'bin_{i + 1}', obs_func)
        block.put_double_array_1d(output_section_name, f'obs_{i + 1}', obs_arr[i])

    block[output_section_name, 'nbin'] = nbins
    block[output_section_name, 'sample'] = config['sample'] if config['sample'] is not None else 'None'

    return 0

def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
