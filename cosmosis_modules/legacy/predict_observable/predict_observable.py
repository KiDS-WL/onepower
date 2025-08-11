"""
This module combines tomographic / stellar mass bins of the individually calculated observables.
It produces the theoretical prediction for the observable for the full survey.
The number of bins and the mass range can be different to what is calculated in the hod_interface.py module.

Furthermore it corrects the individually calculated observables (stellar mass function)
for the difference in input data cosmology to the predicted output cosmology
by multiplication of ratio of volumes according to More et al. 2013 and More et al. 2015
"""

import ast
import astropy.cosmology as cosmology_classes
import numpy as np
from astropy.cosmology import Flatw0waCDM
from cosmosis.datablock import names, option_section
from scipy.integrate import simpson
from scipy.interpolate import interp1d


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
    config['correct_cosmo'] = options.get_bool(option_section, 'correct_cosmo', default=False)

    # Check if suffixes exists in the extrapolate_obs section of pipeline.ini
    if options.has_value(option_section, 'suffixes'):
        config['suffixes'] = np.asarray([options[option_section, 'suffixes']]).flatten()
        config['nbins'] = len(config['suffixes'])
        config['sample'] = options.get_string(option_section, 'sample')
    else:
        config['nbins'] = 1
        config['sample'] = None
        config['suffixes'] = ['_1']

    obs_dist_file = options.get_string(option_section, 'obs_dist_file', default='')
    config['weighted_binning'] = options.get_bool(option_section, 'weighted_binning', default=False)
    config['log10_obs_min'] = np.asarray([options[option_section, 'log10_obs_min']]).flatten()
    config['log10_obs_max'] = np.asarray([options[option_section, 'log10_obs_max']]).flatten()
    config['zmin'] = np.asarray([options[option_section, 'zmin']]).flatten()
    config['zmax'] = np.asarray([options[option_section, 'zmax']]).flatten()
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

    if config['correct_cosmo']:
        # Maybe we should do this also for the model cosmology and in halo_model_ingredients?
        # At least to specify the exact cosmology model, even though it should be as close as general
        # as in CAMB, for which we can safely assume Flatw0waCDM does the job...

        # cosmo_kwargs is to be a string containing a dictionary with all the arguments the
        # requested cosmology accepts (see default)!
        cosmo_kwargs = ast.literal_eval(
            options.get_string(
                option_section, 'cosmo_kwargs', default="{'H0':70.0, 'Om0':0.3, 'Ode0':0.7}"
            )
        )

        # Requested cosmology class from astropy:
        cosmo_class = options.get_string(
            option_section, 'astropy_cosmology_class', default='LambdaCDM'
        )
        cosmo_class_init = getattr(cosmology_classes, cosmo_class)
        cosmo_model_data = cosmo_class_init(**cosmo_kwargs)

        config['cosmo_model_data'] = cosmo_model_data
        config['h_data'] = cosmo_model_data.h

    return config

def execute(block, config):
    input_section_name = config['input_section_name']
    output_section_name = config['output_section_name']
    obs_arr = config['obs_arr']
    suffixes = config['suffixes']
    nbins = config['nbins']

    if config['correct_cosmo']:
        zmin = config['zmin']
        zmax = config['zmax']
        h_data = config['h_data']
        cosmo_model_data = config['cosmo_model_data']

        # Check if the length of zmin, zmax, nbins match
        if len(zmin) != nbins or len(zmax) != nbins:
            raise ValueError('Error: zmin, zmax need to be of the same length as the number of bins provided.')

        # Adopting the same cosmology object as in halo_model_ingredients module
        tcmb = block.get_double(names.cosmological_parameters, 'TCMB', default=2.7255)
        cosmo_model_run = Flatw0waCDM(
            H0=block[names.cosmological_parameters, 'hubble'],
            Ob0=block[names.cosmological_parameters, 'omega_b'],
            Om0=block[names.cosmological_parameters, 'omega_m'],
            m_nu=[0, 0, block[names.cosmological_parameters, 'mnu']],
            Tcmb0=tcmb, w0=block[names.cosmological_parameters, 'w'],
            wa=block[names.cosmological_parameters, 'wa']
        )
        h_run = cosmo_model_run.h

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
            obs_func = simpson(nz[:, np.newaxis] * obs_func, z_obs, axis=0)

        if config['correct_cosmo']:
            obs_func_in = obs_func.copy()
            comoving_volume_data = ((cosmo_model_data.comoving_distance(zmax[i])**3.0
                                 - cosmo_model_data.comoving_distance(zmin[i])**3.0)
                                * h_data**3.0)
            comoving_volume_model = ((cosmo_model_run.comoving_distance(zmax[i])**3.0
                                  - cosmo_model_run.comoving_distance(zmin[i])**3.0)
                                 * h_run**3.0)

            ratio_obs = comoving_volume_model / comoving_volume_data
            obs_func = obs_func_in * ratio_obs

        block.put_double_array_1d(output_section_name, f'bin_{i + 1}', obs_func)
        block.put_double_array_1d(output_section_name, f'obs_{i + 1}', obs_arr[i])
        block.put_double_array_1d(output_section_name, f'mass_{i + 1}', obs_arr[i])

    block[output_section_name, 'nbin'] = nbins
    block[output_section_name, 'sample'] = config['sample'] if config['sample'] is not None else 'None'

    return 0

def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
