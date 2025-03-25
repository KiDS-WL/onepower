"""
This module corrects the individually calculated observables (stellar mass function)
for the difference in input data cosmology to the predicted output cosmology
by multiplication of ratio of volumes according to More et al. 2013 and More et al. 2015
"""

from cosmosis.datablock import names, option_section
import numpy as np
import ast
import astropy
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM, LambdaCDM

cosmo_params = names.cosmological_parameters

def setup(options):
    config = {}
    # Input and output section names
    config['section_name'] = options.get_string(option_section, 'section_name', default='obs_out')

    config['zmin'] = np.asarray([options[option_section, 'zmin']]).flatten()
    config['zmax'] = np.asarray([options[option_section, 'zmax']]).flatten()

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
    cosmo_class_init = getattr(astropy.cosmology, cosmo_class)
    cosmo_model_data = cosmo_class_init(**cosmo_kwargs)

    config['cosmo_model_data'] = cosmo_model_data
    config['h_data'] = cosmo_model_data.h

    return config

def execute(block, config):
    section_name = config['section_name']
    zmin = config['zmin']
    zmax = config['zmax']
    nbins = block[section_name, 'nbin']
    h_data = config['h_data']
    cosmo_model_data = config['cosmo_model_data']

    # Check if the length of zmin, zmax, nbins match
    if len(zmin) != nbins or len(zmax) != nbins:
        raise ValueError('Error: zmin, zmax need to be of the same length as the number of bins provided.')

    # Adopting the same cosmology object as in halo_model_ingredients module
    tcmb = block.get_double(cosmo_params, 'TCMB', default=2.7255)
    cosmo_model_run = Flatw0waCDM(
        H0=block[cosmo_params, 'hubble'],
        Ob0=block[cosmo_params, 'omega_b'],
        Om0=block[cosmo_params, 'omega_m'],
        m_nu=[0, 0, block[cosmo_params, 'mnu']],
        Tcmb0=tcmb, w0=block[cosmo_params, 'w'],
        wa=block[cosmo_params, 'wa']
    )
    h_run = cosmo_model_run.h

    # number of bins for the observable this is given via saved nbins value
    for i in range(nbins):
        obs_func = block[section_name, f'bin_{i+1}']
        #obs_arr = block[section_name, f'obs_{i+1}']

        comoving_volume_data = ((cosmo_model_data.comoving_distance(zmax[i])**3.0
                                 - cosmo_model_data.comoving_distance(zmin[i])**3.0)
                                * h_data**3.0)
        comoving_volume_model = ((cosmo_model_run.comoving_distance(zmax[i])**3.0
                                  - cosmo_model_run.comoving_distance(zmin[i])**3.0)
                                 * h_run**3.0)

        ratio_obs = comoving_volume_model / comoving_volume_data
        obs_func_new = obs_func * ratio_obs

        block.replace_double_array_1d(section_name, f'bin_{i+1}', obs_func_new)
        # AD: I think there is no change to the obs values (the stellar masses)
        #block.replace_double_array_1d(section_name, f'obs_{i+1}', obs_arr_new)

    return 0

def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
