"""
This module corrects the individually calculated observables (stellar mass function)
for the difference in input data cosmology to the predicted output cosmology
by multiplication of the ratio of volumes according to More et al. 2013 and More et al. 2015.
"""

import ast
import astropy
import numpy as np
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM, LambdaCDM
from cosmosis.datablock import names, option_section

cosmo_params = names.cosmological_parameters

def setup(options):
    config = {}

    # Input and output section names
    config['section_name'] = options.get_string(option_section, 'section_name')
    config['z'] = options[option_section, 'z']
    if np.isscalar(config['z']):
        config['z'] = [config['z']]

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

    # Requested cosmology class from astropy
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
    nbins = block[section_name, 'nbin']
    h_data = config['h_data']
    cosmo_model_data = config['cosmo_model_data']
    corr_type = block[section_name, 'save_name']
    z = config['z']

    if len(z) != nbins:
        raise ValueError('Error: z needs to be of the same length as the number of bins provided.')

    # Adopting the same cosmology object as in halo_model_ingredients module
    tcmb = block.get_double(cosmo_params, 'TCMB', default=2.7255)
    cosmo_model_run = Flatw0waCDM(
        H0=block[cosmo_params, 'hubble'],
        Ob0=block[cosmo_params, 'omega_b'],
        Om0=block[cosmo_params, 'omega_m'],
        m_nu=[0, 0, block[cosmo_params, 'mnu']],
        Tcmb0=tcmb,
        w0=block[cosmo_params, 'w'],
        wa=block[cosmo_params, 'wa']
    )
    h_run = cosmo_model_run.h
    z_s = 0.6

    # Number of bins for the observable this is given via saved nbins value
    for i in range(nbins):
        func = block[section_name, f'bin_{i + 1}']
        arr = block[section_name, 'rp']
        zi = z[i]

        ratio = (
            (cosmo_model_run.comoving_distance(zi)) /
            (cosmo_model_data.comoving_distance(zi))
        ) * (h_run / h_data)

        if corr_type == 'wp':
            ratio_corr = (
                cosmo_model_data.efunc(zi) / cosmo_model_run.efunc(zi)
            ) * (h_data / h_run)
        elif corr_type == 'ds':
            ratio_corr = (
                (cosmo_model_data.angular_diameter_distance(z_s) *
                 cosmo_model_run.angular_diameter_distance_z1z2(zi, z_s) *
                 cosmo_model_run.angular_diameter_distance(zi)) /
                (cosmo_model_run.angular_diameter_distance(z_s) *
                 cosmo_model_data.angular_diameter_distance_z1z2(zi, z_s) *
                 cosmo_model_data.angular_diameter_distance(zi))
            ) * (h_run / h_data)
        else:
            raise ValueError('Unrecognized correlation type')

        func_new = func * ratio_corr
        arr_new = arr * ratio

        block.replace_double_array_1d(section_name, f'bin_{i + 1}', func_new)
        block.replace_double_array_1d(section_name, 'rp', arr_new)

    return 0

def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
