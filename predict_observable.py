"""
This module combines tomographic / stellar mass bins of the individually calculated observables
It produces the theoretical prediction for the observable for the full survey. 
The number of bins and the mass range can be different to what is calculated in the hod_interface.py module.
"""

from cosmosis.datablock import names, option_section
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps


def load_and_extrapolate_obs(block, obs_section, suffix_in, x_ext, extrapolate_option):

    x_obs = block[obs_section, 'obs_val' + suffix_in]
    if block.has_value(obs_section, 'z_bin' + suffix_in):
        z_obs = block[obs_section, 'z_bin' + suffix_in]
        obs_in = block[obs_section, 'obs_func' + suffix_in]
        inter_func = interp1d(x_obs, obs_in, kind='linear', fill_value=extrapolate_option, bounds_error=False, axis=1)
        obs_ext = inter_func(x_ext)
    else:
        obs_in = block[obs_section, 'obs_func' + suffix_in]
        inter_func = interp1d(x_obs, obs_in, kind='linear', fill_value=extrapolate_option, bounds_error=False)
        obs_ext = inter_func(x_ext)
        z_obs = None
    
    return z_obs, obs_ext
    
def load_kernel(block, kernel_section, bin, z_ext, extrapolate_option):

    z_obs = block[kernel_section, 'z']
    obs_in = block[kernel_section, 'bin_{}'.format(bin)]
    inter_func = interp1d(z_obs, obs_in, kind='linear', fill_value=extrapolate_option, bounds_error=False)
    kernel_ext = inter_func(z_ext)
    
    return kernel_ext


# ;extrapolates the observable (e.g. stellar mass function) for 
# [extrapolate_obs]
# file = %(HM_PATH)s/extrapolate_obs.py
# input_section_name  = stellar_mass_function
# output_section_name = smf_red
# ;suffixes = 1 2
# sample  = nz_lens
# obs_min = 9.3 ;10.4
# obs_max = 11.3 ;10.1 
# n_obs = 12 ;6
# ; Stellar mass ranges

def setup(options):
	
    config = {}
    # input and output section names
    config['input_section_name']  = options.get_string(option_section, 'input_section_name',  default='stellar_mass_function')
    config['output_section_name'] = options.get_string(option_section, 'output_section_name', default='obs_out')
    
    # check if suffixes exists in the extrapolate_obs section of pipeline.ini
    if options.has_value(option_section, 'suffixes'):
        config['suffixes'] = options[option_section, 'suffixes']
        config['nbins']    = len(config['suffixes'])
        config['sample']   = options.get_string(option_section, 'sample')
    else:
        config['nbins']    = 1
        config['sample']   = None
        config['suffixes'] = ['med']
        
    config['obs_min'] = np.asarray([options[option_section, 'obs_min']]).flatten()
    config['obs_max'] = np.asarray([options[option_section, 'obs_max']]).flatten()
    config['n_obs']   = np.asarray([options[option_section, 'n_obs']]).flatten()

    # Check if the legth of obs_min, obs_max, n_obs match
    if not np.all(np.array([len(config['obs_min']), len(config['obs_max']), len(config['n_obs'])]) == len(config['suffixes'])):
        raise Exception('Error: obs_min, obs_max, n_obs need to be of same length as the number of suffixes provided or equal to one.')

    # observable array
    config['obs_arr']   = []
    for i in range(config['nbins']):
        config['obs_arr'].append(np.logspace(config['obs_min'][i], config['obs_max'][i], config['n_obs'][i]))
    
    return config
	

def execute(block, config):
	
    input_section_name  = config['input_section_name']
    output_section_name = config['output_section_name']
    obs_arr  = config['obs_arr']
    suffixes = config['suffixes']
    nbins    = config['nbins']

    # number of bins for the observable
    for i in range(nbins):
        z_obs, obs_ext = load_and_extrapolate_obs(block, input_section_name, suffixes[i], obs_arr[i], 0.0)
    #                def load_and_extrapolate_obs(block, obs_section, suffix_in, x_ext, extrapolate_option):
        print(z_obs)
        if z_obs is not None:
            # Load kernel if exists
            print('z_obs is not None')
            nz = load_kernel(block, config['sample'], i+1, z_obs, 0.0)
            # Integrate over n(z)
            obs_out = simps(nz[:,np.newaxis]*obs_ext, z_obs, axis=0)
        else:
            # Just use interpolated result at the median redshift for the output
            obs_out = obs_ext
        block.put_double_array_1d(output_section_name, 'bin_{}'.format(i+1), obs_out)
        block.put_double_array_1d(output_section_name, 'obs_{}'.format(i+1), obs_arr[i])
    
    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass


