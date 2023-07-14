"""
This module combines tomographic / stellar mass bins of the individually calculated C_ell
Assumes that the C_ell calculated from corresponding galaxy-matter and galaxy-galaxy
powerspectra are ordered according to the suffixes returned by the
add_red_and_blue_power_spectra.py module
"""

from cosmosis.datablock import names, option_section
import sys
import numpy as np
from cosmosis.datablock import names, option_section
from scipy import interp
from scipy.interpolate import interp1d
from scipy.integrate import simps

import time

def extrapolate_obs(x_ext, x_obs, obs_in, nz, extrapolate_option):
    nx_ext = len(x_ext)
    obs_extk = np.empty([nz, nx_ext])
    for jz in range(0,nz):
        inter_func = interp1d(x_obs, obs_in[jz,:], kind='linear', fill_value=extrapolate_option, bounds_error=False)
        obs_extk[jz,:] = inter_func(x_ext)
    return obs_extk
    
def extrapolate(x_ext, x_obs, obs_in, extrapolate_option):
    inter_func = interp1d(x_obs, obs_in, kind='linear', fill_value=extrapolate_option, bounds_error=False)
    obs_extk = inter_func(x_ext)
    return obs_extk

def load_and_extrapolate_obs(block, obs_section, suffix_in, x_ext, extrapolate_option):

    x_obs = block[obs_section, 'obs_' + suffix_in]
    if block.has_value(obs_section, 'z_bin_' + suffix_in):
        z_obs = block[obs_section, 'z_bin_' + suffix_in]
        nz = len(z_obs)
        obs_in = block[obs_section, 'obs_func_' + suffix_in]
        obs_ext = extrapolate_obs(x_ext, x_obs, obs_in, nz, extrapolate_option)
    else:
        obs_in = block[obs_section, 'obs_func_' + suffix_in]
        obs_ext = extrapolate(x_ext, x_obs, obs_in, extrapolate_option)
        z_obs = None
    
    return z_obs, obs_ext
    
def load_kernel(block, kernel_section, bin, z_ext, extrapolate_option):

    z_obs = block[kernel_section, 'z']
    obs_in = block[kernel_section, 'bin_{}'.format(bin)]
    kernel_ext = extrapolate(z_ext, z_obs, obs_in, extrapolate_option)
    
    return kernel_ext


def setup(options):
    #This function is called once per processor per chain.
    #It is a chance to read any fixed options from the configuration file,
    #load any data, or do any calculations that are fixed once.
	
    config = {}
        
    config['input_section_name'] = options.get_string(option_section, 'input_section_name')
    config['output_section_name'] = options.get_string(option_section, 'output_section_name')
    
    if options.has_value(option_section, 'suffixes'):
        config['suffixes'] = [str_val for str_val in str(options[option_section, 'suffixes']).split(',')]
        config['nbins'] = len(config['suffixes'])
        config['sample'] = options.get_string(option_section, 'sample')
    else:
        config['nbins'] = 1
        config['sample'] = None
        config['suffixes'] = ['med']
        
    config['obs_min'] = [np.float64(str_val) for str_val in str(options[option_section, 'obs_min']).split(',')]
    config['obs_max'] = [np.float64(str_val) for str_val in str(options[option_section, 'obs_max']).split(',')]
    config['n_obs'] = [np.int(str_val) for str_val in str(options[option_section, 'n_obs']).split(',')]
    config['x_arr'] = []
    for i in range(config['nbins']):
        config['x_arr'].append(np.logspace(config['obs_min'][i], config['obs_max'][i], config['n_obs'][i]))
    
    return config
	

def execute(block, config):
    #This function is called every time you have a new sample of cosmological and other parameters.
    #It is the main workhorse of the code. The block contains the parameters and results of any
    #earlier modules, and the config is what we loaded earlier.
	
    input_section_name = config['input_section_name']
    output_section_name = config['output_section_name']
    obs_min = config['obs_min']
    obs_max = config['obs_max']
    n_obs = config['n_obs']
    x_arr = config['x_arr']
    suffixes = config['suffixes']
    
    nbins = config['nbins']
    for i in range(nbins):
        z_obs, obs_ext = load_and_extrapolate_obs(block, input_section_name, suffixes[i], x_arr[i], 0.0)
    
        if z_obs is not None:
            # Load kernel if exists
            nz = load_kernel(block, config['sample'], i+1, z_obs, 0.0)
            # Integrate over n(z)
            obs_out = simps(nz[:,np.newaxis]*obs_ext, z_obs, axis=0)
        else:
            # Just use interpolated result at the median redshift for the output
            obs_out = obs_ext
        block.put_double_array_1d(output_section_name, 'bin_{}'.format(i+1), obs_out)
        block.put_double_array_1d(output_section_name, 'obs_{}'.format(i+1), x_arr[i])
    
    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass


