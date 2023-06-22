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

import time


def setup(options):
    #This function is called once per processor per chain.
    #It is a chance to read any fixed options from the configuration file,
    #load any data, or do any calculations that are fixed once.
	
    config = {}
        
    config['input_section_name'] = options.get_string(option_section, 'input_section_name')
    config['output_section_name'] = options.get_string(option_section, 'output_section_name')
    config['suffixes'] = [str_val for str_val in str(options[option_section, 'suffixes']).split(',')]
    
    config['input_section_names'] = [config['input_section_name'] + '_' + str_val for str_val in config['suffixes']]
    config['nbins'] = len(config['suffixes'])
    
    return config
	

def execute(block, config):
    #This function is called every time you have a new sample of cosmological and other parameters.
    #It is the main workhorse of the code. The block contains the parameters and results of any
    #earlier modules, and the config is what we loaded earlier.
	
    input_section_name = config['input_section_name']
    input_section_names = config['input_section_names']
    output_section_name = config['output_section_name']

    nbins = config['nbins']
    for i in range(nbins):
        keys_bin = block.keys(input_section_names[i])
        nbins_i = block[input_section_names[i], 'nbin_a']
        nbins_j = block[input_section_names[i], 'nbin_b']
        auto_only = block.get_bool(input_section_names[i], 'auto_only')
        jmax = nbins if auto_only else nbins_j
        for j in range(jmax):
            if auto_only:
                c_ell = block[input_section_names[i], 'bin_{0}_{1}'.format(1, 1)]
                if j!=i:
                    continue
            else:
                c_ell = block[input_section_names[i], 'bin_{0}_{1}'.format(1, j+1)]
            block.put_double_array_1d(output_section_name, 'bin_{0}_{1}'.format(i+1, j+1), c_ell)
				
    # copy the remaining info from the first bin
    block.put_double_array_1d(output_section_name, 'ell', block[input_section_names[0], 'ell'])
    block[output_section_name, 'auto_only'] = block.get_bool(input_section_names[0], 'auto_only')
    block[output_section_name, 'is_auto'] = block.get_bool(input_section_names[0], 'is_auto')
    if block.has_value(input_section_names[0], 'nbin'):
        block[output_section_name, 'nbin'] = nbins
    block[output_section_name, 'nbin_a'] = nbins
    if block.get_bool(input_section_names[0], 'auto_only'):
        block[output_section_name, 'nbin_b'] = nbins
    else:
        block[output_section_name, 'nbin_b'] = block[input_section_names[0], 'nbin_b']
    block[output_section_name, 'sep_name'] = block[input_section_names[0], 'sep_name']
    
    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass


