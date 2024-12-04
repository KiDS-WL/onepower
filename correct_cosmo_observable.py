"""
This module combines tomographic / stellar mass bins of the individually calculated observables
It produces the theoretical prediction for the observable for the full survey. 
The number of bins and the mass range can be different to what is calculated in the hod_interface.py module.
"""

from cosmosis.datablock import names, option_section
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM, LambdaCDM

cosmo_params = names.cosmological_parameters

"""
    !!!WORK IN PROGRESS!!!
    Currently non functional
"""

def setup(options):
	
    config = {}
    # input and output section names
    config['section_name']  = options.get_string(option_section, 'section_name',  default='obs_out')
    
    config['zmin'] = np.asarray([options[option_section, 'zmin']]).flatten()
    config['zmax'] = np.asarray([options[option_section, 'zmax']]).flatten()
    
    omegam = options[option_section, 'omega_m']
    omegav = options[option_section, 'omega_lambda']
    h_data = options[option_section, 'h0']
    
    config['cosmo_model_data'] = LambdaCDM(H0=h_data*100., Om0=omegam, Ode0=omegav)
    config['h_data'] = h_data
    
    
    return config
	

def execute(block, config):

    section_name  = config['section_name']
    zmin  = config['zmin']
    zmax  = config['zmax']
    nbins    = block[section_name, 'nbins']
    h_data = config['h_data']
    cosmo_model_data = config['cosmo_model_data']

    # Check if the length of zmin, zmax, nbins match
    if not np.all(np.array([len(config['zmin']), len(config['zmax'])]) == nbins):
        raise Exception('Error: zmin, zmax need to be of same length as \
                         the number of bins provided.')
                         
    cosmo_model_run = Flatw0waCDM(
        H0=block[cosmo_params, 'hubble'], Ob0=block[cosmo_params, 'omega_b'],
        Om0=block[cosmo_params, 'omega_m'], m_nu=[0, 0, block[cosmo_params, 'mnu']], Tcmb0=tcmb,
        w0=block[cosmo_params, 'w'], wa=block[cosmo_params, 'wa'] )
    h_run = cosmo_model_run.h

    # number of bins for the observable this is given via len(suffixes)
    for i in range(nbins):
        #z_obs, obs_func = load_and_interpolate_obs(block, input_section_name, suffixes[i], obs_arr[i], extrapolate_option=0.0)
        
        obs_func = block[section_name, f'bin_{i+1}']
        obs_arr = block[section_name, f'obs_{i+1}']
        
        comoving_data = cosmo_model_data.comoving_distance(zmax[i])**3.0 - cosmo_model_data.comoving_distance(zmin[i])**3.0 * h_run**3.0
        comoving_model = cosmo_model_run.comoving_distance(zmax[i])**3.0 - cosmo_model_run.comoving_distance(zmin[i])**3.0 * h_data**3.0
        
        ratio_obs = comoving_model / comoving_data
        obs_func_new = obs_func * ratio_obs
        obs_arr_new = obs_arr # / h_run**2.0 ??
        
        block.replace_double_array_1d(output_section_name, f'bin_{i+1}', obs_func_new)
        block.replace_double_array_1d(output_section_name, f'obs_{i+1}', obs_arr_new)

    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass


