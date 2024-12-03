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

# z_obs, obs_ext = load_and_interpolate_obs(block, input_section_name, suffixes[i], obs_arr[i], 0.0)
def load_and_interpolate_obs(block, obs_section, suffix_in, obs, extrapolate_option=0.0):
    """
    Loads the observable, e.g. stellar mass, the observable function, e.g. stellar mass function,
    and the redshift bins for the observable. Interpolates the observable function for the obs values 
    that are given.
    """
    # load observable values from observable section name, suffix_in is either med for median
    #  or a number showing the observable-redshift bin index
    obs_in = block[obs_section, f'obs_val_{suffix_in}']
    obs_func_in = block[obs_section, f'obs_func_{suffix_in}']
    # If there are any observable-redshift bins in the observable section:
    # If there are no bins z_bin_{suffix_in} does not exist
    if block.has_value(obs_section, f'z_bin_{suffix_in}'):
        z_obs = block[obs_section, f'z_bin_{suffix_in}']
        obs_func_interp = interp1d(obs_in, obs_func_in, kind='linear', fill_value=extrapolate_option, bounds_error=False, axis=1)
    else:
        z_obs = None
        obs_func_interp = interp1d(obs_in, obs_func_in, kind='linear', fill_value=extrapolate_option, bounds_error=False)
    obs_func = obs_func_interp(obs)

    return z_obs, obs_func

def load_redshift(block, redshift_section, bin, z, extrapolate_option=0.0):
    """
    Loads the redshift distribution in the redshift section. 
    Note: This should match the redshift distribution of the observable sample.
    Then interpolates the redshift distribution for z.
    This is only used if we are not in med (median) mode. 
    """
    z_in  = block[redshift_section, 'z']
    nz_in = block[redshift_section, f'bin_{bin}']
    nz_interp = interp1d(z_in, nz_in, kind='linear', fill_value=extrapolate_option, bounds_error=False)
    nz = nz_interp(z)

    return nz

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


