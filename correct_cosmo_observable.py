"""
This module combines tomographic / stellar mass bins of the individually calculated observables
It produces the theoretical prediction for the observable for the full survey. 
The number of bins and the mass range can be different to what is calculated in the hod_interface.py module.
"""

from cosmosis.datablock import names, option_section
import numpy as np
import ast
import astropy
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM, LambdaCDM

cosmo_params = names.cosmological_parameters

"""
    !!!WORK IN PROGRESS!!!
    Currently non functional
"""

def setup(options):
	
    config = {}
    # input and output section names
    config['section_name'] = options.get_string(option_section, 'section_name',  default='obs_out')
    
    config['zmin'] = np.asarray([options[option_section, 'zmin']]).flatten()
    config['zmax'] = np.asarray([options[option_section, 'zmax']]).flatten()
    
    # Maybe we should do this also for the model cosmology and in halo_model_ingredients?
    # At least to specifiy the exact cosmology model, even though it should be a s close as general
    # as in CAMB, for which we can safely assume Flatw0waCDM does the job...
    
    cosmo_kwargs = ast.literal_eval(options.get_string(option_section, 'cosmo_kwargs',  default="{'H0':70.0, 'Om0':0.7, 'Ode0':0.3}"))
    # cosmo_kwargs is to be a string containing a dictionary!
    # ast.literal_eval("{'H0':h*100.0, 'Om0':omegav, 'Ode0':omegav}")
    
    cosmo_class = options.get_string(option_section, 'astropy_cosmology_class',  default='LambdaCDM')
    cosmo_class_init = getattr(astropy.cosmology, cosmo_class)
    cosmo_model_data = cosmo_class_init(**cosmo_kwargs)
    
    config['cosmo_model_data'] = cosmo_model_data
    config['h_data'] = cosmo_model_data.h
    
    return config
	

def execute(block, config):

    section_name  = config['section_name']
    zmin  = config['zmin']
    zmax  = config['zmax']
    nbins    = block[section_name, 'nbin']
    h_data = config['h_data']
    cosmo_model_data = config['cosmo_model_data']

    # Check if the length of zmin, zmax, nbins match
    if not np.all(np.array([len(config['zmin']), len(config['zmax'])]) == nbins):
        raise Exception('Error: zmin, zmax need to be of same length as \
                         the number of bins provided.')
                        
    # Adopting the same cosmology object as in halo_model_ingredients module
    try:
        tcmb = block[cosmo_params, 'TCMB']
    except:
        tcmb = 2.7255
    cosmo_model_run = Flatw0waCDM(
        H0=block[cosmo_params, 'hubble'], Ob0=block[cosmo_params, 'omega_b'],
        Om0=block[cosmo_params, 'omega_m'], m_nu=[0, 0, block[cosmo_params, 'mnu']], Tcmb0=tcmb,
        w0=block[cosmo_params, 'w'], wa=block[cosmo_params, 'wa'] )
    h_run = cosmo_model_run.h
    
    import matplotlib.pyplot as pl
    # number of bins for the observable this is given via len(suffixes)
    for i in range(nbins):
        
        obs_func = block[section_name, f'bin_{i+1}']
        obs_arr = block[section_name, f'obs_{i+1}']
        pl.plot(obs_arr, obs_func, color='black')
        comoving_volume_data = (cosmo_model_data.comoving_distance(zmax[i])**3.0 - cosmo_model_data.comoving_distance(zmin[i])**3.0) * h_data**3.0
        comoving_volume_model = (cosmo_model_run.comoving_distance(zmax[i])**3.0 - cosmo_model_run.comoving_distance(zmin[i])**3.0) * h_run**3.0
        
        ratio_obs = comoving_volume_model / comoving_volume_data
        obs_func_new = obs_func * ratio_obs
        
        pl.plot(obs_arr, obs_func_new, color='red')
        block.replace_double_array_1d(section_name, f'bin_{i+1}', obs_func_new)
        # AD: I think there is no change to the obs values (the stellar masses)
        #block.replace_double_array_1d(section_name, f'obs_{i+1}', obs_arr_new)

    pl.xscale('log')
    pl.xscale('log')
    pl.show()
    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass


