"""
This module extrapolates the power spectra. It interpolates and extrapolates the
different power spectra in input to match the range and sampling of the matter_power_nl from CAMB.
The extrapolation method is not particurlarly advanced (numpy.interp) and would be good
to replace it with something more robust. 

Step 1: interpolate f_red to the z-bins of the pk of interest
Step 2: interpolate to the z and k-binning of the matter_power_nl
"""

from cosmosis.datablock import names, option_section
import sys
import numpy as np
from cosmosis.datablock import names, option_section
from scipy import interp
from scipy.interpolate import interp1d

import time


# We have a collection of commonly used pre-defined block section names.
# If none of the names here is relevant for your calculation you can use any
# string you want instead.
cosmo = names.cosmological_parameters

def	extrapolate_z(z_ext, z_vec, pk, nk):
	nz_ext = len(z_ext)
	pk_extz = np.empty([nz_ext, nk])
	for ik in range(0,nk):
		#pk_kfixed = pk[:,ik]
		pk_extz[:,ik] = interp(z_ext, z_vec, pk[:,ik])
	return pk_extz

def	extrapolate_k(k_ext, k, pk, nz):
	nk_ext = len(k_ext)
	pk_extk = np.empty([nz, nk_ext])
	for jz in range(0,nz):
		#pk_zfixed = pk[jz,:]
		pk_extk[jz,:] = interp(k_ext, k, pk[jz,:])
	return pk_extk

def extrapolate_power(block, power_section, z_ext, k_ext):
    # Note that we have first interpolated the f_red to the halo model pipeline z range
    k = block[power_section, 'k_h']
    z = block[power_section, 'z']
    nz = len(z)
    nk = len(k)
    pk_in = block[power_section, 'p_k']
    nz_ext = len(z_ext)
    pk_tot_ext_z = extrapolate_z(z_ext, z, pk_in, nk)
    pk_tot_ext = extrapolate_k(k_ext, k, pk_tot_ext_z, nz_ext)
        
    block.put_grid(power_section, 'z', z_ext, 'k_h', k_ext, 'p_k', pk_tot_ext)
		
#--------------------------------------------------------------------------------#	

def setup(options):
    #This function is called once per processor per chain.
    #It is a chance to read any fixed options from the configuration file,
    #load any data, or do any calculations that are fixed once.
		
    # matter
    p_mm_option = options[option_section, 'do_p_mm']
    # clustering
    p_gg_option = options[option_section, 'do_p_gg']
    # galaxy lensing
    p_gm_option = options[option_section, 'do_p_gm']
    # intrinsic alignment
    p_mI_option = options[option_section, 'do_p_mI']
    p_II_option = options[option_section, 'do_p_II']
    p_gI_option = options[option_section, 'do_p_gI']

    zmax =  options[option_section, 'zmax']
			
    return p_mm_option, p_gg_option, p_gm_option, p_mI_option, p_II_option, p_gI_option, zmax
	

def execute(block, config):
    #This function is called every time you have a new sample of cosmological and other parameters.
    #It is the main workhorse of the code. The block contains the parameters and results of any
    #earlier modules, and the config is what we loaded earlier.
	
    p_mm_option, p_gg_option, p_gm_option, p_mI_option, p_II_option, p_gI_option, zmax = config

    # load matter_power_nl k and z:
    z_lin = block['matter_power_lin', 'z']
    k_lin = block['matter_power_lin', 'k_h']
    
    if p_mm_option:
        extrapolate_power(block, 'matter_power_nl', z_lin, k_lin)
    if p_gg_option:
        extrapolate_power(block, 'galaxy_power', z_lin, k_lin)
    if p_gm_option:
        extrapolate_power(block, 'matter_galaxy_power', z_lin, k_lin)
    if p_mI_option:
        extrapolate_power(block, 'matter_intrinsic_power', z_lin, k_lin)
    if p_II_option:
        extrapolate_power(block, 'intrinsic_power', z_lin, k_lin)
    if p_gI_option:
        extrapolate_power(block, 'galaxy_intrinsic_power', z_lin, k_lin)
				
    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass


