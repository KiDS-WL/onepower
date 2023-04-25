"""
This module combines the red and blue power spectra. It interpolates and extrapolates the 
different power spectra in input to match the range and sampling of the matter_power_nl.
The extrapolation method is not particurlarly advanced (numpy.interp) and would be good
to replace it with something more robust. 

The red fraction as a function of redshift must be provided by the user ad a txt file with
columns (z, f_red(z)). The z-binning can be arbitrary (it is interpolated inside the code)
but it safe to provide the largest range possible to avoid substantial extrapolations. 

The code assume the red and blue power spectra to be computed on the same z and k binning.

Step 1: interpolate f_red to the z-bins of the pk of interest
Step 2: add red and blue power spectra
Step 3: interpolate to the z and k-binning of the matter_power_nl

NO CROSS TERMS ARE CURRENTLY IMPLEMENTED.

For each redshift, the power spectra are combined as following:

GI -> pk_tot = f_red * pk_red + (1-f_red) * pk_blue 
II -> pk_tot = f_red**2. * pk_red + (1-f_red)**2. * pk_blue
gI -> pk_tot = f_red**2. * pk_red + (1-f_red)**2. * pk_blue
gg -> pk_tot = f_red**2. * pk_red + (1-f_red)**2. * pk_blue
gm -> pk_tot = f_red * pk_red + (1-f_red) * pk_blue

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

def add_red_and_blue_power(block, f_red, power_section, z_ext, k_ext):
        # Note that we have first interpolated the f_red to the halo model pipeline z range
        k = block[power_section+'_red', 'k_h']
        z = block[power_section+'_red', 'z']
        nz = len(z)
        nk = len(k)
        pk_tot = np.zeros([nz,nk])
        pk_red = block[power_section+'_red', 'p_k']
        pk_blue = block[power_section+'_blue', 'p_k']
		
        # TODO: Add the cross terms
        # This is not optimised, but it is good to first choose what do we want to implement
        # in terms of cross terms.
        """
        if (power_section == 'intrinsic_power'):
            for jz in range(nz):
                pk_tot[jz] = f_red[jz]**2.*pk_red[jz] + (1.-f_red[jz])**2.*pk_blue[jz]
        if (power_section == 'galaxy_power'):
            for jz in range(nz):
                pk_tot[jz] = f_red[jz]**2.*pk_red[jz] + (1.-f_red[jz])**2.*pk_blue[jz]
        if (power_section == 'galaxy_intrinsic_power'):
            for jz in range(nz):
                pk_tot[jz] = f_red[jz]**2.*pk_red[jz] + (1.-f_red[jz])**2.*pk_blue[jz]
        else:
            #for jz in range(nz):
            #    pk_tot[jz] = f_red[jz]*pk_red[jz] + (1.-f_red[jz])*pk_blue[jz]
        """
        if power_section in ['intrinsic_power', 'galaxy_power', 'galaxy_intrinsic_power']:
            pk_tot = f_red[:,np.newaxis]**2.*pk_red + (1.-f_red[:,np.newaxis])**2.*pk_blue
        else:
            pk_tot = f_red[:,np.newaxis]*pk_red + (1.-f_red[:,np.newaxis])*pk_blue
        
        #warnings.warn('No cross terms between red and blue galaxies implemented.\nThis is only valid for IA in the regime of negligible blue galaxy alignment.')
        #IT 02/03/22: Commented line 86 to execute the code
        # extrapolate
        nz_ext = len(z_ext)
        pk_tot_ext_z = extrapolate_z(z_ext, z, pk_tot, nk)
        pk_tot_ext = extrapolate_k(k_ext, k, pk_tot_ext_z, nz_ext)
        
        block.put_grid(power_section, 'z', z_ext, 'k_h', k_ext, 'p_k', pk_tot_ext)
		
#--------------------------------------------------------------------------------#	

def setup(options):
    #This function is called once per processor per chain.
    #It is a chance to read any fixed options from the configuration file,
    #load any data, or do any calculations that are fixed once.
		
    f_red_file = options[option_section, 'f_red_file']
    z_fred, f_red = np.loadtxt(f_red_file, unpack=True)
    print (z_fred, f_red)
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
			
    return z_fred, f_red, p_mm_option, p_gg_option, p_gm_option, p_mI_option, p_II_option, p_gI_option, zmax
	

def execute(block, config):
    #This function is called every time you have a new sample of cosmological and other parameters.
    #It is the main workhorse of the code. The block contains the parameters and results of any
    #earlier modules, and the config is what we loaded earlier.
	
    z_fred_file, f_red_file, p_mm_option, p_gg_option, p_gm_option, p_mI_option, p_II_option, p_gI_option, zmax = config

    # load matter_power_nl k and z:
    z_nl = block['matter_power_lin', 'z']
    k_nl = block['matter_power_lin', 'k_h']
    
    #"""
    if p_mm_option:
        # load halo model k and z (red and blue are expected to be with the same red/blue ranges and z,k-samplings!):
        z_hm = block['matter_power_nl', 'z']
        k_hm = block['matter_power_nl', 'k_h']
        nz = len(z_hm)
        nk = len(k_hm)
        pk_nl = block['matter_power_nl', 'p_k']
        nz_ext = len(z_nl)
        pk_tot_ext_z = extrapolate_z(z_nl, z_hm, pk_nl, nk)
        pk_tot_ext = extrapolate_k(k_nl, k_hm, pk_tot_ext_z, nz_ext)
        
        block.replace_grid('matter_power_nl', 'z', z_nl, 'k_h', k_nl, 'p_k', pk_tot_ext)
        #add_red_and_blue_power(block, f_red(z_hm), 'matter_power', z_nl, k_nl)
    #"""
    if p_gg_option:
        # load halo model k and z (red and blue are expected to be with the same red/blue ranges and z,k-samplings!):
        z_hm = block['galaxy_power_red', 'z']
        f_red = interp1d(z_fred_file, f_red_file, 'linear', bounds_error=False, fill_value='extrapolate')
        add_red_and_blue_power(block, f_red(z_hm), 'galaxy_power', z_nl, k_nl)
    if p_gm_option:
        # load halo model k and z (red and blue are expected to be with the same red/blue ranges and z,k-samplings!):
        z_hm = block['matter_galaxy_power_red', 'z']
        #IT Added bounds_error=False and fill_value extrapolate
        f_red = interp1d(z_fred_file, f_red_file, 'linear', bounds_error=False, fill_value='extrapolate')
        add_red_and_blue_power(block, f_red(z_hm), 'matter_galaxy_power', z_nl, k_nl)
    if p_mI_option:
        # load halo model k and z (red and blue are expected to be with the same red/blue ranges and z,k-samplings!):
        z_hm = block['matter_intrinsic_power_red', 'z']
        f_red = interp1d(z_fred_file, f_red_file, 'linear', bounds_error=False, fill_value='extrapolate')
        add_red_and_blue_power(block, f_red(z_hm), 'matter_intrinsic_power', z_nl, k_nl)
    if p_II_option:
        # load halo model k and z (red and blue are expected to be with the same red/blue ranges and z,k-samplings!):
        z_hm = block['intrinsic_power_red', 'z']
        f_red = interp1d(z_fred_file, f_red_file, 'linear', bounds_error=False, fill_value='extrapolate')
        add_red_and_blue_power(block, f_red(z_hm), 'intrinsic_power', z_nl, k_nl)
    if p_gI_option:
        # load halo model k and z (red and blue are expected to be with the same red/blue ranges and z,k-samplings!):
        z_hm = block['galaxy_intrinsic_power_red', 'z']
        f_red = interp1d(z_fred_file, f_red_file, 'linear', bounds_error=False, fill_value='extrapolate')
        add_red_and_blue_power(block, f_red(z_hm), 'galaxy_intrinsic_power', z_nl, k_nl)
				
    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass


