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
import numpy as np
from scipy.interpolate import interp1d

# We have a collection of commonly used pre-defined block section names.
# If none of the names here is relevant for your calculation you can use any
# string you want instead.
cosmo = names.cosmological_parameters


def add_red_and_blue_power(block, suffix_red, suffix_blue, suffix_out, f_red, power_section, z_ext, k_ext, extrapolate_option):
    # Note that we have first interpolated the f_red to the halo model pipeline z range
    k = block[power_section + suffix_red, 'k_h']
    z = block[power_section + suffix_red, 'z']
    pk_red = block[power_section + suffix_red, 'p_k']
    pk_blue = block[power_section + suffix_blue, 'p_k']
		
    # TODO: Add the cross terms
    # This is not optimised, but it is good to first choose what do we want to implement
    # in terms of cross terms.
    if power_section in ['intrinsic_power', 'galaxy_power', 'galaxy_intrinsic_power']:
        pk_tot = f_red[:,np.newaxis]**2.*pk_red + (1.-f_red[:,np.newaxis])**2.*pk_blue
    else:
        pk_tot = f_red[:,np.newaxis]*pk_red + (1.-f_red[:,np.newaxis])*pk_blue
        
    #warnings.warn('No cross terms between red and blue galaxies implemented.\nThis is only valid for IA in the regime of negligible blue galaxy alignment.')
    #IT 02/03/22: Commented line 86 to execute the code
    
    # extrapolate
    inter_func_z = interp1d(z, np.nan_to_num(np.log10(pk_tot)), kind='linear', fill_value=extrapolate_option, bounds_error=False, axis=0)
    pk_tot_ext_z = 10.0**inter_func_z(z_ext)
    
    inter_func_k = interp1d(np.log10(k), np.nan_to_num(np.log10(pk_tot_ext_z)), kind='linear', fill_value='extrapolate', bounds_error=False, axis=1)
    pk_tot_ext = 10.0**inter_func_k(np.log10(k_ext))
        
    block.put_grid(power_section + suffix_out, 'z', z_ext, 'k_h', k_ext, 'p_k', pk_tot_ext)


def extrapolate_power(block, suffix_out, suffix_in, power_section, z_ext, k_ext, extrapolate_option):
    k = block[power_section + suffix_in, 'k_h']
    z = block[power_section + suffix_in, 'z']
    pk_in = block[power_section + suffix_in, 'p_k']
    
    inter_func_z = interp1d(z, np.nan_to_num(np.log10(pk_in)), kind='linear', fill_value=extrapolate_option, bounds_error=False, axis=0)
    pk_tot_ext_z = 10.0**inter_func_z(z_ext)
        
    inter_func_k = interp1d(np.log10(k), np.nan_to_num(np.log10(pk_tot_ext_z)), kind='linear', fill_value='extrapolate', bounds_error=False, axis=1)
    pk_tot_ext = 10.0**inter_func_k(np.log10(k_ext))
        
    block.put_grid(power_section + suffix_out, 'z', z_ext, 'k_h', k_ext, 'p_k', pk_tot_ext)
 
 
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

    if any(option == 'add_and_extrapolate' for option in [p_gg_option, p_gm_option, p_mI_option, p_II_option, p_gI_option]):
        f_red_file = options[option_section, 'f_red_file']
        z_fred, f_red = np.loadtxt(f_red_file, unpack=True)
        print(z_fred, f_red)
    else:
        print('Only extrapolating power spectra.')
        z_fred, f_red = None, None

    name_extrap = options.get_string(option_section, 'input_suffix_extrap', default='').lower()
    name_red = options.get_string(option_section, 'input_suffix_red', default='').lower()
    name_blue = options.get_string(option_section, 'input_suffix_blue', default='').lower()
    if name_extrap != '':
        suffix_extrap = '_' + name_extrap
    else:
        suffix_extrap = ''
    if name_red != '':
        suffix_red = '_' + name_red
    else:
        suffix_red = ''
    if name_blue != '':
        suffix_blue = '_' + name_blue
    else:
        suffix_blue = ''
			
    return z_fred, f_red, p_mm_option, p_gg_option, p_gm_option, p_mI_option, p_II_option, p_gI_option, suffix_extrap, suffix_red, suffix_blue
	

def execute(block, config):
    #This function is called every time you have a new sample of cosmological and other parameters.
    #It is the main workhorse of the code. The block contains the parameters and results of any
    #earlier modules, and the config is what we loaded earlier.
	
    z_fred_file, f_red_file, p_mm_option, p_gg_option, p_gm_option, p_mI_option, p_II_option, p_gI_option, suffix0_extrap, suffix0_red, suffix0_blue = config

    # load matter_power_nl k and z:
    z_lin = block['matter_power_lin', 'z']
    k_lin = block['matter_power_lin', 'k_h']
    
    
    if p_mm_option == 'extrapolate':
        extrapolate_power(block, '','', 'matter_power_nl', z_lin, k_lin, 'extrapolate')
        # TODO: Remove  once extrapolation of NL power spectra is validated
        try:
            extrapolate_power(block, '','', 'matter_power_nl_mead', z_lin, k_lin, 'extrapolate')
        except:
            pass
    
    if any(option == 'extrapolate' for option in [p_gg_option, p_gm_option, p_mI_option, p_II_option, p_gI_option]):
        hod_bins_extrap = block['hod' + suffix0_extrap + '_params', 'nbins']
        observables_z = block['hod' + suffix0_extrap + '_params', 'option']
        
        if observables_z == True:
            extrapolate_option = 'extrapolate'
        if observables_z == False:
            extrapolate_option = 0.0
        
        for nb in range(0,hod_bins_extrap):
            if hod_bins_extrap != 1:
                suffix_extrap = suffix0_extrap + '_{}'.format(nb+1)
                suffix_out = '_{}'.format(nb+1)
            else:
                suffix_extrap = suffix0_extrap
                suffix_out = ''
                
            if p_gg_option == 'extrapolate':
                extrapolate_power(block, suffix_out, suffix_extrap, 'galaxy_power', z_lin, k_lin, extrapolate_option)
                
            if p_gm_option == 'extrapolate':
                extrapolate_power(block, suffix_out, suffix_extrap, 'matter_galaxy_power', z_lin, k_lin, extrapolate_option)
                
            if p_mI_option == 'extrapolate':
                extrapolate_power(block, suffix_out, suffix_extrap, 'matter_intrinsic_power', z_lin, k_lin, extrapolate_option)
                
            if p_II_option == 'extrapolate':
                extrapolate_power(block, suffix_out, suffix_extrap, 'intrinsic_power', z_lin, k_lin, extrapolate_option)
                
            if p_gI_option == 'extrapolate':
                extrapolate_power(block, suffix_out, suffix_extrap, 'galaxy_intrinsic_power', z_lin, k_lin, extrapolate_option)
        
        
    if any(option == 'add_and_extrapolate' for option in [p_gg_option, p_gm_option, p_mI_option, p_II_option, p_gI_option]):
        hod_bins_red = block['hod' + suffix0_red + '_params', 'nbins']
        hod_bins_blue = block['hod' + suffix0_blue + '_params', 'nbins']
        
        observables_z_red = block['hod' + suffix0_red + '_params', 'option']
        if observables_z_red == True:
            extrapolate_option = 'extrapolate'
        if observables_z_red == False:
            extrapolate_option = 0
        
        if not hod_bins_red == hod_bins_blue:
            raise Exception('Error: number of red and blue stellar mass bins should be the same.')
    
        for nb in range(0,hod_bins_red):
            if hod_bins_red != 1:
                suffix_red = suffix0_red + '_{}'.format(nb+1)
                suffix_blue = suffix0_blue + '_{}'.format(nb+1)
                suffix_out = '_{}'.format(nb+1)
            else:
                suffix_red = suffix0_red
                suffix_blue = suffix0_blue
                suffix_out = ''
        
            if p_gg_option == 'add_and_extrapolate':
                # load halo model k and z (red and blue are expected to be with the same red/blue ranges and z,k-samplings!):
                z_hm = block['galaxy_power' + suffix_red, 'z']
                f_red = interp1d(z_fred_file, f_red_file, 'linear', bounds_error=False, fill_value='extrapolate')
                add_red_and_blue_power(block, suffix_red, suffix_blue, suffix_out, f_red(z_hm), 'galaxy_power', z_lin, k_lin, extrapolate_option)
                
            if p_gm_option == 'add_and_extrapolate':
                # load halo model k and z (red and blue are expected to be with the same red/blue ranges and z,k-samplings!):
                z_hm = block['matter_galaxy_power' + suffix_red, 'z']
                #IT Added bounds_error=False and fill_value extrapolate
                f_red = interp1d(z_fred_file, f_red_file, 'linear', bounds_error=False, fill_value='extrapolate')
                add_red_and_blue_power(block, suffix_red, suffix_blue, suffix_out, f_red(z_hm), 'matter_galaxy_power', z_lin, k_lin, extrapolate_option)
                
            if p_mI_option == 'add_and_extrapolate':
                # load halo model k and z (red and blue are expected to be with the same red/blue ranges and z,k-samplings!):
                z_hm = block['matter_intrinsic_power' + suffix_red, 'z']
                f_red = interp1d(z_fred_file, f_red_file, 'linear', bounds_error=False, fill_value='extrapolate')
                add_red_and_blue_power(block, suffix_red, suffix_blue, suffix_out, f_red(z_hm), 'matter_intrinsic_power', z_lin, k_lin, extrapolate_option)
                
            if p_II_option == 'add_and_extrapolate':
                # load halo model k and z (red and blue are expected to be with the same red/blue ranges and z,k-samplings!):
                z_hm = block['intrinsic_power' + suffix_red, 'z']
                f_red = interp1d(z_fred_file, f_red_file, 'linear', bounds_error=False, fill_value='extrapolate')
                add_red_and_blue_power(block, suffix_red, suffix_blue, suffix_out, f_red(z_hm), 'intrinsic_power', z_lin, k_lin, extrapolate_option)
                
            if p_gI_option == 'add_and_extrapolate':
                # load halo model k and z (red and blue are expected to be with the same red/blue ranges and z,k-samplings!):
                z_hm = block['galaxy_intrinsic_power' + suffix_red, 'z']
                f_red = interp1d(z_fred_file, f_red_file, 'linear', bounds_error=False, fill_value='extrapolate')
                add_red_and_blue_power(block, suffix_red, suffix_blue, suffix_out, f_red(z_hm), 'galaxy_intrinsic_power', z_lin, k_lin, extrapolate_option)
				
    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass


