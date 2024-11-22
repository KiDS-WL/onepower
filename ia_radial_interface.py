from cosmosis.datablock import option_section
import numpy as np

from uell_radial_dependent_alignment_lib import IA_uell_gamma_r_hankel, wkm_f_ell
from hankel import HankelTransform

#--------------------------------------------------------------------------------#	

def setup(options):
    #This function is called once per processor per chain.
    #It is a chance to read any fixed options from the configuration file,
    #load any data, or do any calculations that are fixed once.
    
    # Here we're setting the resolution of the redshift, mass and k grid that we'll use
    # to calculate the w(k,z|m) function which is slow: the lower the resolution the better
    # and we'll interpolate over this later to get the
    # right resolution for the power spectrum calculation

    # The set of default parameters here are fast and reasonably accurate
    # Needs more testing to be sure what the optimal defaults are though
    # TODO:  I have not yet tested the resolution in the z-dimension

    nmass = options.get_int(option_section, 'nmass',default=5)
    kmin = options.get_double(option_section, 'kmin',default=1e-3)
    kmax = options.get_double(option_section, 'kmax',default=1e3)
    nk = options.get_int(option_section, 'nk', default=10)
    k_vec = np.logspace(np.log10(kmin), np.log10(kmax), nk)

    # Are we calculating the alignment for say red or blue galaxies?
    name = options.get_string(option_section, 'output_suffix', default='').lower()
    if name != '':
        suffix = f'_{name}'
    else:
        suffix = ''
  
    # CCL and Fortuna use ell_max=6.  SB10 uses ell_max = 2.  
    # Higher/lower increases/decreases accuracy but slows/speeds the code
    ell_max = options.get_int(option_section, 'ell_max', default=6)
    #if ell_max > 11 then return a warning and stop
    if ell_max > 11:
        raise ValueError("Please reduce ell_max<11 or update ia_radial_interface.py")
    
    # initialise Hankel transform
    #HankelTransform(nu, # The order of the bessel function
    #                N,  # Number of steps in the integration
    #                h   # Proxy for "size" of steps in integration)
    # We've used hankel.get_h to set h, N is then h=pi/N, finding best_h = 0.05, best_N=62
    #If you want perfect agreement with CCL use: N=50000, h=0.00006 (VERY SLOW!!)

    N_hankel = options.get_int(option_section, 'N_hankel', default=350)
    h_hankel = np.pi/N_hankel
    h_transform = [HankelTransform(ell+0.5,N_hankel,h_hankel) for ell in range(0,ell_max+1,2)]

    return k_vec, nmass, suffix, h_transform, ell_max

def execute(block, config):
    #This function is called every time you have a new sample of cosmological and other parameters.
    #It is the main workhorse of the code. The block contains the parameters and results of any 
    #earlier modules, and the config is what we loaded earlier.
    
    k_setup, nmass_setup, suffix, h_transform, ell_max = config

    # Load slope of the power law that describes the satellite alignment
    gamma_1h_slope = block[f'intrinsic_alignment_parameters{suffix}', 'gamma_1h_radial_slope']
    # This already contains the luminosity dependence if there
    gamma_1h_amplitude = block[f'ia_small_scale_alignment{suffix}', 'alignment_1h']
    # Also load the redshift dimension 
    z = block['concentration_m', 'z'] #This dimension/resolution here has been set by the nz in halo_model_ingredients.py
    nz=np.size(z)

    # Now I want to load the high resolution halo parameters calculated with the halo model module
    # and then downsample them to a lower resolution grid for the radial IA calculation
    # When downsampling we note that this doesn't need to be perfect, our final resolution does not need to 
    # perfectly match the user input value - just as close as possible

    mass_halo = block['concentration_m', 'm_h']    
    nmass_halo = np.size(mass_halo) #The dimension/resolution here has been set by the nmass in halo_model_ingredients.py
    c_halo = block['concentration_m', 'c']  #This has dimension nz,nmass_halo
    r_s_halo = block['nfw_scale_radius_dm', 'rs'] #This has dimension nz,nmass_halo
    rvir_halo = block['virial_radius', 'rvir_m'] #This has dimension nmass_halo : rvir doesn't change with z, hence no z-dimension

    if nmass_halo < nmass_setup:
        raise ValueError("The halo mass resolution is too low for the radial IA calculation. Please increase nmass when you run halo_model_ingredients.py")
    elif nmass_halo == nmass_setup:
        mass = mass_halo
        c = c_halo
        r_s = r_s_halo
        rvir = rvir_halo
    else:
        downsample_factor = int(nmass_halo/nmass_setup)
        mass= mass_halo[::downsample_factor]
        c = c_halo[:,::downsample_factor]
        r_s = r_s_halo[:,::downsample_factor]
        rvir = rvir_halo[::downsample_factor]
        # and we need to make sure that the highest mass is included to avoid extrapolation issues
        if mass[-1] != mass_halo[-1]:
            mass = np.append(mass, mass_halo[-1])
            c = np.concatenate((c,np.atleast_2d(c_halo[:,-1]).T),axis=1)
            r_s = np.concatenate((r_s, np.atleast_2d(r_s_halo[:,-1]).T), axis=1) #Make sure we include the highest scale radius in there
            rvir = np.append(rvir, rvir_halo[-1]) #Make sure we have the highest virial radius in there

    k=k_setup
    # uell[l,z,m,k]
    # AD: THIS FUNCTION IS THE SLOWEST PART!
    uell = IA_uell_gamma_r_hankel(gamma_1h_amplitude, gamma_1h_slope, k, c, z, r_s, rvir, mass, ell_max, h_transform)
    theta_k = np.pi/2.
    phi_k = 0.
    wkm = wkm_f_ell(uell, theta_k, phi_k, ell_max, gamma_1h_slope)  #This has low-res dimension [nz,nmass,nk]

    for jz in range(0,nz):
        block.put_grid( 'wkm', f'mass_{jz}{suffix}', mass, f'k_h_{jz}{suffix}', k, f'w_km_{jz}{suffix}', wkm[jz,:,:])
    block.put_double_array_1d('wkm', f'z{suffix}', z)

    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
