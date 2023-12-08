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

    #TODO - is 6 enough?
    ell_max = 6
    # WARNING: the function calculate_f_ell is only valid for ell_max <= 11
    # If you want to increase ell_max, you need to extend the function

    #TODO: what are these numbers in the Hankel transform?    
    # initialise Hankel transform
    h_transform = [HankelTransform(ell+0.5,300,0.01) for ell in range(0,ell_max+1,2)]
    # integration step size and number of r-bins to be reviewed! We might want to have more precise evaluation!

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
    # Also load the redshift dimension #TODO: can this be optimised?
    z = block['concentration_dm', 'z'] #This dimension/resolution here has been set by the nz in hmf_and_hbf.py
    nz=np.size(z)

    # Now I want to load the high resolution halo parameters calculated with the halo model module
    # and then downsample them to a lower resolution grid for the radial IA calculation
    # When downsampling we note that this doesn't need to be perfect, our final resolution does not need to 
    # perfectly match the user input value - just as close as possible

    mass_halo = block['concentration_dm', 'm_h']    
    nmass_halo = np.size(mass_halo) #The dimension/resolution here has been set by the nmass in hmf_and_hbf.py
    downsample_factor = int(nmass_halo/nmass_setup)  
    mass= mass_halo[::downsample_factor]      

    c_halo = block['concentration_dm', 'c']  #This has dimension nz,nmass_halo
    c = c_halo[:,::downsample_factor]

    r_s_halo = block['nfw_scale_radius_dm', 'rs'] #This has dimension nz,nmass_halo
    r_s = r_s_halo[:,::downsample_factor]

    rvir_halo = block['virial_radius', 'rvir_dm'] #This has dimension nmass_halo  #TODO:  why no redshift dimension?
    rvir = rvir_halo[::downsample_factor]

    k=k_setup

    #ell_max = 6
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
