from cosmosis.datablock import names, option_section
import sys
import numpy as np
import hankel
import math

import time
from itertools import count
from numpy import absolute, array, arange, cos, exp, linspace, log10, \
                  logspace, max as npmax, median, newaxis, ones, outer,\
                  pi, sin,  sum as npsum, zeros
from scipy.integrate import quad, simps, trapz
from scipy.interpolate import interp1d, interp2d
from scipy.special import legendre, sici, binom
import math


from uell_radial_dependent_alignment_lib_v3_eps import radvir, IA_uell_gamma_r_hankel, wkm_my_fell
from hankel import HankelTransform

# We have a collection of commonly used pre-defined block section names.
# If none of the names here is relevant for your calculation you can use any
# string you want instead.

cosmo = names.cosmological_parameters


#--------------------------------------------------------------------------------#	

def setup(options):
    #This function is called once per processor per chain.
    #It is a chance to read any fixed options from the configuration file,
    #load any data, or do any calculations that are fixed once.

    log_mass_min = options[option_section, 'log_mass_min']
    log_mass_max = options[option_section, 'log_mass_max']
    nmass = options[option_section, 'nmass']
    #log-spaced mass in units of M_sun/h
    dlog10m = (log_mass_max-log_mass_min)/nmass
    mass = 10.0 ** np.arange(log_mass_min, log_mass_max, dlog10m)

    zmin = options[option_section, 'zmin']
    zmax = options[option_section, 'zmax']
    nz = options[option_section, 'nz']
    z_vec = np.linspace(zmin, zmax, nz)

    kmin = options[option_section, 'kmin']
    kmax = options[option_section, 'kmax']
    nk = options[option_section, 'nk']
    k_vec = np.logspace(np.log10(kmin), np.log10(kmax), nk)

    name = options.get_string(option_section, 'name', default='').lower()
    if name:
        suffix = '_' + name
    else:
        suffix = ''
    ell_max = 6
    # initialise Hankel transform
    h_transform = [HankelTransform(ell+0.5,300,0.01) for ell in range(0,ell_max+1,2)]
    # integration step size and number of r-bins to be reviewed! We might want to have more precise evaluation!
    #print(h_transform)

    return z_vec, nz, mass, nmass, k_vec, nk, suffix, h_transform, ell_max

def execute(block, config):
    #This function is called every time you have a new sample of cosmological and other parameters.
    #It is the main workhorse of the code. The block contains the parameters and results of any 
    #earlier modules, and the config is what we loaded earlier.
    
    z_setup, nz_setup, mass_setup, nmass_setup, k_setup, nk_setup, suffix, h_transform, ell_max = config
    start_time = time.time()

    # create intermediate variables to speed up the calculation
    nz = nz_setup
    z = z_setup
    
    #nmass = 100
    #mass = np.logspace(np.log10(mass_setup.min()), np.log10(mass_setup.max()), nmass)
    nk = 80 #500
    #k = np.logspace(np.log10(k_setup.min()), np.log10(k_setup.max()), nk)
    k = np.logspace(-1., 3., nk)

    #k = np.logspace(-2., np.log10(k_setup.max()), nk)

    # Load slope of the power law that describes the satellite alignment
    gamma_1h_slope = block['intrinsic_alignment_parameters' + suffix, 'gamma_1h_radial_slope']
    # This already contains the luminosity dependence if there
    gamma_1h_amplitude = block['ia_small_scale_alignment' + suffix, 'alignment_1h']

    mass_halo = block['concentration_dm', 'm_h']
    z_halo = block['concentration_dm', 'z']
    c = block['concentration_dm', 'c']
    r_s = block['nfw_scale_radius_dm', 'rs']
    rvir = block['virial_radius', 'rvir_dm']
    mass = mass_halo


    #print('r_s.shape = ', r_s.shape)
    #print('rvir.shape = ', rvir.shape)
    #print('c.shape = ', c.shape)
    #print('mass.shape = ', mass.shape)

    #ell_max = 6
    # uell[l,z,m,k]
    # AD: THIS FUNCTION IS THE SLOWEST PART!
    uell = IA_uell_gamma_r_hankel(gamma_1h_amplitude, gamma_1h_slope, k, c, z, r_s, rvir, mass, ell_max, h_transform)
    #print (uell.shape)
    # interpolate
    # Do we need this interpolation???
    uell_interpolated = np.empty([int(ell_max/2+1), nz, nmass_setup, nk_setup])
    for il in range(0,int(ell_max/2+1)):
        for jz in range(0,nz):
            f_interp = interp2d(k, mass, uell[il, jz], kind='linear', bounds_error=False) #, fill_value=0)
            uell_interpolated[il,jz] = f_interp(k_setup, mass_setup)
    #print ('interp ok')
    
    
    # wkm[nz,nmass,nk]
    theta_k = np.pi/2.
    phi_k = 0.
    wkm = wkm_my_fell(uell_interpolated, theta_k, phi_k, ell_max, gamma_1h_slope)
    #print('wkm.shape = ', wkm.shape)
    for jz in range(0,nz):
        block.put_grid( 'wkm_z%d'%jz+suffix, 'mass', mass_setup, 'k_h', k_setup, 'w_km', wkm[jz,:,:])
    block.put_double_array_1d('wkm'+suffix, 'z', z)

    #print('--- wkm: %s seconds ---' % (time.time() - start_time))


    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
