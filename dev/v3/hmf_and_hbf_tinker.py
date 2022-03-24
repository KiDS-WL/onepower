# Compute the HMF through the Steven Murray python package hmf
# At the moment this module is ONLY designed to work in this particular pipeline
# To wrap it properly in CosmoSIS more work is needed. It also requires a
# permission from the authors -> I haven't contact them yet, so this must be
# considered only for a private use.

# The module also includes an option to compute the halo bias from Tinker+10.



from cosmosis.datablock import names, option_section
import sys
import numpy as np
from scipy.interpolate import interp1d, interp2d
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM
import astropy.units as u
from scipy.integrate import simps
from hmf import MassFunction
from halomod import bias as bias_func

# AD: Leaving in for now...
def tinker_bias(nu, Delta=200., delta_c=1.686):
    nu = nu**0.5
    # Table 2, Tinker+2010
    y = np.log10(Delta)
    expvar = np.exp(-(4./y)**4.)
    A = 1.+0.24*y*expvar
    a = 0.44*y-0.88
    B = 0.183
    b = 1.5
    C = 0.019+0.107*y+0.19*expvar
    c = 2.4
    # equation 6
    bias = 1.-A*(nu**a)/(nu**a+delta_c**a) + B*nu**b + C*nu**c
    return bias


# We have a collection of commonly used pre-defined block section names.
# If none of the names here is relevant for your calculation you can use any
# string you want instead.
cosmo_names = "cosmological_parameters"

def setup(options):
    #This function is called once per processor per chain.
    #It is a chance to read any fixed options from the configuration file,
    #load any data, or do any calculations that are fixed once.

    log_mass_min = options[option_section, "log_mass_min"]
    log_mass_max = options[option_section, "log_mass_max"]
    nmass = options[option_section, "nmass"]

    zmin = options[option_section, "zmin"]
    zmax = options[option_section, "zmax"]
    nz = options[option_section, "nz"]
    z_vec = np.linspace(zmin, zmax, nz)
    print(z_vec)

    dlog10m = (log_mass_max-log_mass_min)/nmass

    #hmf_model = options[option_section, "hmf_model"]
    halo_bias_option = options[option_section, "do_halo_bias"]


    initialise_cosmo=Flatw0waCDM(
        H0=70., Ob0=0.044, Om0=0.3, Tcmb0=2.725, w0=-1., wa=0.)
    

    mf = MassFunction(z=0., cosmo_model=initialise_cosmo, Mmin=log_mass_min, Mmax=log_mass_max, dlog10m=dlog10m, sigma_8=0.8, n=0.96,
					  hmf_model=options[option_section, "hmf"], mdef_model=options[option_section, "mdef_model"], mdef_params={'overdensity':options[option_section, "overdensity"]}, transfer_model='EH', delta_c=options[option_section, "delta_c"])
    # This mf parameters that are fixed here now need to be read from the ini files! Need to make sure camb is not called when initialising the mf!
    print( mf.cosmo)
    
    mass = mf.m

    return log_mass_min, log_mass_max, nmass, dlog10m, z_vec, nz, halo_bias_option, mass, mf, options[option_section, "overdensity"], options[option_section, "delta_c"], options[option_section, "hmf"]


def execute(block, config):
    #This function is called every time you have a new sample of cosmological and other parameters.
    #It is the main workhorse of the code. The block contains the parameters and results of any 
    #earlier modules, and the config is what we loaded earlier.

    log_mass_min, log_mass_max, nmass, dlog10m, z_vec, nz, halo_bias_option, mass, mf, overdensity, delta_c, bias_model = config

    # Update the cosmological parameters
    #this_cosmo.update(cosmo_params={"H0":block[cosmo_names, "hubble"], "Om0":block[cosmo_names, "omega_m"], "Ob0":block[cosmo_names, "omega_b"]})
    this_cosmo_run=Flatw0waCDM(
        H0=block[cosmo_names, "hubble"], Ob0=block[cosmo_names, "omega_b"], Om0=block[cosmo_names, "omega_m"], Tcmb0=2.725,
		w0=block[cosmo_names, "w"], wa=block[cosmo_names, "wa"])
    ns = block[cosmo_names, "n_s"]

    #--------------------------------------#
    # read sigma_8 for the given cosmology
    #--------------------------------------#

    # Note that CAMB does not return the sigma_8 at z=0, as it might seem from the documentation, but sigma_8(z),
    # so the user should always start from z=0
    sigma_8 = block[cosmo_names, 'sigma_8']
    #print ('sigma_8 = ', sigma_8)

    nmass_hmf = len(mf.m)

    dndlnmh = np.empty([nz, nmass_hmf])
    nu = np.empty([nz,nmass_hmf])
    mean_density0 = np.empty([nz])
    mean_density_z = np.empty([nz])
   
    for jz in range(0,nz):
        mf.update(z=z_vec[jz], cosmo_model=this_cosmo_run, sigma_8=sigma_8, n=ns)

        print ( 'mf.mean_density0 = ', mf.mean_density0 )

        # Tinker assumes a different definition of nu:
        nu[jz] = mf.nu #sqrt not needed, done internally in bias function!
        dndlnmh[jz] = mf.dndlnm
        mean_density0[jz] = mf.mean_density0
        mean_density_z[jz] = mf.mean_density
        #matter_power_lin[jz+1] = mf.power
        # AD: add here the mean_density at z=0 as output, growth factor, etc!

    block.put_grid("hmf", "z", z_vec, "m_h", mass, "dndlnmh", dndlnmh)
    block.put_double_array_1d("density", "mean_density0", mean_density0)
    block.put_double_array_1d("density", "mean_density_z", mean_density_z)
    block.put_double_array_1d("density", "rho_crit", mean_density0/this_cosmo_run.Om0)


    #--------------------------------------#
    # HALO BIAS
    #--------------------------------------#
    

    if halo_bias_option:
        b_nu = np.empty([nz,len(mf.m)])

        # AD: remove this loop?
        for jz in range(0,nz):
            bias = getattr(bias_func, bias_model)(nu[jz], delta_c=delta_c, delta_halo=overdensity, sigma_8=sigma_8, n=ns, cosmo=this_cosmo_run, m=mass)
            b_nu[jz] = bias.bias()
        block.put_grid("halobias", "z", z_vec, "m_h", mass, "b_hb", b_nu)


    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
