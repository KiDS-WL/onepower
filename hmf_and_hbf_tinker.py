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
from halomod.concentration import make_colossus_cm
from halomod import concentration as conc_func
from colossus.cosmology import cosmology as colossus_cosmology
from colossus.halo import concentration as colossus_concentration
import hmf.halos.mass_definitions as md
import time

def concentration_colossus_old(block, mass, z_vec, model, mdef, overdensity):
    # calculates concentration given halo mass, using the colossus model provided in config
    # furthermore it converts to halomod instance to be used with the halomodel, consistenly with halo mass function
    # and halo bias function
    nz = len(z_vec)
    nmass = len(mass)
    c = np.empty([nz, nmass])
    
    conc_func = make_colossus_cm(model=model)()
    conc_func.mdef = getattr(md, mdef)()
    if not mdef in ['SOVirial']:
        conc_func.overdensity = overdensity
    for i,zi in enumerate(z_vec):
        c[i,:] = np.abs(conc_func.cm(mass, zi))
    #c = conc_func.cm(mass[np.newaxis,:], z_vec[:,np.newaxis])
    return c
    
def concentration_colossus_new(block, cosmo, mass, z_vec, model, mdef, overdensity):
    # calculates concentration given halo mass, using the halomod model provided in config
    # furthermore it converts to halomod instance to be used with the halomodel, consistenly with halo mass function
    # and halo bias function
    nz = len(z_vec)
    nmass = len(mass)
    c = np.empty([nz, nmass])
    this_cosmo = colossus_cosmology.fromAstropy(astropy_cosmo=cosmo, cosmo_name='custom', sigma8=block[cosmo_names, 'sigma_8'], ns=block[cosmo_names, 'n_s'])
    mdef = getattr(md, mdef)() if mdef in ['SOVirial'] else getattr(md, mdef)(overdensity=overdensity)
    for i,zi in enumerate(z_vec):
        c[i,:] = np.abs(colossus_concentration.concentration(M=mass, z=zi, mdef=mdef.colossus_name, model=model))
    return c


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
cosmo_names = 'cosmological_parameters'

def setup(options):
    #This function is called once per processor per chain.
    #It is a chance to read any fixed options from the configuration file,
    #load any data, or do any calculations that are fixed once.

    log_mass_min = options[option_section, 'log_mass_min']
    log_mass_max = options[option_section, 'log_mass_max']
    nmass = options[option_section, 'nmass']

    zmin = options[option_section, 'zmin']
    zmax = options[option_section, 'zmax']
    nz = options[option_section, 'nz']
    z_vec = np.linspace(zmin, zmax, nz)
    
    dlog10m = (log_mass_max-log_mass_min)/nmass
    
    # most general astropy cosmology initialisation, gets updated as sampler runs with camb provided cosmology parameters.
    # setting some values to generate instance
    initialise_cosmo=Flatw0waCDM(
        H0=100., Ob0=0.044, Om0=0.3, Tcmb0=2.7255, w0=-1., wa=0.)
    mdef_model = options[option_section, 'mdef_model']
    overdensity = options[option_section, 'overdensity']
    delta_c = options[option_section, 'delta_c']
    mdef_params = {} if mdef_model in ['SOVirial'] else {'overdensity':overdensity}
        
    mf = MassFunction(z=0., cosmo_model=initialise_cosmo, Mmin=log_mass_min, Mmax=log_mass_max, dlog10m=dlog10m, sigma_8=0.8, n=0.96,
					  hmf_model=options[option_section, 'hmf_model'], mdef_model=mdef_model, mdef_params=mdef_params, transfer_model='CAMB', delta_c=delta_c, disable_mass_conversion=False)
    # This mf parameters that are fixed here now need to be read from the ini files! Need to make sure camb is not called when initialising the mf!
    #print( mf.cosmo)
    
    mass = mf.m

    return log_mass_min, log_mass_max, nmass, dlog10m, z_vec, nz, mass, mf, options[option_section, 'cm_model'], mdef_model, overdensity, delta_c, options[option_section, 'bias_model']


def execute(block, config):
    #This function is called every time you have a new sample of cosmological and other parameters.
    #It is the main workhorse of the code. The block contains the parameters and results of any 
    #earlier modules, and the config is what we loaded earlier.

    log_mass_min, log_mass_max, nmass, dlog10m, z_vec, nz, mass, mf, model_cm, mdef, overdensity, delta_c, bias_model = config

    # Update the cosmological parameters
    #this_cosmo.update(cosmo_params={'H0':block[cosmo_names, 'hubble'], 'Om0':block[cosmo_names, 'omega_m'], 'Ob0':block[cosmo_names, 'omega_b']})
    this_cosmo_run=Flatw0waCDM(
        H0=block[cosmo_names, 'hubble'], Ob0=block[cosmo_names, 'omega_b'], Om0=block[cosmo_names, 'omega_m'], m_nu=block[cosmo_names, 'mnu'], Tcmb0=block[cosmo_names, 'TCMB'],
		w0=block[cosmo_names, 'w'], wa=block[cosmo_names, 'wa'])
    ns = block[cosmo_names, 'n_s']

    # Note that CAMB does not return the sigma_8 at z=0, as it might seem from the documentation, but sigma_8(z),
    # so the user should always start from z=0
    sigma_8 = block[cosmo_names, 'sigma_8']
    
    nmass_hmf = len(mf.m)

    dndlnmh = np.empty([nz, nmass_hmf])
    nu = np.empty([nz,nmass_hmf])
    b_nu = np.empty([nz,nmass_hmf])
    mean_density0 = np.empty([nz])
    mean_density_z = np.empty([nz])
    rho_halo = np.empty([nz])
    neff = np.empty([nz])
    sigma_var = np.empty([nz])
    for jz in range(0,nz):
        if mdef in ['SOVirial']:
            x = this_cosmo_run.Om(z_vec[jz]) - 1.0
            overdensity_i = (18 * np.pi**2 + 82 * x - 39 * x**2) / this_cosmo_run.Om(z_vec[jz])
            delta_c_i = (3.0/20.0) * (12.0*np.pi)**(2.0/3.0) * (1.0 + 0.0123*np.log10(this_cosmo_run.Om(z_vec[jz])))
        else:
            overdensity_i = overdensity
            delta_c_i = delta_c
            
        mf.update(z=z_vec[jz], cosmo_model=this_cosmo_run, sigma_8=sigma_8, n=ns, delta_c=delta_c_i)
        nu[jz] = mf.nu
        #idx_sigma = np.argmin(np.abs(mf.sigma/(np.pi**2.0) - delta_c_i))
        idx_neff = np.argmin(np.abs(mf.nu**0.5 - 1.0))
        sigma_var[jz] = mf.normalised_filter.sigma(8.0)
        neff[jz] = mf.n_eff[idx_neff]
        dndlnmh[jz] = mf.dndlnm
        mean_density0[jz] = mf.mean_density0
        mean_density_z[jz] = mf.mean_density
        rho_halo[jz] = overdensity_i * mf.mean_density0
        bias = getattr(bias_func, bias_model)(mf.nu, delta_c=delta_c_i, delta_halo=overdensity_i, sigma_8=sigma_8, n=ns, cosmo=this_cosmo_run, m=mass)
        b_nu[jz] = bias.bias()
        #matter_power_lin[jz+1] = mf.power # we rather read linear power spectrum directly from camb, even though hmf does the same... (avoiding double work)

    block.put_grid('hmf', 'z', z_vec, 'm_h', mass, 'dndlnmh', dndlnmh)
    block.put_double_array_1d('density', 'mean_density0', mean_density0)#(mean_density0/this_cosmo_run.Om0)*this_cosmo_run.Odm0)
    block.put_double_array_1d('density', 'mean_density_z', mean_density_z)
    block.put_double_array_1d('density', 'rho_crit', mean_density0/this_cosmo_run.Om0)
    block.put_grid('halobias', 'z', z_vec, 'm_h', mass, 'b_hb', b_nu)
    
    conc = concentration_colossus_new(block, this_cosmo_run, mass, z_vec, model_cm, mdef, overdensity)
    block.put_grid('concentration', 'z', z_vec, 'm_h', mass, 'c', conc)
    block.put_double_array_1d('density', 'rho_halo', rho_halo)
    block.put_grid('hmf', 'z', z_vec, 'm_h', mass, 'nu', nu**0.5)
    block.put_double_array_1d('hmf', 'neff', neff)
    block.put_double_array_1d('hmf', 'sigma_var', sigma_var)

    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
