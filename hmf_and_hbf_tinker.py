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
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM, LambdaCDM
import astropy.units as u
from scipy.integrate import simps
from hmf import MassFunction
from halomod import bias as bias_func
from halomod.concentration import make_colossus_cm
from halomod import concentration as conc_func
from colossus.cosmology import cosmology as colossus_cosmology
from colossus.halo import concentration as colossus_concentration
import hmf.halos.mass_definitions as md
import hmf.cosmology.growth_factor as gf
import time
import pk_lib
import copy

    
def concentration_colossus(block, cosmo, mass, z_vec, model, mdef, overdensity):
    # calculates concentration given halo mass, using the halomod model provided in config
    # furthermore it converts to halomod instance to be used with the halomodel, consistenly with halo mass function
    # and halo bias function
    #nz = len(z_vec)
    nmass = len(mass)
    c = np.empty([nmass])
    this_cosmo = colossus_cosmology.fromAstropy(astropy_cosmo=cosmo, cosmo_name='custom', sigma8=block[cosmo_names, 'sigma_8'], ns=block[cosmo_names, 'n_s'])
    mdef = getattr(md, mdef)() if mdef in ['SOVirial'] else getattr(md, mdef)(overdensity=overdensity)
    #for i,zi in enumerate(z_vec):
    c = np.abs(colossus_concentration.concentration(M=mass, z=z_vec, mdef=mdef.colossus_name, model=model))
    #c[c<=1.0]=1.0#e-4
    return c
    
    
def concentration_halomod(block, cosmo, mass, z, model, mdef, overdensity, mf, delta_c):
    # calculates concentration given halo mass, using the halomod model provided in config
    # furthermore it converts to halomod instance to be used with the halomodel, consistenly with halo mass function
    # and halo bias function
    mdef = getattr(md, mdef)() if mdef in ['SOVirial'] else getattr(md, mdef)(overdensity=overdensity)
    cm = getattr(conc_func, model)(cosmo=mf, filter0=mf.filter, delta_c=delta_c, mdef=mdef)
    
    c = cm.cm(mass, z)
    #print(cm.zc(mass, z))
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
    
def acceleration_parameter(cosmo, z):
    return -0.5*(cosmo.Om(z) + (1.0 + 3.0*cosmo.w(z))*cosmo.Ode(z))
    
def get_growth_interpolator(cosmo):
    """
    Solve the linear growth ODE and returns an interpolating function for the solution
    LCDM = True forces w = -1 and imposes flatness by modifying the dark-energy density
    TODO: w dependence for initial conditions; f here is correct for w=0 only
    TODO: Could use d_init = a(1+(w-1)/(w(6w-5))*(Om_w/Om_m)*a**-3w) at early times with w = w(a<<1)
    """
    #a_init = 1e-4
    #z_init = (1.0/a_init) - 1.0
    z_init = 500.0
    a_init = 1.0/(1.0+z_init)
    from scipy.integrate import solve_ivp
    na = 129 # Number of scale factors used to construct interpolator
    a = np.linspace(a_init, 1.0, na)
    f = 1.0 - cosmo.Om(z_init) # Early mass density
    d_init = a_init**(1.0 - ((3.0/5.0) * f))            # Initial condition (~ a_init; but f factor accounts for EDE-ish)
    v_init = (1.0 - ((3.0/5.0) * f))*a_init**(-((3.0/5.0) * f)) # Initial condition (~ 1; but f factor accounts for EDE-ish)
    #print(d_init, v_init)
    y0 = (d_init, v_init)
    def fun(ax, y):
        d, v = y[0], y[1]
        dxda = v
        zx = (1.0/ax) - 1.0
        fv = -(2.0 + acceleration_parameter(cosmo, zx)*cosmo.inv_efunc(zx)**2.0)*v/ax
        fd = 1.5*cosmo.Om(zx)*d/ax**2
        dvda = fv+fd
        return dxda, dvda
        
    g = solve_ivp(fun, (a[0], a[-1]), y0, t_eval=a, atol=1e-8, rtol=1e-8, vectorized=True).y[0]
    g_interp = interp1d(a, g, kind='linear', assume_sorted=True)
    return g_interp

def get_accumulated_growth(a, g):
    """
    Calculates the accumulated growth at scale factor 'a'
    """
    
    z_init = 500.0
    a_init = 1.0/(1.0+z_init)
    from scipy.integrate import quad
    missing = g(a_init) # Integeral from 0 to ai of g(ai)/ai ~ g(ai) for ai << 1
    G, _ = quad(lambda a: g(a)/a, a_init, a) + missing
    return G

def f_Mead(x, y, p0, p1, p2, p3):
    return p0 + p1*(1.0-x) + p2*(1.0-x)**2.0 + p3*(1.0-y)

def dc_Mead(a, Om, f_nu, g, G):
    """
    delta_c fitting function from Mead (2017; 1606.05345)
    All input parameters should be evaluated as functions of a/z
    """
    
    # See Appendix A of Mead (2017) for naming convention
    p10, p11, p12, p13 = -0.0069, -0.0208, 0.0312, 0.0021
    p20, p21, p22, p23 = 0.0001, -0.0647, -0.0417, 0.0646
    a1, a2 = 1, 0
    
    dc0 = (3.0/20.0)*(12.0*np.pi)**(2.0/3.0) # delta_c = ~1.686' EdS linear collapse threshold
    # Linear collapse threshold
    dc_Mead = 1.0 + f_Mead(g/a, G/a, p10, p11, p12, p13)*np.log10(Om)**a1 + f_Mead(g/a, G/a, p20, p21, p22, p23)*np.log10(Om)**a2
    return dc_Mead * dc0 * (1.0 - 0.041*f_nu)

def Dv_Mead(a, Om, f_nu, g, G):
    """
    Delta_v fitting function from Mead (2017; 1606.05345)
    All input parameters should be evaluated as functions of a/z
    """
    
    # See Appendix A of Mead (2017) for naming convention
    p30, p31, p32, p33 = -0.79, -10.17, 2.51, 6.51
    p40, p41, p42, p43 = -1.89, 0.38, 18.8, -15.87
    a3, a4 = 1, 2
    
    Dv0 = 18.0*np.pi**2.0  # Delta_v = ~178, EdS halo virial overdensity
    # Halo virial overdensity
    Dv_Mead = 1.0 + f_Mead(g/a, G/a, p30, p31, p32, p33)*np.log10(Om)**a3 + f_Mead(g/a, G/a, p40, p41, p42, p43)*np.log10(Om)**a4
    return Dv_Mead * Dv0 * (1.0 + 0.763*f_nu)
    

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
    
    gf._GrowthFactor.supported_cosmos = (FlatLambdaCDM, Flatw0waCDM, LambdaCDM)
    mf = MassFunction(z=0., cosmo_model=initialise_cosmo, Mmin=log_mass_min, Mmax=log_mass_max, dlog10m=dlog10m, sigma_8=0.8, n=0.96,
					  hmf_model=options[option_section, 'hmf_model'], mdef_model=mdef_model, mdef_params=mdef_params, transfer_model='CAMB', delta_c=delta_c, disable_mass_conversion=False, lnk_min=-18.0, lnk_max=18.0)
    # This mf parameters that are fixed here now need to be read from the ini files! Need to make sure camb is not called when initialising the mf!
    #print( mf.cosmo)
    
    mass = mf.m
    
    check_mead = options.has_value(option_section, 'use_mead2020_corrections')
    if check_mead:
        use_mead = options[option_section, 'use_mead2020_corrections']
        if use_mead == 'mead2020':
            mead_correction = 'nofeedback'
        elif use_mead == 'mead2020_feedback':
            mead_correction = 'feedback'
        elif use_mead == 'fit':
            mead_correction = 'fit'
    else:
        mead_correction = None

    return log_mass_min, log_mass_max, nmass, dlog10m, z_vec, nz, mass, mf, options[option_section, 'cm_model'], mdef_model, overdensity, delta_c, options[option_section, 'bias_model'], mead_correction


def execute(block, config):
    #This function is called every time you have a new sample of cosmological and other parameters.
    #It is the main workhorse of the code. The block contains the parameters and results of any 
    #earlier modules, and the config is what we loaded earlier.

    log_mass_min, log_mass_max, nmass, dlog10m, z_vec, nz, mass, mf, model_cm, mdef, overdensity, delta_c, bias_model, mead_correction = config

    # Update the cosmological parameters
    #this_cosmo.update(cosmo_params={'H0':block[cosmo_names, 'hubble'], 'Om0':block[cosmo_names, 'omega_m'], 'Ob0':block[cosmo_names, 'omega_b']})
    this_cosmo_run=Flatw0waCDM(
        H0=block[cosmo_names, 'hubble'], Ob0=block[cosmo_names, 'omega_b'], Om0=block[cosmo_names, 'omega_m'], m_nu=block[cosmo_names, 'mnu'], Tcmb0=block[cosmo_names, 'TCMB'],
    		w0=block[cosmo_names, 'w'], wa=block[cosmo_names, 'wa'])
    ns = block[cosmo_names, 'n_s']

    # Note that CAMB does not return the sigma_8 at z=0, as it might seem from the documentation, but sigma_8(z),
    # so the user should always start from z=0
    samplesig8 = block.has_value(cosmo_names, 'sigma_8_input')
    if samplesig8:
        sigma_8 = block[cosmo_names, 'sigma_8_input']
    else:
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
    f_nu = np.empty([nz])
    h_z = np.empty([nz])
    conc = np.empty([nz,nmass_hmf])
    if mead_correction:
        growth = get_growth_interpolator(this_cosmo_run)
    for jz in range(0,nz):
        if mdef in ['SOVirial'] and mead_correction is None:
            #x = this_cosmo_run.Om(z_vec[jz]) - 1.0
            #overdensity_i = (18 * np.pi**2 + 82 * x - 39 * x**2) / this_cosmo_run.Om(z_vec[jz]) # Calculated within HMF!
            delta_c_i = (3.0/20.0) * (12.0*np.pi)**(2.0/3.0) * (1.0 + 0.0123*np.log10(this_cosmo_run.Om(z_vec[jz])))
            mf.update(z=z_vec[jz], cosmo_model=this_cosmo_run, sigma_8=sigma_8, n=ns, delta_c=delta_c_i)
            overdensity_i = mf.halo_overdensity_mean
            conc[jz,:] = concentration_colossus(block, this_cosmo_run, mass, z_vec[jz], model_cm, mdef, overdensity_i)
        elif mead_correction is not None:
            ajz = this_cosmo_run.scale_factor(z_vec[jz])
            g = growth(ajz)
            G = get_accumulated_growth(ajz, growth)
            delta_c_i = dc_Mead(ajz, this_cosmo_run.Om(z_vec[jz]), this_cosmo_run.Onu0/this_cosmo_run.Om0, g, G)
            overdensity_i = Dv_Mead(ajz, this_cosmo_run.Om(z_vec[jz]), this_cosmo_run.Onu0/this_cosmo_run.Om0, g, G)
            mdef_mead = 'SOMean' # Need to use SOMean to correcly parse the Mead overdensity as calculated above! Otherwise the code again uses the Bryan & Norman function!
            mf.update(z=z_vec[jz], cosmo_model=this_cosmo_run, sigma_8=sigma_8, n=ns, delta_c=delta_c_i, mdef_params={'overdensity':overdensity_i}, mdef_model=mdef_mead)
            conc[jz,:] = concentration_colossus(block, this_cosmo_run, mass, z_vec[jz], model_cm, mdef_mead, overdensity_i)
        else:
            overdensity_i = overdensity
            delta_c_i = delta_c
            mf.update(z=z_vec[jz], cosmo_model=this_cosmo_run, sigma_8=sigma_8, n=ns, delta_c=delta_c_i, mdef_params={'overdensity':overdensity_i})
            conc[jz,:] = concentration_colossus(block, this_cosmo_run, mass, z_vec[jz], model_cm, mdef, overdensity_i)
            
        nu[jz] = mf.nu
        idx_neff = np.argmin(np.abs(mf.nu**0.5 - 1.0))
        sigma_var[jz] = mf.normalised_filter.sigma(8.0)
        neff[jz] = mf.n_eff[idx_neff]
        dndlnmh[jz] = mf.dndlnm
        mean_density0[jz] = mf.mean_density0
        mean_density_z[jz] = mf.mean_density
        rho_halo[jz] = overdensity_i * mf.mean_density0
        bias = getattr(bias_func, bias_model)(mf.nu, delta_c=delta_c_i, delta_halo=overdensity_i, sigma_8=sigma_8, n=ns, cosmo=this_cosmo_run, m=mass)
        b_nu[jz] = bias.bias()
        f_nu[jz] = this_cosmo_run.Onu0/this_cosmo_run.Om0
        #matter_power_lin[jz] = mf.power # we rather read linear power spectrum directly from camb, even though hmf does the same... (avoiding double work)
        #conc[jz,:] = concentration_halomod(block, this_cosmo_run, mass, z_vec[jz], model_cm, mdef, overdensity_i, mf, delta_c_i)
    block.put_grid('hmf', 'z', z_vec, 'm_h', mass, 'dndlnmh', dndlnmh)
    block.put_double_array_1d('density', 'mean_density0', mean_density0) #(mean_density0/this_cosmo_run.Om0)*this_cosmo_run.Odm0)
    block.put_double_array_1d('density', 'mean_density_z', mean_density_z)
    block.put_double_array_1d('cosmological_parameters', 'fnu', f_nu)
    block.put_double_array_1d('density', 'rho_crit', mean_density0/this_cosmo_run.Om0)
    block.put_grid('halobias', 'z', z_vec, 'm_h', mass, 'b_hb', b_nu)
    
    #conc = concentration_colossus(block, this_cosmo_run, mass, z_vec, model_cm, mdef, overdensity)
    block.put_grid('concentration', 'z', z_vec, 'm_h', mass, 'c', conc)
    block.put_double_array_1d('density', 'rho_halo', rho_halo)
    block.put_grid('hmf', 'z', z_vec, 'm_h', mass, 'nu', nu**0.5)
    block.put_double_array_1d('hmf', 'neff', neff)
    block.put_double_array_1d('hmf', 'sigma_var', sigma_var)
    h_z = this_cosmo_run.H(z_vec).value/100.0
    block.put_double_array_1d('cosmological_parameters', 'h_z', h_z)
    #print(h_z)
    #quit()

    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
