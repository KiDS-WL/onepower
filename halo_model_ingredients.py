from cosmosis.datablock import names, option_section
import warnings
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps, solve_ivp, quad
from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM, LambdaCDM
import astropy.units as u
from halomod import bias as bias_func
from halomod.concentration import make_colossus_cm
from halomod import concentration as conc_func
from hmf import MassFunction
import hmf.halos.mass_definitions as md
from hmf.halos.mass_definitions import SphericalOverdensity
import hmf.cosmology.growth_factor as gf
from darkmatter_lib import radvir_from_mass, scale_radius, compute_u_dm
import halo_model_utility as hmu
from colossus.cosmology import cosmology as colossus_cosmology
from colossus.halo import concentration as colossus_concentration
import time

from astropy.cosmology import Planck15

# cosmological parameters section name in block
cosmo_params = names.cosmological_parameters


class SOVirial_Mead(SphericalOverdensity):
    """
    SOVirial overdensity definition from Mead et al. 2020
    """
    _defaults = {"overdensity": 200}
        
    def halo_density(self, z=0, cosmo=Planck15):
        """The density of haloes under this definition."""
        return self.params["overdensity"] * self.mean_density(z, cosmo)
        
    @property
    def colossus_name(self):
        return "200c"
            
    def __str__(self):
        """Describe the halo definition in standard notation."""
        return "SOVirial"


# TODO: concentration is saved into multiple folders. Check if these can be merged.
def concentration_colossus(block, cosmo, mass, z, model, mdef, overdensity):
    """
    calculates concentration given halo mass, using the halomod model provided in config
    furthermore it converts to halomod instance to be used with the halomodel,
    consistenly with halo mass function and halo bias function
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        # This dissables the warning from colossus that is just telling us what we know
        # colossus warns us about massive neutrinos, but that is also ok
        # as we do not use cosmology from it, but it requires it to setup the instance!
        this_cosmo = colossus_cosmology.fromAstropy(astropy_cosmo=cosmo, cosmo_name='custom',
                     sigma8=block[cosmo_params, 'sigma_8'], ns=block[cosmo_params, 'n_s'])

    if isinstance(mdef, str):
        mdef = getattr(md, mdef)() if mdef in ['SOVirial'] else getattr(md, mdef)(overdensity=overdensity)
    else:
        mdef = SOVirial_Mead(overdensity=overdensity)
    
    # This is the slow part: 0.4-0.5 seconds per call, called separately for each redshift. 
    # MA: Possible solution: See if we can get away with a smaller numbr of redshifts and interpolate.
    #tic = time.perf_counter()
    c, ms = colossus_concentration.concentration(M=mass, z=z, mdef=mdef.colossus_name, model=model,
            range_return=True, range_warning=False)
    #toc = time.perf_counter()
    #print(" colossus_concentration.concentration: "+'%.4f' %(toc - tic)+ "s")

    c_interp = interp1d(mass[c>0], c[c>0], kind='linear', bounds_error=False, fill_value=1.0)

    
    return c_interp(mass)



def setup(options):

    # Read in from the halo_model_ingredients section of the ini file
    # log-spaced mass in units of M_sun/h
    log_mass_min = options[option_section, 'log_mass_min']
    log_mass_max = options[option_section, 'log_mass_max']
    nmass        = options[option_section, 'nmass']
    dlog10m      = (log_mass_max-log_mass_min)/nmass

    zmin  = options[option_section, 'zmin']
    zmax  = options[option_section, 'zmax']
    nz    = options[option_section, 'nz']
    z_vec = np.linspace(zmin, zmax, nz)
    nz_conc = options.get_int(option_section, 'nz_conc', default=5)

    # Profile
    nk      = options[option_section, 'nk']
    # currently we have only implmented NFW. 
    profile = options.get_string(option_section, 'profile', default='nfw')
    profile = profile.lower()
    profile_value_name = options.get_string(option_section, 'profile_value_name', default='profile_parameters')

    # Type of mass definition for Haloes
    mdef_model = options[option_section, 'mdef_model']
    # Over density threshold
    overdensity = options[option_section, 'overdensity']
    # Concentration mass relation model
    cm_model = options[option_section, 'cm_model']
    # critical density contrast
    delta_c = options[option_section, 'delta_c']
    # Linear halo bias model
    bias_model = options[option_section, 'bias_model']
    mdef_params = {} if mdef_model in ['SOVirial'] else {'overdensity':overdensity}

    #Choice of code to define Halo Mass Function
    hmf_code = options.get_string(option_section, 'hmf_code', default='HMF')   #Options are CCL, HMF
    
    # most general astropy cosmology initialisation, 
    # gets updated as sampler runs with camb provided cosmology parameters.
    # setting some values to generate instance
    initialise_cosmo=Flatw0waCDM(H0=100., Ob0=0.044, Om0=0.3, Tcmb0=2.7255, w0=-1., wa=0.)

    # Growth Factor from hmf
    #gf._GrowthFactor.supported_cosmos = (FlatLambdaCDM, Flatw0waCDM, LambdaCDM)

    # Halo Mass function from hmf
    # This is the slow part it take 1.58/1.67
    mf = MassFunction(z=0., cosmo_model=initialise_cosmo, Mmin=log_mass_min, 
                        Mmax=log_mass_max, dlog10m=dlog10m, sigma_8=0.8, n=0.96,
                        hmf_model=options[option_section, 'hmf_model'],
                        mdef_model=mdef_model, mdef_params=mdef_params,
                        transfer_model='CAMB',
                        delta_c=delta_c, disable_mass_conversion=False,
                        growth_model='CambGrowth',
                        lnk_min=-18.0, lnk_max=18.0)

    # Array of halo masses - equally spaced in logM
    mass = mf.m   

    # Option to set similar corrections to HMcode2020
    # TODO: stellar_fraction_from_observable_feedback option is not used here
    use_mead = options.get_string(option_section, 'use_mead2020_corrections', default='None')
    if use_mead == 'mead2020':
        mead_correction = 'nofeedback'
    elif use_mead == 'mead2020_feedback':
        mead_correction = 'feedback'
    #elif use_mead == 'fit_feedback':
    #    mead_correction = 'fit'
    else:
        mead_correction = None

    # config ={}
    # config['log_mass_min'] =log_mass_min
    # config['log_mass_max'] =log_mass_max
    return log_mass_min, log_mass_max, nmass, z_vec, nz, nz_conc, mass, mf, cm_model, mdef_model, overdensity, delta_c, bias_model, hmf_code, mead_correction, nk, profile, profile_value_name

def execute(block, config):

    # Read in the config as returned by setup
    log_mass_min, log_mass_max, nmass, z_vec, nz, nz_conc, mass, mf, model_cm, mdef, overdensity, delta_c, bias_model, hmf_code, mead_correction, nk, profile, profile_value_name = config

    # astropy cosmology requires the CMB temprature as an input. 
    # If it exists in the values file read it from there otherwise set to its default value
    try:
        tcmb = block[cosmo_params, 'TCMB']
    except:
        tcmb = 2.7255

    # Update the cosmological parameters
    this_cosmo_run=Flatw0waCDM(
        H0=block[cosmo_params, 'hubble'], Ob0=block[cosmo_params, 'omega_b'],
        Om0=block[cosmo_params, 'omega_m'], m_nu=[0, 0, block[cosmo_params, 'mnu']], Tcmb0=tcmb,
    	w0=block[cosmo_params, 'w'], wa=block[cosmo_params, 'wa'] )
     
    #LCDMcosmo = FlatLambdaCDM(
    #    H0=block[cosmo_params, 'hubble'], Ob0=block[cosmo_params, 'omega_b'],
    #    Om0=block[cosmo_params, 'omega_m'], m_nu=block[cosmo_params, 'mnu'], Tcmb0=tcmb)

    ns      = block[cosmo_params, 'n_s']
    sigma_8 = block[cosmo_params, 'sigma_8']
    
    transfer_k = block['matter_power_transfer_func', 'k_h']
    transfer_func = block['matter_power_transfer_func', 't_k']

    # initialise arrays
    nmass_hmf = len(mass)
    dndlnmh   = np.empty([nz,nmass_hmf])
    nu        = np.empty([nz,nmass_hmf])
    b_nu      = np.empty([nz,nmass_hmf])
    rho_halo  = np.empty([nz])
    neff      = np.empty([nz])
    sigma8_z  = np.empty([nz])
    # Fraction of neutrinos to total matter,  f_nu = Ω_nu /Ω_m
    f_nu      = np.empty([nz])
    h_z       = np.empty([nz])
    mean_density0  = np.empty([nz])
    mean_density_z = np.empty([nz])
    overdensity_z  = np.empty([nz])
    
    downsample_factor = int(nz/nz_conc)
    if downsample_factor > 0 :
        z_conc = z_vec[::downsample_factor]
    else:
        z_conc = z_vec
    conc = np.empty([z_conc.size, nmass_hmf])

    if mead_correction:
        growth = hmu.get_growth_interpolator(this_cosmo_run)
        zf = np.empty([nz,nmass_hmf])
        #growth_LCDM = hmu.get_growth_interpolator(LCDMcosmo)

    # About 1.8 seconds for mf.update.About 7 seconds for concentration_colossus

    # loop over a series of redshift values defined by z_vec = np.linspace(zmin, zmax, nz)
    for jz,z_iter in enumerate(z_vec):
        if mdef in ['SOVirial'] and mead_correction is None:
            delta_c_z = (3.0/20.0) * (12.0*np.pi)**(2.0/3.0) * (1.0 + 0.0123*np.log10(this_cosmo_run.Om(z_iter)))
            # Update the cosmology for the halo mass function, this takes a little while the first time it is called
            # Then it is faster becayse it only updates the redshift and the corresponding delta_c
            mf.update(z=z_iter, cosmo_model=this_cosmo_run, sigma_8=sigma_8, n=ns, delta_c=delta_c_z, transfer_model='FromArray', transfer_params={'k':transfer_k, 'T':transfer_func})
            overdensity_z[jz] = mf.halo_overdensity_mean
            mdef_conc = mdef
        elif mead_correction is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                # This disables the warning from hmf is just telling us what we know
                # hmf's internal way of calculating the overdensity and the collapse threshold are fixed. 
                # When we use the mead correction we want to define the haloes using the virial definition. 
                # To avoid conflicts we manually pass the overdensity and the collapse threshold, 
                # but for that we need to set the mass definition to be "mean", 
                # so that it is compared to the mean density of the Universe rather than critical density. 
                # hmf warns us that the value is not a native definition for the given halo mass function, 
                # but will interpolate between the known ones (this is happening when one uses Tinker hmf for instance). 
                a = this_cosmo_run.scale_factor(z_iter)
                g = growth(a)
                G = hmu.get_accumulated_growth(a, growth)
                delta_c_z = hmu.dc_Mead(a, this_cosmo_run.Om(z_iter)+this_cosmo_run.Onu(z_iter), this_cosmo_run.Onu0/(this_cosmo_run.Om0+this_cosmo_run.Onu0), g, G)
                overdensity_z[jz] = hmu.Dv_Mead(a, this_cosmo_run.Om(z_iter)+this_cosmo_run.Onu(z_iter), this_cosmo_run.Onu0/(this_cosmo_run.Om0+this_cosmo_run.Onu0), g, G)
                #dolag = (growth(LCDMcosmo.scale_factor(10.0))/growth_LCDM(LCDMcosmo.scale_factor(10.0)))*(growth_LCDM(a)/growth(a))
                mdef_mead = SOVirial_Mead#'SOMean' # Need to use SOMean to correcly parse the Mead overdensity as calculated above! Otherwise the code again uses the Bryan & Norman function!
                mdef_conc = mdef_mead
                mf.ERROR_ON_BAD_MDEF = False
                mf.update(z=z_iter, cosmo_model=this_cosmo_run, sigma_8=sigma_8, n=ns, delta_c=delta_c_z, mdef_model=mdef_mead,  mdef_params={'overdensity':overdensity_z[jz]}, disable_mass_conversion=True, transfer_model='FromArray', transfer_params={'k':transfer_k, 'T':transfer_func})
                if mead_correction in ['feedback', 'nofeedback']:
                    zf[jz,:] = hmu.get_halo_collapse_redshifts(mass, z_iter, delta_c_z, growth, this_cosmo_run, mf)
        else:
            overdensity_z[jz] = overdensity
            delta_c_z = delta_c
            mf.update(z=z_iter, cosmo_model=this_cosmo_run, sigma_8=sigma_8, n=ns, delta_c=delta_c_z, mdef_params={'overdensity':overdensity_z[jz]}, transfer_model='FromArray', transfer_params={'k':transfer_k, 'T':transfer_func})
            mdef_conc = mdef
    
        # What code do you want to use to define the halo mass function
        if hmf_code == 'CCL':
            #This stays or goes depending on whether we want to keep using the CCL version of the halo mass function
            import pyccl as ccl
            #halo_model_ingredients took: 2.503 seconds
            cosmo_ccl = ccl.Cosmology(Omega_c=block[cosmo_params, 'omega_m']-block[cosmo_params, 'omega_b'], 
                                      Omega_b=block[cosmo_params, 'omega_b'], h=block[cosmo_params, 'h0'], 
                                      sigma8=block[cosmo_params, 'sigma_8'], n_s=block[cosmo_params, 'n_s'])
            # Here we use a mass definition with Delta = 200 times the matter density,
            hmd_200m = ccl.halos.MassDef200m
            # the Tinker 2010 halo mass function,   This is dmdlog10M
            nM = ccl.halos.MassFuncTinker10(mass_def=hmd_200m)
            h = block[cosmo_params, 'h0']
            dndlnmh[jz] = nM(cosmo_ccl, mass, 1/(1+z_iter)) /(h**3*np.log(10))
        else:   
            # halo_model_ingredients took: 0.613 seconds
            # The differential mass function in terms of natural log of m, len=len(m) [units \(h^3 Mpc^{-3}\)]
            # dn(m)/ dln m eq1 of 1306.6721
            dndlnmh[jz]   = mf.dndlnm 

        #Peak height, mf.nu from hmf is \left(\frac{\delta_c}{\sigma}\right)^2\), but we want \frac{\delta_c}{\sigma}
        nu[jz]        = mf.nu**0.5

        # TODO: This is the mean matter density at z=0, change to be just one value.
        mean_density0[jz]  = mf.mean_density0
        # This is the redshift dependant mean density
        mean_density_z[jz] = mf.mean_density
        rho_halo[jz]  = overdensity_z[jz] * mf.mean_density0
        bias     = getattr(bias_func, bias_model)(mf.nu, delta_c=delta_c_z, delta_halo=overdensity_z[jz],
                                                 sigma_8=sigma_8, n=ns, cosmo=this_cosmo_run, m=mass)
        b_nu[jz] = bias.bias()
        # fraction of neutrinos to matter
        f_nu[jz] = this_cosmo_run.Onu0/this_cosmo_run.Om0

        # These are only used for mead_corrections
        # index of 
        idx_neff      = np.argmin(np.abs(mf.nu - 1.0))
        # effective power spectrum index at the collapse scale, 
        # Question: n_eff is just used in the transition_smoothing module for mead_corrections. It is called n^eff_cc in table 2 of https://arxiv.org/pdf/2009.01858.pdf . But it doesn't explain what it is. Do we know if this is the correct one to use? 
        neff[jz]      = mf.n_eff[idx_neff]
        # Only used for mead_corrections
        sigma8_z[jz] = mf.normalised_filter.sigma(8.0)
    
    downsample_factor = int(nz/nz_conc)
    if downsample_factor > 0 :
        overdensity_conc = overdensity_z[::downsample_factor]
    else:
        overdensity_conc = overdensity_z
    
    for i,z_iter in enumerate(z_conc):
        conc[i,:] = concentration_colossus(block, this_cosmo_run, mass, z_iter, model_cm, mdef_conc, overdensity_conc[i])
   
    # Upsample concentration
    if downsample_factor > 0 :
        conc_func = interp1d(np.log10(z_conc+1.0), conc, axis=0, kind='cubic', fill_value='extrapolate', bounds_error=False)
        conc = conc_func(np.log10(z_vec+1.0))
    
    
    ###################################################################################################################
    # Halo Profile

    # get these from the values.ini file
    norm_cen = block[profile_value_name, 'norm_cen']
    norm_sat = block[profile_value_name, 'norm_sat']
    eta_cen  = block[profile_value_name, 'eta_cen']
    eta_sat  = block[profile_value_name, 'eta_sat']

    if mead_correction == 'nofeedback':
        norm_cen  =  5.196#1.0 #(5.196/3.85)#0.85*1.299
        eta_cen   = (0.1281 * sigma8_z[:,np.newaxis]**(-0.3644))
        conc = (1.0+zf)/(1.0+z_vec[:,np.newaxis])
        
    if mead_correction == 'feedback':
        theta_agn = block['halo_model_parameters', 'logT_AGN'] - 7.8
        norm_cen  = ((5.196/4.0) * (3.44 - 0.496*theta_agn) * np.power(10.0, z_vec*(-0.0671 - 0.0371*theta_agn)))[:,np.newaxis]
        eta_cen   = (0.1281 * sigma8_z[:,np.newaxis]**(-0.3644))
        conc = (1.0+zf)/(1.0+z_vec[:,np.newaxis])
    # TODO: what happends if mead_correction is stellar_fraction_from_observable_feedback?

    conc_cen = norm_cen * conc
    conc_sat = norm_sat * conc
    rvir_cen = radvir_from_mass(mass, rho_halo)
    rvir_sat = radvir_from_mass(mass, rho_halo)
    
    r_s_cen  = scale_radius(rvir_cen, conc_cen) * nu**eta_cen
    r_s_sat  = scale_radius(rvir_sat, conc_sat) * nu**eta_sat

    # compute the Fourier-transform of the NFW profile (normalised to the mass of the halo)
    #k = np.logspace(-2,1,200) # AD: inherit the range from Plin? That would avoid intepolations ...
    k_vec_original = block['matter_power_lin', 'k_h']
    k = np.logspace(np.log10(k_vec_original[0]), np.log10(k_vec_original[-1]), num=nk)

    # Allow for different profiles, 
    # TODO: Look into pyhalomodel.
    if profile == 'nfw':
        u_dm_cen = compute_u_dm(k, r_s_cen, conc_cen, mass)
        u_dm_sat = compute_u_dm(k, r_s_sat, conc_sat, mass)
    else:
        warnings.warn(f'Currently the only supported profile is "nfw". You have chosen {profile} which is not supported. Returning NFW results')
        u_dm_cen = compute_u_dm(k, r_s_cen, conc_cen, mass)
        u_dm_sat = compute_u_dm(k, r_s_sat, conc_sat, mass)

    #  TODO: Clean these up. Put more of them into the same folder
    block.put_grid('concentration_dm', 'z', z_vec, 'm_h', mass, 'c', conc_cen)
    block.put_grid('concentration_sat', 'z', z_vec, 'm_h', mass, 'c', conc_sat)
    block.put_grid('nfw_scale_radius_dm', 'z', z_vec, 'm_h', mass, 'rs', r_s_cen)
    block.put_grid('nfw_scale_radius_sat', 'z', z_vec, 'm_h', mass, 'rs', r_s_sat)
    block.put_double_array_1d('virial_radius', 'm_h', mass)
    block.put_double_array_1d('virial_radius', 'rvir_dm', rvir_cen[0])  #rvir doesn't change with z, hence no z-dimension
    block.put_double_array_1d('virial_radius', 'rvir_sat', rvir_sat[0])


    block.put_double_array_1d('fourier_nfw_profile', 'z', z_vec)
    block.put_double_array_1d('fourier_nfw_profile', 'm_h', mass)
    block.put_double_array_1d('fourier_nfw_profile', 'k_h', k)
    block.put_double_array_nd('fourier_nfw_profile', 'ukm', u_dm_cen)
    block.put_double_array_nd('fourier_nfw_profile', 'uksat', u_dm_sat)
    ###################################################################################################################


    # density 
    # These two do not depend on redshift so should only be saved as a single number
    block.put_double_array_1d('density', 'mean_density0', mean_density0) #(mean_density0/this_cosmo_run.Om0)*this_cosmo_run.Odm0)
    block.put_double_array_1d('density', 'rho_crit', mean_density0/this_cosmo_run.Om0)
    # These depend on redshift
    block.put_double_array_1d('density', 'mean_density_z', mean_density_z)
    block.put_double_array_1d('density', 'rho_halo', rho_halo)
    block.put_double_array_1d('density', 'z', z_vec)

    # hmf
    block.put_grid('hmf', 'z', z_vec, 'm_h', mass, 'dndlnmh', dndlnmh)
    block.put_grid('hmf', 'z', z_vec, 'm_h', mass, 'nu', nu)
    block.put_double_array_1d('hmf', 'neff', neff)
    block.put_double_array_1d('hmf', 'sigma8_z', sigma8_z)

    # halobias
    block.put_grid('halobias', 'z', z_vec, 'm_h', mass, 'b_hb', b_nu)
    
    # concentration
    block.put_grid('concentration', 'z', z_vec, 'm_h', mass, 'c', conc)

    # cosmological parameters
    h_z = this_cosmo_run.H(z_vec).value/100.0
    block.put_double_array_1d(cosmo_params, 'h_z', h_z)
    # f_nu = Omega_nu/Omega_m
    block.put_double_array_1d(cosmo_params, 'fnu', f_nu)
    block.put_double_array_1d(cosmo_params, 'z',z_vec)

    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
