from cosmosis.datablock import names, option_section
import warnings
import numpy as np
from astropy.cosmology import Flatw0waCDM
import halo_model_utility as hmu
import hmf
from halomod.halo_model import DMHaloModel
from halomod.concentration import make_colossus_cm
import halomod.profiles as profile_classes

#import hmf.cosmology.growth_factor as gf
# from hmf import MassFunction
# from darkmatter_lib import radvir_from_mass, scale_radius, compute_u_dm
#from scipy.optimize import root_scalar
# from scipy.integrate import simps, solve_ivp, quad
# from astropy.cosmology import FlatLambdaCDM, LambdaCDM
#import time
# import astropy.units as u
# from halomod import bias as bias_func
# from halomod.concentration import CMRelation, Bullock01
# from halomod import concentration as conc_func
# from halomod.profiles import Profile, NFWInf
# from types import MethodType
# from colossus.cosmology import cosmology as colossus_cosmology
# from colossus.halo import concentration as colossus_concentration
# import hmf.halos.mass_definitions as md

# cosmological parameters section name in block
cosmo_params = names.cosmological_parameters

# This patches hmf caching!
def obj_eq_fix(ob1, ob2):
    """Test equality of objects that is numpy-aware."""
    try:
        return bool(ob1 == ob2)
    except ValueError:
        # Could be a numpy array.
        #return np.array_equiv(ob1, ob2)#(ob1 == ob2).all()
        try:
            return np.array_equal(ob1, ob2)
        except ValueError:
            if ob1.keys() != ob2.keys():
                return False
            return all(np.array_equal(ob1[key], ob2[key]) for key in ob1)

# Redefine obj_eq using the function above.
hmf._internals._cache.obj_eq = obj_eq_fix


def setup(options):

    # log10 Minimum, Maximum and number of log10 mass bins for halo masses: M_halo
    # The units are in log10(M_sun h^-1)
    log_mass_min = options[option_section, 'log_mass_min']
    log_mass_max = options[option_section, 'log_mass_max']
    nmass        = options[option_section, 'nmass']
    dlog10m      = (log_mass_max-log_mass_min)/nmass

    # Minimum and Maximum redshift and number of redshift bins for calculating the ingredients
    zmin    = options[option_section, 'zmin']
    zmax    = options[option_section, 'zmax']
    nz      = options[option_section, 'nz']
    z_vec   = np.linspace(zmin, zmax, nz)
    # If this is smaller than nz then downsample for concentraion to speed it up. 
    # The concentration function is slow!
    # nz_conc = options.get_int(option_section, 'nz_conc', default=5)

    # Profile
    nk      = options[option_section, 'nk']
    profile = options.get_string(option_section, 'profile', default='NFW')
    profile_value_name = options.get_string(option_section, 'profile_value_name', default='profile_parameters')

    # model name for halo mass functions
    hmf_model = options[option_section, 'hmf_model']
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
    # set mdef_params to {} for SOVirial model. Otherwise set it to use the overdensity {'overdensity':overdensity}
    mdef_params = {} if mdef_model in ['SOVirial'] else {'overdensity':overdensity}
    
    # Option to set similar corrections to HMcode2020
    # MA question: What do these different options do? It doesn't look like there is a difference between them.
    use_mead = options.get_string(option_section, 'use_mead2020_corrections', default='None')
    if use_mead == 'mead2020':
        mead_correction = 'nofeedback'
    elif use_mead == 'mead2020_feedback':
        mead_correction = 'feedback'
    #elif use_mead == 'fit_feedback':
    #    mead_correction = 'fit'
    else:
        mead_correction = None
    
    # If mead correction is applied set the ingredients to match Mead et al. (2021)
    if mead_correction is not None:
        hmf_model = 'ST'
        bias_model = 'ST99'
        mdef_model = 'SOVirial'
        mdef_params = {}
        cm_model = 'bullock01'
    
    # most general astropy cosmology initialisation,
    # gets updated as sampler runs with camb provided cosmology parameters.
    # setting some values to generate instance
    initialise_cosmo=Flatw0waCDM(
        H0=100.0, Ob0=0.044, Om0=0.3, Tcmb0=2.7255, w0=-1., wa=0.)

    # Growth Factor from hmf
    #gf._GrowthFactor.supported_cosmos = (FlatLambdaCDM, Flatw0waCDM, LambdaCDM)

    # Halo Mass function from hmf
    # This is the slow part it take 1.58/1.67
    DM_hmf = DMHaloModel(z=0., cosmo_model=initialise_cosmo, sigma_8=0.8, n=0.96,
                        transfer_model='CAMB',growth_model='CambGrowth',
                        lnk_min=-18.0, lnk_max=18.0, dlnk=0.001,
                        Mmin=log_mass_min, Mmax=log_mass_max, dlog10m=dlog10m, 
                        hmf_model=hmf_model, mdef_model=mdef_model, mdef_params=mdef_params,
                        delta_c=delta_c, disable_mass_conversion=False,
                        bias_model=bias_model,
                        #halo_profile_model=profile,
                        #halo_concentration_model=make_colossus_cm(cm_model))
                        halo_profile_model=hmu.get_bloated_profile(getattr(profile_classes, profile)),
                        halo_concentration_model=hmu.get_modified_concentration(make_colossus_cm(cm_model)))

    # DM_hmf.cmz_relation
    # print(DM_hmf.m,DM_hmf.cmz_relation)
    # exit(1)
    # Array of halo masses 
    mass = DM_hmf.m

    return {"z_vec": z_vec,
            "nz": nz,
            "mass": mass,
            "DM_hmf": DM_hmf,
            "mdef_model": mdef_model,
            "overdensity": overdensity,
            "delta_c": delta_c,
            "mead_correction": mead_correction,
            "nk":nk,
            "profile_value_name":profile_value_name}

def execute(block, config):

    # Read in the config as returned by setup
    z_vec = config["z_vec"]
    nz = config["nz"]
    mass = config["mass"]
    DM_hmf = config["DM_hmf"]
    mdef_model = config["mdef_model"]
    overdensity = config["overdensity"]
    delta_c = config["delta_c"]
    mead_correction= config["mead_correction"]
    nk = config["nk"]
    profile_value_name = config["profile_value_name"]

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

    ns      = block[cosmo_params, 'n_s']
    sigma_8 = block[cosmo_params, 'sigma_8']
    
    # TODO: will the inputs depend on the profile model? 
    norm_cen = block[profile_value_name, 'norm_cen']
    norm_sat = block[profile_value_name, 'norm_sat']
    eta_cen  = block[profile_value_name, 'eta_cen']
    eta_sat  = block[profile_value_name, 'eta_sat']
    
    # initialise arrays
    nmass_hmf = len(mass)
    dndlnmh   = np.empty([nz,nmass_hmf])
    nu        = np.empty([nz,nmass_hmf])
    b_nu      = np.empty([nz,nmass_hmf])
    rho_halo  = np.empty([nz])
    neff      = np.empty([nz])
    sigma8_z  = np.empty([nz])
    f_nu      = np.empty([nz])
    h_z       = np.empty([nz])
    mean_density0  = np.empty([nz])
    mean_density_z = np.empty([nz])
    overdensity_z  = np.empty([nz])
    u_dm_cen  = np.empty([nz,nk,nmass_hmf])
    u_dm_sat  = np.empty([nz,nk,nmass_hmf])
    conc_cen  = np.empty([nz,nmass_hmf])
    conc_sat  = np.empty([nz,nmass_hmf])
    r_s_cen   = np.empty([nz,nmass_hmf])
    r_s_sat   = np.empty([nz,nmass_hmf])
    rvir_cen  = np.empty([nz,nmass_hmf])
    rvir_sat  = np.empty([nz,nmass_hmf])

    if mead_correction:
        growth = hmu.get_growth_interpolator(this_cosmo_run)
        #growth_LCDM = hmu.get_growth_interpolator(LCDMcosmo)

    k_vec_original = block['matter_power_lin', 'k_h']
    k = np.logspace(np.log10(k_vec_original[0]), np.log10(k_vec_original[-1]), num=nk)
    
    # Power spectrum transfer function used to update the transfer function in hmf
    transfer_k    = block['matter_power_transfer_func', 'k_h']
    transfer_func = block['matter_power_transfer_func', 't_k']

    # loop over a series of redshift values defined by z_vec = np.linspace(zmin, zmax, nz)
    for jz,z_iter in enumerate(z_vec):
        if mdef_model in ['SOVirial'] and mead_correction is None:
            # The critical overdensity for collapse for a given redshift and Omega_m
            delta_c_z = (3.0/20.0) * (12.0*np.pi)**(2.0/3.0) * (1.0 + 0.0123*np.log10(this_cosmo_run.Om(z_iter)))
            # Update the cosmology for the halo mass function, this takes a little while the first time it is called
            # Then it is faster becayse it only updates the redshift and the corresponding delta_c
            DM_hmf.update(z=z_iter, cosmo_model=this_cosmo_run, sigma_8=sigma_8, n=ns, delta_c=delta_c_z,
                        transfer_model='FromArray', transfer_params={'k':transfer_k, 'T':transfer_func})
            overdensity_z[jz] = DM_hmf.halo_overdensity_mean
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
                delta_c_z = hmu.dc_Mead(a, this_cosmo_run.Om(z_iter)+this_cosmo_run.Onu(z_iter), 
                                    this_cosmo_run.Onu0/(this_cosmo_run.Om0+this_cosmo_run.Onu0), g, G)
                overdensity_z[jz] = hmu.Dv_Mead(a, this_cosmo_run.Om(z_iter)+this_cosmo_run.Onu(z_iter), 
                                        this_cosmo_run.Onu0/(this_cosmo_run.Om0+this_cosmo_run.Onu0), g, G)
                mdef_mead = hmu.SOVirial_Mead #'SOMean' # Need to use SOMean to correcly parse the Mead overdensity as calculated above! Otherwise the code again uses the Bryan & Norman function!
                DM_hmf.ERROR_ON_BAD_MDEF = False
                DM_hmf.update(z=z_iter, cosmo_model=this_cosmo_run, sigma_8=sigma_8, 
                              n=ns, delta_c=delta_c_z, mdef_model=mdef_mead,  
                              mdef_params={'overdensity':overdensity_z[jz]}, disable_mass_conversion=True, 
                              transfer_model='FromArray', transfer_params={'k':transfer_k, 'T':transfer_func})
        else:
            overdensity_z[jz] = overdensity
            delta_c_z = delta_c
            DM_hmf.update(z=z_iter, cosmo_model=this_cosmo_run, sigma_8=sigma_8, n=ns, 
                          delta_c=delta_c_z, mdef_params={'overdensity':overdensity_z[jz]}, 
                          transfer_model='FromArray', transfer_params={'k':transfer_k, 'T':transfer_func})
        
        #Peak height, DM_hmf.nu from hmf is \left(\frac{\delta_c}{\sigma}\right)^2\), but we want \frac{\delta_c}{\sigma}
        nu[jz]        = DM_hmf.nu**0.5
        dndlnmh[jz]   = DM_hmf.dndlnm
        mean_density0 = DM_hmf.mean_density0
        mean_density_z[jz] = DM_hmf.mean_density
        rho_halo[jz]  = overdensity_z[jz] * DM_hmf.mean_density0 
        b_nu[jz] = DM_hmf.halo_bias
    
        # These are only used for mead_corrections
        # index of 
        idx_neff      = np.argmin(np.abs(DM_hmf.nu - 1.0))
        # effective power spectrum index at the collapse scale,
        # Question: n_eff is just used in the transition_smoothing module for mead_corrections. 
        # It is called n^eff_cc in table 2 of https://arxiv.org/pdf/2009.01858.pdf . 
        # But it doesn't explain what it is. Do we know if this is the correct one to use? 
        neff[jz]      = DM_hmf.n_eff[idx_neff]
        #Rnl = DM_hmf.filter.mass_to_radius(DM_hmf.mass_nonlinear, DM_hmf.mean_density0)
        #neff[jz] = -3.0 - 2.0*DM_hmf.normalised_filter.dlnss_dlnm(Rnl)
        
        # Only used for mead_corrections
        sigma8_z[jz] = DM_hmf.normalised_filter.sigma(8.0)
        #pk_cold = DM_hmf.power * hmu.Tk_cold_ratio(DM_hmf.k, g, block[cosmo_params, 'ommh2'], block[cosmo_params, 'h0'], this_cosmo_run.Onu0/this_cosmo_run.Om0, this_cosmo_run.Neff, T_CMB=tcmb)**2.0
        #sigma8_z[jz] = hmu.sigmaR_cc(pk_cold, DM_hmf.k, 8.0)
        
        if mead_correction == 'nofeedback':
            norm_cen  = 5.196 #/3.85#1.0#(5.196/3.85) #0.85*1.299
            eta_cen   = 0.1281 * sigma8_z[jz]**(-0.3644)
            
            zf = hmu.get_halo_collapse_redshifts(mass, z_iter, delta_c_z, growth, this_cosmo_run, DM_hmf)
            conc_cen[jz,:] = norm_cen * (1.0+zf)/(1.0+z_iter)
            conc_sat[jz,:] = norm_sat * (1.0+zf)/(1.0+z_iter)
            
            # The only difference is the concentration values. Here is it is conc_cen and for the next block it is conc_sat
            DM_hmf.update(halo_profile_params={'eta_bloat':eta_cen, 'nu':list(nu[jz]), 'cosmo':this_cosmo_run})
            nfw_cen = DM_hmf.halo_profile.u(k, DM_hmf.m, c=conc_cen[jz,:], norm='m', coord='k')
            u_dm_cen[jz,:,:] = nfw_cen/np.expand_dims(nfw_cen[0,:], 0)
            r_s_cen[jz,:] = DM_hmf.halo_profile._rs_from_m(DM_hmf.m)
            rvir_cen[jz,:] = DM_hmf.halo_profile.halo_mass_to_radius(DM_hmf.m)
        
            DM_hmf.update(halo_profile_params={'eta_bloat':eta_sat, 'nu':list(nu[jz]), 'cosmo':this_cosmo_run})
            nfw_sat = DM_hmf.halo_profile.u(k, DM_hmf.m, c=conc_sat[jz,:], norm='m', coord='k')
            u_dm_sat[jz,:,:] = nfw_sat/np.expand_dims(nfw_sat[0,:], 0)
            r_s_sat[jz,:] = DM_hmf.halo_profile._rs_from_m(DM_hmf.m)
            rvir_sat[jz,:] = DM_hmf.halo_profile.halo_mass_to_radius(DM_hmf.m)
            
            
        elif mead_correction == 'feedback':
            theta_agn = block['halo_model_parameters', 'logT_AGN'] - 7.8
            norm_cen  = (5.196/4.0) * ((3.44 - 0.496*theta_agn) * np.power(10.0, z_iter*(-0.0671 - 0.0371*theta_agn)))
            eta_cen   = 0.1281 * sigma8_z[jz]**(-0.3644)
            
            zf = hmu.get_halo_collapse_redshifts(mass, z_iter, delta_c_z, growth, this_cosmo_run, DM_hmf)
            conc_cen[jz,:] = norm_cen * (1.0+zf)/(1.0+z_iter)
            conc_sat[jz,:] = norm_sat * (1.0+zf)/(1.0+z_iter)
            
            # The only difference is the concentration values. Here is it is conc_cen and for the next block it is conc_sat
            DM_hmf.update(halo_profile_params={'eta_bloat':eta_cen, 'nu':list(nu[jz]), 'cosmo':this_cosmo_run})
            nfw_cen = DM_hmf.halo_profile.u(k, DM_hmf.m, c=conc_cen[jz,:], norm='m', coord='k')
            u_dm_cen[jz,:,:] = nfw_cen/np.expand_dims(nfw_cen[0,:], 0)
            r_s_cen[jz,:] = DM_hmf.halo_profile._rs_from_m(DM_hmf.m)
            rvir_cen[jz,:] = DM_hmf.halo_profile.halo_mass_to_radius(DM_hmf.m)
        
            DM_hmf.update(halo_profile_params={'eta_bloat':eta_sat, 'nu':list(nu[jz]), 'cosmo':this_cosmo_run})
            nfw_sat = DM_hmf.halo_profile.u(k, DM_hmf.m, c=conc_sat[jz,:], norm='m', coord='k')
            u_dm_sat[jz,:,:] = nfw_sat/np.expand_dims(nfw_sat[0,:], 0)
            r_s_sat[jz,:] = DM_hmf.halo_profile._rs_from_m(DM_hmf.m)
            rvir_sat[jz,:] = DM_hmf.halo_profile.halo_mass_to_radius(DM_hmf.m)
        
        else:
            DM_hmf.update(halo_profile_params={'eta_bloat':eta_cen, 'nu':list(nu[jz]), 'cosmo':this_cosmo_run}, halo_concentration_params={'norm':norm_cen, 'sigma8':sigma_8, 'ns':ns})
            conc_cen[jz,:] = DM_hmf.cmz_relation
            nfw_cen = DM_hmf.halo_profile.u(k, DM_hmf.m, c=conc_cen[jz,:], norm='m', coord='k')
            u_dm_cen[jz,:,:] = nfw_cen/np.expand_dims(nfw_cen[0,:], 0)
            r_s_cen[jz,:] = DM_hmf.halo_profile._rs_from_m(DM_hmf.m)
            rvir_cen[jz,:] = DM_hmf.halo_profile.halo_mass_to_radius(DM_hmf.m)
        
            DM_hmf.update(halo_profile_params={'eta_bloat':eta_sat, 'nu':list(nu[jz]), 'cosmo':this_cosmo_run}, halo_concentration_params={'norm':norm_sat, 'sigma8':sigma_8, 'ns':ns})
            conc_sat[jz,:] = DM_hmf.cmz_relation
            nfw_sat = DM_hmf.halo_profile.u(k, DM_hmf.m, c=conc_sat[jz,:], norm='m', coord='k')
            u_dm_sat[jz,:,:] = nfw_sat/np.expand_dims(nfw_sat[0,:], 0)
            r_s_sat[jz,:] = DM_hmf.halo_profile._rs_from_m(DM_hmf.m)
            rvir_sat[jz,:] = DM_hmf.halo_profile.halo_mass_to_radius(DM_hmf.m)
    
    

    #  TODO: Clean these up. Put more of them into the same folder
    block.put_grid('concentration_dm', 'z', z_vec, 'm_h', mass, 'c', conc_cen)
    block.put_grid('concentration_sat', 'z', z_vec, 'm_h', mass, 'c', conc_sat)
    block.put_grid('nfw_scale_radius_dm', 'z', z_vec, 'm_h', mass, 'rs', r_s_cen)
    block.put_grid('nfw_scale_radius_sat', 'z', z_vec, 'm_h', mass, 'rs', r_s_sat)

    block.put_double_array_1d('virial_radius', 'm_h', mass)
    block.put_double_array_1d('virial_radius', 'rvir_dm', rvir_cen[0])
    block.put_double_array_1d('virial_radius', 'rvir_sat', rvir_sat[0])

    block.put_double_array_1d('fourier_nfw_profile', 'z', z_vec)
    block.put_double_array_1d('fourier_nfw_profile', 'm_h', mass)
    block.put_double_array_1d('fourier_nfw_profile', 'k_h', k)
    block.put_double_array_nd('fourier_nfw_profile', 'ukm', u_dm_cen)
    block.put_double_array_nd('fourier_nfw_profile', 'uksat', u_dm_sat)
    ###################################################################################################################

    # density
    block['density', 'mean_density0'] = mean_density0
    block['density', 'rho_crit'] = mean_density0/this_cosmo_run.Om0
    block.put_double_array_1d('density', 'mean_density_z', mean_density_z)
    block.put_double_array_1d('density', 'rho_halo', rho_halo)

    # hmf
    block.put_grid('hmf', 'z', z_vec, 'm_h', mass, 'dndlnmh', dndlnmh)
    block.put_grid('hmf', 'z', z_vec, 'm_h', mass, 'nu', nu)
    block.put_double_array_1d('hmf', 'neff', neff)
    block.put_double_array_1d('hmf', 'sigma8_z', sigma8_z)

    # halobias
    block.put_grid('halobias', 'z', z_vec, 'm_h', mass, 'b_hb', b_nu)
    
    # cosmological parameters
    f_nu = this_cosmo_run.Onu0/this_cosmo_run.Om0
    # h_z = this_cosmo_run.H(z_vec).value/100.0
    # block.put_double_array_1d(cosmo_params, 'h_z', h_z)
    # block.put_double_array_1d(cosmo_params, 'z',z_vec)
    block[cosmo_params, 'fnu'] = f_nu

    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass


