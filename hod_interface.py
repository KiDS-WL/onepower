# CosmoSiS module to compute the halo occupation distribution (HOD) given the conditional
# luminosity function, as described in Cacciato et al. 2009,2013. The functions that describes
# the predictions of the conditional luminosity function are computed in lf_lib_simps.py


# The halo occupation distribution predicts the number of galaxies that populate a halo of mass M:
#
# N_gal(M) = N_cen(M) + N_sat(M)
#
# In this formalism, such prediction comes from the conditional luminosity function
# which describes the number of galaxies of luminosity L in [L-dL/2, L+dL/2] in a halo of mass M,
# Phi(L|M). The number of galaxies is then given by
#
# N_j(M,z) = \int \Phi_j(L|M) n(M,z) dL 
#
# where j=cen,sat.

# TODO: Generalise the functional forms for HOD relations!

from cosmosis.datablock import names, option_section
import numpy as np
import cf_lib as cf
from scipy.interpolate import interp1d, interp2d, RegularGridInterpolator

# We have a collection of commonly used pre-defined block section names.
# If none of the names here is relevant for your calculation you can use any
# string you want instead.
cosmo = names.cosmological_parameters

#--------------------------------------------------------------------------------#	

class HODpar :
    def __init__(self, norm_c, ml_0, ml_1, g1, g2, sigma_c, norm_s, pivot, alpha_star, b0, b1, b2):
        #centrals
        self.norm_c = norm_c
        self.ml_1 = ml_1
        self.ml_0 = ml_0
        self.g_1 = g1
        self.g_2 = g2
        self.sigma_c = sigma_c
        #satellites
        self.norm_s = norm_s
        self.pivot = pivot
        self.alpha_star = alpha_star
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2

def load_data(file_name):
    z_data, min_magnitude, max_magnitude = np.loadtxt(file_name, usecols = (0,1,2), unpack=True, dtype=float)
    if (min_magnitude[0]>max_magnitude[0]):
        raise ErrorValue('Error: in the magnitues_file, the minimum magnitude must be more negative than the maximum magnitude.')
    return z_data, min_magnitude, max_magnitude


def setup(options):
    #This function is called once per processor per chain.
    #It is a chance to read any fixed options from the configuration file,
    #load any data, or do any calculations that are fixed once.

    observables_z = options[option_section, 'observables_z']

    if observables_z:
        file_name = options[option_section, 'observables_file'] # in units of L_sun/h2
        z_bins, obs_min, obs_max = load_data(file_name)
        nobs = options[option_section, 'nobs']
        nz = len(z_bins)
        log_obs_min = np.log10(obs_min)[np.newaxis,:]
        log_obs_max = np.log10(obs_max)[np.newaxis,:]
        z_bins = z_bins[np.newaxis,:]
        nbins = 1
    else:
        obs_min = np.array([np.float64(str_val) for str_val in str(options[option_section, 'obs_min']).split(',')])
        obs_max = np.array([np.float64(str_val) for str_val in str(options[option_section, 'obs_max']).split(',')])
        zmin = np.array([np.float64(str_val) for str_val in str(options[option_section, 'zmin']).split(',')])
        zmax = np.array([np.float64(str_val) for str_val in str(options[option_section, 'zmax']).split(',')])
        nobs = options[option_section, 'nobs']
        nz = options[option_section, 'nz']
        
        if not np.all(np.array([len(obs_min), len(obs_max), len(zmin), len(zmax)]) == len(obs_min)):
            raise Exception('Error: obs_min, obs_max, zmin and zmax need to be of same length.')
        else:
            nbins = len(obs_min)
        
        z_bins = np.array([np.linspace(zmin_i, zmax_i, nz, endpoint=True) for zmin_i, zmax_i in zip(zmin, zmax)])
        #z_bins_tmp = np.array([np.linspace(zmin_i, zmax_i, nz+1, endpoint=True) for zmin_i, zmax_i in zip(zmin, zmax)])
        #z_bins = np.array([(b[1:] + b[:-1])/2 for b in z_bins_tmp])
        log_obs_min = np.array([np.repeat(obs_min_i,nz) for obs_min_i in obs_min])
        log_obs_max = np.array([np.repeat(obs_max_i,nz) for obs_max_i in obs_max])

    log_mass_min = options[option_section, 'log_mass_min']
    log_mass_max = options[option_section, 'log_mass_max']
    nmass = options[option_section, 'nmass']

    #---- log-spaced mass sample ----#
    dlog10m = (log_mass_max-log_mass_min)/nmass
    mass = 10.0 ** np.arange(log_mass_min, log_mass_max, dlog10m)

    hod_option = options[option_section, 'do_hod']
    galaxy_bias_option = options[option_section, 'do_galaxy_linear_bias']

    if (hod_option == False) and (galaxy_bias_option == True):
        raise ValueError('Error, if you want to compute the galaxy linear bias,'
        'please, select the hod option too.')

    observable_option = options[option_section, 'do_observable_function']
    observable_mode = options[option_section, 'observable_mode']
    z_picked = options[option_section, 'z_median']


    #abs_mag_sun = options[option_section, 'abs_mag_sun']

    name = options.get_string(option_section, 'output_suffix', default='').lower()
    if name != '':
        suffix = '_' + name
    else:
        suffix = ''
        
    name_params = options.get_string(option_section, 'params_suffix', default='').lower()
    if name_params != '':
        suffix_params = '_' + name_params
    else:
        suffix_params = ''

    # per each redshift bin, the range of observables over which we can integrate the conditional function changes, due to the
    # flux lim of the survey. This means that per each redshift, we have a different luminosity array to be
    # employed in the log-simpson integration.
    # AD: For stellar masses it holds the same, but we can also employ this to construct more complex samples/bins. Can pick lower redshift limit for particulare stellar mass, etc...

    print('z\t log OBS_min(z)\t log OBS_max(z)\n')
    obs_simps = np.empty([nbins,nz,nobs])
    for nb in range(0,nbins):
        for jz in range(0,nz):
            obs_minz = log_obs_min[nb,jz]
            obs_maxz = log_obs_max[nb,jz]
            obs_simps[nb,jz] = np.logspace(obs_minz, obs_maxz, nobs)
            print ('%f %f %f' %(z_bins[nb,jz], obs_minz, obs_maxz))
    
    return obs_simps, nbins, nz, nobs, z_bins, log_mass_min, log_mass_max, nmass, mass, z_picked, hod_option, galaxy_bias_option, observable_option, observable_mode, suffix, suffix_params, observables_z




def execute(block, config):
    #This function is called every time you have a new sample of cosmological and other parameters.
    #It is the main workhorse of the code. The block contains the parameters and results of any 
    #earlier modules, and the config is what we loaded earlier.

    obs_simps, nbins, nz, nobs, z_bins, log_mass_min, log_mass_max, nmass, mass, z_picked, hod_option, galaxy_bias_option, observable_option, observable_mode, suffix0, suffix_params, observables_z = config

    #---- loading hod from the datablock ----#

    #centrals
    norm_c = block['hod_parameters' + suffix_params, 'norm_c']
    log_ml_0 = block['hod_parameters' + suffix_params, 'log_ml_0']
    log_ml_1 = block['hod_parameters' + suffix_params, 'log_ml_1']
    g1 = block['hod_parameters' + suffix_params, 'g1']
    g2 = block['hod_parameters' + suffix_params, 'g2']
    scatter=block['hod_parameters' + suffix_params, 'scatter']
    #satellites
    norm_s = block['hod_parameters' + suffix_params, 'norm_s']
    pivot = block['hod_parameters' + suffix_params, 'pivot']
    alpha_s = block['hod_parameters' + suffix_params, 'alpha_s']
    b0 = block['hod_parameters' + suffix_params, 'b0']
    b1 = block['hod_parameters' + suffix_params, 'b1']
    b2 = block['hod_parameters' + suffix_params, 'b2']
    if block.has_value('hod_parameters' + suffix_params, 'A_cen'):
        A_cen = block['hod_parameters' + suffix_params, 'A_cen']
    else:
        A_cen = None
    if block.has_value('hod_parameters' + suffix_params, 'A_sat'):
        A_sat = block['hod_parameters' + suffix_params, 'A_sat']
    else:
        A_sat = None

    hod = HODpar(norm_c, 10.**log_ml_0, 10.**log_ml_1, g1, g2, scatter, norm_s, pivot, alpha_s, b0, b1, b2)

    block.put_int('hod' + suffix0 + '_metadata', 'nbins', nbins)
    block.put_bool('hod' + suffix0 + '_metadata', 'option', observables_z)

    #---- loading the halo mass function ----#

    dndlnM_grid = block['hmf','dndlnmh']
    mass_dn = block['hmf','m_h']
    z_dn = block['hmf','z']
    
    for nb in range(0,nbins):
        if nbins != 1:
            suffix = suffix0 + '_{}'.format(nb+1)
        else:
            suffix = suffix0
            
        f_int_dndlnM = RegularGridInterpolator((mass_dn.T, z_dn.T), dndlnM_grid.T, bounds_error=False, fill_value=None)
        mass_i, z_bins_i = np.meshgrid(mass, z_bins[nb], sparse=True)
        dndlnM = f_int_dndlnM((mass_i.T, z_bins_i.T)).T
    
        phi_c = np.empty([nz, nmass, nobs])
        phi_s = np.empty([nz, nmass, nobs])
        
        phi_c = cf.cf_cen(obs_simps[nb,:,np.newaxis], mass[:,np.newaxis], hod)
        phi_s = cf.cf_sat(obs_simps[nb,:,np.newaxis], mass[:,np.newaxis], hod)
        
        phi = phi_c + phi_s


        ###################################   HALO OCCUPATION DISTRIBUTION   #########################################
    
        # Since the luminosity bins are a function of redshift, Phi(L(z)|M) is a 3-dim array in L, z, M. The
        # resulting HODs are a function of mass and redshift. Note that they would only be a function of mass in theory.
        # The dependence on redshift comes as a result of the flux-lim of the survey. It's not physical at this stage and
        # does not capture the passive evolution of galaxies, that has to be modelled in an independent way.
    
        if hod_option:
            n_sat = np.array([cf.compute_hod(obs_simps_z, phi_s_z) for obs_simps_z, phi_s_z in zip(obs_simps[nb], phi_s)])
            n_cen = np.array([cf.compute_hod(obs_simps_z, phi_c_z) for obs_simps_z, phi_c_z in zip(obs_simps[nb], phi_c)])
            f_star = np.array([cf.compute_stellar_fraction(obs_simps_z, phi_z_i)/mass for obs_simps_z, phi_z_i in zip(obs_simps[nb], phi)])

            # Assembly bias (using the decorated HOD formalism for concentration as a secondary parameter):
            # arXiv:1512.03050
            
            if A_cen is not None:
                delta_pop_c = A_cen * np.fmin(n_cen, 1.0-n_cen)
                n_cen = n_cen + delta_pop_c
            if A_sat is not None:
                delta_pop_s = A_sat * n_sat
                n_sat = n_sat + delta_pop_s
    
            n_tot = n_cen + n_sat
    
            block.put_grid('hod' + suffix, 'z', z_bins[nb], 'mass', mass, 'n_sat', n_sat)
            block.put_grid('hod' + suffix, 'z', z_bins[nb], 'mass', mass, 'n_cen', n_cen)
            block.put_grid('hod' + suffix, 'z', z_bins[nb], 'mass', mass, 'n_tot', n_tot)
            block.put_grid('hod' + suffix, 'z', z_bins[nb], 'mass', mass, 'f_star', f_star)
        
            numdens_cen = cf.compute_number_density(mass, n_cen, dndlnM)
            numdens_sat = cf.compute_number_density(mass, n_sat, dndlnM)
    
            numdens_tot = numdens_cen + numdens_sat
            fraction_cen = numdens_cen/numdens_tot
            fraction_sat = numdens_sat/numdens_tot
    
            # save on datablock
            block.put_double_array_1d('hod' + suffix, 'number_density_cen', numdens_cen)
            block.put_double_array_1d('hod' + suffix, 'number_density_sat', numdens_sat)
            block.put_double_array_1d('hod' + suffix, 'number_density_tot', numdens_tot)
            block.put_double_array_1d('hod' + suffix, 'central_fraction', fraction_cen)
            block.put_double_array_1d('hod' + suffix, 'satellite_fraction', fraction_sat)
            
            # compute average halo mass per bin
            mass_avg = cf.compute_avg_halo_mass(mass, n_cen, dndlnM)/numdens_cen
            block.put_double_array_1d('hod' + suffix, 'average_halo_mass', mass_avg)
            
            if galaxy_bias_option:
                #---- loading the halo bias function ----#
                mass_hbf = block['halobias', 'm_h']
                z_hbf = block['halobias', 'z']
                halobias_hbf = block['halobias', 'b_hb']
    
                f_interp_halobias = RegularGridInterpolator((mass_hbf.T, z_hbf.T), halobias_hbf.T, bounds_error=False, fill_value=None)
                mass_i, z_bins_i = np.meshgrid(mass, z_bins[nb], sparse=True)
                hbias = f_interp_halobias((mass_i.T,z_bins_i.T)).T
    
                galaxybias_cen = cf.compute_galaxy_linear_bias(mass[np.newaxis,:], n_cen, hbias, dndlnM)/numdens_tot
                galaxybias_sat = cf.compute_galaxy_linear_bias(mass[np.newaxis,:], n_sat, hbias, dndlnM)/numdens_tot
                galaxybias_tot = cf.compute_galaxy_linear_bias(mass[np.newaxis,:], n_tot, hbias, dndlnM)/numdens_tot
    
                block.put_double_array_1d('galaxy_bias' + suffix, 'galaxy_bias_centrals', galaxybias_cen)
                block.put_double_array_1d('galaxy_bias' + suffix, 'galaxy_bias_satellites', galaxybias_sat)
                # this can be useful in case you want to use the constant bias module to compute p_gg
                block.put_double_array_1d('galaxy_bias' + suffix, 'b', galaxybias_tot)
    
        #######################################   OBSERVABLE FUNCTION   #############################################
    
        if observable_option and observable_mode == 'obs_z':
            nl_obs = 100
            obs_range_h = np.logspace(np.log10(obs_simps[nb].min()),np.log10(obs_simps[nb].max()), nl_obs)
            obs_func_h = np.empty([nz,nl_obs])
                
            obs_func_tmp = cf.obs_func(mass[np.newaxis,:,np.newaxis], phi, dndlnM[:,:,np.newaxis], axis=-2)
    
            # interpolate in L_obs to have a consistent grid
            for jz in range(0,nz):
                interp = interp1d(obs_simps[nb,jz], obs_func_tmp[jz], kind='linear', bounds_error=False, fill_value=(0,0))
                obs_func_h[jz] = interp(obs_range_h)
                    
            #save on datablock
            block.put_grid('observable_function' + suffix0, 'z_bin_{}'.format(nb+1), z_bins[nb], 'obs_{}'.format(nb+1), obs_range_h, 'obs_func_{}'.format(nb+1), np.log(10.0)*obs_func_h*obs_range_h)
            
            
    # Calculating the full stellar mass fraction and if desired the observable function for one bin case
    nl_obs = 100
    nl_z = 15
    z_bins_one = np.linspace(z_bins.min(), z_bins.max(), nl_z)
        
    f_mass_z_one = RegularGridInterpolator((mass_dn.T, z_dn.T), dndlnM_grid.T, bounds_error=False, fill_value=None)
    mass_one_i, z_one_i = np.meshgrid(mass_dn, z_bins_one)
    dn_dlnM_one = f_mass_z_one((mass_one_i.T, z_one_i.T)).T
        
    obs_range_h = np.empty([nl_z,nl_obs])
    for jz in range(0,nl_z):
        obs_range_h[jz] = np.logspace(np.log10(obs_simps.min()),np.log10(obs_simps.max()), nl_obs)
    obs_func_h = np.empty([nl_z,nl_obs])
        
    phi_c = cf.cf_cen(obs_range_h[:,np.newaxis], mass[:,np.newaxis], hod)
    phi_s = cf.cf_sat(obs_range_h[:,np.newaxis], mass[:,np.newaxis], hod)
    phi = phi_c + phi_s
    
    f_star = np.array([cf.compute_stellar_fraction(obs_range_h_i, phi_z_i)/mass for obs_range_h_i, phi_z_i in zip(obs_range_h, phi)])
    block.put_grid('hod' + suffix0 + '_metadata', 'z', z_bins_one, 'mass', mass, 'f_star', f_star)
    
    if observable_option and observable_mode == 'obs_onebin':
        obs_func_h = cf.obs_func(mass[np.newaxis,:,np.newaxis], phi, dn_dlnM_one[:,:,np.newaxis], axis=-2)

        #save on datablock
        block.put_grid('observable_function' + suffix0, 'z_bin_{}'.format(1), z_bins_one, 'obs_{}'.format(1), obs_range_h[0], 'obs_func_{}'.format(1),np.log(10.0)*obs_func_h*obs_range_h[0])
        
    
    
    #########################
    
    # The following applies for a sigle computation of the Luminosity Function (at the z_median of the sample)
    
    # Compute luminosity function: note that since the luminosity function is computed for a single value of z and
    # might have a different range in magnitudes with respect to the case of the hod section (independent
    # options), we decided to re-compute phi rather than interpolating on the previous one.
    # At the moment, we assume the LF to be computed on the largest possible range of absolute magnitudes.
    
    if observable_option and observable_mode == 'obs_zmed':
        #interpolate the hmf at the redshift where the luminosity function is evaluated
        #f_mass_z_dn = interp2d(mass_dn, z_bins, dndlnM)
        #dn_dlnM_zmedian = f_mass_z_dn(mass_dn, z_picked)
        
        f_mass_z_dn = RegularGridInterpolator((mass_dn.T, z_dn.T), dndlnM_grid.T, bounds_error=False, fill_value=None)
        mass_dn_i, z_picked_i = np.meshgrid(mass_dn, z_picked)
        dn_dlnM_zmedian = f_mass_z_dn((mass_dn_i.T, z_picked_i.T)).T
    
        obs_range = np.logspace(np.log10(obs_simps.min()), np.log10(obs_simps.max()), nobs)
    
        phi_c_lf = cf.cf_cen(obs_range[:,np.newaxis], mass, hod)
        phi_s_lf = cf.cf_sat(obs_range[:,np.newaxis], mass, hod)
        phi_lf = phi_c_lf+phi_s_lf

        obs_func = cf.obs_func(mass, phi_lf, dn_dlnM_zmedian)
            
        # AD: CHECK THE h HERE!
        # AD: Should be without as the h is carried through in the first place!
        # AD: ln(10) factor added to the output and multiplication with M/L to get to the usual units data are in 99% reported in!
            
        #If required, convert everything to the h from the cosmological parameter section, otherwise keep h=1
        #h=1.
        #if rescale_to_h == True:
        #	h = block['cosmological_parameters', 'h0']
        #	print h

        # go back to the observed magnitudes
        obs_h = obs_range#/(h**2.) #note that the _h subscript avoids mixing h conventions while computing the clf_quantities
        obs_func_h = obs_func#*(h**5.)
        
        #mr_obs = cf.convert_to_magnitudes(obs_range, abs_mag_sun)

        #save on datablock
        block.put_double_array_1d('observable_function' + suffix0,'obs_med',obs_h)
        #block.put_double_array_1d('observable_function' + suffix,'obs_func_med',obs_func_h)
        block.put_double_array_1d('observable_function' + suffix0,'obs_func_med',np.log(10.)*obs_func_h*obs_h)

        #Back to magnitudes
        # It doesn't mean anything if the observable is stellar mass!
        #Lf_in_mags = 0.4*np.log(10.)*obs_h*obs_func_h

        #block.put_double_array_1d('observable_function' + suffix0,'mr_med', mr_obs)
        #block.put_double_array_1d('observable_function' + suffix0,'obs_mr_med',Lf_in_mags)

        #Characteristic luminosity of central galaxies
        obs_cen = cf.mor(mass, hod, norm_c)

        block.put_double_array_1d('observable_function' + suffix0,'halo_mass_med',mass)
        block.put_double_array_1d('observable_function' + suffix0,'obs_halo_mass_relation',obs_cen)
    
    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
