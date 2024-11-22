# CosmoSiS module to compute the halo occupation distribution (HOD) conditional on a galaxy observable
# such as luminosity or stellar mass function.
# The formalism is described in Cacciato (2009, thesis), Cacciato et al. (2013, application to SDSS). 
# This is based on the parametrisation and functional forms first proposed by Yang et al. 2008

# The halo occupation distribution predicts the number of galaxies that populate a halo of mass M:
#
# N_gal(M) = N_cen(M) + N_sat(M),
#
# where the galaxies are divided into centrals and satellites. 
# The prediction depends on the number of galaxies with the given observable, O, 
# in an observable bin, [O-Delta_O/2, O+Delta_O/2] 
# given their halo mass, M: \Phi_x(O|M)
# The number of galaxies is then given by
#
# N_x(M,z) = \int \Phi_x(O|M) n(M,z) dL 
#
# where x=cen,sat.

from cosmosis.datablock import names, option_section
import numpy as np
from scipy.interpolate import interp1d
# halo model library with HOD and conditional functions
import hod_lib as hod

cosmo = names.cosmological_parameters

#--------------------------------------------------------------------------------#	

# a class with all the HOD parameters
class HODpar :
    """
    The conditional observable function provide a distribution for the galaxies populating a halo of mass M.
    For centrals this is a lognormal distribution, with a mean that is a function of halo mass and is
    described by a double power law.  
    For satellites this is a generalised Schechter function with two free powers. 
    M_char is characteristic halo mass that divides the two regimes of the double powerlaw.
    Obs_norm_c is the normalisation factor for central galaxies

    """
    def __init__(self, Obs_norm_c, M_char, g1, g2, sigma_log10_O_c, norm_s, pivot, alpha_s, beta_s, b0, b1, b2):
        # general parameters
        self.M_char = M_char 
        self.g_1 = g1
        self.g_2 = g2
        #centrals
        self.Obs_norm_c = Obs_norm_c
        self.sigma_log10_O_c = sigma_log10_O_c
        #satellites
        self.norm_s = norm_s  # extra normalisation factor for satellites
        self.pivot = pivot
        self.alpha_s = alpha_s #low obs slope 
        self.beta_s = beta_s #slope in the exponent
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2


# Used for reading data from a text file.
def load_data(file_name):
    z_data, min_magnitude, max_magnitude = np.loadtxt(file_name, usecols = (0,1,2), unpack=True, dtype=float)
    if (min_magnitude[0]>max_magnitude[0]):
        raise ErrorValue('Error: in the magnitues_file, the minimum magnitude must be more negative than the maximum magnitude.')
    return z_data, min_magnitude, max_magnitude


def setup(options):

    # TODO: Change the bining of the observable such that nbins can be larger than 1 if inputs are read through a file.

    # output section name for HOD related outputs.
    hod_section_name = options.get_string(option_section, 'hod_section_name').lower()
    # where to read the values of parameters in the value.ini
    values_name      = options.get_string(option_section, 'values_name','hod_parameters').lower()
    # output section name for the observable related quantities.
    observable_section_name  = options.get_string(option_section, 'observable_section_name', default='stellar_mass_function').lower()
    # TODO: check where this is used and if we need it
    # output section name for galaxy bias
    galaxy_bias_section_name = options.get_string(option_section, 'galaxy_bias_section_name', default='galaxy_bias').lower()


    # Checks units of h
    #observable_h_unit = options.get_string(option_section, 'observable_h_unit', default='1/h').lower()
    observable_h_unit = options.get_string(option_section, 'observable_h_unit', default='1/h^2').lower()
    valid_units = ['1/h', '1/h^2']
    if not observable_h_unit in valid_units:
        raise Exception('Currently supported h factors in obserable are {valid_units}')

    # if file name is given then use it otherwise use values in the ini file # in units of O_sun/h2
    if options.has_value(option_section, 'observables_file'):
        observables_z = True
        file_name     = options.get_string(option_section, 'observables_file')
        z_bins, obs_min, obs_max = load_data(file_name)
        nz     = len(z_bins)
        log_obs_min = np.log10(obs_min)[np.newaxis,:]
        log_obs_max = np.log10(obs_max)[np.newaxis,:]
        z_bins = z_bins[np.newaxis,:]
        # number of bins in the observable, for example you might have divided your sample into 3 stellar mass bins
        # With a file input we are assuming that eveything is part of the same bin currently. 
        nbins  = 1
    elif options.has_value(option_section, 'mass_lim') and options.has_value(option_section, 'mass_lim_low'):
        observables_z = True
        file_name     = options.get_string(option_section, 'mass_lim')
        file_name_low = options.get_string(option_section, 'mass_lim_low')
        z_bins = np.linspace(options[option_section, 'zmin'], options[option_section, 'zmax'], options[option_section, 'nz'])
        
        with open(file_name, 'rb') as dill_file:
            fit_func_inv = pickle.load(dill_file)
        
        with open(file_name_low, 'rb') as dill_file:
            fit_func_low = pickle.load(dill_file)
        
        obs_min = fit_func_inv(z_bins)
        obs_max = fit_func_low(z_bins)
        nz     = options[option_section, 'nz']
        log_obs_min = np.log10(obs_min)[np.newaxis,:]
        log_obs_max = np.log10(obs_max)[np.newaxis,:]
        z_bins = z_bins[np.newaxis,:]
        # number of bins in the observable, for example you might have divided your sample into 3 stellar mass bins
        # With a file input we are assuming that eveything is part of the same bin currently.
        nbins  = 1
    else:
        observables_z = False
        # These are the values used to define the edges of the observable and redshift bins. 
        # They are used to create volume limites samples of the galaxies.
        # 1 or more values can be given for each min max value, as long as they are all the same length.
        obs_min = np.asarray([options[option_section, 'obs_min']]).flatten()
        obs_max = np.asarray([options[option_section, 'obs_max']]).flatten()
        zmin = np.asarray([options[option_section, 'zmin']]).flatten()
        zmax = np.asarray([options[option_section, 'zmax']]).flatten()
        # TODO: number of redshift bins used for ...
        nz      = options[option_section, 'nz']
        
        # Check if the length of obs_min, obs_max, zmin and zmax match.
        if not np.all(np.array([len(obs_min), len(obs_max), len(zmin), len(zmax)]) == len(obs_min)):
            raise Exception('Error: obs_min, obs_max, zmin and zmax need to be of same length.')
        else:
            # nbins is the number of observable bins.
            nbins = len(obs_min)
        
        # Arrays using starting and end values of zmin and zmax 
        # TODO: Check if this is what we want
        z_bins = np.array([np.linspace(zmin_i, zmax_i, nz) for zmin_i, zmax_i in zip(zmin, zmax)])
        #z_bins = np.array([np.linspace(zmin_i, zmax_i, nz, endpoint=True) for zmin_i, zmax_i in zip(zmin, zmax)])
        # This simply repeats the min and max values for the observables for all redshift bins
        log_obs_min = np.array([np.repeat(obs_min_i,nz) for obs_min_i in obs_min])
        log_obs_max = np.array([np.repeat(obs_max_i,nz) for obs_max_i in obs_max])

    # read this from the ini file
    # number of bins used for defining observable functions, usually a larger number
    nobs = options.get_int(option_section, 'nobs', 200)



    # TODO: check if we need this
    #It just outputs estimates of the linear bias for the HOD which isn't necessarily a bad thing?
    galaxy_bias_option = options.get_bool(option_section, 'do_galaxy_linear_bias', False)

    save_observable   = options.get_bool(option_section, 'save_observable',True)
    # options are: "obs_z" or "obs_zmed" or "obs_onebin" depending if you want to calculate 
    # the observable function per each redshift or on the median one or per one big bin
    observable_mode   = options.get_string(option_section, 'observable_mode','obs_z')
    # TODO: Check if this is z_median or if it is used in the other options
    z_picked          = options.get_double(option_section, 'z_input',0.1)


    # per each redshift bin, the range of observables over which we can integrate the conditional function changes, 
    # due to the flux lim of the survey. 
    # This means that per each redshift, we have a different observable values to be
    # employed in the log-simpson integration.
    # For stellar masses it holds the same, but we can also employ this to construct more complex samples/bins. 
    # Can pick lower redshift limit for particulare stellar mass, etc...

    # TODO: Check what this does
    # log-simpson integration
    # print('z\t log OBS_min(z)\t log OBS_max(z)\n')
    # A 3D array with nobs log-binned observables 
    obs_simps = np.empty([nbins,nz,nobs])
    for nb in range(0,nbins):
        for jz in range(0,nz):
            obs_minz = log_obs_min[nb,jz]
            obs_maxz = log_obs_max[nb,jz]
            obs_simps[nb,jz] = np.logspace(obs_minz, obs_maxz, nobs)
            # print ('%f %f %f' %(z_bins[nb,jz], obs_minz, obs_maxz))

    return  obs_simps, nbins, nz, nobs, z_bins,\
            z_picked, galaxy_bias_option, save_observable, observable_mode, hod_section_name,\
            values_name, observables_z, observable_section_name, galaxy_bias_section_name,\
            observable_h_unit, valid_units



# TODO: log_mass_min, log_mass_max not used here
def execute(block, config):

    obs_simps, nbins, nz, nobs, z_bins, z_picked, galaxy_bias_option, save_observable, \
    observable_mode, hod_section_name, values_name, observables_z, observable_section_name,\
    galaxy_bias_section_name, observable_h_unit, valid_units = config

    #---- loading hod value from the values.ini file ----#
    #centrals

    # all observable masses in units of log10(M_sun h^-2)
    log10_obs_norm_c = block[values_name, 'log10_obs_norm_c'] #O_0, O_norm_c
    log10_M_ch       = block[values_name, 'log10_m_ch'] # log10 M_char
    g1               = block[values_name, 'g1'] # gamma_1
    g2               = block[values_name, 'g2'] # gamma_2
    sigma_log10_O_c  = block[values_name, 'sigma_log10_O_c'] # sigma_log10_O_c

    # TODO: check how this works
    if block.has_value(values_name, 'A_cen'):
        A_cen = block[values_name, 'A_cen']
    else:
        A_cen = None

    #satellites
    norm_s   = block[values_name, 'norm_s'] # normalisation
    alpha_s  = block[values_name, 'alpha_s'] # goes into the conditional stellar mass function Phi_sat(M*|M)
    beta_s  = block[values_name, 'beta_s'] # goes into the conditional stellar mass function Phi_sat(M*|M)
    pivot    = block[values_name, 'pivot']  # pivot mass for the normalisation of the stellar mass function: ϕ∗s
    # log10[ϕ∗s(M)] = b0 + b1(log10 m_p)+ b2(log10 m_p)^2, m_p = M/pivot
    b0 = block[values_name, 'b0'] 
    b1 = block[values_name, 'b1']
    b2 = block[values_name, 'b2']
    
    # TODO: check how this works
    if block.has_value(values_name, 'A_sat'):
        A_sat = block[values_name, 'A_sat']
    else:
        A_sat = None

    hod_par = HODpar(10.**log10_obs_norm_c, 10.**log10_M_ch, g1, g2, sigma_log10_O_c, norm_s, pivot, alpha_s, beta_s, b0, b1, b2)

    block.put_int(hod_section_name, 'nbins', nbins)
    block.put_bool(hod_section_name, 'option', observables_z)

    #---- loading the halo mass function ----#
    dndlnM_grid = block['hmf','dndlnmh']
    mass        = block['hmf','m_h']
    z_dn        = block['hmf','z']
    
    for nb in range(0,nbins):
        if nbins != 1:
            suffix = f'_{nb+1}'
        else:
            suffix = ''
            
        # set interpolator for the halo mass function
        f_int_dndlnM = interp1d(z_dn, dndlnM_grid, kind='linear', fill_value='extrapolate', bounds_error=False, axis=0)
        dndlnM = f_int_dndlnM(z_bins[nb])
        nmass = mass.size
    
        phi_c = np.empty([nz, nmass, nobs])
        phi_s = np.empty([nz, nmass, nobs])
        
        phi_c = hod.phi_cen(obs_simps[nb,:,np.newaxis], mass[:,np.newaxis], hod_par)
        phi_s = hod.phi_sat(obs_simps[nb,:,np.newaxis], mass[:,np.newaxis], hod_par)
        
        phi = phi_c + phi_s


        ###################################   HALO OCCUPATION DISTRIBUTION   #########################################
    
        # Since the bins are a function of redshift, Phi(O(z)|M) is a 3-dim array in O, z, M. The
        # resulting HODs are a function of mass and redshift. Note that they would only be a function of mass in theory.
        # The dependence on redshift comes as a result of the flux-lim of the survey. It's not physical at this stage and
        # does not capture the passive evolution of galaxies, that has to be modelled in an independent way.
    
        # ⟨Nx|M⟩ =int_{O_low}^{O_high} Φx(O|M) dO
        n_sat  = np.array([hod.compute_hod(obs_simps_z, phi_s_z) for obs_simps_z, phi_s_z in zip(obs_simps[nb], phi_s)])
        n_cen  = np.array([hod.compute_hod(obs_simps_z, phi_c_z) for obs_simps_z, phi_c_z in zip(obs_simps[nb], phi_c)])
        
        # f_star = int_{O_low}^{O_high} Φx(O|M) O dO
        # TODO: check the h units are correct
        # valid_units[1] is 1/h^2
        f_star = np.array([hod.compute_stellar_fraction(obs_simps_z, phi_z_i)/mass for obs_simps_z, phi_z_i in zip(obs_simps[nb], phi)])
        if observable_h_unit == valid_units[1]:
            f_star = f_star * block['cosmological_parameters', 'h0']

        # TODO:check this
        # Assembly bias (using the decorated HOD formalism for concentration as a secondary parameter):
        # arXiv:1512.03050
        
        if A_cen is not None:
            delta_pop_c = A_cen * np.fmin(n_cen, 1.0-n_cen)
            n_cen = n_cen + delta_pop_c
        if A_sat is not None:
            delta_pop_s = A_sat * n_sat
            n_sat = n_sat + delta_pop_s

        n_tot = n_cen + n_sat

        # TODO: Is there a better way to do this? The z dependence doesn't do anything. They are exactly the same values.
        # Do we need the z dependence? 
        block.put_grid(hod_section_name, f'z{suffix}', z_bins[nb], f'mass{suffix}', mass, f'n_sat{suffix}', n_sat)
        block.put_grid(hod_section_name, f'z{suffix}', z_bins[nb], f'mass{suffix}', mass, f'n_cen{suffix}', n_cen)
        block.put_grid(hod_section_name, f'z{suffix}', z_bins[nb], f'mass{suffix}', mass, f'n_tot{suffix}', n_tot)
        block.put_grid(hod_section_name, f'z{suffix}', z_bins[nb], f'mass{suffix}', mass, f'f_star{suffix}', f_star)
    
        # Nx = int ⟨Nx|M⟩ n(M) dM
        numdens_cen = hod.compute_number_density(mass, n_cen, dndlnM)
        numdens_sat = hod.compute_number_density(mass, n_sat, dndlnM)

        numdens_tot = numdens_cen + numdens_sat
        fraction_cen = numdens_cen/numdens_tot
        fraction_sat = numdens_sat/numdens_tot
        # compute average halo mass per bin
        # M_mean = int ⟨Nx|M⟩ M n(M) dM
        mass_avg = hod.compute_avg_halo_mass(mass, n_cen, dndlnM)/numdens_cen

        block.put_double_array_1d(hod_section_name, f'number_density_cen{suffix}', numdens_cen)
        block.put_double_array_1d(hod_section_name, f'number_density_sat{suffix}', numdens_sat)
        block.put_double_array_1d(hod_section_name, f'number_density_tot{suffix}', numdens_tot)
        block.put_double_array_1d(hod_section_name, f'central_fraction{suffix}', fraction_cen)
        block.put_double_array_1d(hod_section_name, f'satellite_fraction{suffix}', fraction_sat)
        block.put_double_array_1d(hod_section_name, f'average_halo_mass{suffix}', mass_avg)
        
        # Very important, the RegularGridInterpolator creates oscillations in the hmf. So we change this to interp1d
        # Need to either remove interpolation or use a different interpolation method.
        if galaxy_bias_option:
            #---- loading the halo bias function ----#
            z_hbf        = block['halobias', 'z']
            halobias_hbf = block['halobias', 'b_hb']

            f_interp_halobias = interp1d(z_hbf, halobias_hbf, kind='linear', fill_value='extrapolate', bounds_error=False, axis=0)
            hbias = f_interp_halobias(z_bins[nb])
            
            galaxybias_cen = hod.compute_galaxy_linear_bias(mass[np.newaxis,:], n_cen, hbias, dndlnM)/numdens_tot
            galaxybias_sat = hod.compute_galaxy_linear_bias(mass[np.newaxis,:], n_sat, hbias, dndlnM)/numdens_tot
            galaxybias_tot = hod.compute_galaxy_linear_bias(mass[np.newaxis,:], n_tot, hbias, dndlnM)/numdens_tot
            
            # TODO:Put these into a different section
            block.put_double_array_1d(hod_section_name, f'galaxy_bias_centrals{suffix}', galaxybias_cen)
            block.put_double_array_1d(hod_section_name, f'galaxy_bias_satellites{suffix}', galaxybias_sat)
            # # this can be useful in case you want to use the constant bias module to compute p_gg
            block.put_double_array_1d(hod_section_name, f'b{suffix}', galaxybias_tot)
    
        #######################################   OBSERVABLE FUNCTION   #############################################
    
        if save_observable and observable_mode == 'obs_z':
            suffix_obs = f'_{nb+1}'
            nl_obs = 100
            obs_range_h = np.logspace(np.log10(obs_simps[nb].min()),np.log10(obs_simps[nb].max()), nl_obs)
            obs_func_h = np.empty([nz,nl_obs])
                
            obs_func_tmp = hod.obs_func(mass[np.newaxis,:,np.newaxis], phi, dndlnM[:,:,np.newaxis], axis=-2)
    
            # interpolate in L_obs to have a consistent grid
            for jz in range(0,nz):
                interp = interp1d(obs_simps[nb,jz], obs_func_tmp[jz], kind='linear', bounds_error=False, fill_value=(0,0))
                obs_func_h[jz] = interp(obs_range_h)
                    
            #TODO: put this in a different section
            block.put_grid(observable_section_name, f'z_bin{suffix_obs}', z_bins[nb], f'obs_val{suffix_obs}', obs_range_h, f'obs_func{suffix_obs}', np.log(10.0)*obs_func_h*obs_range_h)
            
    if save_observable:
        block.put(observable_section_name,'observable_mode', observable_mode)
    
    # Calculating the full stellar mass fraction and if desired the observable function for one bin case
    nl_obs = 100
    nl_z = 15
    z_bins_one = np.linspace(z_bins.min(), z_bins.max(), nl_z)
    
    f_mass_z_one = interp1d(z_dn, dndlnM_grid, kind='linear', fill_value='extrapolate', bounds_error=False, axis=0)
    dn_dlnM_one = f_mass_z_one(z_bins_one)
    
    obs_range_h = np.empty([nl_z,nl_obs])
    for jz in range(0,nl_z):
        obs_range_h[jz] = np.logspace(np.log10(obs_simps.min()),np.log10(obs_simps.max()), nl_obs)
    obs_func_h = np.empty([nl_z,nl_obs])
    
    phi_c = hod.phi_cen(obs_range_h[:,np.newaxis], mass[:,np.newaxis], hod_par)
    phi_s = hod.phi_sat(obs_range_h[:,np.newaxis], mass[:,np.newaxis], hod_par)
    phi = phi_c + phi_s
    
    # TODO: What is this one for? There is already f_start for the bins.
    # The TO-DO in this lines needs explanation: f_star here is different then f_star calculate for each bin, 
    # thus it needs to be saved differently. This one is used for the stellar mass contribution in Pmm, 
    # so for baryonic feedback in cosmic shear and thus needs to be calculated for a wide range of halo masses, 
    # unlike the other f_star which are for each stellar mass bin. I would keep the metadata block here 
    # to save all the parameters not directly connected with "per bin" HODs and corresponding products.
    # AD: added suffix here in order to keep track of the right one if multiple hods used!
    f_star_mm = np.array([hod.compute_stellar_fraction(obs_range_h_i, phi_z_i)/mass for obs_range_h_i, phi_z_i in zip(obs_range_h, phi)])
    if observable_h_unit == valid_units[1]:
        f_star_mm = f_star_mm * block['cosmological_parameters', 'h0']
    block.put_grid(hod_section_name, 'z_extended', z_bins_one, 'mass_extended', mass, 'f_star_extended', f_star_mm)
    
    if save_observable and observable_mode == 'obs_onebin':
        obs_func_h = hod.obs_func(mass[np.newaxis,:,np.newaxis], phi, dn_dlnM_one[:,:,np.newaxis], axis=-2)

        #TODO: put this in a different section
        block.put_grid(observable_section_name, 'z_bin_1', z_bins_one, 'obs_val_1', obs_range_h[0], 'obs_func_1', np.log(10.0)*obs_func_h*obs_range_h[0])
    
    
    #########################
    
    # The following applies for a single computation of the observable Function (at the z_median of the sample)
    
    # Compute observable function: note that since the observable function is computed for a single value of z and
    # might have a different range in magnitudes with respect to the case of the hod section (independent
    # options), we decided to re-compute phi rather than interpolating on the previous one.
    # At the moment, we assume the observable function is to be computed on the largest possible range 
    # of absolute magnitudes.
    
    if save_observable and observable_mode == 'obs_zmed':
        #interpolate the hmf at the redshift where the observable function is evaluated
        #f_mass_z_dn = interp2d(mass_dn, z_bins, dndlnM)
        #dn_dlnM_zmedian = f_mass_z_dn(mass_dn, z_picked)
        
        f_mass_z_dn = interp1d(z_dn, dndlnM_grid, kind='linear', fill_value='extrapolate', bounds_error=False, axis=0)
        dn_dlnM_zmedian = f_mass_z_dn(z_picked)
        
    
        # logspace values for the obervable values (e.g. stellar masses)
        obs_range = np.logspace(np.log10(obs_simps.min()), np.log10(obs_simps.max()), nobs)
    
        phi_c_lf = hod.phi_cen(obs_range[:,np.newaxis], mass, hod_par)
        phi_s_lf = hod.phi_sat(obs_range[:,np.newaxis], mass, hod_par)
        phi_lf   = phi_c_lf+phi_s_lf

        obs_func = hod.obs_func(mass, phi_lf, dn_dlnM_zmedian)
            
        # AD: CHECK THE h HERE!
        # AD: Should be without as the h is carried through in the first place!
        # AD: ln(10) factor added to the output and multiplication with M/L to get to the usual units data are in 99% reported in!

        obs_h = obs_range#*(h**2.) #note that the _h subscript avoids mixing h conventions while computing the clf_quantities
        obs_func_h = obs_func

        # x value for the observable function (e.g. stellar masses)
        block.put_double_array_1d(observable_section_name,'obs_val_med',obs_h)
        #block.put_double_array_1d('observable_function' + suffix,'obs_func_med',obs_func_h)
        block.put_double_array_1d(observable_section_name,'obs_func_med',np.log(10.)*obs_func_h*obs_h)

        #Mean value of the observable for central galaxies
        mean_obs_cen = hod.cal_mean_obs_c(mass, hod_par)

        block.put_double_array_1d(observable_section_name,'halo_mass_med',mass)
        block.put_double_array_1d(observable_section_name,'mean_obs_halo_mass_relation',mean_obs_cen)
    
    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
