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

# AD: make this more general for stellar mass function, re-thing the liminusity limits, functional forms for HOD relations!

from cosmosis.datablock import names, option_section
import sys
import numpy as np
import clf_lib as lf
from scipy.interpolate import interp1d, interp2d, RegularGridInterpolator
from itertools import count

import time


# We have a collection of commonly used pre-defined block section names.
# If none of the names here is relevant for your calculation you can use any
# string you want instead.
cosmo = names.cosmological_parameters

#--------------------------------------------------------------------------------#	

class HODpar :
    def __init__(self, m1,l0,g1,g2,sigma_c, alpha_star, b0, b1, b2):
        #centrals
        self.m_1 = m1
        self.l_0 = l0
        self.g_1 = g1
        self.g_2 = g2
        self.sigma_c = sigma_c
        #satellites
        self.alpha_star = alpha_star
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2

def load_data(file_name):
    z_data, min_magnitude, max_magnitude = np.loadtxt(file_name, usecols = (0,1,2), unpack=True, dtype=np.float)
    if (min_magnitude[0]>max_magnitude[0]):
        raise ErrorValue("Error: in the magnitues_file, the minimum magnitude must be more negative than the maximum magnitude.")
    # AD: should support stellar masses ...
    return z_data, min_magnitude, max_magnitude


def setup(options):
    #This function is called once per processor per chain.
    #It is a chance to read any fixed options from the configuration file,
    #load any data, or do any calculations that are fixed once.

    luminosities_z = options[option_section, "luminosities_z"]

    if luminosities_z:
        file_name = options[option_section, "luminosities_file"] # in units of L_sun/h2
        z_bins, lum_min, lum_max = load_data(file_name)
        nl = options[option_section, "nlum"]
        nz = len(z_bins)
        log_lum_min = np.log10(lum_min)
        log_lum_max = np.log10(lum_max)
    else:
        lum_min = options[option_section, "lum_min"]
        lum_max = options[option_section, "lum_max"]
        nl = options[option_section, "nlum"]
        nz = options[option_section, "nz"]
        log_lum_min = np.repeat(lum_min,nz)
        log_lum_max = np.repeat(lum_max,nz)
        zmin = options[option_section, "zmin"]
        zmax = options[option_section, "zmax"]
        z_bins = np.linspace(zmin, zmax, nz)

    #nl = 200

    log_mass_min = options[option_section, "log_mass_min"]
    log_mass_max = options[option_section, "log_mass_max"]
    nmass = options[option_section, "nmass"]

    #---- log-spaced mass sample ----#
    mass = np.logspace(log_mass_min, log_mass_max, nmass) # units of M_sun/h

    hod_option = options[option_section, "do_hod"]
    number_density_option = options[option_section, "do_number_density"]
    galaxy_bias_option = options[option_section, "do_galaxy_linear_bias"]

    if (number_density_option == False) and (galaxy_bias_option == True):
        raise ValueError("Error, if you want to compute the galaxy linear bias,"
        "please, select the number density option too.")

    lf_option = options[option_section, "do_luminosity_function"]
    lf_mode = options[option_section, "lf_mode"] # options.get_string(option_section, "lf_mode", default=None).lower() #
    z_picked = options[option_section, "z_median"]

    clf_quantities = options[option_section, "do_clf_quantities"] 	#compute ancillary quantities
                                    #that comes with the HODs,
                                    #such as the fraction of satellites as
                                    #function of luminosity, the
                                    #mass-to-light-ratio of centrals etc

    abs_mag_sun = options[option_section, "abs_mag_sun"]

    name = options.get_string(option_section, "name", default="").lower()
    if name:
        suffix = "_" + name
    else:
        suffix = ""

    # per each redshift bin, the range of luminosities over which we can integrate the clf changes, due to the
    # flux lim of the survey. This means that per each redshift, we have a different luminosity array to be
    # employed in the log-simpson integration.

    print('z\t log L_min(z)\t log L_max(z)\n')
    l_z_simps = np.empty([nz,nl])
    for jz in range(0,nz):
        lminz = log_lum_min[jz]
        lmaxz = log_lum_max[jz]
        l_z_simps[jz] = np.logspace(lminz, lmaxz, nl)
        print ('%f %f %f' %(z_bins[jz], lminz, lmaxz))

    return l_z_simps, nz, nl, z_bins, abs_mag_sun, log_mass_min, log_mass_max, nmass, mass, z_picked, hod_option, \
    number_density_option, galaxy_bias_option, lf_option, clf_quantities, lf_mode, suffix




def execute(block, config):
    #This function is called every time you have a new sample of cosmological and other parameters.
    #It is the main workhorse of the code. The block contains the parameters and results of any 
    #earlier modules, and the config is what we loaded earlier.

    l_z_simps, nz, nl, z_bins, abs_mag_sun, log_mass_min, log_mass_max, nmass, mass, z_picked, hod_option, \
    number_density_option, galaxy_bias_option, lf_option, clf_quantities, lf_mode, suffix = config

    start_time = time.time()

    #---- loading hod from the datablock ----#

    #centrals
    lgm1=block["hod_parameters" + suffix, "lgm1"]
    lgl0=block["hod_parameters" + suffix, "lgl0"]
    g1=block["hod_parameters" + suffix, "g1"]
    g2=block["hod_parameters" + suffix, "g2"]
    scatter=block["hod_parameters" + suffix, "scatter"]
    #satellites
    alfa_s = block["hod_parameters" + suffix, "alfa_s"]
    b0 = block["hod_parameters" + suffix, "b0"]
    b1 = block["hod_parameters" + suffix, "b1"]
    b2 = block["hod_parameters" + suffix, "b2"]

    hod = HODpar(10.**lgm1, 10.**lgl0, g1, g2, scatter, alfa_s, b0, b1, b2)


    #---- loading the halo mass function ----#

    dndlnM_grid = block["hmf",'dndlnmh']
    mass_dn = block["hmf",'m_h']
    z_dn = block["hmf",'z']

    f_int_dndlnM = interp2d(mass_dn, z_dn, dndlnM_grid)
    dndlnM = f_int_dndlnM(mass, z_bins)

    phi_c = np.empty([nz, nmass, nl])
    phi_s = np.empty([nz, nmass, nl])

    for jz in range(0, nz):
        for im in range(0,nmass):
            phi_c[jz,im] = lf.clf_cen(l_z_simps[jz], mass[im], hod)
            phi_s[jz,im] = lf.clf_sat(l_z_simps[jz], mass[im], hod)

    phi = phi_c + phi_s


    ###################################   HALO OCCUPATION DISTRIBUTION   #########################################

    # Since the luminosity bins are a function of redshift, Phi(L(z)|M) is a 3-dim array in L, z, M. The
    # resulting HODs are a function of mass and redshift. Note that they would only be a function of mass in theory.
    # The dependence on redshift comes as a result of the flux-lim of the survey. It's not physical at this stage and
    # does not capture the passive evolution of galaxies, that has to be modelled in an independent way.

    if hod_option:
        n_sat = np.array([lf.compute_hod(l_z_simps_z, phi_s_z) for l_z_simps_z, phi_s_z in zip(l_z_simps, phi_s)])
        n_cen = np.array([lf.compute_hod(l_z_simps_z, phi_c_z) for l_z_simps_z, phi_c_z in zip(l_z_simps, phi_c)])

        n_tot = n_cen + n_sat

        block.put_grid("hod" + suffix, "z", z_bins, "mass", mass, "n_sat", n_sat)
        block.put_grid("hod" + suffix, "z", z_bins, "mass", mass, "n_cen", n_cen)
        block.put_grid("hod" + suffix, "z", z_bins, "mass", mass, "n_tot", n_tot)

        if number_density_option:

            numdens_cen = np.empty(nz)
            numdens_sat = np.empty(nz)

            for jz in range(0,nz):
                numdens_cen[jz] = lf.compute_number_density(mass, n_cen[jz], dndlnM[jz]) #this is already normalised
                numdens_sat[jz] = lf.compute_number_density(mass, n_sat[jz], dndlnM[jz]) #this is already normalised

            numdens_tot = numdens_cen + numdens_sat
            fraction_cen = numdens_cen/numdens_tot
            fraction_sat = numdens_sat/numdens_tot

            # save on datablock
            block.put_double_array_1d("hod" + suffix, "number_density_cen", numdens_cen)
            block.put_double_array_1d("hod" + suffix, "number_density_sat", numdens_sat)
            block.put_double_array_1d("hod" + suffix, "number_density_tot", numdens_tot)
            block.put_double_array_1d("hod" + suffix, "central_fraction", fraction_cen)
            block.put_double_array_1d("hod" + suffix, "satellite_fraction", fraction_sat)

            #print("--- hod: %s seconds ---" % (time.time() - start_time))


            if galaxy_bias_option:
                #---- loading the halo bias function ----#
                mass_hbf = block["halobias", "m_h"]
                z_hbf = block["halobias", "z"]
                halobias_hbf = block["halobias", "b_hb"]

                f_interp_halobias = interp2d(mass_hbf, z_hbf, halobias_hbf)
                hbias = f_interp_halobias(mass,z_bins)

                galaxybias_cen = np.empty(nz)
                galaxybias_sat = np.empty(nz)
                galaxybias_tot = np.empty(nz)

                for jz in range(0,nz):
                    galaxybias_cen[jz] = lf.compute_galaxy_linear_bias(mass, n_cen[jz], hbias[jz], dndlnM[jz])/numdens_tot[jz]
                    galaxybias_sat[jz] = lf.compute_galaxy_linear_bias(mass, n_sat[jz], hbias[jz], dndlnM[jz])/numdens_tot[jz]
                    galaxybias_tot[jz] = lf.compute_galaxy_linear_bias(mass, n_tot[jz], hbias[jz], dndlnM[jz])/numdens_tot[jz]

                block.put_double_array_1d("galaxy_bias" + suffix, "galaxy_bias_centrals", galaxybias_cen)
                block.put_double_array_1d("galaxy_bias" + suffix, "galaxy_bias_satellites", galaxybias_sat)
                # this can be useful in case you want to use the constant bias module to compute p_gg
                block.put_double_array_1d("galaxy_bias" + suffix, "b", galaxybias_tot)

    #print("--- bias; %s seconds ---" % (time.time() - start_time))


    #######################################   LUMINOSITY FUNCTION   #############################################

    if lf_option:
        if lf_mode == "lf_z":
            nl_obs = 100
            L_h = np.logspace(6.5,12.5, nl_obs)
            Lf_func_h = np.empty([nz,nl_obs])
            Lf_func_tmp = np.empty([nz,nl])

            for jz in range(0,nz):
                for il in range(0,nl):
                    Lf_func_tmp[jz,il] = lf.LF(mass, phi[jz,:,il], dndlnM[jz])

            # interpolate in L_obs to have a consistent grid
            for jz in range(0,nz):
                interp = interp1d(l_z_simps[jz], Lf_func_tmp[jz], kind='linear', bounds_error=False, fill_value=(0,0))
                Lf_func_h[jz] = interp(L_h)

            #save on datablock
            block.put_grid("luminosity_function" + suffix,'z', z_bins, 'lum',L_h, 'lf_l', Lf_func_h)


        #########################

        # The following applies for a sigle computation of the Luminosity Function (at the z_median of the sample)

        # Compute luminosity function: note that since the luminosity function is computed for a single value of z and
        # might have a different range in magnitudes with respect to the case of the hod section (independent
        # options), we decided to re-compute phi rather than interpolating on the previous one.
        # At the moment, we assume the LF to be computed on the largest possible range of absolute magnitudes.

        if lf_mode == "lf_zmed":
            #interpolate the hmf at the redshift where the luminosity function is evaluated
            f_mass_z_dn = interp2d(mass_dn, z_bins, dndlnM)
            dn_dlnM_zmedian = f_mass_z_dn(mass_dn, z_picked)

            f_mass_z_dn = interp2d(mass_dn, z_bins, dndlnM)
            dn_dlnM_zmedian = f_mass_z_dn(mass_dn, z_picked)

            log_lum_min = np.log10(l_z_simps.min())
            log_lum_max = np.log10(l_z_simps.max())

            L_obs = np.logspace(log_lum_min, log_lum_max, nl)

            phi_c_lf = np.empty([nl, nmass])
            phi_s_lf = np.empty([nl, nmass])
            phi_lf = np.empty([nl, nmass])

            for j in range(0, nl):
                phi_c_lf[j] = lf.clf_cen(L_obs[j], mass, hod)
                phi_s_lf[j] = lf.clf_sat(L_obs[j], mass, hod)
                phi_lf[j] = phi_c_lf[j]+phi_s_lf[j]

            Lf_func = np.empty(nl)
            for i in range(0,nl):
                Lf_func[i] = lf.LF(mass, phi_lf[i], dn_dlnM_zmedian)

            #If required, convert everything to the h from the cosmological parameter section, otherwise keep h=1
            h=1.
            #if rescale_to_h == True:
            #	h = block["cosmological_parameters", "h0"]
            #	print h

            # go back to the observed magnitudes
            L_h = L_obs/(h**2.) #note that the _h subscript avoids mixing h conventions while computing the clf_quantities
            Lf_func_h = Lf_func*(h**5.)

            mr_obs = lf.convert_to_magnitudes(L_obs, abs_mag_sun)

            #save on datablock
            block.put_double_array_1d("luminosity_function" + suffix,'lum_med',L_h)
            block.put_double_array_1d("luminosity_function" + suffix,'lf_l_med',Lf_func_h)

            #Back to magnitudes
            Lf_in_mags = 0.4*np.log(10.)*L_h*Lf_func_h

            block.put_double_array_1d("luminosity_function" + suffix,'mr_med', mr_obs)
            block.put_double_array_1d("luminosity_function" + suffix,'lf_mr_med',Lf_in_mags)

            #Characteristic luminosity of central galaxies
            Lc_cen = lf.L_c(mass, hod)

            block.put_double_array_1d("luminosity_function" + suffix,'mass_med',mass)
            block.put_double_array_1d("luminosity_function" + suffix,'Lc_med',Lc_cen)


    ######################################   CLF DERIVED QUANTITIES   #########################################
    '''
    #For testing purposes it is useful to save some hod-derived quantities
    if clf_quantities:
        lf_cen = np.empty(nl)
        lf_sat = np.empty(nl)
        central_fraction_L = np.empty(nl)
        satellite_fraction_L = np.empty(nl)

        phi_star_sat = lf.phi_star(mass, hod)
        for i in range(0,nl):
            lf_sat[i] = lf.LF(mass, phi_s_lf[i], dn_dlnM_zmedian)
            lf_cen[i] = lf.LF(mass, phi_c_lf[i], dn_dlnM_zmedian)
            satellite_fraction_L[i] = lf_sat[i]/Lf_func[i]
            central_fraction_L[i] = lf_cen[i]/Lf_func[i]

        #convert in mags and rescale to the h for the adopted cosmology
        lf_cen_mags = 0.4*np.log(10.)*L_obs*lf_cen*(h**3.)
        lf_sat_mags = 0.4*np.log(10.)*L_obs*lf_sat*(h**3.)

        block.put_double_array_1d("conditional_luminosity_function" + suffix,'phi_star',phi_star_sat)
        block.put_double_array_1d("conditional_luminosity_function" + suffix,'lf_cen_mags',lf_cen_mags)
        block.put_double_array_1d("conditional_luminosity_function" + suffix,'lf_sat_mags',lf_sat_mags)
        block.put_grid("conditional_luminosity_function", "lum" + suffix, L_obs, "mass", mass, "clf_cen", phi_c_lf)
        block.put_grid("conditional_luminosity_function", "lum", L_obs, "mass", mass, "clf_sat", phi_s_lf)
        block.put_double_array_1d("conditional_luminosity_function" + suffix,'central_fraction_L',central_fraction_L)
        block.put_double_array_1d("conditional_luminosity_function" + suffix,'satellite_fraction_L',satellite_fraction_L)
    '''

    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
