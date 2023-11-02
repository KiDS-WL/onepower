# CosmoSiS module to compute the halo occupation distribution (HOD) for models that are not conditioned on 
# galaxy observables

# The halo occupation distribution predicts the number of galaxies that populate a halo of mass M:
#
# N_gal(M) = N_cen(M) + N_sat(M),
#
# where the galaxies are divided into centrals and satellites. 

from cosmosis.datablock import names, option_section
import numpy as np
import hod_lib as hod
from scipy.interpolate import interp1d, interp2d, RegularGridInterpolator
import pyhalomodel.pyhalomodel as halo

cosmo = names.cosmological_parameters

#--------------------------------------------------------------------------------#	

parameters_zheng = ['Mmin','sigma','M0','M1','alpha']
parameters_zhai  = ['Mmin','sigma','Msat','Mcut','alpha']

def setup(options):

    # TODO: Change the bining of the observable such that nbins can be larger than 1 if inputs are read through a file.
    

    # Read in input and output section names
    hod_section_name = options.get_string(option_section, 'hod_section_name','hod').lower()
    values_name      = options.get_string(option_section, 'values_name','hod_parameters').lower()
    hod_type         = options.get_string(option_section, 'hod_name','zheng').lower()
    

    # galaxy_bias_section_name = options.get_string(option_section, 'galaxy_bias_section_name','galaxy_bias').lower()
    # galaxy_bias_option = options[option_section, 'do_galaxy_linear_bias']

    # zmin    = np.asarray([options[option_section, 'zmin']]).flatten()
    # zmax    = np.asarray([options[option_section, 'zmax']]).flatten()
    # nz      = options[option_section, 'nz']
    
    # Check if the legth of obs_min, obs_max, zmin and zmax match.
    # if not np.all(np.array([len(zmin), len(zmax)]) == len(obs_min)):
    #     raise Exception('Error:  zmin and zmax need to be of same length.')
    
    # Arrays using starting and end values of zmin and zmax 
    # z_bins = np.array([np.linspace(zmin_i, zmax_i, nz) for zmin_i, zmax_i in zip(zmin, zmax)])

    # Minimum and maximum halo masses in log10 space
    # TODO: what are the units? 
    log_mass_min = options[option_section, 'log_mass_min']
    log_mass_max = options[option_section, 'log_mass_max']
    # number of halo mass bins
    nmass        = options[option_section, 'nmass']

    #---- log-spaced mass sample ----#
    dlog10m = (log_mass_max-log_mass_min)/nmass
    mass    = 10.0 ** np.arange(log_mass_min, log_mass_max, dlog10m)

    return hod_section_name,values_name,hod_type,mass




def execute(block, config):

    hod_section_name,values_name,hod_type,mass= config

    #---- loading hod value from the values.ini file ----#
    # check the parameter names in the hod section
    # parameters_in_hod_section = block.keys(section=values_name)
    # # read in the parameters
    HOD = {}
    # for param in parameters_in_hod_section:
    #     print(param[1])
    #     HOD[param[1]] = block[values_name,param[1]]

    if hod_type == 'zheng':
        name =  'Zheng et al. (2005)'
        for param in parameters_zheng:
            if block.has_value(values_name,param.lower()):
                HOD[param] = block[values_name,param.lower()]
            else:
                raise Exception('Error:  parameter '+param+' is needed for the Zheng HOD model')
    elif hod_type == 'zhai':
        name =  'Zhai et al. (2017)'
        for param in parameters_zhai:
            if block.has_value(values_name,param.lower()):
                HOD[param] = block[values_name,param.lower()]
            else:
                raise Exception('Error:  parameter '+param+' is needed for the Zhai HOD model')
    else:
        raise Exception('Error:  not a supported HOD type. The value was: '+hod_type)

    # print(HOD)
    # print(hod_type)
    n_cen, n_sat = halo.HOD_mean(mass, method=name, **HOD)
    n_tot = n_cen + n_sat
    # print(Nc)
    # print(Ns)

# 
    block.put_int(hod_section_name, 'nbins', nbins)

    #---- loading the halo mass function ----#
    dndlnM_grid = block['hmf','dndlnmh']
    mass_dn     = block['hmf','m_h']
    z_dn        = block['hmf','z']
    

    # for nb in range(0,nbins):
    #     if nbins != 1:
    #         suffix = str(nb+1)
    #     else:
    #         suffix = ''
    # TODO: Check if n_sat and n_cen are the same as Nc and Ns
    block.put_grid(hod_section_name, 'z', z_bins[nb], 'mass', mass, 'n_sat', n_sat)
    block.put_grid(hod_section_name, 'z', z_bins[nb], 'mass', mass, 'n_cen', n_cen)
    block.put_grid(hod_section_name, 'z', z_bins[nb], 'mass', mass, 'n_tot', n_tot)
    # block.put_grid(hod_section_name, 'z', z_bins[nb], 'mass', mass, 'f_star', f_star)

    # What is this then?
    numdens_cen = hod.compute_number_density(mass, n_cen, dndlnM)
    numdens_sat = hod.compute_number_density(mass, n_sat, dndlnM)

    numdens_tot = numdens_cen + numdens_sat
    fraction_cen = numdens_cen/numdens_tot
    fraction_sat = numdens_sat/numdens_tot
    # compute average halo mass per bin
    mass_avg = hod.compute_avg_halo_mass(mass, n_cen, dndlnM)/numdens_cen

    block.put_double_array_1d(hod_section_name, 'number_density_cen'+suffix, numdens_cen)
    block.put_double_array_1d(hod_section_name, 'number_density_sat'+suffix, numdens_sat)
    block.put_double_array_1d(hod_section_name, 'number_density_tot'+suffix, numdens_tot)
    block.put_double_array_1d(hod_section_name, 'central_fraction'+suffix, fraction_cen)
    block.put_double_array_1d(hod_section_name, 'satellite_fraction'+suffix, fraction_sat)
    block.put_double_array_1d(hod_section_name, 'average_halo_mass'+suffix, mass_avg)
        
        # if galaxy_bias_option:
        #     #---- loading the halo bias function ----#
        #     # TODO: Check where these come from
        #     mass_hbf     = block['halobias', 'm_h']
        #     z_hbf        = block['halobias', 'z']
        #     halobias_hbf = block['halobias', 'b_hb']

        #     f_interp_halobias = RegularGridInterpolator((mass_hbf.T, z_hbf.T), halobias_hbf.T, bounds_error=False, fill_value=None)
        #     mass_i, z_bins_i  = np.meshgrid(mass, z_bins[nb], sparse=True)
        #     hbias = f_interp_halobias((mass_i.T,z_bins_i.T)).T

        #     galaxybias_cen = hod.compute_galaxy_linear_bias(mass[np.newaxis,:], n_cen, hbias, dndlnM)/numdens_tot
        #     galaxybias_sat = hod.compute_galaxy_linear_bias(mass[np.newaxis,:], n_sat, hbias, dndlnM)/numdens_tot
        #     galaxybias_tot = hod.compute_galaxy_linear_bias(mass[np.newaxis,:], n_tot, hbias, dndlnM)/numdens_tot
            
            # TODO:Put these into a different section
            # block.put_double_array_1d('galaxy_bias' + suffix, 'galaxy_bias_centrals', galaxybias_cen)
            # block.put_double_array_1d('galaxy_bias' + suffix, 'galaxy_bias_satellites', galaxybias_sat)
            # # this can be useful in case you want to use the constant bias module to compute p_gg
            # block.put_double_array_1d('galaxy_bias' + suffix, 'b', galaxybias_tot)
    
    # # TODO: What is this one for? There is already f_start for the bins.
    # # The TO-DO in this lines needs explanation: f_star here is different then f_star calculate for each bin, 
    # # thus it needs to be saved differently. This one is used for the stellar mass contribution in Pmm, 
    # # so for baryonic feedback in cosmic shear and thus needs to be calculated for a wide range of halo masses, 
    # # unlike the other f_star which are for each stellar mass bin. I would keep the metadata block here 
    # # to save all the parameters not directly connected with "per bin" HODs and corresponding products.
    # f_star = np.array([hod.compute_stellar_fraction(obs_range_h_i, phi_z_i)/mass for obs_range_h_i, phi_z_i in zip(obs_range_h, phi)])
    # block.put_grid(hod_section_name, 'z_extended', z_bins_one, 'mass_extended', mass, 'f_star_extended', f_star)
    
    return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
