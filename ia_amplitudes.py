# This module computes the average luminosity scaling of the intrinsic alignment 
# following the formalism presented in Joachimi et al. 2011b
# 
#                            A(L) = (L/L_0)^beta
#
# If the luminosity distribution of the sample is not narrow, we need to average
# the contribution from L^beta of the entire sample, i.e.
#
#            <(L/L_0)^beta> = (1/L_0^beta) \int L^beta p(L) dL
#
# where p(L) is the pdf of L.
# We also allow for a double power law, i.e. 
#                            A(L<L0) = (L/L_0)^beta_low
#                            A(L>L0) = (L/L_0)^beta



# -------------------------------------------------------------------------------- #
# IMPORTANT: here we assume luminosities to be in units of L_sun/h2
# -------------------------------------------------------------------------------- #


from cosmosis.datablock import names, option_section
import numpy as np
from scipy.integrate import simps
from astropy.io import fits

cosmo = names.cosmological_parameters


# ===================== square outside the average ========================== #
def mean_L_L0_to_beta(xlum, pdf, l0, beta):
    l_beta_mean = simps(pdf*(xlum/l0)**beta, xlum)
    #l_beta_mean2 = l_beta_mean**2.
    return l_beta_mean
    
def broken_powerlaw(xlum, pdf, gamma_2h_lum, l0, beta, beta_low):
    alignment_ampl = np.zeros(len(xlum))
    # mask the high lum galaxies   
    high_lum_gals = xlum>l0
    # low lum gals are assumed to follow a different power law (xlum[~high_lum_gals]/l0)**beta_low
    alignment_ampl[~high_lum_gals] = gamma_2h_lum * (xlum[~high_lum_gals]/l0)**beta_low
    # high lum galaxies are assumed to follow the power law relation in Joachimi et a. 2011 (also found by Singh et al. 2015)
    alignment_ampl[high_lum_gals] = gamma_2h_lum * (xlum[high_lum_gals]/l0)**beta
    # integrate over the luminosity pdf at that given redshift bin    
    l_beta_mean = simps(pdf*alignment_ampl, xlum)
    return l_beta_mean
# =========================================================================== #
    

def setup(options):
    #This function is called once per processor per chain.
    #It is a chance to read any fixed options from the configuration file,
    #load any data, or do any calculations that are fixed once.

    centrals_luminosity_dependence = options[option_section, 'centrals_luminosity_dependence']
    if centrals_luminosity_dependence not in ['None', 'Joachimi2011', 'double_powerlaw', 'halo_mass']:
        raise ValueError('The luminosity/halo mass dependence can only take one of the following options:\n \
        constant\n \
        Joachimi2011\n \
        double_powerlaw\n \
        satellite_luminosity_dependence\n \
        halo_mass\n')
        
    satellites_luminosity_dependence = options[option_section, 'satellites_luminosity_dependence']
    if satellites_luminosity_dependence not in ['None', 'satellite_luminosity_dependence', 'halo_mass']:
        raise ValueError('The satellites luminosity/halo mass dependence can only take one of the following options:\n \
        constant\n \
        satellite_luminosity_dependence\n \
        halo_mass\n')
        
    zmin = options[option_section, 'zmin']
    zmax = options[option_section, 'zmax']
    nz = options[option_section, 'nz']
 
    if centrals_luminosity_dependence == 'constant':
        # dummy variables
        print ('No luminosity dependence assumed for the centrals IA signal...')
        nlbins = 100000
        lum_centrals = np.ones([nz,nlbins])
        lum_pdf_z_centrals = np.ones([nz, nlbins])
    elif centrals_luminosity_dependence == 'halo_mass':
        print ('Halo mass dependence assumed for the centrals IA signal...')
        nlbins = 100000
        lum_centrals = np.ones([nz,nlbins])
        lum_pdf_z_centrals = np.ones([nz, nlbins])
    else:
        print ('Preparing the luminosities...')
        z_loglum_file_centrals = options[option_section, 'z_loglum_file_centrals']
        galfile_centrals = fits.open(z_loglum_file_centrals)[1].data
        z_gal_centrals = np.array(galfile_centrals['z'])
        loglum_gal_centrals = np.array(galfile_centrals['loglum'])
        print('loglum gals:')
        print(loglum_gal_centrals)
    
        z_bins = np.linspace(zmin, zmax, nz)
        dz = 0.5*(z_bins[1]-z_bins[0])
        z_edges = z_bins-dz
        z_edges = np.append(z_edges, z_bins[-1]+dz)
    
        # divide the galaxies in z-bins and compute the pdf per bins
        nlbins=10000
        bincen_centrals = np.zeros([nz, nlbins])
        pdf_centrals = np.zeros([nz, nlbins])
        for i in range(0,nz):
            mask_z = (z_gal_centrals>=z_edges[i]) & (z_gal_centrals<z_edges[i+1])
            loglum_bin_centrals = loglum_gal_centrals[mask_z]
            if loglum_bin_centrals.size:
                lum = 10.**loglum_bin_centrals
                pdf_tmp, _lum_bins = np.histogram(lum, bins=nlbins, density=True)
                _dbin = (_lum_bins[-1]-_lum_bins[0])/(1.*nlbins)
                bincen_centrals[i] = _lum_bins[0:-1]+0.5*_dbin
                print('check norm: ', np.sum(pdf_tmp*np.diff(_lum_bins)))
                pdf_centrals[i] = pdf_tmp
                print('check mean:', simps(pdf_centrals[i]*bincen_centrals[i], bincen_centrals[i])/np.mean(lum))
            else:
                pdf_centrals[i] = 0
                
        lum_centrals = bincen_centrals
        lum_pdf_z_centrals = pdf_centrals
        print ('pdf:')
        print (pdf_centrals)
        
    if satellites_luminosity_dependence == 'constant':
        # dummy variables
        print ('No luminosity dependence assumed for the satellites IA signal...')
        nlbins = 100000
        lum_satellites = np.ones([nz,nlbins])
        lum_pdf_z_satellites = np.ones([nz, nlbins])
    elif satellites_luminosity_dependence == 'halo_mass':
        print ('Halo mass dependence assumed for the satellites IA signal...')
        nlbins = 100000
        lum_satellites = np.ones([nz,nlbins])
        lum_pdf_z_satellites = np.ones([nz, nlbins])
    else:
        print ('Preparing the luminosities...')
        z_loglum_file_satellites = options[option_section, 'z_loglum_file_satellites']
        galfile_satellites = fits.open(z_loglum_file_satellites)[1].data
        z_gal_satellites = np.array(galfile_satellites['z'])
        loglum_gal_satellites = np.array(galfile_satellites['loglum'])
        print('loglum gals:')
        print(loglum_gal_satellites)
    
        z_bins = np.linspace(zmin, zmax, nz)
        dz = 0.5*(z_bins[1]-z_bins[0])
        z_edges = z_bins-dz
        z_edges = np.append(z_edges, z_bins[-1]+dz)
    
        # divide the galaxies in z-bins and compute the pdf per bins
        nlbins=10000
        bincen_satellites = np.zeros([nz, nlbins])
        pdf_satellites = np.zeros([nz, nlbins])
        for i in range(0,nz):
            mask_z = (z_gal_satellites>=z_edges[i]) & (z_gal_satellites<z_edges[i+1])
            loglum_bin_satellites = loglum_gal_satellites[mask_z]
            if loglum_bin_satellites.size:
                lum = 10.**loglum_bin_satellites
                pdf_tmp, _lum_bins = np.histogram(lum, bins=nlbins, density=True)
                _dbin = (_lum_bins[-1]-_lum_bins[0])/(1.*nlbins)
                bincen_satellites[i] = _lum_bins[0:-1]+0.5*_dbin
                print('check norm: ', np.sum(pdf_tmp*np.diff(_lum_bins)))
                pdf_satellites[i] = pdf_tmp
                print('check mean:', simps(pdf_satellites[i]*bincen_satellites[i], bincen_satellites[i])/np.mean(lum))
            else:
                pdf_satellites[i] = 0
        
        lum_satellites = bincen_satellites
        lum_pdf_z_satellites = pdf_satellites
        print ('pdf:')
        print (pdf_satellites)
            
    name = options.get_string(option_section, 'output_suffix', default='').lower()
    if name != '':
        suffix = '_' + name
    else:
        suffix = ''
    
    return lum_centrals, lum_pdf_z_centrals, lum_satellites, lum_pdf_z_satellites, nz, nlbins, centrals_luminosity_dependence, satellites_luminosity_dependence, suffix


def execute(block, config):

    lum_centrals, lum_pdf_z_centrals, lum_satellites, lum_pdf_z_satellites, nz, nlum, centrals_luminosity_dependence, satellites_luminosity_dependence, suffix = config
    # TODO: Check if all the options work as intended!
    
    if centrals_luminosity_dependence == 'constant':
        gamma_2h = block['intrinsic_alignment_parameters' + suffix, 'A']
        block.put_double_array_1d('ia_large_scale_alignment' + suffix, 'alignment_gi', gamma_2h * np.ones(nz))
        
    if centrals_luminosity_dependence == 'joachimi2011':
        l0 = block['intrinsic_alignment_parameters' + suffix, 'l_0']
        beta_l = block['intrinsic_alignment_parameters' + suffix, 'beta_l']
        gamma_2h = block['intrinsic_alignment_parameters' + suffix, 'gamma_2h_amplitude']
        print ('amplitude_IA = ', gamma_2h)
        mean_lscaling = mean_L_L0_to_beta(lum_centrals, lum_pdf_z_centrals, l0, beta_l)
        block.put_double_array_1d('ia_large_scale_alignment' + suffix, 'alignment_gi', gamma_2h * mean_lscaling)
        
    if centrals_luminosity_dependence == 'double_powerlaw':
        l0 = block['intrinsic_alignment_parameters' + suffix, 'l_0']
        beta_l = block['intrinsic_alignment_parameters' + suffix, 'beta_l']
        beta_low = block['intrinsic_alignment_parameters' + suffix, 'beta_low']
        gamma_2h = block['intrinsic_alignment_parameters' + suffix, 'gamma_2h_amplitude']
        mean_lscaling = np.empty(nz)
        for i in range(0,nz):
            mean_lscaling[i] = broken_powerlaw(lum_centrals[i], lum_pdf_z_centrals[i], gamma_2h, l0, beta_l, beta_low)
        block.put_double_array_1d('ia_large_scale_alignment' + suffix, 'alignment_gi', mean_lscaling)
        
    if centrals_luminosity_dependence == 'halo_mass':
        m0 = block.get_double('intrinsic_alignment_parameters' + suffix, 'M_0')
        beta_m = block.get_double('intrinsic_alignment_parameters' + suffix, 'beta_mh')
        gamma_2h = block.get_double('intrinsic_alignment_parameters' + suffix, 'gamma_2h_amplitude')
        # Technically just repacking the variables, but this is the easiest way to accomodate backwards compatibility and clean pk_lib.py module
        block.put_double_array_1d('ia_large_scale_alignment' + suffix, 'alignment_gi', gamma_2h * np.ones(nz))
        block.put_double('ia_large_scale_alignment' + suffix, 'M_0', m0)
        block.put_double('ia_large_scale_alignment' + suffix, 'beta_mh', beta_m)
        block.put_string('ia_large_scale_alignment' + suffix, 'instance', centrals_luminosity_dependence)
        
    
    
    
    if satellites_luminosity_dependence == 'constant':
        gamma_1h = block['intrinsic_alignment_parameters' + suffix, 'gamma_1h_amplitude']
        block.put_double_array_1d('ia_small_scale_alignment' + suffix, 'alignment_1h', gamma_1h * np.ones(nz))
        
    if satellites_luminosity_dependence == 'satellite_luminosity_dependence':
        l0 = block['intrinsic_alignment_parameters' + suffix, 'l_0']
        zeta_l = block['intrinsic_alignment_parameters' + suffix, 'zeta_l']
        gamma_1h = block['intrinsic_alignment_parameters' + suffix, 'gamma_1h_amplitude']
        mean_lscaling = mean_L_L0_to_beta(lum_satellites, lum_pdf_z_satellites, l0, zeta_l)
        block.put_double_array_1d('ia_small_scale_alignment' + suffix, 'alignment_1h', gamma_1h * mean_lscaling)
        
    if satellites_luminosity_dependence == 'halo_mass':
        m0 = block.get_double('intrinsic_alignment_parameters' + suffix, 'M_0')
        zeta_m = block.get_double('intrinsic_alignment_parameters' + suffix, 'zeta_mh')
        gamma_1h = block.get_double('intrinsic_alignment_parameters' + suffix, 'gamma_1h_amplitude')
        # Technically just repacking the variables, but this is the easiest way to accomodate backwards compatibility and clean pk_lib.py module
        block.put_double_array_1d('ia_small_scale_alignment' + suffix, 'alignment_1h', gamma_1h * np.ones(nz))
        block.put_double('ia_small_scale_alignment' + suffix, 'M_0', m0)
        block.put_double('ia_small_scale_alignment' + suffix, 'zeta_mh', zeta_m)
        block.put_string('ia_small_scale_alignment' + suffix, 'instance', satellites_luminosity_dependence)


    return 0
    

def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
    
    
