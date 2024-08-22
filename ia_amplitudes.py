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


from cosmosis.datablock import option_section
import numpy as np
from scipy.integrate import simps
from astropy.io import fits


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

    central_IA_depends_on = options[option_section, 'central_IA_depends_on']
    if central_IA_depends_on not in ['constant', 'luminosity', 'halo_mass']:
        raise ValueError('Choose one of the following options for the dependence of the central IA model:\n \
        constant\n \
        luminosity\n \
        halo_mass\n')
    
    satellite_IA_depends_on = options[option_section, 'satellite_IA_depends_on']
    if satellite_IA_depends_on not in ['constant', 'luminosity', 'halo_mass']:
        raise ValueError('Choose one of the following options for the dependence of the satellite IA model:\n \
        constant\n \
        luminosity\n \
        halo_mass\n')
        
    zmin = options[option_section, 'zmin']
    zmax = options[option_section, 'zmax']
    nz = options[option_section, 'nz']
 
    if central_IA_depends_on == 'luminosity':
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
                print(f'check norm: {np.sum(pdf_tmp*np.diff(_lum_bins))}')
                pdf_centrals[i] = pdf_tmp
                print(f'check mean: {simps(pdf_centrals[i]*bincen_centrals[i], bincen_centrals[i])/np.mean(lum)}')
            else:
                pdf_centrals[i] = 0
                
        lum_centrals = bincen_centrals
        lum_pdf_z_centrals = pdf_centrals
        # print ('pdf:')
        # print (pdf_centrals)
    else:
        # include dummy variables
        nlbins = 100000
        lum_centrals = np.ones([nz,nlbins])
        lum_pdf_z_centrals = np.ones([nz, nlbins])
        
    if satellite_IA_depends_on == 'luminosity':
        print ('Preparing the luminosities...')
        z_loglum_file_satellites = options[option_section, 'z_loglum_file_satellites']
        galfile_satellites = fits.open(z_loglum_file_satellites)[1].data
        z_gal_satellites = np.array(galfile_satellites['z'])
        loglum_gal_satellites = np.array(galfile_satellites['loglum'])
        # print('loglum gals:')
        # print(loglum_gal_satellites)
    
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
                # print('check norm: ', np.sum(pdf_tmp*np.diff(_lum_bins)))
                pdf_satellites[i] = pdf_tmp
                # print('check mean:', simps(pdf_satellites[i]*bincen_satellites[i], bincen_satellites[i])/np.mean(lum))
            else:
                pdf_satellites[i] = 0
        
        lum_satellites = bincen_satellites
        lum_pdf_z_satellites = pdf_satellites
        # print ('pdf:')
        # print (pdf_satellites)
    else:
        # dummy variables
        nlbins = 100000
        lum_satellites = np.ones([nz,nlbins])
        lum_pdf_z_satellites = np.ones([nz, nlbins])
            
    name = options.get_string(option_section, 'output_suffix', default='').lower()
    if name != '':
        suffix = f'_{name}'
    else:
        suffix = ''
    
    return lum_centrals, lum_pdf_z_centrals, lum_satellites, lum_pdf_z_satellites, nz, nlbins, central_IA_depends_on, satellite_IA_depends_on, suffix


def execute(block, config):

    lum_centrals, lum_pdf_z_centrals, lum_satellites, lum_pdf_z_satellites, nz, nlum, central_IA_depends_on, satellite_IA_depends_on, suffix = config
    # TODO: Check if all the options work as intended!
    # TODO: I think we should re-write this bit so the type of dependence doens't have to be defined
    # Include default values if parameters are not defined in the value file instead.

    # First the Centrals
    # All options require the central 2-halo amplitude to be defined 
    gamma_2h = block[f'intrinsic_alignment_parameters{suffix}', 'gamma_2h_amplitude']

    if central_IA_depends_on == 'constant':
        block.put_double_array_1d(f'ia_large_scale_alignment{suffix}', 'alignment_gi', gamma_2h * np.ones(nz))
        
    if central_IA_depends_on == 'luminosity':
        # Check that the user knows what they're doing:
        if not block.has_value(f'intrinsic_alignment_parameters{suffix}', 'L_pivot'):
            raise ValueError('You have chosen central luminosity scaling without providing a pivot luminosity parameter.  Include L_pivot. "\n ')
        
        # single-power law parameters
        lpiv = block[f'intrinsic_alignment_parameters{suffix}', 'L_pivot']
        beta = block[f'intrinsic_alignment_parameters{suffix}', 'beta']

        # Then lets see whether user wants to implement a double power law based on what they've included in values.ini
        if block.has_value(f'intrinsic_alignment_parameters{suffix}', 'beta_two'):
            print('You have chosen to implement a double power law model for the luminosity dependence of the centrals')
            beta_two = block[f'intrinsic_alignment_parameters{suffix}', 'beta_two']
            mean_lscaling = np.empty(nz)
            for i in range(0,nz):
                mean_lscaling[i] = broken_powerlaw(lum_centrals[i], lum_pdf_z_centrals[i], gamma_2h, lpiv, beta, beta_two)
            block.put_double_array_1d(f'ia_large_scale_alignment{suffix}', 'alignment_gi', mean_lscaling)
        else:
            print('You have chosen to implement a single power law model for the luminosity dependence of the centrals')
            mean_lscaling = mean_L_L0_to_beta(lum_centrals, lum_pdf_z_centrals, lpiv, beta)
            block.put_double_array_1d(f'ia_large_scale_alignment{suffix}', 'alignment_gi', gamma_2h * mean_lscaling)
        
    if central_IA_depends_on  == 'halo_mass':

        # Check that the user knows what they're doing:
        if not block.has_value(f'intrinsic_alignment_parameters{suffix}', 'M_pivot'):
            raise ValueError('You have chosen central halo-mass scaling without providing a pivot mass parameter.  Include M_pivot. "\n ')
        
        mpiv = block.get_double(f'intrinsic_alignment_parameters{suffix}', 'M_pivot')
        beta = block.get_double(f'intrinsic_alignment_parameters{suffix}', 'beta')
        if block.has_value(f'intrinsic_alignment_parameters{suffix}', 'beta_two'):
            raise ValueError('A double power law model for the halo mass dependence of centrals has not been implemented.  Either remove beta_two from your parameter list or select central_IA_depends_on="luminosity"\n ')
        # Technically just repacking the variables, but this is the easiest way to accomodate backwards compatibility and clean pk_lib.py module
        block.put_double_array_1d(f'ia_large_scale_alignment{suffix}', 'alignment_gi', gamma_2h * np.ones(nz))
        block.put_double(f'ia_large_scale_alignment{suffix}', 'M_pivot', mpiv)
        block.put_double(f'ia_large_scale_alignment{suffix}', 'beta', beta)
    
    #Add instance information to block
    block.put_string(f'ia_large_scale_alignment{suffix}', 'instance', central_IA_depends_on )
        
    # Second the Satellites
    # All options require the satellite 1-halo amplitude to be defined 
    gamma_1h = block[f'intrinsic_alignment_parameters{suffix}', 'gamma_1h_amplitude']
    
    if satellite_IA_depends_on == 'constant':
        block.put_double_array_1d(f'ia_small_scale_alignment{suffix}', 'alignment_1h', gamma_1h * np.ones(nz))
        
    if satellite_IA_depends_on == 'luminosity':
        lpiv = block[f'intrinsic_alignment_parameters{suffix}', 'L_pivot']
        beta_sat = block[f'intrinsic_alignment_parameters{suffix}', 'beta_sat']
        mean_lscaling = mean_L_L0_to_beta(lum_satellites, lum_pdf_z_satellites, lpiv, beta_sat)
        block.put_double_array_1d(f'ia_small_scale_alignment{suffix}', 'alignment_1h', gamma_1h * mean_lscaling)
        
    if satellite_IA_depends_on == 'halo_mass':
        mpiv = block.get_double(f'intrinsic_alignment_parameters{suffix}', 'M_pivot')
        beta_sat = block.get_double(f'intrinsic_alignment_parameters{suffix}', 'beta_sat')
        # Technically just repacking the variables, but this is the easiest way to accomodate backwards compatibility and clean pk_lib.py module
        block.put_double_array_1d(f'ia_small_scale_alignment{suffix}', 'alignment_1h', gamma_1h * np.ones(nz))
        block.put_double(f'ia_small_scale_alignment{suffix}', 'M_pivot', mpiv)
        block.put_double(f'ia_small_scale_alignment{suffix}', 'beta_sat', beta_sat)

    #Add instance information to block
    block.put_string(f'ia_small_scale_alignment{suffix}', 'instance', satellite_IA_depends_on)

    return 0
    

def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
    
    
