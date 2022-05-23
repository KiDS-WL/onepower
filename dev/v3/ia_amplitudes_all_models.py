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
import sys
import re
from os import listdir
import numpy as np
import lf_lib_simps as lf
from scipy.interpolate import interp1d, interp2d
from scipy.integrate import simps
from astropy.io import fits

import time

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

    luminosity_dependence = options[option_section, "luminosity_dependence"]
    if luminosity_dependence not in ["None", "Joachimi2011", "double_powerlaw", "satellite_luminosity_dependence"]:
        #IT (02/03/22): Replaced broken_powerlaw by double_powerlaw in line 64
        raise ValueError('The luminosity dependence can only take one of the following options:\n \
        None\n \
        Joachimi2011\n \
        double_powerlaw\n \
        satellite_luminosity_dependence\n')
        
    galaxy_type_option = options[option_section, "galaxy_type"]
    if galaxy_type_option == "centrals":
        galaxy_type = 0
    elif galaxy_type_option == "satellites":
        galaxy_type = 1
    else:
        raise ValueError("Please, select galaxy type. Options are: 'centrals' or 'satellites'.")
 
    zmin = options[option_section, "zmin"]
    zmax = options[option_section, "zmax"]
    nz = options[option_section, "nz"]
 
    if luminosity_dependence=="None":
        # dummy variables
        print ("No luminosity dependence assumed for the IA signal...")
        nlbins = 100000
        lum = np.ones([nz,nlbins])
        lum_pdf_z = np.ones([nz, nlbins])
    else:
        print ("Preparing the luminosities...")
        z_loglum_file = options[option_section, "z_loglum_file"] 
        galfile = fits.open(z_loglum_file)[1].data
        z_gal = np.array(galfile['z'])
        loglum_gal = np.array(galfile['loglum'])
        print('loglum gals:')
        print(loglum_gal)
    
        z_bins = np.linspace(zmin, zmax, nz)
        dz = 0.5*(z_bins[1]-z_bins[0])
        z_edges = z_bins-dz
        z_edges = np.append(z_edges, z_bins[-1]+dz)
    
        # divide the galaxies in z-bins and compute the pdf per bins
        nlbins=10000
        bincen = np.zeros([nz, nlbins])
        pdf = np.zeros([nz, nlbins])
        for i in range(0,nz):
            mask_z = (z_gal>=z_edges[i]) & (z_gal<z_edges[i+1])
            loglum_bin = loglum_gal[mask_z]
            if loglum_bin.size:
                #print (i)
                lum = 10.**loglum_bin
                pdf_tmp, _lum_bins = np.histogram(lum, bins=nlbins, density=True)
                _dbin = (_lum_bins[-1]-_lum_bins[0])/(1.*nlbins)
                bincen[i] = _lum_bins[0:-1]+0.5*_dbin
                print('check norm: ', np.sum(pdf_tmp*np.diff(_lum_bins)))
                pdf[i] = pdf_tmp 
                print('check mean:', simps(pdf[i]*bincen[i], bincen[i])/np.mean(lum))
                #import matplotlib.pyplot as plt
                #plt.plot(bincen[i], pdf[i])
                #plt.xscale('log')
            else:
                pdf[i] = 0 
        #plt.show()       
        lum = bincen
        lum_pdf_z = pdf
        print ('pdf:')
        print (pdf)
            
    name = options.get_string(option_section, "name", default="").lower()
    if name:
        suffix = "_" + name
    else:
        suffix = ""
    
    return lum, lum_pdf_z, nz, nlbins, galaxy_type, luminosity_dependence, suffix


def execute(block, config):

    lum, lum_pdf_z, nz, nlum, galaxy_type, luminosity_dependence, suffix = config
            
    if galaxy_type==0:
        if luminosity_dependence == 'None':
            gamma_2h = block["intrinsic_alignment_parameters" + suffix, "A"]     
            block.put_double_array_1d("ia_large_scale_alignment" + suffix, "alignment_gi", gamma_2h * np.ones(nz))
        if luminosity_dependence == 'Joachimi2011':        
            l0 = block["intrinsic_alignment_parameters" + suffix, "l_0"]
            beta_l = block["intrinsic_alignment_parameters" + suffix, "beta_l"] 
            gamma_2h = block["intrinsic_alignment_parameters" + suffix, "gamma_2h_amplitude"] 
            print ('gamma_2h = ', gamma_2h)          
            #mean_lscaling = np.empty(nz)
            #mean_lscaling_beta2 = np.empty(nz)
            #for i in range(0,nz):
            #    mean_lscaling[i] = mean_L_L0_to_beta(lum[i], lum_pdf_z[i], l0, beta_l)
            mean_lscaling = mean_L_L0_to_beta(lum, lum_pdf_z, l0, beta_l)
            block.put_double_array_1d("ia_large_scale_alignment" + suffix, "alignment_gi", gamma_2h * mean_lscaling)
        if luminosity_dependence == 'double_powerlaw':        
            l0 = block["intrinsic_alignment_parameters" + suffix, "l_0"]
            beta_l = block["intrinsic_alignment_parameters" + suffix, "beta_l"] 
            beta_low = block["intrinsic_alignment_parameters" + suffix, "beta_low"] 
            gamma_2h = block["intrinsic_alignment_parameters" + suffix, "gamma_2h_amplitude"]           
            mean_lscaling = np.empty(nz)
            for i in range(0,nz):
                mean_lscaling[i] = broken_powerlaw(lum[i], lum_pdf_z[i], gamma_2h, l0, beta_l, beta_low)
            block.put_double_array_1d("ia_large_scale_alignment" + suffix, "alignment_gi", mean_lscaling)
        
    if galaxy_type==1:
        if luminosity_dependence == 'None':
            gamma_1h = block["intrinsic_alignment_parameters" + suffix, "gamma_1h_amplitude"]             
            block.put_double_array_1d("ia_small_scale_alignment" + suffix, "alignment_1h", gamma_1h * np.ones(nz))
        if luminosity_dependence == 'satellite_luminosity_dependence':
            l0 = block["intrinsic_alignment_parameters" + suffix, "l_0"]
            zeta_l = block["intrinsic_alignment_parameters" + suffix, "zeta_l"]  
            gamma_1h = block["intrinsic_alignment_parameters" + suffix, "gamma_1h_amplitude"]             
            #mean_lscaling = np.empty(nz)
            #for i in range(0,nz):
            #    mean_lscaling[i] = mean_L_L0_to_beta(lum[i], lum_pdf_z[i], l0, zeta_l)
            mean_lscaling = mean_L_L0_to_beta(lum, lum_pdf_z, l0, zeta_l)
            block.put_double_array_1d("ia_small_scale_alignment" + suffix, "alignment_1h", gamma_1h * mean_lscaling)

    return 0
    

def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
    
    
