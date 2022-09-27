from cosmosis.datablock import names, option_section
import sys
import numpy as np
from os import listdir
import re

from cosmosis.datablock import names, option_section
import lf_lib_simps as lf
from scipy.interpolate import interp1d, interp2d, RegularGridInterpolator
from itertools import count

import time



# We have a collection of commonly used pre-defined block section names.
# If none of the names here is relevant for your calculation you can use any
# string you want instead.
cosmo = names.cosmological_parameters


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [ atoi(c) for c in re.split('(\d+)', text) ]
    
def load_data(file_name): 
	logmass_bins = np.loadtxt(file_name+'/logmass.txt')
	mass_bins = 10.**logmass_bins
	#mass_bins = np.loadtxt(file_name+'/mass.txt')
	ncen = np.loadtxt(file_name+'/ncen.txt')
	nsat = np.loadtxt(file_name+'/nsat.txt')
	z_bins = np.loadtxt(file_name+'/z.txt')
	nz_file = len(z_bins)
	numdens_cen = np.ones(nz_file)
	numdens_sat = np.ones(nz_file)
	numdens_tot = np.ones(nz_file)


	print('ncen.shape = ', ncen.shape)

	print('nz = ', nz_file)

	return mass_bins, z_bins, ncen, nsat, numdens_tot, numdens_cen, numdens_sat #, f_red #, f_red_cen #, f_sat, f_cen, numdenscenred, numdenssatred
	
#--------------------------------------------------------------------------------#	

def setup(options):
    #This function is called once per processor per chain.
    #It is a chance to read any fixed options from the configuration file,
    #load any data, or do any calculations that are fixed once.
		
	log_mass_min = options[option_section, 'log_mass_min']
	log_mass_max = options[option_section, 'log_mass_max']
	nmass = options[option_section, 'nmass']
	#log-spaced mass in units of M_sun/h
    dlog10m = (log_mass_max-log_mass_min)/nmass
    mass = 10.0 ** np.arange(log_mass_min, log_mass_max, dlog10m)
	
	zmin = options[option_section, 'zmin']
	zmax = options[option_section, 'zmax']
	nz = options[option_section, 'nz']
	z_vec = np.linspace(zmin, zmax, nz)	

	number_density_option = options[option_section, 'number_density']
	galaxy_bias_option = options[option_section, 'galaxy_linear_bias']
		
	filename = options[option_section, 'folder_path']
	mass_bins, z_bins, ncen_mice, nsat_mice, numdens_tot, numdens_cen, numdens_sat = load_data(filename) #, red_fraction_option)
		
	name = options.get_string(option_section, 'name', default='').lower()
	if name:
		suffix = '_' + name
	else:
		suffix = ''
									
	return mass, z_vec, nz, mass_bins, z_bins, ncen_mice, nsat_mice, numdens_tot, numdens_cen, numdens_sat, number_density_option, galaxy_bias_option, suffix
	

def execute(block, config):
    #This function is called every time you have a new sample of cosmological and other parameters.
    #It is the main workhorse of the code. The block contains the parameters and results of any 
    #earlier modules, and the config is what we loaded earlier.
	
	mass, z_vec, nz, mass_bins, z_bins, ncen_mice, nsat_mice, numdens_tot_mice, numdens_cen_mice, numdens_sat_mice, number_density_option, galaxy_bias_option, suffix = config

	# interpolate to the same mass and redshifts used in the rest of the pipeline
	#print ('mass.shape = ', mass.shape)
	#print ('z_bins.shape = ', z_bins.shape)
	#print ('ncen_mice.shape = ', ncen_mice.shape)
	find_ncen = interp2d(mass_bins, z_bins, ncen_mice, kind='linear', bounds_error=False)
	#print('ok')
	n_cen = find_ncen(mass, z_vec)
	#print('ok')
	#print ('n_cen.shape = ', n_cen.shape)
	find_nsat = interp2d(mass_bins, z_bins, nsat_mice, kind='linear', bounds_error=False)
	n_sat = find_nsat(mass, z_vec)		
	n_tot = n_cen + n_sat	
	'''
	import matplotlib.pyplot as plt
	for i in range(0,nz):
		plt.plot(mass, n_cen[i], label='z=%.2lf' %z_vec[i])
	for iz in range(0,len(z_bins)):	
		plt.plot(mass_bins, ncen_mice[iz], 'ro', ms=1.5, ls='')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.show()
	for i in range(0,nz):
		plt.plot(mass, n_sat[i], label='z=%.2lf' %z_vec[i])
	for iz in range(0,len(z_bins)):	
		plt.plot(mass_bins, nsat_mice[iz], 'bo', ms=1.5, ls='')
	plt.xscale('log')
	plt.yscale('log')
	plt.legend()
	plt.show()
	'''
	print('mass_min = ', mass.min())


	if (number_density_option == True) or (galaxy_bias_option == True):	
		
		# The halo mass function of a flux limited survey is complete only at high masses, where the redshift dependent selection
		# of the fain galaxies does not imposes any cut. For this reason, we do re-normalise the HODs to match the 'observed' number densities, 
		# allowing us to recover the correct galaxy fractions (note that this is true by construction, can't be use as a test).
		# IMPORTANT: this procedure does not touch the shape of the HMF (so does not say anything about completeness), but only the
		# amplitude of the HODs. -> we might still over-predict things at low mass. 
		
		#---- loading the halo mass function ----#
		
		dndlnM_grid = block['hmf','dndlnmh']
		mass_dn = block['hmf','m_h']
		z_dn = block['hmf','z']
		f_int_dndlnM = interp2d(mass_dn, z_dn, dndlnM_grid)
		dndlnM = f_int_dndlnM(mass, z_vec)
		
		numdens_cen = np.empty(nz)
		numdens_sat = np.empty(nz)

		for jz in range(0,nz):
			numdens_cen[jz] = lf.compute_number_density(mass, n_cen[jz], dndlnM[jz]) #this is already normalised
			numdens_sat[jz] = lf.compute_number_density(mass, n_sat[jz], dndlnM[jz]) #this is already normalised
		numdens_tot = numdens_cen + numdens_sat
			
		fraction_cen = numdens_cen/numdens_tot
		fraction_sat = numdens_sat/numdens_tot
		
		#print('f_cen+f_sat = ', fraction_cen + fraction_sat)
		
		if number_density_option == True:
			# save on datablock
			block.put_grid('hod' + suffix, 'z', z_vec, 'mass', mass, 'n_sat', n_sat)
			block.put_grid('hod' + suffix, 'z', z_vec, 'mass', mass, 'n_cen', n_cen)
			block.put_double_array_1d('hod' + suffix, 'redshifts', z_vec)
			block.put_double_array_1d('hod' + suffix, 'number_density_cen', numdens_cen)
			block.put_double_array_1d('hod' + suffix, 'number_density_sat', numdens_sat)
			block.put_double_array_1d('hod' + suffix, 'central_fraction', fraction_cen)
			block.put_double_array_1d('hod' + suffix, 'satellite_fraction', fraction_sat)

				
			if galaxy_bias_option == True:
				#print('entering galaxy bias')
				#---- loading the halo bias function ----#
				mass_hbf = block['halobias', 'm_h']
				z_hbf = block['halobias', 'z']
				halobias_hbf = block['halobias', 'b_hb']
				#print('problem loading the halobias?')
				f_interp_halobias = interp2d(mass_hbf, z_hbf, halobias_hbf)
				hbias = f_interp_halobias(mass,z_vec)	
			
				galaxybias_cen = np.empty(nz)
				galaxybias_sat = np.empty(nz)
				galaxybias_tot = np.empty(nz)

				for jz in range(0,nz):
					galaxybias_cen[jz] = lf.compute_galaxy_linear_bias(mass, n_cen[jz], hbias[jz], dndlnM[jz])/numdens_tot[jz]
					galaxybias_sat[jz] = lf.compute_galaxy_linear_bias(mass, n_sat[jz], hbias[jz], dndlnM[jz])/numdens_tot[jz]
					galaxybias_tot[jz] = lf.compute_galaxy_linear_bias(mass, n_tot[jz], hbias[jz], dndlnM[jz])/numdens_tot[jz]

				print('ok')

				block.put_double_array_1d('hod' + suffix, 'galaxy_bias_centrals', galaxybias_cen)
				block.put_double_array_1d('hod' + suffix, 'galaxy_bias_satellites', galaxybias_sat)
				block.put_double_array_1d('hod' + suffix, 'galaxy_bias_total', galaxybias_tot)
				block.put_double_array_1d('galaxy_bias' + suffix, 'b', galaxybias_tot)

	return 0


def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass


