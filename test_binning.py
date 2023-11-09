

import numpy as np


zmin = np.asarray([0.1,0.2])
zmax = np.asarray([0.2,0.3])
obs_min = np.asarray([10.0,11.0])
obs_max = np.asarray([11.0,12.0])
nz = 10
nbins = len(zmin)

# TODO: number of bins for z_bins is set to nz why? 
z_bins = np.array([np.linspace(zmin_i, zmax_i, nz) for zmin_i, zmax_i in zip(zmin, zmax)])
#       z_bins = np.array([np.linspace(zmin_i, zmax_i, nz, endpoint=True) for zmin_i, zmax_i in zip(zmin, zmax)])
# This simply repeats the min and max values for the observables for all redshift bins
log_obs_min = np.array([np.repeat(obs_min_i,nz) for obs_min_i in obs_min])
log_obs_max = np.array([np.repeat(obs_max_i,nz) for obs_max_i in obs_max])


log_mass_min = 10.0
log_mass_max = 16.0
nmass        = 20

#---- log-spaced mass sample ----#
dlog10m = (log_mass_max-log_mass_min)/nmass
mass    = 10.0 ** np.arange(log_mass_min, log_mass_max, dlog10m)


for nb in range(0,nbins):
        mass_i, z_bins_i = np.meshgrid(mass, z_bins[nb], sparse=True)
        print(nb,mass_i,z_bins_i)


# to multiply two arrays element by element do this:
arr1 = np.array([1,2,3])
arr2 = np.array([1,3,9,27])
# elements are
# ij: arr1[i]*arr2[j]
arr_mult = arr1[:,np.newaxis]*arr2