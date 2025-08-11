"""
This module combines the red and blue power spectra. It interpolates and extrapolates the
different power spectra in input to match the range and sampling of the matter_power_nl.
The extrapolation method is not particularly advanced (numpy.interp) and would be good
to replace it with something more robust.

The red fraction as a function of redshift must be provided by the user as a txt file with
columns (z, f_red(z)). The z-binning can be arbitrary (it is interpolated inside the code)
but it is safe to provide the largest range possible to avoid substantial extrapolations.

The code assumes the red and blue power spectra to be computed on the same z and k binning.

Step 1: interpolate f_red to the z-bins of the pk of interest
Step 2: add red and blue power spectra
Step 3: interpolate to the z and k-binning of the matter_power_nl

NO CROSS TERMS ARE CURRENTLY IMPLEMENTED.

For each redshift, the power spectra are combined as follows:

GI -> pk_tot = f_red * pk_red + (1-f_red) * pk_blue
II -> pk_tot = f_red**2. * pk_red + (1-f_red)**2. * pk_blue
gI -> pk_tot = f_red**2. * pk_red + (1-f_red)**2. * pk_blue
gg -> pk_tot = f_red**2. * pk_red + (1-f_red)**2. * pk_blue
gm -> pk_tot = f_red * pk_red + (1-f_red) * pk_blue
"""

import numpy as np
from cosmosis.datablock import names, option_section
from scipy.interpolate import interp1d

# We have a collection of commonly used pre-defined block section names.
# If none of the names here is relevant for your calculation you can use any
# string you want instead.
cosmo = names.cosmological_parameters


def extrapolate_nan(z, k, z_ext, k_ext, pk_tot, extrapolate_option):
    pk_extz = np.empty([len(z_ext), len(k)])
    for i in range(len(k)):
        pki = pk_tot[:, i][np.isfinite(pk_tot[:, i])]
        zi = z[np.isfinite(pk_tot[:, i])]
        inter_func = interp1d(zi, pki, kind='linear', fill_value=extrapolate_option, bounds_error=False)
        pk_extz[:, i] = inter_func(z_ext)

    pk_extk = np.empty([len(z), len(k_ext)])
    for j in range(len(z)):
        pki = pk_extz[j, :][np.isfinite(pk_extz[j, :])]
        ki = k[np.isfinite(pk_extz[j, :])]
        inter_func = interp1d(np.log10(ki), np.log10(pki) + 1.0, kind='linear', fill_value='extrapolate', bounds_error=False)
        pk_extk[j, :] = 10.0**inter_func(np.log10(k_ext)) - 1.0

    return pk_extk


def add_red_and_blue_power(block, suffix_red, suffix_blue, suffix_out, f_red, power_section, z_ext, k_ext, extrapolate_option):
    # Note that we have first interpolated the f_red to the halo model pipeline z range
    k = block[f'{power_section}{suffix_red}', 'k_h']
    z = block[f'{power_section}{suffix_red}', 'z']
    pk_red = block[f'{power_section}{suffix_red}', 'p_k']
    pk_blue = block[f'{power_section}{suffix_blue}', 'p_k']

    # TODO: Add the cross terms
    # This is not optimised, but it is good to first choose what we want to implement
    # in terms of cross terms.
    if power_section in ['intrinsic_power', 'galaxy_power', 'galaxy_intrinsic_power']:
        pk_tot = f_red[:, np.newaxis]**2.0 * pk_red + (1.0 - f_red[:, np.newaxis])**2.0 * pk_blue
    else:
        pk_tot = f_red[:, np.newaxis] * pk_red + (1.0 - f_red[:, np.newaxis]) * pk_blue

    # For matter-intrinsic and galaxy-intrinsic, pk_tot will usually be negative (for A_IA > 0)
    # And at very high k can be as large as -100!
    # If we're interpolating over log10(pk_tot) negative power is problematic
    # Check to see if it is negative, and take the absolute value
    if np.sum(pk_tot) < 0:
        pk_tot = pk_tot * -1
        changed_sign = True
    else:
        changed_sign = False

    # warnings.warn('No cross terms between red and blue galaxies implemented.
    # This is only valid for IA in the regime of negligible blue galaxy alignment.')
    # IT 02/03/22: Commented line 86 to execute the code

    # extrapolate
    inter_func_z = interp1d(z, pk_tot, kind='linear', fill_value=extrapolate_option, bounds_error=False, axis=0)
    pk_tot_ext_z = inter_func_z(z_ext)

    inter_func_k = interp1d(np.log10(k), pk_tot_ext_z, kind='linear', fill_value='extrapolate', bounds_error=False, axis=1)
    pk_tot_ext = inter_func_k(np.log10(k_ext))

    # pk_tot_ext = extrapolate_nan(z, k, z_ext, k_ext, pk_tot, extrapolate_option)

    """
    zz, kk = np.meshgrid(z, np.log10(k))
    zz2, kk2 = np.meshgrid(z_ext, np.log10(k_ext))
    import matplotlib.pyplot as pl
    fig = pl.figure()
    ax = fig.add_subplot(projection='3d')
    # ax.scatter(xg.ravel(), yg.ravel(), data.ravel(), s=60, c='k', label='data')
    ax.plot_wireframe(zz.T, kk.T, np.log10(pk_tot), color='red')  # ,rstride=3, cstride=3, alpha=0.4, label='Spectra')
    ax.plot_wireframe(zz2.T, kk2.T, np.log10(pk_tot_ext))
    pl.legend()
    pl.show()
    """

    # Introduce the sign convention back for the GI terms
    if changed_sign:
        pk_tot_ext = pk_tot_ext * -1

    block.put_grid(f'{power_section}{suffix_out}', 'z', z_ext, 'k_h', k_ext, 'p_k', pk_tot_ext)



def extrapolate_power(block, suffix_out, suffix_in, power_section, z_ext, k_ext, extrapolate_option):
    k = block[f'{power_section}{suffix_in}', 'k_h']
    z = block[f'{power_section}{suffix_in}', 'z']
    pk_in = block[f'{power_section}{suffix_in}', 'p_k']

    # For matter-intrinsic and galaxy-intrinsic, pk_tot will usually be negative (for A_IA > 0)
    # If we're interpolating over log10(pk_tot) negative power is problematic
    # Check to see if it is negative, and take the absolute value
    if np.sum(pk_in) < 0:
        pk_in = pk_in * -1
        changed_sign = True
    else:
        changed_sign = False

    inter_func_z = interp1d(z, pk_in, kind='linear', fill_value=extrapolate_option, bounds_error=False, axis=0)
    pk_tot_ext_z = inter_func_z(z_ext)

    inter_func_k = interp1d(np.log10(k), pk_tot_ext_z, kind='linear', fill_value='extrapolate', bounds_error=False, axis=1)
    pk_tot_ext = inter_func_k(np.log10(k_ext))

    # pk_tot_ext = extrapolate_nan(z, k, z_ext, k_ext, pk_in, extrapolate_option)

    """
    zz, kk = np.meshgrid(z, np.log10(k))
    zz2, kk2 = np.meshgrid(z_ext, np.log10(k_ext))
    import matplotlib.pyplot as pl
    fig = pl.figure()
    ax = fig.add_subplot(projection='3d')
    # ax.scatter(xg.ravel(), yg.ravel(), data.ravel(), s=60, c='k', label='data')
    ax.plot_wireframe(zz.T, kk.T, np.log10(pk_in), color='red')  # ,rstride=3, cstride=3, alpha=0.4, label='Spectra')
    ax.plot_wireframe(zz2.T, kk2.T, np.log10(pk_tot_ext))
    pl.legend()
    pl.show()
    """

    # Introduce the sign convention back for the GI terms
    if changed_sign:
        pk_tot_ext = pk_tot_ext * -1

    block.put_grid(f'{power_section}{suffix_out}', 'z', z_ext, 'k_h', k_ext, 'p_k', pk_tot_ext)


#--------------------------------------------------------------------------------#

def setup(options):
    # This function is called once per processor per chain.
    # It is a chance to read any fixed options from the configuration file,
    # load any data, or do any calculations that are fixed once.

    # matter
    p_mm_option = options[option_section, 'do_p_mm']
    # clustering
    p_gg_option = options[option_section, 'do_p_gg']
    # galaxy lensing
    p_gm_option = options[option_section, 'do_p_gm']
    # intrinsic alignment
    p_mI_option = options[option_section, 'do_p_mI']
    p_II_option = options[option_section, 'do_p_II']
    p_gI_option = options[option_section, 'do_p_gI']

    if any(option == 'add_and_extrapolate' for option in [p_gg_option, p_gm_option, p_mI_option, p_II_option, p_gI_option]):
        f_red_file = options[option_section, 'f_red_file']
        z_fred, f_red = np.loadtxt(f_red_file, unpack=True)
    else:
        print('Only extrapolating power spectra.')
        z_fred, f_red = None, None

    hod_section_name_extrap = options.get_string(option_section, 'hod_section_name_extrap', default='hod').lower()
    hod_section_name_red = options.get_string(option_section, 'hod_section_name_red', default='hod_red').lower()
    hod_section_name_blue = options.get_string(option_section, 'hod_section_name_blue', default='hod_blue').lower()

    name_extrap = options.get_string(option_section, 'input_power_suffix_extrap', default='').lower()
    name_red = options.get_string(option_section, 'input_power_suffix_red', default='red').lower()
    name_blue = options.get_string(option_section, 'input_power_suffix_blue', default='blue').lower()

    suffix_extrap = f'_{name_extrap}' if name_extrap != '' else ''
    suffix_red = f'_{name_red}' if name_red != '' else ''
    suffix_blue = f'_{name_blue}' if name_blue != '' else ''

    return z_fred, f_red, p_mm_option, p_gg_option, p_gm_option, p_mI_option, p_II_option, p_gI_option, suffix_extrap, suffix_red, suffix_blue, hod_section_name_extrap, hod_section_name_red, hod_section_name_blue

def execute(block, config):
    # This function is called every time you have a new sample of cosmological and other parameters.
    # It is the main workhorse of the code. The block contains the parameters and results of any
    # earlier modules, and the config is what we loaded earlier.

    z_fred_file, f_red_file, p_mm_option, p_gg_option, p_gm_option, p_mI_option, p_II_option, p_gI_option, suffix0_extrap, suffix0_red, suffix0_blue, hod_section_name_extrap, hod_section_name_red, hod_section_name_blue = config

    # load matter_power_nl k and z:
    z_lin = block['matter_power_lin', 'z']
    k_lin = block['matter_power_lin', 'k_h']

    if p_mm_option == 'extrapolate':
        extrapolate_power(block, '', '', 'matter_power_nl', z_lin, k_lin, 'extrapolate')
        # TODO: Remove once extrapolation of NL power spectra is validated
        try:
            extrapolate_power(block, '', '', 'matter_power_nl_mead', z_lin, k_lin, 'extrapolate')
        except:
            pass

    if any(option == 'extrapolate' for option in [p_gg_option, p_gm_option, p_mI_option, p_II_option, p_gI_option]):

        hod_bins_extrap = block[hod_section_name_extrap, 'nbins']
        observables_z = block[hod_section_name_extrap, 'observable_z']
        extrapolate_option = 'extrapolate' if observables_z else 0.0

        for nb in range(0, hod_bins_extrap):
            suffix_extrap = f'{suffix0_extrap}_{nb+1}' if hod_bins_extrap != 1 else suffix0_extrap
            suffix_out = f'_{nb+1}' if hod_bins_extrap != 1 else ''

            if p_gg_option == 'extrapolate':
                extrapolate_power(block, suffix_out, suffix_extrap, 'galaxy_power', z_lin, k_lin, extrapolate_option)
            if p_gm_option == 'extrapolate':
                extrapolate_power(block, suffix_out, suffix_extrap, 'matter_galaxy_power', z_lin, k_lin, extrapolate_option)
            if p_mI_option == 'extrapolate':
                extrapolate_power(block, suffix_out, suffix_extrap, 'matter_intrinsic_power', z_lin, k_lin, extrapolate_option)
            if p_II_option == 'extrapolate':
                extrapolate_power(block, suffix_out, suffix_extrap, 'intrinsic_power', z_lin, k_lin, extrapolate_option)
            if p_gI_option == 'extrapolate':
                extrapolate_power(block, suffix_out, suffix_extrap, 'galaxy_intrinsic_power', z_lin, k_lin, extrapolate_option)

    if any(option == 'add_and_extrapolate' for option in [p_gg_option, p_gm_option, p_mI_option, p_II_option, p_gI_option]):

        hod_bins_red = block[hod_section_name_red, 'nbins']
        hod_bins_blue = block[hod_section_name_blue, 'nbins']

        if hod_bins_red != hod_bins_blue:
            raise Exception('Error: number of red and blue stellar mass bins should be the same.')

        observables_z_red = block[hod_section_name_red, 'observable_z']
        extrapolate_option = 'extrapolate' if observables_z_red else 0

        for nb in range(0, hod_bins_red):
            suffix_red = f'{suffix0_red}_{nb+1}' if hod_bins_red != 1 else suffix0_red
            suffix_blue = f'{suffix0_blue}_{nb+1}' if hod_bins_red != 1 else suffix0_blue
            suffix_out = f'_{nb+1}' if hod_bins_red != 1 else ''

            if p_gg_option == 'add_and_extrapolate':
                # load halo model k and z (red and blue are expected to be with the same red/blue ranges and z,k-samplings!):
                z_hm = block[f'galaxy_power{suffix_red}', 'z']
                f_red = interp1d(z_fred_file, f_red_file, 'linear', bounds_error=False, fill_value='extrapolate')
                add_red_and_blue_power(block, suffix_red, suffix_blue, suffix_out, f_red(z_hm), 'galaxy_power', z_lin, k_lin, extrapolate_option)

            if p_gm_option == 'add_and_extrapolate':
                # load halo model k and z (red and blue are expected to be with the same red/blue ranges and z,k-samplings!):
                z_hm = block[f'matter_galaxy_power{suffix_red}', 'z']
                # IT Added bounds_error=False and fill_value extrapolate
                f_red = interp1d(z_fred_file, f_red_file, 'linear', bounds_error=False, fill_value='extrapolate')
                add_red_and_blue_power(block, suffix_red, suffix_blue, suffix_out, f_red(z_hm), 'matter_galaxy_power', z_lin, k_lin, extrapolate_option)

            if p_mI_option == 'add_and_extrapolate':
                # load halo model k and z (red and blue are expected to be with the same red/blue ranges and z,k-samplings!):
                z_hm = block[f'matter_intrinsic_power{suffix_red}', 'z']
                f_red = interp1d(z_fred_file, f_red_file, 'linear', bounds_error=False, fill_value='extrapolate')
                add_red_and_blue_power(block, suffix_red, suffix_blue, suffix_out, f_red(z_hm), 'matter_intrinsic_power', z_lin, k_lin, extrapolate_option)

            if p_II_option == 'add_and_extrapolate':
                # load halo model k and z (red and blue are expected to be with the same red/blue ranges and z,k-samplings!):
                z_hm = block[f'intrinsic_power{suffix_red}', 'z']
                f_red = interp1d(z_fred_file, f_red_file, 'linear', bounds_error=False, fill_value='extrapolate')
                add_red_and_blue_power(block, suffix_red, suffix_blue, suffix_out, f_red(z_hm), 'intrinsic_power', z_lin, k_lin, extrapolate_option)

            if p_gI_option == 'add_and_extrapolate':
                # load halo model k and z (red and blue are expected to be with the same red/blue ranges and z,k-samplings!):
                z_hm = block[f'galaxy_intrinsic_power{suffix_red}', 'z']
                f_red = interp1d(z_fred_file, f_red_file, 'linear', bounds_error=False, fill_value='extrapolate')
                add_red_and_blue_power(block, suffix_red, suffix_blue, suffix_out, f_red(z_hm), 'galaxy_intrinsic_power', z_lin, k_lin, extrapolate_option)

    return 0

def cleanup(config):
    # Usually python modules do not need to do anything here.
    # We just leave it in out of pedantic completeness.
    pass
