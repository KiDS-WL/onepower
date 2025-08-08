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

import numpy as np
from astropy.io import fits
from cosmosis.datablock import option_section
from scipy.integrate import simpson


def mean_l_l0_to_beta(xlum, pdf, l0, beta):
    return simpson(pdf * (xlum / l0) ** beta, xlum)

def broken_powerlaw(xlum, pdf, gamma_2h_lum, l0, beta, beta_low):
    alignment_ampl = np.where(xlum > l0, gamma_2h_lum * (xlum / l0) ** beta,
                              gamma_2h_lum * (xlum / l0) ** beta_low)
    return simpson(pdf * alignment_ampl, xlum)

def compute_luminosity_pdf(z_loglum_file, zmin, zmax, nz, nlbins):
    galfile = fits.open(z_loglum_file)[1].data
    z_gal = np.array(galfile['z'])
    loglum_gal = np.array(galfile['loglum'])

    z_bins = np.linspace(zmin, zmax, nz)
    dz = 0.5 * (z_bins[1] - z_bins[0])
    z_edges = np.append(z_bins - dz, z_bins[-1] + dz)

    bincen = np.zeros([nz, nlbins])
    pdf = np.zeros([nz, nlbins])

    for i in range(nz):
        mask_z = (z_gal >= z_edges[i]) & (z_gal < z_edges[i + 1])
        loglum_bin = loglum_gal[mask_z]
        if loglum_bin.size:
            lum = 10.0 ** loglum_bin
            pdf_tmp, _lum_bins = np.histogram(lum, bins=nlbins, density=True)
            _dbin = (_lum_bins[-1] - _lum_bins[0]) / (1.0 * nlbins)
            bincen[i] = _lum_bins[:-1] + 0.5 * _dbin
            pdf[i] = pdf_tmp

    return bincen, pdf

def setup(options):
    central_ia_depends_on = options[option_section, 'central_IA_depends_on']
    satellite_ia_depends_on = options[option_section, 'satellite_IA_depends_on']

    if central_ia_depends_on not in ['constant', 'luminosity', 'halo_mass']:
        raise ValueError('Choose one of the following options for central_IA_depends_on: '
                         'constant, luminosity, halo_mass')

    if satellite_ia_depends_on not in ['constant', 'luminosity', 'halo_mass']:
        raise ValueError('Choose one of the following options for satellite_IA_depends_on: '
                         'constant, luminosity, halo_mass')

    zmin = options[option_section, 'zmin']
    zmax = options[option_section, 'zmax']
    nz = options[option_section, 'nz']

    if central_ia_depends_on == 'luminosity':
        nlbins = 10000
        z_loglum_file_centrals = options[option_section, 'z_loglum_file_centrals']
        lum_centrals, lum_pdf_z_centrals = compute_luminosity_pdf(
            z_loglum_file_centrals, zmin, zmax, nz, nlbins)
    else:
        nlbins = 100000
        lum_centrals = np.ones([nz, 100000])
        lum_pdf_z_centrals = np.ones([nz, 100000])

    if satellite_ia_depends_on == 'luminosity':
        nlbins = 10000
        z_loglum_file_satellites = options[option_section, 'z_loglum_file_satellites']
        lum_satellites, lum_pdf_z_satellites = compute_luminosity_pdf(
            z_loglum_file_satellites, zmin, zmax, nz, nlbins)
    else:
        nlbins = 100000
        lum_satellites = np.ones([nz, nlbins])
        lum_pdf_z_satellites = np.ones([nz, nlbins])

    name = options.get_string(option_section, 'output_suffix', default='').lower()
    suffix = f'_{name}' if name else ''

    return lum_centrals, lum_pdf_z_centrals, lum_satellites, lum_pdf_z_satellites, nz, 10000, central_ia_depends_on, satellite_ia_depends_on, suffix

def execute(block, config):
    lum_centrals, lum_pdf_z_centrals, lum_satellites, lum_pdf_z_satellites, nz, nlum, central_ia_depends_on, satellite_ia_depends_on, suffix = config

    # TODO: Check if all the options work as intended!
    # TODO: I think we should re-write this bit so the type of dependence doesn't have to be defined
    # Include default values if parameters are not defined in the value file instead.

    # First the Centrals
    # All options require the central 2-halo amplitude to be defined
    gamma_2h = block[f'intrinsic_alignment_parameters{suffix}', 'gamma_2h_amplitude']

    if central_ia_depends_on == 'constant':
        block.put_double_array_1d(f'ia_large_scale_alignment{suffix}', 'alignment_gi',
                                  gamma_2h * np.ones(nz))

    elif central_ia_depends_on == 'luminosity':
        # Check that the user knows what they're doing:
        if not block.has_value(f'intrinsic_alignment_parameters{suffix}', 'L_pivot'):
            raise ValueError('You have chosen central luminosity scaling without providing a '
                             'pivot luminosity parameter. Include L_pivot. "\n ')
        lpiv = block[f'intrinsic_alignment_parameters{suffix}', 'L_pivot']
        beta = block[f'intrinsic_alignment_parameters{suffix}', 'beta']

        if block.has_value(f'intrinsic_alignment_parameters{suffix}', 'beta_two'):
            beta_two = block[f'intrinsic_alignment_parameters{suffix}', 'beta_two']
            mean_lscaling = np.array([broken_powerlaw(lum_centrals[i], lum_pdf_z_centrals[i],
                                                      gamma_2h, 10.0**lpiv, beta, beta_two)
                                      for i in range(nz)])
        else:
            mean_lscaling = gamma_2h * mean_l_l0_to_beta(lum_centrals, lum_pdf_z_centrals,
                                                         10.0**lpiv, beta)

        block.put_double_array_1d(f'ia_large_scale_alignment{suffix}', 'alignment_gi',
                                  mean_lscaling)

    elif central_ia_depends_on == 'halo_mass':
        # Check that the user knows what they're doing:
        if not block.has_value(f'intrinsic_alignment_parameters{suffix}', 'M_pivot'):
            raise ValueError('You have chosen central halo-mass scaling without providing a '
                             'pivot mass parameter. Include M_pivot. "\n ')
        mpiv = block.get_double(f'intrinsic_alignment_parameters{suffix}', 'M_pivot')
        beta = block.get_double(f'intrinsic_alignment_parameters{suffix}', 'beta')

        if block.has_value(f'intrinsic_alignment_parameters{suffix}', 'beta_two'):
            raise ValueError('A double power law model for the halo mass dependence of centrals '
                             'has not been implemented.')
        # Technically just repacking the variables, but this is the easiest way to accommodate
        # backwards compatibility and clean pk_lib.py module
        block.put_double_array_1d(f'ia_large_scale_alignment{suffix}', 'alignment_gi',
                                  gamma_2h * np.ones(nz))
        block.put_double(f'ia_large_scale_alignment{suffix}', 'M_pivot', 10.0**mpiv)
        block.put_double(f'ia_large_scale_alignment{suffix}', 'beta', beta)

    # Add instance information to block
    block.put_string(f'ia_large_scale_alignment{suffix}', 'instance', central_ia_depends_on)

    # Second the Satellites
    # All options require the satellite 1-halo amplitude to be defined
    gamma_1h = block[f'intrinsic_alignment_parameters{suffix}', 'gamma_1h_amplitude']

    if satellite_ia_depends_on == 'constant':
        block.put_double_array_1d(f'ia_small_scale_alignment{suffix}', 'alignment_1h',
                                  gamma_1h * np.ones(nz))

    elif satellite_ia_depends_on == 'luminosity':
        lpiv = block[f'intrinsic_alignment_parameters{suffix}', 'L_pivot']
        beta_sat = block[f'intrinsic_alignment_parameters{suffix}', 'beta_sat']
        mean_lscaling = mean_l_l0_to_beta(lum_satellites, lum_pdf_z_satellites, 10.0**lpiv,
                                          beta_sat)
        block.put_double_array_1d(f'ia_small_scale_alignment{suffix}', 'alignment_1h',
                                  gamma_1h * mean_lscaling)

    elif satellite_ia_depends_on == 'halo_mass':
        mpiv = block.get_double(f'intrinsic_alignment_parameters{suffix}', 'M_pivot')
        beta_sat = block.get_double(f'intrinsic_alignment_parameters{suffix}', 'beta_sat')
        # Technically just repacking the variables, but this is the easiest way to accommodate
        # backwards compatibility and clean pk_lib.py module
        block.put_double_array_1d(f'ia_small_scale_alignment{suffix}', 'alignment_1h',
                                  gamma_1h * np.ones(nz))
        block.put_double(f'ia_small_scale_alignment{suffix}', 'M_pivot', 10.0**mpiv)
        block.put_double(f'ia_small_scale_alignment{suffix}', 'beta_sat', beta_sat)

    # Add instance information to block
    block.put_string(f'ia_small_scale_alignment{suffix}', 'instance', satellite_ia_depends_on)

    return 0

def cleanup(config):
    pass
