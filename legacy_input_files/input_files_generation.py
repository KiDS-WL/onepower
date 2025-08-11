import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import interp1d

# This script needs to be integrated with CosmoPipe and run on the KiDS catalogues. Question remains how to get cen/sat determination of KiDS galaxies, if at all, but if using the halo mass dependent IA that should not matter.

if __name__ == '__main__':

    # Settings
    split_value = 1.75
    nzbins = 30
    plots = True

    hdul = fits.open('KIDS_cat.fits')
    hdul[1].header

    df = hdul[1].data

    # Computing the luminosity from the r band absolute magnitude
    df_obs = np.power(10.0, -0.4 * (df['sdss_r_abs_mag'] - 4.76))

    # For the moment we use a cut in the u-r absolute color vs redshift
    df_ur = df['sdss_u_abs_mag'] - df['sdss_r_abs_mag']

    if plots:
        plt.scatter(df_ur, df['Z_B'], s=0.000005)
        plt.axvline(split_value)
        plt.xlabel('u-r', fontsize=15)
        plt.ylabel('Z_B', fontsize=15)
        plt.show()
        plt.clf()

        plt.hist(df_ur, histtype='step', bins=150, density=True)
        plt.axvline(split_value)
        plt.xlabel('u-r', fontsize=15)
        plt.show()
        plt.clf()

    # Redshift edges for the red/blue split histograms
    edges = np.linspace(np.min(df['Z_B']), np.max(df['Z_B']), nzbins + 1, endpoint=True)
    nblue = np.histogram(df['Z_B'][df_ur <= split_value], bins=edges)[0]
    nred = np.histogram(df['Z_B'][df_ur > split_value], bins=edges)[0]
    total = nblue + nred
    red_fraction = nred / total
    blue_fraction = nblue / total
    zbins = (edges[1:] + edges[:-1]) / 2.0
    # Fraction of red galaxies as a function of redshift
    np.savetxt('f_red.txt', np.column_stack([z_bins, red_fraction]))

    if plots:
        plt.plot(zbins, blue_fraction)
        plt.xlabel('Z_B', fontsize=15)
        plt.ylabel('fraction of blue galaxies', fontsize=15)
        plt.show()
        plt.clf()

    # Luminosity limits for red centrals as a function of redshift
    obs_min = np.empty(nzbins)
    obs_max = np.empty(nzbins)
    for i in range(len(zbins)):
        selection = df_obs[
            (df['Z_B'] < edges[i])
            & (df['Z_B'] >= edges[i - 1])
            & (df_ur > split_value)
            & (df['flag_central'] == 0)
        ]
        obs_min[i] = np.min(selection)
        obs_max[i] = np.max(selection)

    np.savetxt(
        'red_cen_lum_pdf.txt',
        np.column_stack([z_bins, obs_min, obs_max]),
        header='z lum_min lum_max',
    )

    # Luminosity limits for blue centrals as a function of redshift
    obs_min = np.empty(nzbins)
    obs_max = np.empty(nzbins)
    for i in range(len(zbins)):
        selection = df_obs[
            (df['Z_B'] < edges[i])
            & (df['Z_B'] >= edges[i - 1])
            & (df_ur <= split_value)
            & (df['flag_central'] == 0)
        ]
        obs_min[i] = np.min(selection)
        obs_max[i] = np.max(selection)

    np.savetxt(
        'blue_cen_lum_pdf.txt',
        np.column_stack([z_bins, obs_min, obs_max]),
        header='z lum_min lum_max',
    )

    if plots:
        # Check of the luminosity pdfs for red centrals at 4 different redshifts
        for i in [0, 10, 20, 30]:
            selection = df_obs[
                (df['Z_B'] < edges[i])
                & (df['Z_B'] >= edges[i - 1])
                & (df_ur > split_value)
                & (df['flag_central'] == 0)
            ]
            plt.hist(np.log10(selection), bins=10, histtype='step', density=True)
        plt.xlabel('log_lum', fontsize=15)
        plt.show()
        plt.clf()

    # Luminosity pdfs for blue satellites, red satellites, and red centrals as a function of redshift
    # Used for luminosity dependent IA halo model only
    c1 = fits.Column(
        name='z',
        array=df['Z_B'][(df_ur <= split_value) & (df['flag_central'] == 1)],
        format='E',
    )
    c2 = fits.Column(
        name='loglum',
        array=np.log10(df_obs[(df_ur <= split_value) & (df['flag_central'] == 1)]),
        format='E',
    )
    t = fits.BinTableHDU.from_columns([c1, c2])
    t.writeto('bluesat_lum.fits', overwrite=True)

    c1 = fits.Column(
        name='z',
        array=df['Z_B'][(df_ur > split_value) & (df['flag_central'] == 1)],
        format='E',
    )
    c2 = fits.Column(
        name='loglum',
        array=np.log10(df_obs[(df_ur > split_value) & (df['flag_central'] == 1)]),
        format='E',
    )
    t = fits.BinTableHDU.from_columns([c1, c2])
    t.writeto('redsat_lum.fits', overwrite=True)

    c1 = fits.Column(
        name='z',
        array=df['Z_B'][(df_ur > split_value) & (df['flag_central'] == 0)],
        format='E',
    )
    c2 = fits.Column(
        name='loglum',
        array=np.log10(df_obs[(df_ur > split_value) & (df['flag_central'] == 0)]),
        format='E',
    )
    t = fits.BinTableHDU.from_columns([c1, c2])
    t.writeto('redcen_lum.fits', overwrite=True)
