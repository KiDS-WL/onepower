import pytest
import numpy as np
from unittest.mock import MagicMock
from pathlib import Path
from onepower import AlignmentAmplitudes, SatelliteAlignment


@pytest.fixture
def setup_data():
    rng = np.random.default_rng(seed=42)
    z_vec = np.linspace(0, 3, 15)
    mass_in = np.logspace(12, 15, 100)
    c_in = rng.random((15, 100))
    r_s_in = rng.random((15, 100))
    rvir_in = rng.random(100)
    return z_vec, mass_in, c_in, r_s_in, rvir_in


def test_alignment_amplitudes_initialization_and_properties(setup_data):
    z_vec, _, _, _, _ = setup_data
    alignment_amps = AlignmentAmplitudes(z_vec=z_vec)

    assert np.allclose(alignment_amps.z_vec, z_vec)
    assert alignment_amps.central_ia_depends_on == 'halo_mass'
    assert alignment_amps.satellite_ia_depends_on == 'halo_mass'
    assert alignment_amps.gamma_2h_amplitude == 5.33
    assert alignment_amps.beta_cen == 0.44
    assert alignment_amps.gamma_1h_amplitude == 0.0015
    assert alignment_amps.gamma_1h_slope == -2.0
    assert alignment_amps.beta_sat == 0.44


def test_alignment_amplitudes_luminosity_dependencies(setup_data, datadir):
    z_vec, _, _, _, _ = setup_data
    filename_centrals = f'{datadir}/redcen_lum_tests.fits'
    filename_satellites = f'{datadir}/redsat_lum_tests.fits'

    alignment_amps_centrals = AlignmentAmplitudes(
        z_vec=z_vec,
        central_ia_depends_on='luminosity',
        z_loglum_file_centrals=filename_centrals,
    )
    lum_centrals = alignment_amps_centrals.lum_centrals
    lum_centrals_pdf = alignment_amps_centrals.lum_pdf_z_centrals
    assert lum_centrals.shape == (len(z_vec), 10000)
    assert lum_centrals_pdf.shape == (len(z_vec), 10000)

    alignment_amps_satellites = AlignmentAmplitudes(
        z_vec=z_vec,
        satellite_ia_depends_on='luminosity',
        z_loglum_file_satellites=filename_satellites,
    )
    lum_satellites = alignment_amps_satellites.lum_satellites
    lum_satellites_pdf = alignment_amps_satellites.lum_pdf_z_satellites
    assert lum_satellites.shape == (len(z_vec), 10000)
    assert lum_satellites_pdf.shape == (len(z_vec), 10000)


def test_alignment_amplitudes_wrong_case(setup_data):
    z_vec, _, _, _, _ = setup_data

    with pytest.raises(ValueError):
        AlignmentAmplitudes(z_vec=z_vec, central_ia_depends_on='something')

    with pytest.raises(ValueError):
        AlignmentAmplitudes(z_vec=z_vec, satellite_ia_depends_on='something')

    with pytest.raises(ValueError):
        alignment = AlignmentAmplitudes(
            z_vec=z_vec, central_ia_depends_on='luminosity', z_loglum_file_centrals=None
        )
        alignment.lum_centrals

    with pytest.raises(ValueError):
        alignment = AlignmentAmplitudes(
            z_vec=z_vec,
            satellite_ia_depends_on='luminosity',
            z_loglum_file_satellites=None,
        )
        alignment.lum_satellites


def test_alignment_amplitudes_alignment_gi(setup_data, datadir):
    z_vec, _, _, _, _ = setup_data
    filename = f'{datadir}/redcen_lum_tests.fits'

    alignment_amps = AlignmentAmplitudes(z_vec=z_vec, central_ia_depends_on='constant')
    alignment_gi = alignment_amps.alignment_gi
    assert alignment_gi.shape == z_vec.shape

    alignment_amps = AlignmentAmplitudes(
        z_vec=z_vec,
        central_ia_depends_on='luminosity',
        z_loglum_file_centrals=filename,
        lpivot_cen=13.0,
    )
    alignment_gi = alignment_amps.alignment_gi
    assert alignment_gi.shape == z_vec.shape

    alignment_amps = AlignmentAmplitudes(
        z_vec=z_vec,
        central_ia_depends_on='luminosity',
        z_loglum_file_centrals=filename,
        lpivot_cen=13.0,
        beta_two=2.0,
    )
    alignment_gi = alignment_amps.alignment_gi
    assert alignment_gi.shape == z_vec.shape

    with pytest.raises(ValueError):
        alignment_amps = AlignmentAmplitudes(
            z_vec=z_vec,
            central_ia_depends_on='luminosity',
            z_loglum_file_centrals=filename,
        )
        alignment_amps.alignment_gi

    with pytest.raises(ValueError):
        alignment_amps = AlignmentAmplitudes(
            z_vec=z_vec, central_ia_depends_on='halo_mass', mpivot_cen=None
        )
        alignment_amps.alignment_gi

    with pytest.raises(ValueError):
        alignment_amps = AlignmentAmplitudes(
            z_vec=z_vec, central_ia_depends_on='halo_mass', beta_two=2.0
        )
        alignment_amps.alignment_gi

    alignment_amps = AlignmentAmplitudes(z_vec=z_vec, central_ia_depends_on='halo_mass')
    alignment_gi = alignment_amps.alignment_gi
    assert alignment_gi.shape == z_vec.shape


def test_alignment_amplitudes_gamma_1h_amp(setup_data, datadir):
    z_vec, _, _, _, _ = setup_data
    filename = f'{datadir}/redsat_lum_tests.fits'

    alignment_amps = AlignmentAmplitudes(
        z_vec=z_vec, satellite_ia_depends_on='constant'
    )
    gamma_1h_amp = alignment_amps.gamma_1h_amp
    assert gamma_1h_amp.shape == z_vec.shape

    alignment_amps = AlignmentAmplitudes(
        z_vec=z_vec,
        satellite_ia_depends_on='luminosity',
        z_loglum_file_satellites=filename,
        lpivot_sat=13.0,
    )
    gamma_1h_amp = alignment_amps.gamma_1h_amp
    assert gamma_1h_amp.shape == z_vec.shape

    with pytest.raises(ValueError):
        alignment_amps = AlignmentAmplitudes(
            z_vec=z_vec,
            satellite_ia_depends_on='luminosity',
            z_loglum_file_satellites=filename,
        )
        alignment_amps.gamma_1h_amp

    with pytest.raises(ValueError):
        alignment_amps = AlignmentAmplitudes(
            z_vec=z_vec, satellite_ia_depends_on='halo_mass', mpivot_sat=None
        )
        alignment_amps.gamma_1h_amp


def test_satellite_alignment_initialization_and_properties(setup_data):
    z_vec, mass_in, c_in, r_s_in, rvir_in = setup_data
    satellite_alignment = SatelliteAlignment(
        z_vec=z_vec, mass_in=mass_in, c_in=c_in, r_s_in=r_s_in, rvir_in=rvir_in
    )

    assert np.allclose(satellite_alignment.z_vec, z_vec)
    assert np.allclose(satellite_alignment.mass_in, mass_in)
    assert np.allclose(satellite_alignment.c_in, c_in)
    assert np.allclose(satellite_alignment.r_s_in, r_s_in)
    assert np.allclose(satellite_alignment.rvir_in, rvir_in)
    assert satellite_alignment.n_hankel == 350
    assert satellite_alignment.nmass == 5
    assert satellite_alignment.nk == 10
    assert satellite_alignment.ell_max == 6
    assert satellite_alignment.truncate is False
    assert satellite_alignment.method == 'fftlog'
    assert satellite_alignment.hankel is None


def test_satellite_alignment_ell_values_and_methods(setup_data):
    z_vec, mass_in, c_in, r_s_in, rvir_in = setup_data

    satellite_alignment = SatelliteAlignment(
        z_vec=z_vec, mass_in=mass_in, c_in=c_in, r_s_in=r_s_in, rvir_in=rvir_in
    )
    ell_values = satellite_alignment.ell_values
    assert ell_values.tolist() == [0, 2, 4, 6]

    with pytest.raises(ValueError):
        SatelliteAlignment(
            z_vec=z_vec,
            mass_in=mass_in,
            c_in=c_in,
            r_s_in=r_s_in,
            rvir_in=rvir_in,
            ell_max=12,
        )

    satellite_alignment = SatelliteAlignment(
        z_vec=z_vec,
        mass_in=mass_in,
        c_in=c_in,
        r_s_in=r_s_in,
        rvir_in=rvir_in,
        method='hankel',
    )
    hankel = satellite_alignment.hankel
    assert len(hankel) == len(satellite_alignment.ell_values)

    with pytest.raises(ValueError):
        SatelliteAlignment(
            z_vec=z_vec,
            mass_in=mass_in,
            c_in=c_in,
            r_s_in=r_s_in,
            rvir_in=rvir_in,
            method='notsupportedmethod',
        )


def test_satellite_alignment_k_vec_and_mass(setup_data):
    z_vec, mass_in, c_in, r_s_in, rvir_in = setup_data

    satellite_alignment = SatelliteAlignment(
        z_vec=z_vec, mass_in=mass_in, c_in=c_in, r_s_in=r_s_in, rvir_in=rvir_in
    )
    k_vec = satellite_alignment.k_vec
    assert len(k_vec) == 100

    mass = satellite_alignment.mass
    assert mass.shape == mass_in.shape


def test_satellite_alignment_c_and_r_s(setup_data):
    z_vec, mass_in, c_in, r_s_in, rvir_in = setup_data

    satellite_alignment = SatelliteAlignment(
        z_vec=z_vec, mass_in=mass_in, c_in=c_in, r_s_in=r_s_in, rvir_in=rvir_in
    )
    c = satellite_alignment.c
    assert c.shape == c_in.shape

    r_s = satellite_alignment.r_s
    assert r_s.shape == r_s_in.shape


def test_satellite_alignment_rvir(setup_data):
    z_vec, mass_in, c_in, r_s_in, rvir_in = setup_data

    satellite_alignment = SatelliteAlignment(
        z_vec=z_vec, mass_in=mass_in, c_in=c_in, r_s_in=r_s_in, rvir_in=rvir_in
    )
    rvir = satellite_alignment.rvir
    assert rvir.shape == rvir_in.shape


def test_satellite_alignment_wkm_and_upsampled_wkm(setup_data):
    z_vec, mass_in, c_in, r_s_in, rvir_in = setup_data

    satellite_alignment = SatelliteAlignment(
        z_vec=z_vec, mass_in=mass_in, c_in=c_in, r_s_in=r_s_in, rvir_in=rvir_in
    )
    wkm = satellite_alignment.wkm
    assert len(wkm) == 4
    assert wkm[0].shape == (len(z_vec), len(mass_in), len(satellite_alignment.k_vec))

    satellite_alignment = SatelliteAlignment(
        z_vec=z_vec,
        mass_in=mass_in,
        c_in=c_in,
        r_s_in=r_s_in,
        rvir_in=rvir_in,
        gamma_1h_slope=-1.99999999,
    )
    wkm_2 = satellite_alignment.wkm
    assert len(wkm_2) == 4
    assert wkm_2[0].shape == (len(z_vec), len(mass_in), len(satellite_alignment.k_vec))
    assert np.allclose(wkm[0], wkm_2[0], atol=1e-3)
    assert np.allclose(wkm[1], wkm_2[1], atol=1e-3)

    k_vec_out = np.logspace(np.log10(1e-3), np.log10(1e3), 200)
    mass_out = np.logspace(12, 15, 200)

    satellite_alignment = SatelliteAlignment(
        z_vec=z_vec,
        mass_in=mass_in,
        c_in=c_in,
        r_s_in=r_s_in,
        rvir_in=rvir_in,
        method='hankel',
    )
    upsampled_wkm = satellite_alignment.upsampled_wkm(k_vec_out, mass_out)
    assert upsampled_wkm.shape == (len(z_vec), len(mass_out), len(k_vec_out))

    satellite_alignment = SatelliteAlignment(
        z_vec=z_vec,
        mass_in=mass_in,
        c_in=c_in,
        r_s_in=r_s_in,
        rvir_in=rvir_in,
        method='fftlog',
    )
    upsampled_wkm = satellite_alignment.upsampled_wkm(k_vec_out, mass_in)
    assert upsampled_wkm.shape == (len(z_vec), len(mass_in), len(k_vec_out))
