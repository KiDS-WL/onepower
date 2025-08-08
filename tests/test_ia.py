import pytest

import numpy as np
from unittest.mock import MagicMock

from onepower import AlignmentAmplitudes, SatelliteAlignment

# Just some base tests for now, no specific scientific calculations tested yet!

@pytest.fixture
def setup_data():
    z_vec = np.linspace(0, 3, 15)
    mass_in = np.logspace(12, 15, 100)
    c_in = np.random.rand(15, 100)
    r_s_in = np.random.rand(15, 100)
    rvir_in = np.random.rand(100)
    return z_vec, mass_in, c_in, r_s_in, rvir_in

def test_alignment_amplitudes_initialization(setup_data):
    z_vec, _, _, _, _ = setup_data
    alignment_amps = AlignmentAmplitudes(z_vec=z_vec)
    assert alignment_amps.z_vec.all() == z_vec.all()
    assert alignment_amps.central_ia_depends_on == 'halo_mass'
    assert alignment_amps.satellite_ia_depends_on == 'halo_mass'
    assert alignment_amps.gamma_2h_amplitude == 5.33
    assert alignment_amps.beta_cen == 0.44
    assert alignment_amps.gamma_1h_amplitude == 0.0015
    assert alignment_amps.gamma_1h_slope == -2.0
    assert alignment_amps.beta_sat == 0.44

#def test_alignment_amplitudes_lum_centrals(setup_data):
#    z_vec, _, _, _, _ = setup_data
#    alignment_amps = AlignmentAmplitudes(z_vec=z_vec, central_ia_depends_on='luminosity')
#    lum_centrals = alignment_amps.lum_centrals
#    assert lum_centrals.shape == (len(z_vec), 10000)
#
#def test_alignment_amplitudes_lum_satellites(setup_data):
#    z_vec, _, _, _, _ = setup_data
#    alignment_amps = AlignmentAmplitudes(z_vec=z_vec, satellite_ia_depends_on='luminosity')
#    lum_satellites = alignment_amps.lum_satellites
#    assert lum_satellites.shape == (len(z_vec), 10000)

def test_alignment_amplitudes_alignment_gi(setup_data):
    z_vec, _, _, _, _ = setup_data
    alignment_amps = AlignmentAmplitudes(z_vec=z_vec, central_ia_depends_on='constant')
    alignment_gi = alignment_amps.alignment_gi
    assert alignment_gi.shape == z_vec.shape

def test_alignment_amplitudes_gamma_1h_amp(setup_data):
    z_vec, _, _, _, _ = setup_data
    alignment_amps = AlignmentAmplitudes(z_vec=z_vec, satellite_ia_depends_on='constant')
    gamma_1h_amp = alignment_amps.gamma_1h_amp
    assert gamma_1h_amp.shape == z_vec.shape

def test_satellite_alignment_initialization(setup_data):
    z_vec, mass_in, c_in, r_s_in, rvir_in = setup_data
    satellite_alignment = SatelliteAlignment(z_vec=z_vec, mass_in=mass_in, c_in=c_in, r_s_in=r_s_in, rvir_in=rvir_in)
    assert satellite_alignment.z_vec.all() == z_vec.all()
    assert satellite_alignment.mass_in.all() == mass_in.all()
    assert satellite_alignment.c_in.all() == c_in.all()
    assert satellite_alignment.r_s_in.all() == r_s_in.all()
    assert satellite_alignment.rvir_in.all() == rvir_in.all()
    assert satellite_alignment.n_hankel == 350
    assert satellite_alignment.nmass == 5
    assert satellite_alignment.nk == 10
    assert satellite_alignment.ell_max == 6
    assert satellite_alignment.truncate == False
    assert satellite_alignment.method == 'fftlog'

def test_satellite_alignment_ell_values(setup_data):
    z_vec, mass_in, c_in, r_s_in, rvir_in = setup_data
    satellite_alignment = SatelliteAlignment(z_vec=z_vec, mass_in=mass_in, c_in=c_in, r_s_in=r_s_in, rvir_in=rvir_in)
    ell_values = satellite_alignment.ell_values
    assert ell_values.tolist() == [0, 2, 4, 6]

def test_satellite_alignment_k_vec(setup_data):
    z_vec, mass_in, c_in, r_s_in, rvir_in = setup_data
    satellite_alignment = SatelliteAlignment(z_vec=z_vec, mass_in=mass_in, c_in=c_in, r_s_in=r_s_in, rvir_in=rvir_in)
    k_vec = satellite_alignment.k_vec
    assert len(k_vec) == 100

def test_satellite_alignment_mass(setup_data):
    z_vec, mass_in, c_in, r_s_in, rvir_in = setup_data
    satellite_alignment = SatelliteAlignment(z_vec=z_vec, mass_in=mass_in, c_in=c_in, r_s_in=r_s_in, rvir_in=rvir_in)
    mass = satellite_alignment.mass
    assert mass.shape == mass_in.shape

def test_satellite_alignment_c(setup_data):
    z_vec, mass_in, c_in, r_s_in, rvir_in = setup_data
    satellite_alignment = SatelliteAlignment(z_vec=z_vec, mass_in=mass_in, c_in=c_in, r_s_in=r_s_in, rvir_in=rvir_in)
    c = satellite_alignment.c
    assert c.shape == c_in.shape

def test_satellite_alignment_r_s(setup_data):
    z_vec, mass_in, c_in, r_s_in, rvir_in = setup_data
    satellite_alignment = SatelliteAlignment(z_vec=z_vec, mass_in=mass_in, c_in=c_in, r_s_in=r_s_in, rvir_in=rvir_in)
    r_s = satellite_alignment.r_s
    assert r_s.shape == r_s_in.shape

def test_satellite_alignment_rvir(setup_data):
    z_vec, mass_in, c_in, r_s_in, rvir_in = setup_data
    satellite_alignment = SatelliteAlignment(z_vec=z_vec, mass_in=mass_in, c_in=c_in, r_s_in=r_s_in, rvir_in=rvir_in)
    rvir = satellite_alignment.rvir
    assert rvir.shape == rvir_in.shape

def test_satellite_alignment_hankel(setup_data):
    z_vec, mass_in, c_in, r_s_in, rvir_in = setup_data
    satellite_alignment = SatelliteAlignment(z_vec=z_vec, mass_in=mass_in, c_in=c_in, r_s_in=r_s_in, rvir_in=rvir_in, method='hankel')
    hankel = satellite_alignment.hankel
    assert len(hankel) == len(satellite_alignment.ell_values)

def test_satellite_alignment_wkm_f_ell(setup_data):
    z_vec, mass_in, c_in, r_s_in, rvir_in = setup_data
    satellite_alignment = SatelliteAlignment(z_vec=z_vec, mass_in=mass_in, c_in=c_in, r_s_in=r_s_in, rvir_in=rvir_in)
    wkm_f_ell = satellite_alignment.wkm_f_ell
    assert wkm_f_ell.shape == (len(z_vec), len(mass_in), len(satellite_alignment.k_vec))

def test_satellite_alignment_wkm(setup_data):
    z_vec, mass_in, c_in, r_s_in, rvir_in = setup_data
    satellite_alignment = SatelliteAlignment(z_vec=z_vec, mass_in=mass_in, c_in=c_in, r_s_in=r_s_in, rvir_in=rvir_in)
    wkm = satellite_alignment.wkm
    assert len(wkm) == 4
    assert wkm[0].shape == (len(z_vec), len(mass_in), len(satellite_alignment.k_vec))

def test_satellite_alignment_upsampled_wkm_hankel(setup_data):
    z_vec, mass_in, c_in, r_s_in, rvir_in = setup_data
    satellite_alignment = SatelliteAlignment(z_vec=z_vec, mass_in=mass_in, c_in=c_in, r_s_in=r_s_in, rvir_in=rvir_in, method='hankel')
    k_vec_out = np.logspace(np.log10(1e-3), np.log10(1e3), 200)
    mass_out = np.logspace(12, 15, 200)
    upsampled_wkm = satellite_alignment.upsampled_wkm(k_vec_out, mass_out)
    assert upsampled_wkm.shape == (len(z_vec), len(mass_out), len(k_vec_out))

def test_satellite_alignment_upsampled_wkm_fftlog(setup_data):
    z_vec, mass_in, c_in, r_s_in, rvir_in = setup_data
    satellite_alignment = SatelliteAlignment(z_vec=z_vec, mass_in=mass_in, c_in=c_in, r_s_in=r_s_in, rvir_in=rvir_in, method='fftlog')
    k_vec_out = np.logspace(np.log10(1e-3), np.log10(1e3), 200)
    upsampled_wkm = satellite_alignment.upsampled_wkm(k_vec_out, mass_in)
    assert upsampled_wkm.shape == (len(z_vec), len(mass_in), len(k_vec_out))
