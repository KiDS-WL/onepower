import pytest

import numpy as np
from unittest.mock import MagicMock

from onepower import PowerSpectrumResult, Spectra

# Just some base tests for now, no specific scientific calculations tested yet!


@pytest.fixture
def setup_data():
    k_vec = np.logspace(-4, 4, 100)
    z_vec = np.linspace(0.0, 3.0, 15)
    mass = np.logspace(12, 15, 100)
    dndlnm = np.random.rand(15, 100)
    halo_bias = np.random.rand(15, 100)
    matter_power_lin = np.random.rand(15, 100)
    matter_power_nl = np.random.rand(15, 100)
    return k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl


@pytest.fixture
def spectra(setup_data):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    spectra = Spectra(
        Mmin=12,
        Mmax=15,
        dlog10m=(15 - 12) / 100,
        matter_power_lin=matter_power_lin,
        matter_power_nl=matter_power_nl,
    )
    return spectra


def test_spectra_initialization(setup_data):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    spectra = Spectra(
        matter_power_lin=matter_power_lin, matter_power_nl=matter_power_nl
    )
    assert spectra.matter_power_lin.shape == matter_power_lin.shape
    assert spectra.matter_power_nl.shape == matter_power_nl.shape


def test_spectra_beta_nl_array(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    beta_nl_array = spectra._beta_nl_array
    assert beta_nl_array is None


def test_spectra_pk_lin(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    pk_lin = spectra._pk_lin
    assert pk_lin.shape == (len(z_vec), len(k_vec))


def test_spectra_pk_nl(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    pk_nl = spectra._pk_nl
    assert pk_nl.shape == (len(z_vec), len(k_vec))


def test_spectra_peff(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    peff = spectra.peff
    assert peff is None


def test_spectra_calc_bnl(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    spectra.update(bnl=True)
    calc_bnl = spectra.calc_bnl
    spectra.update(bnl=False)
    assert calc_bnl.shape == (1, len(mass), len(mass), len(k_vec))


def test_spectra_I12(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    spectra.update(bnl=True)
    I12 = spectra.I12
    spectra.update(bnl=False)
    assert I12 is not None


def test_spectra_I21(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    spectra.update(bnl=True)
    I21 = spectra.I21
    spectra.update(bnl=False)
    assert I21 is not None


def test_spectra_I22(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    spectra.update(bnl=True)
    I22 = spectra.I22
    spectra.update(bnl=False)
    assert I22 is not None


def test_spectra_hod_mm(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    hod_mm = spectra.hod_mm
    assert hod_mm is None


def test_spectra_fstar_mm(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    fstar_mm = spectra.fstar_mm
    assert fstar_mm.shape == (1, len(z_vec), len(mass))


def test_spectra_matter_profile(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    matter_profile = spectra.matter_profile
    assert matter_profile.shape == (1, len(z_vec), len(k_vec), len(mass))


def test_spectra_matter_profile_2h(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    matter_profile_2h = spectra.matter_profile_2h
    assert matter_profile_2h.shape == (1, len(z_vec), len(k_vec), len(mass))


# def test_spectra_matter_profile_with_feedback(setup_data, spectra):
#    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = setup_data
#    spectra.update(mead_correction='feedback')
#    matter_profile_with_feedback = spectra.matter_profile_with_feedback
#    spectra.update(mead_correction=None)
#    assert matter_profile_with_feedback.shape == (1, len(z_vec), len(k_vec), len(mass))


def test_spectra_one_halo_truncation(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    one_halo_truncation = spectra.one_halo_truncation
    assert one_halo_truncation.shape == (len(k_vec),)


def test_spectra_two_halo_truncation(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    two_halo_truncation = spectra.two_halo_truncation
    assert two_halo_truncation.shape == (len(k_vec),)


def test_spectra_one_halo_truncation_ia(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    one_halo_truncation_ia = spectra.one_halo_truncation_ia
    assert one_halo_truncation_ia.shape == (len(k_vec),)


def test_spectra_two_halo_truncation_ia(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    two_halo_truncation_ia = spectra.two_halo_truncation_ia
    assert two_halo_truncation_ia.shape == (len(k_vec),)


def test_spectra_power_spectrum_lin(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    power_spectrum_lin = spectra.power_spectrum_lin
    assert isinstance(power_spectrum_lin, PowerSpectrumResult)


def test_spectra_power_spectrum_mm(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    power_spectrum_mm = spectra.power_spectrum_mm
    assert isinstance(power_spectrum_mm, PowerSpectrumResult)


def test_spectra_hod(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    hod = spectra.hod
    assert hod is not None


def test_spectra_fstar(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    fstar = spectra.fstar
    assert fstar.shape == (1, len(z_vec), len(mass))


def test_spectra_mass_avg(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    mass_avg = spectra.mass_avg
    assert mass_avg.shape == (1, len(z_vec))


def test_spectra_obs(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    spectra.update(compute_observable=True)
    obs = spectra.obs
    spectra.update(compute_observable=False)
    assert obs is not None


def test_spectra_obs_func(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    spectra.update(compute_observable=True)
    obs_func = spectra.obs_func
    spectra.update(compute_observable=False)
    assert obs_func is not None


def test_spectra_central_galaxy_profile(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    central_galaxy_profile = spectra.central_galaxy_profile
    assert central_galaxy_profile.shape == (1, len(z_vec), len(k_vec), len(mass))


def test_spectra_satellite_galaxy_profile(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    satellite_galaxy_profile = spectra.satellite_galaxy_profile
    assert satellite_galaxy_profile.shape == (1, len(z_vec), len(k_vec), len(mass))


def test_spectra_Ic_term(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    Ic_term = spectra.Ic_term
    assert Ic_term.shape == (1, len(z_vec), len(k_vec))


def test_spectra_Is_term(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    Is_term = spectra.Is_term
    assert Is_term.shape == (1, len(z_vec), len(k_vec))


def test_spectra_power_spectrum_gg(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    power_spectrum_gg = spectra.power_spectrum_gg
    assert isinstance(power_spectrum_gg, PowerSpectrumResult)


def test_spectra_power_spectrum_gm(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    power_spectrum_gm = spectra.power_spectrum_gm
    assert isinstance(power_spectrum_gm, PowerSpectrumResult)


def test_spectra_alignment_class(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    alignment_class = spectra.alignment_class
    assert alignment_class is not None


def test_spectra_beta_cen(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    beta_cen = spectra.beta_cen
    assert beta_cen is not None


def test_spectra_beta_sat(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    beta_sat = spectra.beta_sat
    assert beta_sat is not None


def test_spectra_mpivot_cen(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    mpivot_cen = spectra.mpivot_cen
    assert mpivot_cen is not None


def test_spectra_mpivot_sat(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    mpivot_sat = spectra.mpivot_sat
    assert mpivot_sat is not None


def test_spectra_alignment_gi(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    alignment_gi = spectra.alignment_gi
    assert alignment_gi is not None


def test_spectra_alignment_amplitude_2h(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    alignment_amplitude_2h = spectra.alignment_amplitude_2h
    assert alignment_amplitude_2h.shape == (len(z_vec), 1)


def test_spectra_alignment_amplitude_2h_II(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    alignment_amplitude_2h_II = spectra.alignment_amplitude_2h_II
    assert alignment_amplitude_2h_II.shape == (len(z_vec), 1)


def test_spectra_C1(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    C1 = spectra.C1
    assert C1.shape == (len(z_vec), 1, 1)


def test_spectra_wkm_sat(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    wkm_sat = spectra.wkm_sat
    assert wkm_sat.shape == (len(z_vec), len(mass), len(k_vec))


def test_spectra_central_alignment_profile(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    central_alignment_profile = spectra.central_alignment_profile
    assert central_alignment_profile.shape == (1, len(z_vec), 1, len(mass))


def test_spectra_satellite_alignment_profile(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    satellite_alignment_profile = spectra.satellite_alignment_profile
    assert satellite_alignment_profile.shape == (1, len(z_vec), len(k_vec), len(mass))


def test_spectra_Ic_align_term(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    Ic_align_term = spectra.Ic_align_term
    assert Ic_align_term.shape == (1, len(z_vec), 1)


def test_spectra_Is_align_term(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    Is_align_term = spectra.Is_align_term
    assert Is_align_term.shape == (1, len(z_vec), len(k_vec))


def test_spectra_power_spectrum_mi(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    power_spectrum_mi = spectra.power_spectrum_mi
    power_spectrum_mi = spectra.power_spectrum_mi
    assert isinstance(power_spectrum_mi, PowerSpectrumResult)


def test_spectra_power_spectrum_ii(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    power_spectrum_ii = spectra.power_spectrum_ii
    assert isinstance(power_spectrum_ii, PowerSpectrumResult)


def test_spectra_power_spectrum_gi(setup_data, spectra):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    power_spectrum_gi = spectra.power_spectrum_gi
    assert isinstance(power_spectrum_gi, PowerSpectrumResult)
