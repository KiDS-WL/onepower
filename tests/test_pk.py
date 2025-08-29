import pytest
import numpy as np
from onepower import PowerSpectrumResult, Spectra


@pytest.fixture
def setup_data():
    rng = np.random.default_rng(seed=42)
    k_vec = np.logspace(-4, 4, 100)
    z_vec = np.linspace(0.0, 3.0, 15)
    mass = np.logspace(12, 15, 100)
    dndlnm = rng.random((15, 100))
    halo_bias = rng.random((15, 100))
    matter_power_lin = rng.random((15, 100))
    matter_power_nl = rng.random((15, 100))
    return k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl


@pytest.fixture
def spectra(setup_data):
    k_vec, z_vec, mass, dndlnm, halo_bias, matter_power_lin, matter_power_nl = (
        setup_data
    )
    return Spectra(
        Mmin=12,
        Mmax=15,
        dlog10m=(15 - 12) / 100,
        matter_power_lin=matter_power_lin,
        matter_power_nl=matter_power_nl,
    )


def test_spectra_initialization_and_params_none(setup_data):
    _, _, _, _, _, matter_power_lin, matter_power_nl = setup_data
    spectra = Spectra(
        matter_power_lin=matter_power_lin, matter_power_nl=matter_power_nl
    )
    assert spectra.matter_power_lin.shape == matter_power_lin.shape
    assert spectra.matter_power_nl.shape == matter_power_nl.shape

    spectra_none = Spectra(hod_model=None, poisson_model=None)
    assert spectra_none.hod_model is None
    assert spectra_none.poisson_model is None


def test_spectra_properties_and_zvec_kvec_shape(setup_data, spectra):
    k_vec, z_vec, mass, _, _, matter_power_lin, matter_power_nl = setup_data

    assert spectra._beta_nl_array is None
    assert spectra._pk_lin.shape == (len(z_vec), len(k_vec))
    assert spectra._pk_nl.shape == (len(z_vec), len(k_vec))
    assert spectra.peff is None

    spectra_zvec_kvec = Spectra(
        k_vec=k_vec[:-2],
        z_vec=z_vec[:-2],
        matter_power_lin=matter_power_lin,
        matter_power_nl=matter_power_nl,
        fortuna=True,
        response=True,
    )
    with pytest.raises(ValueError):
        spectra_zvec_kvec._pk_lin
    with pytest.raises(ValueError):
        spectra_zvec_kvec._pk_nl


def test_spectra_bnl_related_properties(setup_data, spectra):
    k_vec, _, mass, _, _, _, _ = setup_data

    spectra.update(bnl=True)
    assert spectra.calc_bnl.shape == (1, len(mass), len(mass), len(k_vec))
    assert spectra.I12 is not None
    assert spectra.I21 is not None
    assert spectra.I22 is not None
    assert isinstance(spectra.power_spectrum_mm, PowerSpectrumResult)
    assert isinstance(spectra.power_spectrum_gg, PowerSpectrumResult)
    assert isinstance(spectra.power_spectrum_gm, PowerSpectrumResult)
    assert isinstance(spectra.power_spectrum_mi, PowerSpectrumResult)
    assert isinstance(spectra.power_spectrum_ii, PowerSpectrumResult)
    assert isinstance(spectra.power_spectrum_gi, PowerSpectrumResult)

    spectra.update(bnl=False)
    assert spectra.I12 is None
    assert spectra.I21 is None
    assert spectra.I22 is None


def test_spectra_hod_and_fstar_matter(setup_data, spectra):
    _, _, mass, _, _, _, _ = setup_data

    assert spectra.hod_mm is None
    assert spectra.fstar_mm.shape == (1, len(spectra.z_vec), len(mass))


def test_spectra_matter_profiles_and_truncation_properties(setup_data, spectra):
    _, z_vec, mass, _, _, _, _ = setup_data

    assert spectra.matter_profile.shape == (
        1,
        len(z_vec),
        len(spectra.k_vec),
        len(mass),
    )
    assert spectra.matter_profile_2h.shape == (
        1,
        len(z_vec),
        len(spectra.k_vec),
        len(mass),
    )

    k_vec, _, _, _, _, _, _ = setup_data
    assert spectra.one_halo_truncation.shape == (len(k_vec),)
    assert spectra.two_halo_truncation.shape == (len(k_vec),)
    assert spectra.one_halo_truncation_ia.shape == (len(k_vec),)
    assert spectra.two_halo_truncation_ia.shape == (len(k_vec),)

    spectra.update(
        one_halo_ktrunc=None,
        two_halo_ktrunc=None,
        one_halo_ktrunc_ia=None,
        two_halo_ktrunc_ia=None,
    )
    assert np.allclose(spectra.one_halo_truncation, np.ones_like(k_vec))
    assert np.allclose(spectra.two_halo_truncation, np.ones_like(k_vec))
    assert np.allclose(spectra.one_halo_truncation_ia, np.ones_like(k_vec))
    assert np.allclose(spectra.two_halo_truncation_ia, np.ones_like(k_vec))


def test_spectra_power_spectrum_properties_and_hod_observable_properties(
    setup_data, spectra
):
    _, z_vec, mass, _, _, _, _ = setup_data

    assert isinstance(spectra.power_spectrum_lin, PowerSpectrumResult)
    assert isinstance(spectra.power_spectrum_mm, PowerSpectrumResult)

    assert spectra.hod is not None
    assert spectra.fstar.shape == (1, len(z_vec), len(mass))
    assert spectra.mass_avg.shape == (1, len(z_vec))

    spectra.update(compute_observable=True)
    assert spectra.obs is not None
    assert spectra.obs_func is not None
    spectra.update(compute_observable=False)


def test_spectra_galaxy_profiles_and_terms_and_power_spectra(setup_data, spectra):
    _, z_vec, mass, _, _, _, _ = setup_data

    assert spectra.central_galaxy_profile.shape == (
        1,
        len(z_vec),
        len(spectra.k_vec),
        len(mass),
    )
    assert spectra.satellite_galaxy_profile.shape == (
        1,
        len(z_vec),
        len(spectra.k_vec),
        len(mass),
    )

    assert spectra.Ic_term.shape == (1, len(z_vec), len(spectra.k_vec))
    assert spectra.Is_term.shape == (1, len(z_vec), len(spectra.k_vec))
    assert isinstance(spectra.power_spectrum_gg, PowerSpectrumResult)
    assert isinstance(spectra.power_spectrum_gm, PowerSpectrumResult)


def test_spectra_alignment_properties_and_profiles_and_terms(setup_data, spectra):
    _, z_vec, mass, _, _, _, _ = setup_data

    assert spectra.alignment_class is not None
    assert spectra.beta_cen is not None
    assert spectra.beta_sat is not None
    assert spectra.mpivot_cen is not None
    assert spectra.mpivot_sat is not None
    assert spectra.alignment_gi is not None
    assert spectra.alignment_amplitude_2h.shape == (len(z_vec), 1)
    assert spectra.alignment_amplitude_2h_II.shape == (len(z_vec), 1)
    assert spectra.C1.shape == (len(z_vec), 1, 1)

    assert spectra.wkm_sat.shape == (1, len(z_vec), len(mass), len(spectra.k_vec))
    assert spectra.central_alignment_profile.shape == (1, len(z_vec), 1, len(mass))
    assert spectra.satellite_alignment_profile.shape == (
        1,
        len(z_vec),
        len(spectra.k_vec),
        len(mass),
    )
    assert spectra.Ic_align_term.shape == (1, len(z_vec), 1)
    assert spectra.Is_align_term.shape == (1, len(z_vec), len(spectra.k_vec))


def test_spectra_alignment_power_spectra_and_mead(setup_data):
    _, _, mass, _, _, _, _ = setup_data

    spectra_mead = Spectra(mead_correction='nofeedback')
    assert isinstance(spectra_mead.power_spectrum_mm, PowerSpectrumResult)
    assert isinstance(spectra_mead.power_spectrum_gg, PowerSpectrumResult)
    assert isinstance(spectra_mead.power_spectrum_gm, PowerSpectrumResult)
    assert isinstance(spectra_mead.power_spectrum_mi, PowerSpectrumResult)
    assert isinstance(spectra_mead.power_spectrum_ii, PowerSpectrumResult)
    assert isinstance(spectra_mead.power_spectrum_gi, PowerSpectrumResult)

    spectra_mead = Spectra(mead_correction='feedback')
    assert isinstance(spectra_mead.power_spectrum_mm, PowerSpectrumResult)
    assert isinstance(spectra_mead.power_spectrum_gg, PowerSpectrumResult)
    assert isinstance(spectra_mead.power_spectrum_gm, PowerSpectrumResult)
    assert isinstance(spectra_mead.power_spectrum_mi, PowerSpectrumResult)
    assert isinstance(spectra_mead.power_spectrum_ii, PowerSpectrumResult)
    assert isinstance(spectra_mead.power_spectrum_gi, PowerSpectrumResult)

    spectra_mead = Spectra(
        mead_correction='fit', Mmin=12, Mmax=15, dlog10m=(15 - 12) / 100
    )
    assert spectra_mead.hod_mm is not None
    assert spectra_mead.fstar_mm.shape == (1, len(spectra_mead.z_vec), len(mass))
    assert isinstance(spectra_mead.power_spectrum_mm, PowerSpectrumResult)
    assert isinstance(spectra_mead.power_spectrum_gm, PowerSpectrumResult)
    assert isinstance(spectra_mead.power_spectrum_mi, PowerSpectrumResult)
    assert isinstance(spectra_mead.power_spectrum_ii, PowerSpectrumResult)
    assert isinstance(spectra_mead.power_spectrum_gi, PowerSpectrumResult)


def test_fortuna_and_response(setup_data):
    k_vec, z_vec, mass, _, _, matter_power_lin, matter_power_nl = setup_data
    spectra = Spectra(
        k_vec=k_vec,
        z_vec=z_vec,
        matter_power_lin=matter_power_lin,
        matter_power_nl=matter_power_nl,
        fortuna=True,
        response=True,
    )

    assert spectra._beta_nl_array is None
    assert spectra._pk_lin.shape == (len(z_vec), len(k_vec))
    assert spectra._pk_nl.shape == (len(z_vec), len(k_vec))
    assert spectra.peff is not None
    assert isinstance(spectra.power_spectrum_mm, PowerSpectrumResult)
    assert isinstance(spectra.power_spectrum_gg, PowerSpectrumResult)
    assert isinstance(spectra.power_spectrum_gm, PowerSpectrumResult)
    assert isinstance(spectra.power_spectrum_mi, PowerSpectrumResult)
    assert isinstance(spectra.power_spectrum_ii, PowerSpectrumResult)
    assert isinstance(spectra.power_spectrum_gi, PowerSpectrumResult)
