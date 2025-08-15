import pytest
import numpy as np
from unittest.mock import MagicMock
from onepower import (
    HaloOccupationDistribution,
    Cacciato,
    Simple,
    Zehavi,
    Zheng,
    Zhai,
    load_data,
)


@pytest.fixture
def setup_data():
    rng = np.random.default_rng(seed=42)
    mass = np.logspace(12, 15, 100)
    dndlnm = rng.random((15, 100))
    halo_bias = rng.random((15, 100))
    z_vec = np.linspace(0, 3, 15)
    cosmo = MagicMock()
    cosmo.h = 0.7
    hod_settings = {
        'observables_file': None,
        'obs_min': np.atleast_1d(8.0),
        'obs_max': np.atleast_1d(12.0),
        'zmin': np.atleast_1d(0.0),
        'zmax': np.atleast_1d(0.2),
        'nz': 15,
        'nobs': 300,
        'observable_h_unit': '1/h^2',
    }
    return mass, dndlnm, halo_bias, z_vec, cosmo, hod_settings


def test_load_data(datadir):
    file_name = f'{datadir}/test_data.txt'
    data = np.array([[0.1, 8.0, 12.0], [0.2, 8.5, 12.5]])
    np.savetxt(file_name, data)
    z_data, obs_min, obs_max = load_data(file_name)
    assert np.allclose(z_data, data[:, 0])
    assert np.allclose(obs_min, data[:, 1])
    assert np.allclose(obs_max, data[:, 2])


def test_hod_initialization_and_quantities(setup_data):
    mass, dndlnm, halo_bias, z_vec, cosmo, hod_settings = setup_data
    hod = HaloOccupationDistribution(
        cosmo=cosmo,
        mass=mass,
        dndlnm=dndlnm,
        halo_bias=halo_bias,
        z_vec=z_vec,
        hod_settings=hod_settings,
    )

    assert hod.cosmo == cosmo
    assert hod.mass.shape == (1, 1, len(mass))
    assert np.allclose(hod.z_vec, z_vec)
    assert hod.dndlnm.shape == dndlnm.shape
    assert hod.halo_bias.shape == halo_bias.shape

    with pytest.raises(NotImplementedError):
        hod._compute_hod_cen
    with pytest.raises(NotImplementedError):
        hod._compute_hod_sat

    obs = hod.obs
    assert obs.shape == (hod.nbins, hod.nz, 1, hod.nobs)

    dndlnm_int = hod.dndlnm_int
    assert dndlnm_int.shape == (1, hod.nz, len(mass))

    halo_bias_int = hod.halo_bias_int
    assert halo_bias_int.shape == (1, hod.nz, len(mass))

    assert hod.data is None
    assert hod.nobs == hod_settings['nobs']
    assert hod.nbins == len(hod_settings['obs_min'])
    assert hod.nz == hod_settings['nz']

    z = hod.z
    assert z.shape == (hod.nbins, hod.nz)

    log_obs_min = hod.log_obs_min
    assert log_obs_min.shape == (hod.nbins, hod.nz)

    log_obs_max = hod.log_obs_max
    assert log_obs_max.shape == (hod.nbins, hod.nz)

    rng = np.random.default_rng(seed=42)
    data = rng.random((hod.nbins, hod.nz))
    interpolated_data = hod._interpolate(data)
    assert interpolated_data.shape == (hod.nbins, len(z_vec))


def test_hod_quantities_with_file(setup_data, datadir):
    mass, dndlnm, halo_bias, z_vec, cosmo, hod_settings = setup_data
    file_name = f'{datadir}/test_data.txt'
    data = np.array([[0.1, 8.0, 12.0], [0.2, 8.5, 12.5]])
    np.savetxt(file_name, data)
    hod_settings['observables_file'] = file_name

    hod = HaloOccupationDistribution(
        cosmo=cosmo,
        mass=mass,
        dndlnm=dndlnm,
        halo_bias=halo_bias,
        z_vec=z_vec,
        hod_settings=hod_settings,
    )

    obs = hod.obs
    assert obs.shape == (hod.nbins, hod.nz, 1, hod.nobs)

    dndlnm_int = hod.dndlnm_int
    assert dndlnm_int.shape == (1, hod.nz, len(mass))

    halo_bias_int = hod.halo_bias_int
    assert halo_bias_int.shape == (1, hod.nz, len(mass))

    assert hod.nobs == hod_settings['nobs']
    assert hod.nbins == len(hod_settings['obs_min'])
    assert hod.nz == len(data)

    z = hod.z
    assert z.shape == (hod.nbins, hod.nz)

    log_obs_min = hod.log_obs_min
    assert log_obs_min.shape == (hod.nbins, hod.nz)

    log_obs_max = hod.log_obs_max
    assert log_obs_max.shape == (hod.nbins, hod.nz)

    rng = np.random.default_rng(seed=42)
    data = rng.random((hod.nbins, hod.nz))
    interpolated_data = hod._interpolate(data)
    assert interpolated_data.shape == (hod.nbins, len(z_vec))


def test_cacciato_properties(setup_data):
    mass, dndlnm, halo_bias, z_vec, cosmo, hod_settings = setup_data
    cacciato = Cacciato(
        cosmo=cosmo,
        mass=mass,
        dndlnm=dndlnm,
        halo_bias=halo_bias,
        z_vec=z_vec,
        hod_settings=hod_settings,
        A_cen=0.0,
        A_sat=0.0,
    )

    assert isinstance(cacciato, Cacciato)
    assert cacciato.Obs_norm_c == 10.0**9.95
    assert cacciato.M_char == 10.0**11.24

    assert cacciato.number_density.shape == (cacciato.nbins, cacciato.nz)
    assert cacciato.number_density_cen.shape == (cacciato.nbins, cacciato.nz)
    assert cacciato.number_density_sat.shape == (cacciato.nbins, cacciato.nz)
    assert cacciato.avg_halo_mass.shape == (cacciato.nbins, cacciato.nz)
    assert cacciato.avg_halo_mass_cen.shape == (cacciato.nbins, cacciato.nz)
    assert cacciato.avg_halo_mass_sat.shape == (cacciato.nbins, cacciato.nz)
    assert cacciato.galaxy_linear_bias.shape == (cacciato.nbins, cacciato.nz)
    assert cacciato.galaxy_linear_bias_cen.shape == (cacciato.nbins, cacciato.nz)
    assert cacciato.galaxy_linear_bias_sat.shape == (cacciato.nbins, cacciato.nz)
    assert cacciato.f_c.shape == (cacciato.nbins, cacciato.nz)
    assert cacciato.f_s.shape == (cacciato.nbins, cacciato.nz)
    assert cacciato.hod.shape == (cacciato.nbins, cacciato.nz, mass.size)
    assert cacciato.hod_cen.shape == (cacciato.nbins, cacciato.nz, mass.size)
    assert cacciato.hod_sat.shape == (cacciato.nbins, cacciato.nz, mass.size)
    assert cacciato.stellar_fraction.shape == (cacciato.nbins, cacciato.nz, mass.size)
    assert cacciato.stellar_fraction_cen.shape == (
        cacciato.nbins,
        cacciato.nz,
        mass.size,
    )
    assert cacciato.stellar_fraction_sat.shape == (
        cacciato.nbins,
        cacciato.nz,
        mass.size,
    )
    assert cacciato.obs_func.shape == (cacciato.nbins, cacciato.nz, cacciato.nobs)
    assert cacciato.obs_func_cen.shape == (cacciato.nbins, cacciato.nz, cacciato.nobs)
    assert cacciato.obs_func_sat.shape == (cacciato.nbins, cacciato.nz, cacciato.nobs)


def test_simple_properties(setup_data):
    mass, dndlnm, halo_bias, z_vec, cosmo, hod_settings = setup_data
    simple = Simple(
        cosmo=cosmo,
        mass=mass,
        dndlnm=dndlnm,
        halo_bias=halo_bias,
        z_vec=z_vec,
        hod_settings=hod_settings,
        A_cen=0.0,
        A_sat=0.0,
    )

    assert isinstance(simple, Simple)
    assert simple.log10_Mmin == 12.0
    assert simple.log10_Msat == 13.0
    assert simple.alpha == 1.0
    assert simple._compute_hod_cen.shape == (1, z_vec.size, mass.size)
    assert simple._compute_hod_sat.shape == (1, z_vec.size, mass.size)
    assert np.allclose(
        simple.stellar_fraction, np.zeros((simple.nbins, simple.nz, mass.size))
    )
    assert np.allclose(
        simple.stellar_fraction_cen, np.zeros((simple.nbins, simple.nz, mass.size))
    )
    assert np.allclose(
        simple.stellar_fraction_sat, np.zeros((simple.nbins, simple.nz, mass.size))
    )


def test_zehavi_properties(setup_data):
    mass, dndlnm, halo_bias, z_vec, cosmo, hod_settings = setup_data
    zehavi = Zehavi(
        cosmo=cosmo,
        mass=mass,
        dndlnm=dndlnm,
        halo_bias=halo_bias,
        z_vec=z_vec,
        hod_settings=hod_settings,
        A_cen=0.0,
        A_sat=0.0,
    )

    assert isinstance(zehavi, Zehavi)
    assert zehavi.log10_Mmin == 12.0
    assert zehavi.log10_Msat == 13.0
    assert zehavi.alpha == 1.0
    assert zehavi._compute_hod_cen.shape == (1, z_vec.size, mass.size)
    assert zehavi._compute_hod_sat.shape == (1, z_vec.size, mass.size)


def test_zheng_properties(setup_data):
    mass, dndlnm, halo_bias, z_vec, cosmo, hod_settings = setup_data
    zheng = Zheng(
        cosmo=cosmo,
        mass=mass,
        dndlnm=dndlnm,
        halo_bias=halo_bias,
        z_vec=z_vec,
        hod_settings=hod_settings,
        A_cen=0.0,
        A_sat=0.0,
    )

    assert isinstance(zheng, Zheng)
    assert zheng.log10_Mmin == 12.0
    assert zheng.log10_M0 == 12.0
    assert zheng.log10_M1 == 13.0
    assert zheng.alpha == 1.0
    assert zheng.sigma == 0.15
    assert zheng._compute_hod_cen.shape == (1, z_vec.size, mass.size)
    assert zheng._compute_hod_sat.shape == (1, z_vec.size, mass.size)


def test_zhai_properties(setup_data):
    mass, dndlnm, halo_bias, z_vec, cosmo, hod_settings = setup_data
    zhai = Zhai(
        cosmo=cosmo,
        mass=mass,
        dndlnm=dndlnm,
        halo_bias=halo_bias,
        z_vec=z_vec,
        hod_settings=hod_settings,
        A_cen=0.0,
        A_sat=0.0,
    )

    assert isinstance(zhai, Zhai)
    assert zhai.log10_Mmin == 13.68
    assert zhai.log10_Msat == 14.87
    assert zhai.log10_Mcut == 12.32
    assert zhai.alpha == 0.41
    assert zhai.sigma == 0.82
    assert zhai._compute_hod_cen.shape == (1, z_vec.size, mass.size)
    assert zhai._compute_hod_sat.shape == (1, z_vec.size, mass.size)
