import pytest

import numpy as np
from unittest.mock import MagicMock

from onepower import HOD, Cacciato, Simple, Zehavi, Zhai, Zheng, load_data

# Just some base tests for now, no specific scientific calculations tested yet!


@pytest.fixture
def setup_data():
    mass = np.logspace(12, 15, 100)
    dndlnm = np.random.rand(15, 100)
    halo_bias = np.random.rand(15, 100)
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


def test_load_data(tmp_path):
    file_name = tmp_path / "test_data.txt"
    data = np.array([[0.1, 8.0, 12.0], [0.2, 8.5, 12.5]])
    np.savetxt(file_name, data)
    z_data, obs_min, obs_max = load_data(file_name)
    assert np.allclose(z_data, data[:, 0])
    assert np.allclose(obs_min, data[:, 1])
    assert np.allclose(obs_max, data[:, 2])


def test_hod_initialization(setup_data):
    mass, dndlnm, halo_bias, z_vec, cosmo, hod_settings = setup_data
    hod = HOD(
        cosmo=cosmo,
        mass=mass,
        dndlnm=dndlnm,
        halo_bias=halo_bias,
        z_vec=z_vec,
        hod_settings=hod_settings,
    )
    assert hod.cosmo == cosmo
    assert hod.mass.shape == (1, 1, len(mass))
    assert hod.z_vec.all() == z_vec.all()
    assert hod.dndlnm.shape == dndlnm.shape
    assert hod.halo_bias.shape == halo_bias.shape


def test_hod_obs(setup_data):
    mass, dndlnm, halo_bias, z_vec, cosmo, hod_settings = setup_data
    hod = HOD(
        cosmo=cosmo,
        mass=mass,
        dndlnm=dndlnm,
        halo_bias=halo_bias,
        z_vec=z_vec,
        hod_settings=hod_settings,
    )
    obs = hod.obs
    assert obs.shape == (hod.nbins, hod.nz, 1, hod.nobs)


def test_hod_dndlnm_int(setup_data):
    mass, dndlnm, halo_bias, z_vec, cosmo, hod_settings = setup_data
    hod = HOD(
        cosmo=cosmo,
        mass=mass,
        dndlnm=dndlnm,
        halo_bias=halo_bias,
        z_vec=z_vec,
        hod_settings=hod_settings,
    )
    dndlnm_int = hod.dndlnm_int
    assert dndlnm_int.shape == (1, hod.nz, len(mass))


def test_hod_halo_bias_int(setup_data):
    mass, dndlnm, halo_bias, z_vec, cosmo, hod_settings = setup_data
    hod = HOD(
        cosmo=cosmo,
        mass=mass,
        dndlnm=dndlnm,
        halo_bias=halo_bias,
        z_vec=z_vec,
        hod_settings=hod_settings,
    )
    halo_bias_int = hod.halo_bias_int
    assert halo_bias_int.shape == (1, hod.nz, len(mass))


def test_hod_data(setup_data):
    mass, dndlnm, halo_bias, z_vec, cosmo, hod_settings = setup_data
    hod = HOD(
        cosmo=cosmo,
        mass=mass,
        dndlnm=dndlnm,
        halo_bias=halo_bias,
        z_vec=z_vec,
        hod_settings=hod_settings,
    )
    data = hod.data
    assert data is None


def test_hod_nobs(setup_data):
    mass, dndlnm, halo_bias, z_vec, cosmo, hod_settings = setup_data
    hod = HOD(
        cosmo=cosmo,
        mass=mass,
        dndlnm=dndlnm,
        halo_bias=halo_bias,
        z_vec=z_vec,
        hod_settings=hod_settings,
    )
    nobs = hod.nobs
    assert nobs == hod_settings['nobs']


def test_hod_nbins(setup_data):
    mass, dndlnm, halo_bias, z_vec, cosmo, hod_settings = setup_data
    hod = HOD(
        cosmo=cosmo,
        mass=mass,
        dndlnm=dndlnm,
        halo_bias=halo_bias,
        z_vec=z_vec,
        hod_settings=hod_settings,
    )
    nbins = hod.nbins
    assert nbins == len(hod_settings['obs_min'])


def test_hod_nz(setup_data):
    mass, dndlnm, halo_bias, z_vec, cosmo, hod_settings = setup_data
    hod = HOD(
        cosmo=cosmo,
        mass=mass,
        dndlnm=dndlnm,
        halo_bias=halo_bias,
        z_vec=z_vec,
        hod_settings=hod_settings,
    )
    nz = hod.nz
    assert nz == hod_settings['nz']


def test_hod_z(setup_data):
    mass, dndlnm, halo_bias, z_vec, cosmo, hod_settings = setup_data
    hod = HOD(
        cosmo=cosmo,
        mass=mass,
        dndlnm=dndlnm,
        halo_bias=halo_bias,
        z_vec=z_vec,
        hod_settings=hod_settings,
    )
    z = hod.z
    assert z.shape == (hod.nbins, hod.nz)


def test_hod_log_obs_min(setup_data):
    mass, dndlnm, halo_bias, z_vec, cosmo, hod_settings = setup_data
    hod = HOD(
        cosmo=cosmo,
        mass=mass,
        dndlnm=dndlnm,
        halo_bias=halo_bias,
        z_vec=z_vec,
        hod_settings=hod_settings,
    )
    log_obs_min = hod.log_obs_min
    assert log_obs_min.shape == (hod.nbins, hod.nz)


def test_hod_log_obs_max(setup_data):
    mass, dndlnm, halo_bias, z_vec, cosmo, hod_settings = setup_data
    hod = HOD(
        cosmo=cosmo,
        mass=mass,
        dndlnm=dndlnm,
        halo_bias=halo_bias,
        z_vec=z_vec,
        hod_settings=hod_settings,
    )
    log_obs_max = hod.log_obs_max
    assert log_obs_max.shape == (hod.nbins, hod.nz)


def test_hod_interpolate(setup_data):
    mass, dndlnm, halo_bias, z_vec, cosmo, hod_settings = setup_data
    hod = HOD(
        cosmo=cosmo,
        mass=mass,
        dndlnm=dndlnm,
        halo_bias=halo_bias,
        z_vec=z_vec,
        hod_settings=hod_settings,
    )
    data = np.random.rand(hod.nbins, hod.nz)
    interpolated_data = hod._interpolate(data)
    assert interpolated_data.shape == (hod.nbins, len(z_vec))


def test_cacciato_initialization(setup_data):
    mass, dndlnm, halo_bias, z_vec, cosmo, hod_settings = setup_data
    cacciato = Cacciato(
        cosmo=cosmo,
        mass=mass,
        dndlnm=dndlnm,
        halo_bias=halo_bias,
        z_vec=z_vec,
        hod_settings=hod_settings,
    )
    assert isinstance(cacciato, Cacciato)


def test_simple_initialization(setup_data):
    mass, dndlnm, halo_bias, z_vec, cosmo, hod_settings = setup_data
    simple = Simple(
        cosmo=cosmo,
        mass=mass,
        dndlnm=dndlnm,
        halo_bias=halo_bias,
        z_vec=z_vec,
        hod_settings=hod_settings,
    )
    assert isinstance(simple, Simple)


def test_zehavi_initialization(setup_data):
    mass, dndlnm, halo_bias, z_vec, cosmo, hod_settings = setup_data
    zehavi = Zehavi(
        cosmo=cosmo,
        mass=mass,
        dndlnm=dndlnm,
        halo_bias=halo_bias,
        z_vec=z_vec,
        hod_settings=hod_settings,
    )
    assert isinstance(zehavi, Zehavi)


def test_zheng_initialization(setup_data):
    mass, dndlnm, halo_bias, z_vec, cosmo, hod_settings = setup_data
    zheng = Zheng(
        cosmo=cosmo,
        mass=mass,
        dndlnm=dndlnm,
        halo_bias=halo_bias,
        z_vec=z_vec,
        hod_settings=hod_settings,
    )
    assert isinstance(zheng, Zheng)


def test_zhai_initialization(setup_data):
    mass, dndlnm, halo_bias, z_vec, cosmo, hod_settings = setup_data
    zhai = Zhai(
        cosmo=cosmo,
        mass=mass,
        dndlnm=dndlnm,
        halo_bias=halo_bias,
        z_vec=z_vec,
        hod_settings=hod_settings,
    )
    assert isinstance(zhai, Zhai)
