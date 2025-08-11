import pytest

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from onepower import CosmologyBase, HaloModelIngredients
from onepower.hmi import SOVirial_Mead

# Just some base tests for now, no specific scientific calculations tested yet!

# Helper tolerance for float comparisons
TOL = 1e-5


@pytest.fixture
def setup_data():
    k_vec = np.logspace(-4, 4, 100)
    z_vec = np.linspace(0.0, 3.0, 15)
    return k_vec, z_vec


@pytest.fixture
def ingredients(setup_data):
    k_vec, z_vec = setup_data
    return HaloModelIngredients(k_vec=k_vec, z_vec=z_vec)


def test_SOVirial_Mead_halo_density_and_str():
    sov = SOVirial_Mead()
    dens = sov.halo_density(z=0)
    assert np.isclose(dens, sov.params["overdensity"] * sov.mean_density(0))
    assert sov.colossus_name == "200c"
    assert str(sov) == "SOVirial"


def test_cosmology_base_initialization(setup_data):
    k_vec, z_vec = setup_data
    cosmo = CosmologyBase(z_vec=z_vec)
    assert cosmo.z_vec.all() == z_vec.all()
    assert cosmo.h0 == 0.7
    assert cosmo.omega_c == 0.25
    assert cosmo.omega_b == 0.05
    assert cosmo.omega_m == 0.3
    assert cosmo.w0 == -1.0
    assert cosmo.wa == 0.0
    assert cosmo.n_s == 0.9
    assert cosmo.tcmb == 2.7255
    assert cosmo.m_nu == 0.06
    assert cosmo.sigma_8 == 0.8
    assert cosmo.log10T_AGN == 7.8


def test_cosmo_model(setup_data):
    cosmo = CosmologyBase()
    cosmo_model = cosmo.cosmo_model
    assert hasattr(cosmo_model, 'H0')
    assert hasattr(cosmo_model, 'Om0')
    assert hasattr(cosmo_model, 'Ob0')
    assert hasattr(cosmo_model, 'Tcmb0')
    assert hasattr(cosmo_model, 'w0')
    assert hasattr(cosmo_model, 'wa')


def test_scale_factor_consistency():
    cosmo = CosmologyBase()
    a = cosmo.scale_factor
    assert np.isclose(a[0], cosmo.cosmo_model.scale_factor(cosmo.z_vec[0]))
    assert len(a) == len(cosmo.z_vec)
    assert a[-1] < a[0]


@pytest.mark.parametrize("a,Om,Ode,Ok", [(0.5, 0.3, 0.7, 0), (1.0, 0.3, 0.7, 0)])
def test_Hubble_and_Omega_functions(a, Om, Ode, Ok):
    cosmo = CosmologyBase()
    H2 = cosmo._Hubble2(a, Om, Ode, Ok)
    Omega_m = cosmo._Omega_m(a, Om, Ode, Ok)
    AH = cosmo._AH(a, Om, Ode)

    assert H2 > 0
    assert 0 <= Omega_m <= 1
    assert np.isfinite(AH)


def test_get_mead_growth_fnc_and_growth():
    cosmo = CosmologyBase()
    g_fnc = cosmo.get_mead_growth_fnc
    # Should be an interp1d function
    assert isinstance(g_fnc, interp1d)

    a = np.linspace(0.001, 1, 10)
    g_vals = g_fnc(a)
    assert np.all(np.diff(g_vals) >= -1e-8)

    growth = cosmo.get_mead_growth
    assert growth.shape == cosmo.scale_factor.shape


def test_get_mead_accumulated_growth_integration():
    cosmo = CosmologyBase()
    G = cosmo.get_mead_accumulated_growth
    assert G.shape == cosmo.scale_factor.shape
    assert np.all(np.diff(G) < 0)
    a0 = cosmo.scale_factor[0]
    val, err = quad(lambda a: cosmo.get_mead_growth_fnc(a) / a, 1e-4, a0)
    assert np.isclose(G[0], val + cosmo.get_mead_growth_fnc(1e-4), atol=1e-4)


@pytest.mark.parametrize(
    "x,y,p0,p1,p2,p3",
    [
        (0.5, 0.5, 1, 2, 3, 4),
        (1.0, 0.0, 0, -1, 0.5, 0),
        (0.0, 1.0, -1, 0, 0, 1),
    ],
)
def test_f_Mead_function(x, y, p0, p1, p2, p3):
    cosmo = CosmologyBase()
    result = cosmo.f_Mead(x, y, p0, p1, p2, p3)
    expected = p0 + p1 * (1 - x) + p2 * (1 - x) ** 2 + p3 * (1 - y)
    assert np.isclose(result, expected)


def test_dc_Mead_and_Dv_Mead_values():
    cosmo = CosmologyBase()
    dc = cosmo.dc_Mead
    Dv = cosmo.Dv_Mead

    assert dc.shape == cosmo.z_vec.shape
    assert Dv.shape == cosmo.z_vec.shape
    assert np.all((dc > 1.5) & (dc < 1.8))
    assert np.all(Dv > 100)
    assert np.all(Dv < 400)


def test_halo_model_ingredients_initialization(setup_data):
    k_vec, z_vec = setup_data
    ingredients = HaloModelIngredients(k_vec=k_vec)
    assert ingredients.k_vec.all() == k_vec.all()
    assert ingredients.lnk_min == np.log(10 ** (-4.0))
    assert ingredients.lnk_max == np.log(10 ** (4.0))
    assert ingredients.dlnk == (np.log(10 ** (4.0)) - np.log(10 ** (-4.0))) / 100
    assert ingredients.Mmin == 9.0
    assert ingredients.Mmax == 16.0
    assert ingredients.dlog10m == 0.05
    assert ingredients.mdef_model == 'SOMean'
    assert ingredients.hmf_model == 'Tinker10'
    assert ingredients.bias_model == 'Tinker10'
    assert ingredients.halo_profile_model_dm == 'NFW'
    assert ingredients.halo_concentration_model_dm == 'Duffy08'
    assert ingredients.halo_profile_model_sat == 'NFW'
    assert ingredients.halo_concentration_model_sat == 'Duffy08'
    assert ingredients.transfer_model == 'CAMB'
    assert ingredients.transfer_params == {}
    assert ingredients.growth_model == 'CambGrowth'
    assert ingredients.growth_params == {}
    assert ingredients.norm_cen == 1.0
    assert ingredients.norm_sat == 1.0
    assert ingredients.eta_cen == 0.0
    assert ingredients.eta_sat == 0.0
    assert ingredients.overdensity == 200
    assert ingredients.delta_c == 1.686


# For the repacked input/outputs from hmf/halomod we only check if the
# inputs and outputs are in the correct format / shape!
# We trust the authors of hmf/halomod that the scientific accuracy of
# returned quantities is checked!
def test_halo_model_ingredients_norm_c(setup_data, ingredients):
    k_vec, z_vec = setup_data
    norm_c = ingredients._norm_c
    assert norm_c.shape == z_vec.shape


def test_halo_model_ingredients_norm_s(setup_data, ingredients):
    k_vec, z_vec = setup_data
    norm_s = ingredients._norm_s
    assert norm_s.shape == z_vec.shape


def test_halo_model_ingredients_eta_c(setup_data, ingredients):
    k_vec, z_vec = setup_data
    eta_c = ingredients._eta_c
    assert eta_c.shape == z_vec.shape


def test_halo_model_ingredients_eta_s(setup_data, ingredients):
    k_vec, z_vec = setup_data
    eta_s = ingredients._eta_s
    assert eta_s.shape == z_vec.shape


def test_halo_model_ingredients_delta_c_mod(setup_data, ingredients):
    k_vec, z_vec = setup_data
    delta_c_mod = ingredients._delta_c_mod
    ingredients.update(mead_correction='feedback')
    delta_c_mod_mead = ingredients._delta_c_mod
    assert delta_c_mod.shape == z_vec.shape
    assert delta_c_mod_mead.shape == z_vec.shape
    assert delta_c_mod_mead.all() == ingredients.dc_Mead.all()
    ingredients.update(mead_correction=None)


def test_halo_model_ingredients_mdef_mod(setup_data, ingredients):
    k_vec, z_vec = setup_data
    mdef_mod = ingredients._mdef_mod
    ingredients.update(mead_correction='feedback')
    mdef_mod_mead = ingredients._mdef_mod
    assert isinstance(mdef_mod, str)
    assert isinstance(mdef_mod_mead, type(SOVirial_Mead))
    ingredients.update(mead_correction=None)


def test_halo_model_ingredients_hmf_mod(setup_data, ingredients):
    k_vec, z_vec = setup_data
    hmf_mod = ingredients._hmf_mod
    ingredients.update(mead_correction='feedback')
    hmf_mod_mead = ingredients._hmf_mod
    assert isinstance(hmf_mod, str)
    assert isinstance(hmf_mod_mead, str)
    assert hmf_mod_mead == 'ST'
    ingredients.update(mead_correction=None)


def test_halo_model_ingredients_bias_mod(setup_data, ingredients):
    k_vec, z_vec = setup_data
    bias_mod = ingredients._bias_mod
    ingredients.update(mead_correction='feedback')
    bias_mod_mead = ingredients._bias_mod
    assert isinstance(bias_mod, str)
    assert isinstance(bias_mod_mead, str)
    assert bias_mod_mead == 'ST99'
    ingredients.update(mead_correction=None)


def test_halo_model_ingredients_halo_concentration_mod_dm(setup_data, ingredients):
    k_vec, z_vec = setup_data
    halo_concentration_mod_dm = ingredients._halo_concentration_mod_dm
    assert callable(halo_concentration_mod_dm)
    ingredients.update(halo_concentration_model_dm='diemer19')
    halo_concentration_mod_dm = ingredients._halo_concentration_mod_dm
    assert callable(halo_concentration_mod_dm)

    ingredients.update(
        halo_concentration_model_dm='Duffy08', mead_correction='feedback'
    )
    halo_concentration_mod_dm_mead = ingredients._halo_concentration_mod_dm
    assert callable(halo_concentration_mod_dm_mead)
    ingredients.update(mead_correction=None)


def test_halo_model_ingredients_halo_concentration_mod_sat(setup_data, ingredients):
    k_vec, z_vec = setup_data
    halo_concentration_mod_sat = ingredients._halo_concentration_mod_sat
    assert callable(halo_concentration_mod_sat)
    ingredients.update(halo_concentration_model_sat='diemer19')

    halo_concentration_mod_sat = ingredients._halo_concentration_mod_sat
    assert callable(halo_concentration_mod_sat)
    ingredients.update(halo_concentration_model_sat='Duffy08')


def test_halo_model_ingredients_mdef_params(setup_data, ingredients):
    k_vec, z_vec = setup_data
    mdef_params = ingredients.mdef_params
    assert isinstance(mdef_params, list)


def test_halo_model_ingredients_halo_profile_params(setup_data, ingredients):
    k_vec, z_vec = setup_data
    halo_profile_params = ingredients.halo_profile_params
    assert isinstance(halo_profile_params, dict)


def test_halo_model_ingredients_scale_factor(setup_data, ingredients):
    k_vec, z_vec = setup_data
    scale_factor = ingredients.scale_factor
    assert isinstance(scale_factor, np.ndarray)


def test_halo_model_ingredients_disable_mass_conversion(setup_data, ingredients):
    k_vec, z_vec = setup_data
    disable_mass_conversion = ingredients.disable_mass_conversion
    ingredients.update(mead_correction='feedback')
    assert isinstance(disable_mass_conversion, bool)

    ingredients.update(mead_correction=None)
    disable_mass_conversion = ingredients.disable_mass_conversion
    assert isinstance(disable_mass_conversion, bool)


def test_halo_model_ingredients_K(setup_data, ingredients):
    k_vec, z_vec = setup_data
    K = ingredients.K
    assert isinstance(K, np.ndarray)
    assert K.all() == np.zeros_like(z_vec).all()

    ingredients.update(mead_correction='feedback')
    K = ingredients.K
    assert isinstance(K, np.ndarray)

    ingredients.update(mead_correction='nofeedback')
    K = ingredients.K
    assert isinstance(K, np.ndarray)

    ingredients.update(mead_correction=None)


def test_halo_model_ingredients_hmf_generator(setup_data, ingredients):
    k_vec, z_vec = setup_data
    hmf_generator = ingredients._hmf_generator
    assert isinstance(hmf_generator, tuple)
    assert len(hmf_generator) == 2

    ingredients.update(mead_correction='feedback')
    hmf_generator = ingredients._hmf_generator
    assert isinstance(hmf_generator, tuple)
    assert len(hmf_generator) == 2
    ingredients.update(mead_correction=None)


def test_halo_model_ingredients_hmf_cen(setup_data, ingredients):
    k_vec, z_vec = setup_data
    hmf_cen = ingredients._hmf_cen
    assert isinstance(hmf_cen, list)
    assert len(hmf_cen) == z_vec.size


def test_halo_model_ingredients_hmf_sat(setup_data, ingredients):
    k_vec, z_vec = setup_data
    hmf_sat = ingredients._hmf_sat
    assert isinstance(hmf_sat, list)
    assert len(hmf_sat) == z_vec.size


def test_halo_model_ingredients_mass(setup_data, ingredients):
    k_vec, z_vec = setup_data
    mass = ingredients.mass
    assert isinstance(mass, np.ndarray)


def test_halo_model_ingredients_power(setup_data, ingredients):
    k_vec, z_vec = setup_data
    power = ingredients.power
    assert isinstance(power, np.ndarray)
    assert power.shape == (z_vec.size, k_vec.size)


def test_halo_model_ingredients_nonlinear_power(setup_data, ingredients):
    k_vec, z_vec = setup_data
    nonlinear_power = ingredients.nonlinear_power
    assert isinstance(nonlinear_power, np.ndarray)
    assert nonlinear_power.shape == (z_vec.size, k_vec.size)


def test_halo_model_ingredients_kh(setup_data, ingredients):
    k_vec, z_vec = setup_data
    kh = ingredients.kh
    assert isinstance(kh, np.ndarray)
    assert kh.size == k_vec.size


def test_halo_model_ingredients_halo_overdensity_mean(setup_data, ingredients):
    k_vec, z_vec = setup_data
    halo_overdensity_mean = ingredients.halo_overdensity_mean
    assert isinstance(halo_overdensity_mean, np.ndarray)
    assert halo_overdensity_mean.shape[0] == z_vec.size


def test_halo_model_ingredients_nu(setup_data, ingredients):
    k_vec, z_vec = setup_data
    nu = ingredients.nu
    assert isinstance(nu, np.ndarray)
    assert nu.shape[0] == z_vec.size


def test_halo_model_ingredients_dndlnm(setup_data, ingredients):
    k_vec, z_vec = setup_data
    dndlnm = ingredients.dndlnm
    assert isinstance(dndlnm, np.ndarray)
    assert dndlnm.shape[0] == z_vec.size


def test_halo_model_ingredients_mean_density0(setup_data, ingredients):
    k_vec, z_vec = setup_data
    mean_density0 = ingredients.mean_density0
    assert isinstance(mean_density0, np.ndarray)
    assert mean_density0.size == z_vec.size


def test_halo_model_ingredients_mean_density_z(setup_data, ingredients):
    k_vec, z_vec = setup_data
    mean_density_z = ingredients.mean_density_z
    assert isinstance(mean_density_z, np.ndarray)
    assert mean_density_z.size == z_vec.size


def test_halo_model_ingredients_rho_halo(setup_data, ingredients):
    k_vec, z_vec = setup_data
    rho_halo = ingredients.rho_halo
    assert isinstance(rho_halo, np.ndarray)
    assert rho_halo.size == z_vec.size


def test_halo_model_ingredients_halo_bias(setup_data, ingredients):
    k_vec, z_vec = setup_data
    halo_bias = ingredients.halo_bias
    assert isinstance(halo_bias, np.ndarray)
    assert halo_bias.shape[0] == z_vec.size


def test_halo_model_ingredients_neff(setup_data, ingredients):
    k_vec, z_vec = setup_data
    neff = ingredients.neff
    assert isinstance(neff, np.ndarray)
    assert neff.shape[0] == z_vec.size


def test_halo_model_ingredients_sigma8_z(setup_data, ingredients):
    k_vec, z_vec = setup_data
    sigma8_z = ingredients.sigma8_z
    assert isinstance(sigma8_z, np.ndarray)
    assert sigma8_z.shape[0] == z_vec.size


def test_halo_model_ingredients_fnu(setup_data, ingredients):
    k_vec, z_vec = setup_data
    fnu = ingredients.fnu
    assert isinstance(fnu, np.ndarray)
    assert fnu.shape[0] == z_vec.size


def test_halo_model_ingredients_conc_cen(setup_data, ingredients):
    k_vec, z_vec = setup_data
    conc_cen = ingredients.conc_cen
    assert isinstance(conc_cen, np.ndarray)
    assert conc_cen.shape[0] == z_vec.size


def test_halo_model_ingredients_nfw_cen(setup_data, ingredients):
    k_vec, z_vec = setup_data
    nfw_cen = ingredients.nfw_cen
    assert isinstance(nfw_cen, np.ndarray)
    assert nfw_cen.shape[0] == z_vec.size


def test_halo_model_ingredients_u_dm(setup_data, ingredients):
    k_vec, z_vec = setup_data
    u_dm = ingredients.u_dm
    assert isinstance(u_dm, np.ndarray)
    assert u_dm.shape[0] == z_vec.size
    assert u_dm[0, 0, 0] == 1.0


def test_halo_model_ingredients_r_s_cen(setup_data, ingredients):
    k_vec, z_vec = setup_data
    r_s_cen = ingredients.r_s_cen
    assert isinstance(r_s_cen, np.ndarray)
    assert r_s_cen.shape[0] == z_vec.size


def test_halo_model_ingredients_rvir_cen(setup_data, ingredients):
    k_vec, z_vec = setup_data
    rvir_cen = ingredients.rvir_cen
    assert isinstance(rvir_cen, np.ndarray)
    assert rvir_cen.shape[0] == z_vec.size


def test_halo_model_ingredients_conc_sat(setup_data, ingredients):
    k_vec, z_vec = setup_data
    conc_sat = ingredients.conc_sat
    assert isinstance(conc_sat, np.ndarray)
    assert conc_sat.shape[0] == z_vec.size


def test_halo_model_ingredients_nfw_sat(setup_data, ingredients):
    k_vec, z_vec = setup_data
    nfw_sat = ingredients.nfw_sat
    assert isinstance(nfw_sat, np.ndarray)
    assert nfw_sat.shape[0] == z_vec.size


def test_halo_model_ingredients_u_sat(setup_data, ingredients):
    k_vec, z_vec = setup_data
    u_sat = ingredients.u_sat
    assert isinstance(u_sat, np.ndarray)
    assert u_sat.shape[0] == z_vec.size
    assert u_sat[0, 0, 0] == 1.0


def test_halo_model_ingredients_r_s_sat(setup_data, ingredients):
    k_vec, z_vec = setup_data
    r_s_sat = ingredients.r_s_sat
    assert isinstance(r_s_sat, np.ndarray)
    assert r_s_sat.shape[0] == z_vec.size


def test_halo_model_ingredients_rvir_sat(setup_data, ingredients):
    k_vec, z_vec = setup_data
    rvir_sat = ingredients.rvir_sat
    assert isinstance(rvir_sat, np.ndarray)
    assert rvir_sat.shape[0] == z_vec.size


def test_halo_model_ingredients_growth_factor(setup_data, ingredients):
    k_vec, z_vec = setup_data
    growth_factor = ingredients.growth_factor
    assert isinstance(growth_factor, np.ndarray)
    assert growth_factor.shape[0] == z_vec.size
