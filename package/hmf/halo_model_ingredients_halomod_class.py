from functools import cached_property
import warnings
import numpy as np
from astropy.cosmology import Flatw0waCDM, Planck15
import halo_model_utility as hmu
import hmf
from halomod.halo_model import DMHaloModel
from hmf.halos.mass_definitions import SphericalOverdensity
from halomod.concentration import make_colossus_cm, interp_concentration
import halomod.profiles as profile_classes
import halomod.concentration as concentration_classes
import time

from halomod.functional import get_halomodel

# Silencing a warning from hmf for which the nonlinear mass is still correctly calculated
warnings.filterwarnings("ignore", message="Nonlinear mass outside mass range")

DMHaloModel.ERROR_ON_BAD_MDEF = False


class SOVirial_Mead(SphericalOverdensity):
    """
    SOVirial overdensity definition from Mead et al. (2021).
    """
    _defaults = {"overdensity": 200}

    def halo_density(self, z=0, cosmo=Planck15):
        """The density of haloes under this definition."""
        return self.params["overdensity"] * self.mean_density(z, cosmo)

    @property
    def colossus_name(self):
        return "200c"

    def __str__(self):
        """Describe the halo definition in standard notation."""
        return "SOVirial"


class HaloModelIngredients:
    def __init__(self,
            k_vec = None,
            z_vec = None,
            lnk_min = 0.0,
            lnk_max = 0.0,
            dlnk = 0.0,
            Mmin = 0.0,
            Mmax = 0.0,
            dlog10m = 0.0,
            mdef_model = None,
            hmf_model = None,
            bias_model = None,
            halo_profile_model = None,
            halo_concentration_model = None,
            transfer_model = None,
            transfer_params = {},
            growth_model = None,
            growth_params = {},
            h0 = 0.0,
            omega_c = 0.0,
            omega_b = 0.0,
            omega_m = 0.0,
            w0 = 0.0,
            wa = 0.0,
            n_s = 0.0,
            tcmb = 0-0,
            m_nu = 0.0,
            sigma_8 = 0.0,
            log10T_AGN = 0.0,
            norm_cen = 0.0,
            norm_sat = 0.0,
            eta_cen = 0.0,
            eta_sat = 0.0,
            overdensity = 0.0,
            delta_c = 0.0,
            mead_correction = None
        ):
    
        self.k_vec = k_vec
        self.z_vec = z_vec
        self.lnk_min = lnk_min
        self.lnk_max = lnk_max
        self.dlnk = dlnk
        self.Mmin = Mmin
        self.Mmax = Mmax
        self.dlog10m = dlog10m
        self.halo_profile_model = halo_profile_model
        self.omega_c = omega_c
        self.omega_m = omega_m
        self.omega_b = omega_b
        self.w0 = w0
        self.wa = wa
        self.h0 = h0
        self.n_s = n_s
        self.tcmb = tcmb
        self.m_nu = m_nu
        self.sigma_8 = sigma_8
        self.log10T_AGN = log10T_AGN
        self.transfer_model = transfer_model
        self.transfer_params = transfer_params
        self.growth_model = growth_model
        self.growth_params = growth_params
        self.mead_correction = mead_correction

        self.norm_cen = norm_cen * np.ones_like(self.z_vec)
        self.norm_sat = norm_sat * np.ones_like(self.z_vec)

        self.halo_profile_params = {'cosmo': self.cosmo_model}

        if self.mead_correction in ['feedback', 'nofeedback']:
            self._setup_mead_correction(norm_sat, eta_sat, norm_cen)
        else:
            self._setup_default(hmf_model, bias_model, halo_concentration_model, mdef_model, overdensity, delta_c, eta_cen, eta_sat)
        
    def _setup_mead_correction(self, norm_sat, eta_sat, norm_cen):
        self.disable_mass_conversion = True
        self.hmf_model = 'ST'
        self.bias_model = 'ST99'
        self.halo_concentration_model = interp_concentration(getattr(concentration_classes, 'Bullock01'))
        growth = hmu.get_growth_interpolator(self.cosmo_model)
        a = self.cosmo_model.scale_factor(self.z_vec)
        g = growth(a)
        G = np.array([hmu.get_accumulated_growth(ai, growth) for ai in a])
        delta_c_mead = hmu.dc_Mead(a, self.cosmo_model.Om(self.z_vec) + self.cosmo_model.Onu(self.z_vec),
                            self.cosmo_model.Onu0 / (self.cosmo_model.Om0 + self.cosmo_model.Onu0), g, G)
        halo_overdensity_mead = hmu.Dv_Mead(a, self.cosmo_model.Om(self.z_vec) + self.cosmo_model.Onu(self.z_vec),
                                        self.cosmo_model.Onu0 / (self.cosmo_model.Om0 + self.cosmo_model.Onu0), g, G)
        self.delta_c = delta_c_mead
        self.mdef_model = SOVirial_Mead
        self.mdef_params = [{'overdensity': overdensity} for overdensity in halo_overdensity_mead]

        #self.norm_cen = np.ones_like(self.z_vec) # tmp
        #self.norm_sat = np.ones_like(self.z_vec) # tmp
        #self.eta_sat = eta_sat * np.ones_like(self.z_vec) # tmp

        if self.mead_correction == 'nofeedback':
            self.K = 5.196 * np.ones_like(self.z_vec)
        elif self.mead_correction == 'feedback':
            theta_agn = self.log10T_AGN - 7.8
            self.K = (5.196 / 4.0) * ((3.44 - 0.496 * theta_agn) * np.power(10.0, self.z_vec * (-0.0671 - 0.0371 * theta_agn)))

    def _setup_default(self, hmf_model, bias_model, halo_concentration_model, mdef_model, overdensity, delta_c, eta_cen, eta_sat):
        self.disable_mass_conversion = False
        self.hmf_model = hmf_model
        self.bias_model = bias_model
        try:
            self.halo_concentration_model = interp_concentration(getattr(concentration_classes, halo_concentration_model))
        except:
            self.halo_concentration_model = interp_concentration(make_colossus_cm(halo_concentration_model))
        self.mdef_model = mdef_model
        self.mdef_params = [{} if self.mdef_model == 'SOVirial' else {'overdensity': overdensity} for _ in self.z_vec]
        self.delta_c = (3.0 / 20.0) * (12.0 * np.pi) ** (2.0 / 3.0) * (1.0 + 0.0123 * np.log10(self.cosmo_model.Om(self.z_vec))) if self.mdef_model == 'SOVirial' else delta_c * np.ones_like(self.z_vec)
        
        self.eta_cen = eta_cen * np.ones_like(self.z_vec)
        self.eta_sat = eta_sat * np.ones_like(self.z_vec)
            
    @cached_property
    def cosmo_model(self):
        # Update the cosmological parameters
        return Flatw0waCDM(
            H0=self.h0*100.0,
            Ob0=self.omega_b,
            Om0=self.omega_m,
            m_nu=[0, 0, self.m_nu],
            Tcmb0=self.tcmb,
            w0=self.w0,
            wa=self.wa
        )
        
    @cached_property
    def hmf_generator(self):
        x = DMHaloModel(
                z=0.0,
                lnk_min=self.lnk_min,
                lnk_max=self.lnk_max,
                dlnk=self.dlnk,
                Mmin=self.Mmin,
                Mmax=self.Mmax,
                dlog10m=self.dlog10m,
                hmf_model=self.hmf_model,
                mdef_model=self.mdef_model,
                disable_mass_conversion=self.disable_mass_conversion,
                bias_model=self.bias_model,
                halo_profile_model=self.halo_profile_model,
                halo_profile_params=self.halo_profile_params,
                halo_concentration_model=self.halo_concentration_model,
                cosmo_model=self.cosmo_model,
                sigma_8=self.sigma_8,
                n=self.n_s,
                transfer_model=self.transfer_model,
                transfer_params=self.transfer_params,
                growth_model=self.growth_model,
                growth_params=self.growth_params,
                mdef_params=self.mdef_params[0],
                delta_c=self.delta_c[0]
            )
        y = x.clone()
        x_out = []
        y_out = []
        if self.mead_correction in ['feedback', 'nofeedback']:
            # For centrals
            for z, mdef_par, dc, norm_cen, k in zip(self.z_vec, self.mdef_params, self.delta_c, self.norm_cen, self.K):
                x.update(
                    z=z,
                    mdef_params=mdef_par,
                    delta_c=dc,
                )
                eta_cen = 0.1281 * x.sigma8_z**(-0.3644)
                x.update(
                    halo_profile_params={'eta_bloat': eta_cen},
                    halo_concentration_params={'norm': norm_cen, 'K': k}
                )
                #yield x
                x_out.append(x.clone())
            
            # For satellites
            for z, mdef_par, dc, norm_sat, k in zip(self.z_vec, self.mdef_params, self.delta_c, self.norm_sat, self.K):
                y.update(
                    z=z,
                    mdef_params=mdef_par,
                    delta_c=dc
                )
                eta_sat = 0.1281 * y.sigma8_z**(-0.3644)
                y.update(
                    halo_profile_params={'eta_bloat': eta_sat},
                    halo_concentration_params={'norm': norm_sat, 'K': k}
                )
                #yield y
                y_out.append(y.clone())
        else:
            # For centrals
            for z, mdef_par, dc, eta_cen, norm_cen in zip(self.z_vec, self.mdef_params, self.delta_c, self.eta_cen, self.norm_cen):
                x.update(
                    z=z,
                    mdef_params=mdef_par,
                    delta_c=dc,
                    halo_profile_params={'eta_bloat': eta_cen},
                    halo_concentration_params={'norm': norm_cen}
                )
                #yield x
                x_out.append(x.clone())
            
            # For satellites
            for z, mdef_par, dc, eta_sat, norm_sat in zip(self.z_vec, self.mdef_params, self.delta_c, self.eta_sat, self.norm_sat):
                y.update(
                    z=z,
                    mdef_params=mdef_par,
                    delta_c=dc,
                    halo_profile_params={'eta_bloat': eta_sat},
                    halo_concentration_params={'norm': norm_sat}
                )
                #yield y
                y_out.append(y.clone())
        return x_out, y_out

    @cached_property
    def hmf_cen(self):
        return self.hmf_generator[0]

    @cached_property
    def hmf_sat(self):
        return self.hmf_generator[1]

    @property
    def mass(self):
        return self.hmf_cen[0].m

    @property
    def halo_overdensity_mean(self):
        return np.array([x.halo_overdensity_mean for x in self.hmf_cen])

    @property
    def nu(self):
        return np.array([x.nu**0.5 for x in self.hmf_cen])

    @property
    def dndlnm(self):
        return np.array([x.dndlnm for x in self.hmf_cen])

    @property
    def mean_density0(self):
        return np.array([x.mean_density0 for x in self.hmf_cen])

    @property
    def mean_density_z(self):
        return np.array([x.mean_density for x in self.hmf_cen])

    @property
    def rho_halo(self):
        return np.array([x.halo_overdensity_mean * x.mean_density0 for x in self.hmf_cen])

    @property
    def halo_bias(self):
        return np.array([x.halo_bias for x in self.hmf_cen])

    @property
    def neff(self):
        return np.array([x.n_eff_at_collapse for x in self.hmf_cen])

    @property
    def sigma8_z(self):
        return np.squeeze(np.array([x.sigma8_z for x in self.hmf_cen]))

    @property
    def fnu(self):
        return np.array([self.cosmo_model.Onu0 / self.cosmo_model.Om0 for _ in self.z_vec])

    @property
    def conc_cen(self):
        return np.array([x.cmz_relation for x in self.hmf_cen])

    @property
    def nfw_cen(self):
        return np.array([x.halo_profile.u(self.k_vec, x.m) for x in self.hmf_cen])

    @property
    def u_dm(self):
        return self.nfw_cen / np.expand_dims(self.nfw_cen[:, 0, :], 1)

    @property
    def r_s_cen(self):
        return np.array([x.halo_profile._rs_from_m(x.m) for x in self.hmf_cen])

    @property
    def rvir_cen(self):
        return np.array([x.halo_profile.halo_mass_to_radius(x.m) for x in self.hmf_cen])

    @property
    def conc_sat(self):
        return np.array([x.cmz_relation for x in self.hmf_sat])

    @property
    def nfw_sat(self):
        return np.array([x.halo_profile.u(self.k_vec, x.m) for x in self.hmf_sat])

    @property
    def u_sat(self):
        return self.nfw_sat / np.expand_dims(self.nfw_sat[:, 0, :], 1)

    @property
    def r_s_sat(self):
        return np.array([x.halo_profile._rs_from_m(x.m) for x in self.hmf_sat])

    @property
    def rvir_sat(self):
        return np.array([x.halo_profile.halo_mass_to_radius(x.m) for x in self.hmf_sat])
    
    @property
    def growth_factor(self):
        # TO-DO: Check against interpolated one from CAMB!
        return self.hmf_cen[0]._growth_factor_fn(self.z_vec)
        
    # Maybe implement at some point?
    # Rnl = DM_hmf.filter.mass_to_radius(DM_hmf.mass_nonlinear, DM_hmf.mean_density0)
    # neff[jz] = -3.0 - 2.0*DM_hmf.normalised_filter.dlnss_dlnm(Rnl)

    # Only used for mead_corrections
    # pk_cold = DM_hmf.power * hmu.Tk_cold_ratio(DM_hmf.k, g, block[cosmo_params, 'ommh2'], block[cosmo_params, 'h0'], this_cosmo_run.Onu0/this_cosmo_run.Om0, this_cosmo_run.Neff, T_CMB=tcmb)**2.0
    # sigma8_z[jz] = hmu.sigmaR_cc(pk_cold, DM_hmf.k, 8.0)
