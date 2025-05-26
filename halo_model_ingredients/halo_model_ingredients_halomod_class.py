import warnings
import numpy as np
from astropy.cosmology import Flatw0waCDM
import halo_model_utility as hmu
import hmf
from halomod import get_halomodel
from halomod.halo_model import DMHaloModel
from halomod.concentration import make_colossus_cm, interp_concentration
import halomod.profiles as profile_classes
import halomod.concentration as concentration_classes
import time

# Silencing a warning from hmf for which the nonlinear mass is still correctly calculated
warnings.filterwarnings("ignore", message="Nonlinear mass outside mass range")


class HaloModelIngredients:
    def __init__(self,
            **kwargs
        ):
        self.k_vec = k_vec
        self.z_vec = z_vec
        self.lnk_min = lnk_min
        self.lnk_max = lnk_max
        self.dlnk = dlnk
        self.Mmin = Mmin
        self.Mmax = Mmax
        self.dlog10m = dlog10m
        self.disable_mass_conversion = disable_mass_conversion
        self.halo_profile_model = halo_profile_model
        self.halo_concentration_model = halo_concentration_model
        self.omega_c = omega_c
        self.omega_m = omega_m
        self.omega_b = omega_b
        self.w0 = w0
        self.wa = wa
        self.h0 = h0
        self.n_s = n_s
        self.tcmb = tcmb
        self.sigma_8 = sigma_8
        self.log10T_AGN = log10T_AGN
        self.transfer_model = transfer_model
        self.transfer_params = transfer_params
        self.growth_model = growth_model
        self.growth_params = growth_params
        self.mead_correction = mead_correction
        self.hmf_params = hmf_params


        self.halo_profile_params = {'cosmo': self.cosmo_model}
        # If Mead correction is applied, set the ingredients to match Mead et al. (2021)
        if self.mead_correction is not None:
            self.disable_mass_conversion = True
            self.hmf_model = 'ST'
            self.bias_model = 'ST99'
            self.halo_concentration_model = 'Duffy08'  # Dummy cm model, correct one calculated in execute
            growth = hmu.get_growth_interpolator(self.cosmo_model)
            # growth_LCDM = hmu.get_growth_interpolator(LCDMcosmo)
            a = self.cosmo_model.scale_factor(self.z_vec)
            g = growth(a)
            G = np.array([hmu.get_accumulated_growth(ai, growth) for ai in a])
            delta_c_mead = hmu.dc_Mead(a, self.cosmo_model.Om(self.z_vec) + self.cosmo_model.Onu(self.z_vec),
                            self.cosmo_model.Onu0 / (self.cosmo_model.Om0 + self.cosmo_model.Onu0), g, G)
            halo_overdensity_mead = hmu.Dv_Mead(a, self.cosmo_model.Om(self.z_vec) + self.cosmo_model.Onu(self.z_vec),
                                        self.cosmo_model.Onu0 / (elf.cosmo_model.Om0 + self.cosmo_model.Onu0), g, G)
            self.delta_c = delta_c_mead,
            self.mdef_model = hmu.SOVirial_Mead,
            self.mdef_params = {'overdensity': halo_overdensity_mead}
            
        if mead_correction is None:
            self.disable_mass_conversion = False
            self.hmf_model = hmf_model
            self.bias_model = bias_model
            try:
                self.halo_concentration_model = interp_concentration(getattr(concentration_classes, cm_model))
            except:
                self.halo_concentration_model = interp_concentration(make_colossus_cm(cm_model))
            self.delta_c = (
                (3.0 / 20.0) * (12.0 * np.pi) ** (2.0 / 3.0) * (1.0 + 0.0123 * np.log10(self.cosmo_model.Om(self.z_vec)))
                if self.mdef_model == 'SOVirial' else
                delta_c
            )
            self.mdef_model = mdef_model
            self.mdef_params = {} if self.mdef_model == 'SOVirial' else {'overdensity': overdensity}
            
    @cached_property
    def cosmo_model(self):
        # Update the cosmological parameters
        return Flatw0waCDM(
            H0=self.h0*100.0,
            Ob0=self.omega_b,
            Om0=self.omega_m,
            m_nu=[0, 0, self.mnu],
            Tcmb0=self.tcmb,
            w0=self.w0,
            wa=self.wa
        )
      
    @cached_property
    def hmf(self):
        return DMHaloModel(
            lnk_min=self.lnk_min,
            lnk_max=self.lnk_max,
            dlnk=self.dlnk,
            Mmin=self.Mmin,
            Mmax=self.Mmax,
            dlog10m=self.dlog10m,
            hmf_model=self.hmf_model,
            mdef_model=self.mdef_model,
            mdef_params=self.mdef_params,
            delta_c=self.delta_c,
            disable_mass_conversion=self.disable_mass_conversion,
            bias_model=self.bias_model,
            halo_profile_model=self.halo_profile_model,
            halo_concentration_model=self.halo_concentration_model,
            cosmo_model=self.cosmo_model,
            sigma_8=self.sigma_8,
            n=self.n_s,
            transfer_model=self.transfer_model,
            transfer_params=self.transfer_params,
            halo_profile_params=self.halo_profile_params,
            growth_model=self.growth_model,
            growth_params=self.growth_params
        )
        
    # Split update and models for centrals and satellites, so that user can specifiy different profile and concenctration models, not only normalisations and bloating ...
    def hmf_z(self):
        return [self.hmf.update(
            z=z[i],
            mdef_params=self.mdef_params,
            delta_c=self.delta_c[i],
            halo_profile_params=self.halo_profile_params,
        ) for i in range(len(self.z_vec))]
        
    @property
    def mass(self):
        return self.hmf.m

    # This all need z-dependence
    @property
    def halo_overdensity_mean(self):
        return self.hmf.halo_overdensity_mean

    @property
    def nu(self):
        return self.hmf.nu**0.5
           
    @property
    def dndlnm(self):
        return self.hmf.dndlnm
    
    @property
    def mean_density0(self):
        return self.hmf.mean_density0

    @property
    def mean_density_z(self):
        return self.hmf.mean_density

    @property
    def rho_halo(self):
        return self.hmf.halo_overdensity_mean * self.hmf.mean_density0

    @property
    def b_nu(self):
        return sel.hmf.halo_bias

    @property
    def neff(self):
        return self.hmf.n_eff_at_collapse

    @property
    def sigma8_z(self):
        return self.hmf.sigma8_z
        
    """
    # still missing outputs
    f_nu = np.empty([nz])
    u_dm_cen = np.empty([nz, nk, nmass_hmf])
    u_dm_sat = np.empty([nz, nk, nmass_hmf])
    conc_cen = np.empty([nz, nmass_hmf])
    conc_sat = np.empty([nz, nmass_hmf])
    r_s_cen = np.empty([nz, nmass_hmf])
    r_s_sat = np.empty([nz, nmass_hmf])
    rvir_cen = np.empty([nz, nmass_hmf])
    rvir_sat = np.empty([nz, nmass_hmf])
    """
       
       

    norm_cen = self.hmf_params['norm_cen']
    norm_sat = self.hmf_params['norm_sat']
    eta_cen = self.hmf_params['eta_cen']
    eta_sat = self.hmf_params['eta_sat']

    DM_hmf.ERROR_ON_BAD_MDEF = False

    # Loop over a series of redshift values defined by z_vec = np.linspace(zmin, zmax, nz)
    for jz, z_iter in enumerate(z_vec):
        if mead_correction is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                # This disables the warning from hmf. hmf is just telling us what we know
                # hmf's internal way of calculating the overdensity and the collapse threshold are fixed.
                # When we use the mead correction we want to define the haloes using the virial definition.
                # To avoid conflicts we manually pass the overdensity and the collapse threshold,
                # but for that we need to set the mass definition to be "mean",
                # so that it is compared to the mean density of the Universe rather than critical density.
                # hmf warns us that the value is not a native definition for the given halo mass function,
                # but will interpolate between the known ones (this is happening when one uses Tinker hmf for instance).
                a = this_cosmo_run.scale_factor(z_iter)
                g = growth(a)
                G = hmu.get_accumulated_growth(a, growth)
                delta_c_z = hmu.dc_Mead(a, this_cosmo_run.Om(z_iter) + this_cosmo_run.Onu(z_iter),
                                        this_cosmo_run.Onu0 / (this_cosmo_run.Om0 + this_cosmo_run.Onu0), g, G)
                halo_overdensity_mead = hmu.Dv_Mead(a, this_cosmo_run.Om(z_iter) + this_cosmo_run.Onu(z_iter),
                                                    this_cosmo_run.Onu0 / (this_cosmo_run.Om0 + this_cosmo_run.Onu0), g, G)
                DM_hmf.update(
                    z=z_iter,
                    delta_c=delta_c_z,
                    mdef_model=hmu.SOVirial_Mead,
                    mdef_params={'overdensity': halo_overdensity_mead}
                )

                eta_cen = 0.1281 * DM_hmf.sigma8_z**(-0.3644)
                if mead_correction == 'nofeedback':
                    norm_cen = 5.196  # /3.85#1.0#(5.196/3.85) #0.85*1.299
                elif mead_correction == 'feedback':
                    theta_agn = block['halo_model_parameters', 'logT_AGN'] - 7.8
                    norm_cen = (5.196 / 4.0) * ((3.44 - 0.496 * theta_agn) * np.power(10.0, z_iter * (-0.0671 - 0.0371 * theta_agn)))

                # We can do this collapse redshift, or directly use the Bullock01 from halomod! The difference is really in growth rate
                # but we can also try to use the delta_c and overdensity calculations using CAMB growth!
                zf = hmu.get_halo_collapse_redshifts(mass, z_iter, delta_c_z, growth, this_cosmo_run, DM_hmf)
                conc_cen[jz, :] = norm_cen * (1.0 + zf) / (1.0 + z_iter)
                conc_sat[jz, :] = norm_sat * (1.0 + zf) / (1.0 + z_iter)

                DM_hmf.update(halo_profile_params={'eta_bloat': eta_cen})
                nfw_cen = DM_hmf.halo_profile.u(k, DM_hmf.m, c=conc_cen[jz, :])
                u_dm_cen[jz, :, :] = nfw_cen / np.expand_dims(nfw_cen[0, :], 0)
                r_s_cen[jz, :] = DM_hmf.halo_profile._rs_from_m(DM_hmf.m)
                rvir_cen[jz, :] = DM_hmf.halo_profile.halo_mass_to_radius(DM_hmf.m)

                DM_hmf.update(halo_profile_params={'eta_bloat': eta_sat})
                nfw_sat = DM_hmf.halo_profile.u(k, DM_hmf.m, c=conc_sat[jz, :])
                u_dm_sat[jz, :, :] = nfw_sat / np.expand_dims(nfw_sat[0, :], 0)
                r_s_sat[jz, :] = DM_hmf.halo_profile._rs_from_m(DM_hmf.m)
                rvir_sat[jz, :] = DM_hmf.halo_profile.halo_mass_to_radius(DM_hmf.m)

        if mead_correction is None:
            
            DM_hmf.update(
                z=z_iter,
                delta_c=delta_c_z,
                halo_profile_params={'eta_bloat': eta_cen},
                halo_concentration_params={'norm': norm_cen}
            )

            conc_cen[jz, :] = DM_hmf.cmz_relation
            nfw_cen = DM_hmf.halo_profile.u(k, DM_hmf.m)
            u_dm_cen[jz, :, :] = nfw_cen / np.expand_dims(nfw_cen[0, :], 0)
            r_s_cen[jz, :] = DM_hmf.halo_profile._rs_from_m(DM_hmf.m)
            rvir_cen[jz, :] = DM_hmf.halo_profile.halo_mass_to_radius(DM_hmf.m)

            DM_hmf.update(halo_profile_params={'eta_bloat': eta_sat},
                          halo_concentration_params={'norm': norm_sat})
            conc_sat[jz, :] = DM_hmf.cmz_relation
            nfw_sat = DM_hmf.halo_profile.u(k, DM_hmf.m)
            u_dm_sat[jz, :, :] = nfw_sat / np.expand_dims(nfw_sat[0, :], 0)
            r_s_sat[jz, :] = DM_hmf.halo_profile._rs_from_m(DM_hmf.m)
            rvir_sat[jz, :] = DM_hmf.halo_profile.halo_mass_to_radius(DM_hmf.m)
       
        
    # Maybe implement at some point?
    # Rnl = DM_hmf.filter.mass_to_radius(DM_hmf.mass_nonlinear, DM_hmf.mean_density0)
    # neff[jz] = -3.0 - 2.0*DM_hmf.normalised_filter.dlnss_dlnm(Rnl)

    # Only used for mead_corrections
    # pk_cold = DM_hmf.power * hmu.Tk_cold_ratio(DM_hmf.k, g, block[cosmo_params, 'ommh2'], block[cosmo_params, 'h0'], this_cosmo_run.Onu0/this_cosmo_run.Om0, this_cosmo_run.Neff, T_CMB=tcmb)**2.0
    # sigma8_z[jz] = hmu.sigmaR_cc(pk_cold, DM_hmf.k, 8.0)

