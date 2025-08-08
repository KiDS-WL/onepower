import numpy as np
import pyfftlog
import scipy.interpolate
from cosmosis.datablock import option_section
from scipy.integrate import simpson

# These are the ones the user can use
TRANSFORM_WP = "wp"
TRANSFORM_DS = "ds"

DEFAULT_N_TRANSFORM = 8192
DEFAULT_K_MIN = 0.0001
DEFAULT_K_MAX = 5.0e6
DEFAULT_RP_MIN = 0.1
DEFAULT_RP_MAX = 1000.0


TRANSFORMS = [TRANSFORM_WP, TRANSFORM_DS]

DEFAULT_SECTIONS = {
    TRANSFORM_DS: ("matter_galaxy_power", "ds"),
    TRANSFORM_WP: (     "galaxy_power",   "wp"),
}

OUTPUT_NAMES = {
    TRANSFORM_WP:  "bin_{}",
    TRANSFORM_DS:  "bin_{}",
}


# Bias q and order mu parameters for transform
_TRANSFORM_PARAMETERS = {
    TRANSFORM_WP: (0.0, 0.0),
    TRANSFORM_DS: (0.0, 2.0),
}


class LogInterp:
    """
    This is a helper object that interpolates into f(x) where x>0.
    If all f>0 then it interpolates log(f) vs log(x).  If they are all f<0 then it
    interpolate log(-f) vs log(x).  If f is mixed or has some f=0 then it just interpolates
    f vs log(x).

    """

    def __init__(self, angle, spec, kind):
        if np.all(spec > 0):
            self.interp_func = scipy.interpolate.interp1d(
                np.log(angle), np.log(spec), kind, bounds_error=False, fill_value='extrapolate')
            self.interp_type = 'loglog'
        elif np.all(spec < 0):
            self.interp_func = scipy.interpolate.interp1d(
                np.log(angle), np.log(-spec), kind, bounds_error=False, fill_value='extrapolate')
            self.interp_type = 'minus_loglog'
        else:
            self.interp_func = scipy.interpolate.interp1d(
                np.log(angle), spec, kind, bounds_error=False, fill_value='extrapolate')
            self.interp_type = 'log_ang'

    def __call__(self, angle):
        if self.interp_type == 'loglog':
            spec = np.exp(self.interp_func(np.log(angle)))
        elif self.interp_type == 'minus_loglog':
            spec = -np.exp(self.interp_func(np.log(angle)))
        else:
            assert self.interp_type == 'log_ang'
            spec = self.interp_func(np.log(angle))
        return spec


class Transformer:
    """
    Class to build Hankel Transformers that convert from 3D power spectra to correlation functions.
    Several transform types are allowed, depending whether you are using cosmic shear, clustering, or
    galaxy-galaxy lensing.
    """

    def __init__(self, transform_type, n, k_min, k_max,
                 rp_min, rp_max, lower=1.0, upper=-2.0):

        # We use a fixed ell grid in log space and will interpolate/extrapolate our inputs onto this
        # grid. We typically use a maximum ell very much higher than the range we have physical values
        # for.  The exact values there do not matter, but they must be not have a sharp cut-off to avoid
        # oscillations at small angle.
        self.k_min = k_min
        self.k_max = k_max
        k = np.logspace(np.log10(k_min), np.log10(k_max), n)
        self.k = k
        dlogr = np.log(k[1]) - np.log(k[0])

        # pyfftlog has several options about how the theta and ell values used are chosen.
        # This option tells it to pick them to minimize ringing.
        kropt = 1

        # The parameters of the Hankel transform depend on the type.
        # They are defined in a dict at the top of the file
        self.q, self.mu = _TRANSFORM_PARAMETERS[transform_type]

        # Prepare the Hankel transform.
        self.kr, self.xsave = pyfftlog.fhti(
            n, self.mu, dlogr, q=self.q, kropt=kropt)

        # We always to the inverse transform, from Fourier->Real.
        self.direction = -1

        # Some more fixed values.
        self.rp_min = rp_min
        self.rp_max = rp_max
        self.lower = lower
        self.upper = upper

        # work out the effective rp values.
        nc = 0.5 * (n + 1)
        log_kmin = np.log(k_min)
        log_kmax = np.log(k_max)
        log_kmid = 0.5 * (log_kmin + log_kmax)
        k_mid = np.exp(log_kmid)
        r_mid = self.kr / k_mid
        x = np.arange(n)

        # And the effective separations of the output
        self.rp = np.exp((x - nc) * dlogr) * r_mid
        #self.rp = np.degrees(self.rp_rad) * 60.0
        self.range = (self.rp > self.rp_min) & (self.rp < self.rp_max)


    def __call__(self, k_in, pk_in):
        """Convert the input k and P(k) points to the points this transform requires, and then
        transform."""

        # Sample onto self.ell
        pk = self._interpolate_and_extrapolate_pk(k_in, pk_in)

        if self.q == 0:
            xi = pyfftlog.fht(self.k * pk, self.xsave,
                              tdir=self.direction) * (2 * np.pi) / self.rp
        else:
            xi = pyfftlog.fhtq(self.k * pk, self.xsave,
                               tdir=self.direction) * (2 * np.pi) / self.rp

        return self.rp[self.range], xi[self.range]

    def _interpolate_and_extrapolate_pk(self, k, pk):
        """Extrapolate and interpolate the input ell and cl to the default points for this transform"""
        k_min = k[0]
        k_max = k[-1]
        interpolator = LogInterp(k, pk, 'linear')
        pk_out = interpolator(self.k)
        #bad_low = np.isnan(pk_out) & (self.k < k_min)
        #bad_high = np.isnan(pk_out) & (self.k > k_max)

        #pk_out[bad_low] = pk[0] * (self.k[bad_low] / k_min)**self.lower
        #pk_out[bad_high] = pk[-1] * (self.k[bad_high] / k_max)**self.upper

        return pk_out


class CosmosisTransformer(Transformer):
    """
    Subclass of the Transformer object above specialised to cosmosis - gets its configuration
    and input/output from cosmosis data blocks.
    """

    def __init__(self, corr_type, options):
        # The type of transform to perform

        # Where to get/put the input/outputs
        default_input, default_output = DEFAULT_SECTIONS[corr_type]
        self.corr_type = corr_type

        self.input_section = options.get_string(
            option_section, "input_section_name", default_input)

        self.output_section = options.get_string(
            option_section, "output_section_name", default_output)

        if options.has_value(option_section, "suffixes"):
            self.suffixes = np.asarray([options[option_section, "suffixes"]]).flatten()
            self.nbins = len(self.suffixes)
            self.sample = options.get_string(option_section, "sample")
        else:
            self.suffixes = None
            self.nbins = None
            self.sample = options.get_string(option_section, "sample")

        # We don't use the default= keyword above because it breaks
        # for the xip / xim
        if self.input_section == "":
            self.input_section = default_input
        if self.output_section == "":
            self.output_section = default_output

        # Parameters of the transform
        self.n = options.get_int(option_section, "n_transform", DEFAULT_N_TRANSFORM)
        k_min = options.get_double(
            option_section, "k_min_extrapolate", DEFAULT_K_MIN)
        k_max = options.get_double(
            option_section, "k_max_extrapolate", DEFAULT_K_MAX)
        rp_min = options.get_double(
            option_section, "rp_min", DEFAULT_RP_MIN)
        rp_max = options.get_double(
            option_section, "rp_max", DEFAULT_RP_MAX)

        self.output_name = OUTPUT_NAMES[corr_type]

        super().__init__(
            corr_type, self.n, k_min, k_max, rp_min, rp_max)

    def __call__(self, block):

        # Choose the bin values to go up to.  Different modules might specify this in different ways.
        # They might have one nbin value (for cosmic shear and clustering) or two (for GGL)
        if self.corr_type == "ds":
            density = block["density", "mean_density0"]/1e12
        else:
            density = 1.0

        if self.nbins is None:
            nbins = block[self.sample, "nbin"]
        else:
            nbins = self.nbins
        # Loop through bin pairs and see if P(k) exists for all of them
        for i in range(nbins):

            b1 = i + 1

            # The key name for each bin
            output_name = self.output_name.format(b1)

            if self.suffixes is not None:
                input_section = f"{self.input_section}_{self.suffixes[i]}"
            else:
                input_section = self.input_section
            # Read the input k.
            k = block[input_section, "k_h"]
            z = block[input_section, "z"]

            # Read input P(k) from data block.
            pk = block[input_section, "p_k"]

            # Compute the transform.  Calls the earlier __call__ method above.
            xi = np.zeros((z.size, len(self.rp[self.range])))

            for j in range(len(z)):
                rp, xi[j,:] = super().__call__(k, pk[j,:])
            # Integrate over n(z)
            nz = self.load_kernel(block, self.sample, b1, z, 0.0)
            xi = simpson(nz[:,np.newaxis]*xi*density, z, axis=0)
            # Save results back to cosmosis
            block[self.output_section, output_name] = xi

        block[self.output_section, "nbin"] = nbins
        block[self.output_section, "sample"] = self.sample
        block[self.output_section, "rp"] = rp
        block[self.output_section, "sep_name"] = "rp"
        block[self.output_section, "save_name"] = self.corr_type

    def load_kernel(self, block, kernel_section, bin, z_ext, extrapolate_option):

        z_obs = block[kernel_section, "z"]
        obs_in = block[kernel_section, f"bin_{bin}"]
        inter_func = scipy.interpolate.interp1d(z_obs, obs_in, kind="linear", fill_value=extrapolate_option, bounds_error=False)
        kernel_ext = inter_func(z_ext)

        return kernel_ext


def setup(options):
    # xi, gamma, or w - defines what type of transform to do.
    corr_type = options.get_string(option_section, "corr_type")
    if corr_type not in TRANSFORMS:
        raise ValueError("Parameter transform in pk_to_corr must be one of {}".format(
            ", ".join(TRANSFORMS)))

    # The transformer object, which stores all the constants of the transform.
    # Further parameters of the transform are chosen in the __init__ function of CosmosisTransformer
    # in the code above.
    transformer = CosmosisTransformer(corr_type, options)

    return transformer


def execute(block, config):
    transformer = config
    transformer(block)
    return 0
