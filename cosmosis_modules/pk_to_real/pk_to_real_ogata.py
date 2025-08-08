import matplotlib.pyplot as pl
import numpy as np
import pyfftlog
import scipy.interpolate
from cosmosis.datablock import option_section
from hankel import HankelTransform
from scipy.integrate import simpson

# These are the ones the user can use
TRANSFORM_WP = "wp"
TRANSFORM_DS = "ds"

DEFAULT_N_TRANSFORM = 250#8192
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


class projected_corr():
    def __init__(self, transform_type, n, k_min, k_max, rp_min, rp_max):

        self.k_min = k_min
        self.k_max = k_max
        k = np.logspace(np.log10(k_min), np.log10(k_max), n)
        self.k = k
        dlogr = np.log(k[1]) - np.log(k[0])

        # The parameters of the Hankel transform depend on the type.
        # They are defined in a dict at the top of the file
        self.q, self.mu = _TRANSFORM_PARAMETERS[transform_type]

        # Prepare the Hankel transform.
        N_hankel = 20#350
        h_hankel = np.pi/N_hankel
        self.h_transform = HankelTransform(self.mu,N_hankel,h_hankel)

        # Some more fixed values.
        self.rp_min = rp_min
        self.rp_max = rp_max

        # work out the effective rp values.
        nc = 0.5 * (n + 1)
        log_kmin = np.log(k_min)
        log_kmax = np.log(k_max)
        log_kmid = 0.5 * (log_kmin + log_kmax)
        k_mid = np.exp(log_kmid)
        r_mid = 1.0 / k_mid
        x = np.arange(n)

        # And the effective separations of the output
        self.rp = np.exp((x - nc) * dlogr) * r_mid
        #self.rp = np.degrees(self.rp_rad) * 60.0
        self.range = (self.rp > self.rp_min) & (self.rp < self.rp_max)


    def evaluate(self, k_in, pk_in):
        # Finally calculates the Hankel transform
        pk_iter = scipy.interpolate.interp1d(k_in, pk_in, kind="linear", fill_value='extrapolate', bounds_error=False)

        result = np.zeros(self.rp.shape)
        h = self.h_transform
        for i in range(result.size):
            integ = lambda x: pk_iter(x/self.rp[i]) * x
            result[i] = h.transform(integ)[0]
        xi = result / (2.0 * np.pi * self.rp**2)

        return self.rp[self.range], xi[self.range]


class projection():
    """
    Subclass of the Transformer object above specialised to cosmosis - gets its configuration
    and input/output from cosmosis data blocks.
    """

    def __init__(self, corr_type, options):
        # The type of transform to perform

        # Where to get/put the input/outputs
        default_input, default_output = DEFAULT_SECTIONS[corr_type]

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
        self.project = projected_corr(corr_type, self.n, k_min, k_max, rp_min, rp_max)


    def transform(self, block):

        # Choose the bin values to go up to.  Different modules might specify this in different ways.
        # They might have one nbin value (for cosmic shear and clustering) or two (for GGL)

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
            xi = np.zeros((z.size, len(self.project.rp[self.project.range])))

            for j in range(len(z)):
                rp, xi[j,:] = self.project.evaluate(k, pk[j,:])
            # Integrate over n(z)
            nz = self.load_kernel(block, self.sample, b1, z, 0.0)
            xi = simpson(nz[:,np.newaxis]*xi, z, axis=0)
            #pl.plot(rp, xi)

            # Save results back to cosmosis
            block[self.output_section, output_name] = xi
        #pl.xscale('log')
        #pl.yscale('log')
        #pl.show()

        block[self.output_section, "nbin"] = nbins
        block[self.output_section, "sample"] = self.sample
        block[self.output_section, "rp"] = rp
        block[self.output_section, "sep_name"] = "rp"
        block[self.output_section, "save_name"] = ""

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
    setup = projection(corr_type, options)

    return setup


def execute(block, setup):
    result = setup
    result.transform(block)
    return 0
