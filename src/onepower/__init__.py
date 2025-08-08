"""A package for calculating the halo model."""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass

from .add import UpsampledSpectra
from .bnl import NonLinearBias
from .hmi import CosmologyBase, HaloModelIngredients
from .hod import HOD, Cacciato, Simple, Zehavi, Zhai, Zheng, load_data
from .ia import AlignmentAmplitudes, SatelliteAlignment
from .pk import PowerSpectrumResult, Spectra
from .utils import poisson
