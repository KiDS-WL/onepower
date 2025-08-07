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

from .pk import Spectra, PowerSpectrumResult
from .ia import AlignmentAmplitudes, SatelliteAlignment
from .bnl import NonLinearBias
from .hod import HOD, Cacciato, Zheng, Zhai, Zehavi, Simple, load_data
from .hmi import HaloModelIngredients, CosmologyBase
from .add import UpsampledSpectra
from .utils import poisson
