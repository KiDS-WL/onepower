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

from .pk.pk import Spectra#MatterSpectra, GalaxySpectra, AlignmentSpectra
from .pk.ia import AlignmentAmplitudes, SatelliteAlignment
from .pk.bnl import NonLinearBias
from .hod.hod import HOD, Cacciato, Zheng, Zhai, Zehavi
from .hmi.hmi import HaloModelIngredients, CosmologyBase
