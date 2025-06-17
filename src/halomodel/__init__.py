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

from .pk.pk import MatterSpectra, GalaxySpectra, AlignmentSpectra
from .pk.ia_rad import SatelliteAlignment
from .pk.ia_amp import AlignmentAmplitudes
from .hod.hod import HOD, Cacciato, Zheng, Zhai, Zehavi
from .hmi.hmi import HaloModelIngredients, CosmologyBase
