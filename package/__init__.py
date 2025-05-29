"""A package for calculating the halo model."""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

#try:
#    __version__ = version(__name__)
#except PackageNotFoundError:
#    # package is not installed
#    pass

from .pk import MatterSpectra, GalaxySpectra, AlignmentSpectra
from .hod import HOD, Cacciato, Zheng
