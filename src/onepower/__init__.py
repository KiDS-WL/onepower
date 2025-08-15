"""A package for calculating the halo model."""

import contextlib

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

with contextlib.suppress(PackageNotFoundError):
    __version__ = version(__name__)

from .add import UpsampledSpectra
from .bnl import NonLinearBias
from .hmi import CosmologyBase, HaloModelIngredients
from .hod import (
    HaloOccupationDistribution,
    Cacciato,
    Simple,
    Zehavi,
    Zhai,
    Zheng,
    load_data,
)
from .ia import AlignmentAmplitudes, SatelliteAlignment
from .pk import PowerSpectrumResult, Spectra
from .utils import poisson
