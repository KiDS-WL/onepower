"""A package for calculating the halo model."""

#try:
#    from importlib.metadata import PackageNotFoundError, version
#except ImportError:
#    from importlib_metadata import PackageNotFoundError, version

#try:
#    __version__ = version(__name__)
#except PackageNotFoundError:
#    # package is not installed
#    pass

from .pk.pk_lib_class import MatterSpectra, GalaxySpectra, AlignmentSpectra
from .hod.hod_lib_class_no_loop import HOD, Cacciato, Zheng
from .hmf.halo_model_ingredients_halomod_class import HaloModelIngredients
