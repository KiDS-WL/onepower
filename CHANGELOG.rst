Releases
========

dev-version
-----------

**Changes**

* Performance improvements in Bnl non-linear corrections.
* Added more defaults in CosmosSIS interface, so the Pmm case can be run without any HOD parameters needed.
* Deprecated Python 3.10, planning to drop support for 3.11 when ``hmf/halomod`` does the same.

**Fixes**

* Fixed neutrino contributions in DarkQuest Emulator cosmology object.

v0.7.2 [3 Feb 2026]
--------------------

**Fixes**

* Minor fixes to the codebase, mostly stemming from the changes in non-linear corrections and hmcode ingredients.

v0.7.1 [3 Feb 2026]
--------------------

**Fixes**

* Minor fixes to the codebase, mostly stemming from the changes in non-linear corrections and hmcode ingredients.

v0.7.0 [26 Jan 2026]
--------------------

**Changes**

* Changed the behaviour of the bnl, fortuna and hmcode non-linear corrections, and changed the behaviour of the hmcode ingredients, both are now governed by two sepearate variables called ``hmcode_ingredients`` and ``nonlinear_mode``. Backwards compatibility is **not** maintained!
* Updates and fixes to the pk_to_real code and the corresponding likelihood.

**Fixes**

* Minor fixes to the codebase, mostly stemming from the changes to the changes in non-linear corrections and hmcode ingredients.

v0.6.0 [14 Nov 2025]
--------------------

**Changes**

* Changed the behaviour of the matter-matter power spectra when response is True
* Updates to the pre-commit configuration

**Fixes**

* Minor fixes to the codebase, mostly stemming from the changes to the matter-matter power spectra

v0.5.0 [14 Nov 2025]
--------------------

**Changes**

* Added the convert_theta module to cosmosis modules for backwards support of other cosmosis modules
* Updated the documentation
* Updated the benchmarks

**Fixes**

* Minor fixes to the codebase, mostly affecting the cosmosis modules and parameters defaults


v0.3.5 [18 Sep 2025]
--------------------

**Changes**

* Fix to Github actions

v0.3.0 [18 Sep 2025]
--------------------

**Changes**

* Use the updated ``halomod`` and ``hmf`` packages that implement changes required to run ``onepower``
* Updates to the pyproject.toml file to include all dependencies and metadata
* Enables Dependabot and pre-commit.ci

**Fixes**

* Stricter linting

**Updates**

* Updated benchmarks

v0.2.0 [2 Sep 2025]
-------------------

**Changes**

* Cosmology inputs take different parameters, dropping Omega_M
* More numexpr usage to speed up the code
* Update to the documentation theme
* Setting up PyPI support

**Fixes**

* Fixed neutrinos

v0.1.1 [15 Aug 2025]
--------------------

**Changes**

* Unified the settings to ``pyproject.toml``
* Moved from ``flake8`` to ``Ruff``, deprecated ``black`` formatter
* Update to the documentation theme

**Fixes**

* ``numpy`` random generator updated
* Changed ``HOD`` to ``HaloOccupationDistribution`` to remove conflicts with ``halomod``

v0.1.0 [13 Aug 2025]
--------------------

There have have been *so many* changes since the legacy version of the code, that
it is almost pointless to list them. A brief summary is in order:

**Features**

* Fully modularised code and package.
* Installable with ``pip``.
* Majority of documentation written, with examples on how to use the code.
* Majority of functional tests written, with more scientifi tests and benchmars to come.

**Fixes**

* A lot.


Legacy [13 Aug 2025]
--------------------

This is a realese of the code pre-modularisation and unification in a single package. It uses the framwork initially laid out by Maria Cristina Fortuna, by interfacing with CosmoSIS using multiple modules and passing data between them using the CosmosSIS datablock object.
We release this for legacy purposes and for us to more easily compare the changes and solve bugs that the mudularisation might have introduced.
