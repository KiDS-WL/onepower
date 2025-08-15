Releases
========

dev-version
----------------------



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

**Bugfixes**

* A lot.


Legacy [13 Aug 2025]
--------------------

This is a realese of the code pre-modularisation and unification in a single package. It uses the framwork initially laid out by Maria Cristina Fortuna, by interfacing with CosmoSIS using multiple modules and passing data between them using the CosmosSIS datablock object.
We release this for legacy purposes and for us to more easily compare the changes and solve bugs that the mudularisation might have introduced.
