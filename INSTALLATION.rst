Installation
============

This page will guide you through installing ``onepower`` -- either as purely a user, or
as a potential developer.

Dependencies
------------
``onepower`` has a number of dependencies, all of which should be automatically installed
as you install the package itself. You therefore do not need to worry about installing
them yourself, except in some circumstances.

** Note that currently the ``hmf`` and ``halomod`` need to be installed manually from an alternative fork, until pull requests are approved: **

.. code-block:: bash
    pip install hmf @ git+git://github.com/andrejdvornik/hmf.git@main
    pip install halomod git+git://github.com/andrejdvornik/halomod.git@main

The list of dependencies used by ``onepower`` is:

1. The `halo mass function calculator, hmf <https://hmf.readthedocs.io/en/3.3.4/>`_
2. The `halomod <https://github.com/halomod/halomod>`_.
3. The `Dark Emulator <https://dark-emulator.readthedocs.io/en/latest/>`_
4. ...

The optional input fits files as used in Fortuna et al. 2021 are available here: `Luminosity_redshift <https://ruhr-uni-bochum.sciebo.de/s/ZdAE6nTf0OPyV6S>`_.
Those are required for the IA predictions to agree, as they provide sample properties and fractions of red and blue galaxies.
Similar data is required if any IA predictions are to be used.


User Install
------------
You may install the latest release of ``onepower`` using ``pip``

.. code-block:: bash

    pip install onepower

This will install all uninstalled dependencies (see previous section).
Alternatively, for the very bleeding edge, install from the main branch of the repo

.. code-block:: bash

    pip install onepower @ git+git://github.com/KiDS-WL/onepower.git

Developer Install
-----------------
If you intend to develop ``onepower``, clone the repository:

.. code-block:: bash

    git clone https://github.com/KiDS-WL/onepower.git


or your fork of it:

.. code-block:: bash

    git clone https://github.com/<your-username>/onepower.git

Move to the directory and install with

.. code-block:: bash

    pip install -e ".[dev]"

This will install all dependencies -- both for using and developing the package (testing,
creating docs, etc.). Again, see above about dependencies with ``conda`` if you are
using a ``conda`` environment (which is recommended).
