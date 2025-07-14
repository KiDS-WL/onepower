Installation
============

This page will guide you through installing ``onepower`` -- either as purely a user, or
as a potential developer.

Dependencies
------------
``onepower`` has a number of dependencies, all of which should be automatically installed
as you install the package itself. You therefore do not need to worry about installing
them yourself, except in some circumstances.

User Install
------------
You may install the latest release of ``onepower`` using ``pip``::

    pip install onepower

This will install all uninstalled dependencies (see previous section).
Alternatively, for the very bleeding edge, install from the master branch of the repo::

    pip install onepower @ git+git:

Developer Install
-----------------
If you intend to develop ``onepower``, clone the repository (or your fork of it)::

    git clone https://github.com/<your-username>/onepower.git

Move to the directory and install with::

    pip install -e ".[dev]"

This will install all dependencies -- both for using and developing the package (testing,
creating docs, etc.). Again, see above about dependencies with ``conda`` if you are
using a ``conda`` environment (which is recommended).

.. note:: Once the package is installed, you will need to locally run ``pre-commit install``,
          to have constant checks on your code formatting before commits are accepted.
