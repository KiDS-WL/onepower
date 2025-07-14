OnePower
========

.. image:: https://github.com/KiDS-WL/halomodel_for_cosmosis/blob/andrej_dev/logo.png?raw=true
   :alt: OnePower

**OnePower**
*A Python package for calculating power spectra using the halo model approach.*

"The One Tool to Predict All Power Spectra."

OnePower is a Python package for computing power spectra and one-point statistics using the halo model framework. It is designed for studying the galaxy-matter connection, cosmological structure formation, and intrinsic alignments, especially in the non-linear regime.

Features
--------

- Non-linear **matter-matter**, **galaxy-galaxy**, and **galaxy-matter** power spectra
- Predictions of **stellar mass functions** and / or **luminosity functions**
- Modeling of **intrinsic alignments** using the halo model approach
- Built on a flexible, extensible halo model architecture

OnePower is ideal for:

- Modeling of galaxy surveys
- Cosmological parameter inference
- Understanding the galaxy-halo connection in nonlinear regimes

📦 `View on GitHub <https://github.com/yourusername/onepower>`_

📄 `Read the Docs <https://onepower.readthedocs.io>`_

💾 Install via PyPI

Installation
------------

.. code-block:: bash

    pip install onepower

Example usage
-------------

.. code-block:: python

    from onepower import Spectra
    ps = Spectra(...)
    pk_mm = ps.power_spectrum_mm.pk_tot
    pk_mm_1h = ps.power_spectrum_mm.pk_1h
    pk_mm_2h = ps.power_spectrum_mm.pk_2h

One can also use the accompanying CosmoSIS interface and use the OnePower to predict the power spectra in the CosmoSIS framework. That opens up many more options, specifically on the observables and statistics to predict.

If you want to calculate the covariance matrix for the power spectra calculated using OnePower, you can use the sister package `OneCovariance <https://github.com/rreischke/OneCovariance>`_!

Citation
--------

.. code-block:: bibtex

    @misc{tatooine,
      author       = {Your Name and Collaborators},
      title        = {OnePower: A Python package for calculating power spectra using the halo model approach.},
      year         = {2025},
      howpublished = {\url{https://github.com/yourusername/onepower}},
      note         = {Version 1.0},
    }

This code originated from the IA halo model repository of Maria-Cristina Fortuna and it is designed so that it can natively interact with `CosmoSIS standard library <https://github.com/joezuntz/cosmosis-standard-library>`_.

Dependencies:
-------------

1. The `halo mass function calculator, hmf <https://hmf.readthedocs.io/en/3.3.4/>`_
2. The hmf interface `halomod <https://github.com/halomod/halomod>`_.
3. The `Dark Emulator <https://dark-emulator.readthedocs.io/en/latest/>`_
4. The `Cosmopower <https://alessiospuriomancini.github.io/cosmopower/>`_ machine learning library

The fits files are here: `Luminosity_redshift <https://ruhr-uni-bochum.sciebo.de/s/ZdAE6nTf0OPyV6S>`_


Disclaimer
----------

This software is not affiliated with Tolkien Enterprises or any related franchise. The name "OnePower" is used solely as a thematic reference.