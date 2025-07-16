OnePower
========

.. raw:: html

      <div align="center">
        <picture>
          <source
            srcset="https://andrej.dvrnk.si/page/wp-content/uploads/2025/07/logosmall_black.png"
            media="(prefers-color-scheme: light)"
          />
          <source
            srcset="https://andrej.dvrnk.si/page/wp-content/uploads/2025/07/logosmall_white.png"
            media="(prefers-color-scheme: dark)"
          />
        <img
          src="https://andrej.dvrnk.si/page/wp-content/uploads/2025/07/logosmall_black.png"
          alt="Logo"
        />
      </picture>
    <p align="center">
     <i>"The One Tool to Predict All Power Spectra."</i>
    </p>
    </div>

OnePower is a Python package for computing power spectra and one-point statistics using the halo model framework. It is designed for studying the galaxy-matter connection, cosmological structure formation, and intrinsic alignments, especially in the non-linear regime.

Features
--------

- Non-linear **matter-matter**, **galaxy-galaxy**, and **galaxy-matter** power spectra
- Predictions of **stellar mass functions** and / or **luminosity functions**
- Modeling of **intrinsic alignments** using the halo model approach
- Built on a flexible, extensible halo model architecture
- Includes a interface module for interfacing the code with the `CosmoSIS <https://github.com/joezuntz/cosmosis>`_

OnePower is ideal for:

- Modeling of galaxy surveys
- Cosmological parameter inference
- Understanding the galaxy-halo connection in nonlinear regimes

📦 `View on GitHub <https://github.com/yourusername/OnePower>`_

📄 `Read the Docs <https://onepower.readthedocs.io>`_

💾 Install via PyPI

Example usage
-------------

.. code-block:: python

    from onepower import Spectra
    ps = Spectra(...)
    pk_mm = ps.power_spectrum_mm.pk_tot
    pk_mm_1h = ps.power_spectrum_mm.pk_1h
    pk_mm_2h = ps.power_spectrum_mm.pk_2h

One can also use the accompanying CosmoSIS interface and use the OnePower to predict the power spectra in the CosmoSIS framework. That opens up many more options, specifically on the observables and statistics to predict.
See the .yaml file for the use of that specific interface module in CosmoSIS Standard Library or in cosmosis_modules folder

If you want to calculate the covariance matrix for the power spectra calculated using OnePower, you can use the sister package `OneCovariance <https://github.com/rreischke/OneCovariance>`_!


Attribution
-----------

This code originated from the IA halo model repository of Maria-Cristina Fortuna and used in Fortuna et al. 2021, and the halo model code used in Dvornik et al. 2023 and earlier papers. It is designed so that it can natively interact with `CosmoSIS standard library <https://github.com/joezuntz/cosmosis-standard-library>`_.
Please also cite the papers below if you find this code useful in your research:

.. code-block:: bibtex

    @misc{OnePower,
      author       = {Your Name and Collaborators},
      title        = {OnePower: A Python package for calculating power spectra using the halo model approach.},
      year         = {2025},
      howpublished = {\url{https://github.com/yourusername/onepower}},
      note         = {Version 1.0},
    }

Disclaimer
----------

This software is not affiliated with Tolkien Enterprises or any related franchise. The name "OnePower" is used solely as a thematic reference.