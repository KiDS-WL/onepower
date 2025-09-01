
.. image:: https://andrej.dvrnk.si/page/wp-content/uploads/2025/08/logosmall_white_merged.png
    :class: only-dark

.. image:: https://andrej.dvrnk.si/page/wp-content/uploads/2025/08/logosmall_black_merged.png
    :class: only-light

.. raw:: html

  <div align="center">
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
- Includes a interface module for interfacing the code with the `CosmoSIS <https://github.com/joezuntz/cosmosis>`_ (cloning of GitHub repository required for ease of use)

OnePower is ideal for:

- Modeling of galaxy surveys
- Cosmological parameter inference
- Understanding the galaxy-halo connection in nonlinear regimes

📦 `View on GitHub <https://github.com/KiDS-WL/onepower>`_

📄 `Documentation <https://kids-wl.github.io/onepower/index.html>`_

💾 `Install via PyPI <https://github.com/KiDS-WL/onepower>`_

Example usage
-------------

As OnePower has defaults for all of its parameters, a reasonable resulting power spectra can be calculated by passing no parameters:

.. code-block:: python

    from onepower import Spectra
    ps = Spectra()
    pk_mm = ps.power_spectrum_mm.pk_tot
    pk_mm_1h = ps.power_spectrum_mm.pk_1h
    pk_mm_2h = ps.power_spectrum_mm.pk_2h

One can also use the accompanying CosmoSIS interface and use the OnePower to predict the power spectra in the CosmoSIS framework. That opens up many more options, specifically on the observables and statistics to predict.
See the .yaml file for the use of that specific interface module in CosmoSIS Standard Library or in cosmosis_modules folder

If you want to calculate the covariance matrix for the power spectra calculated using OnePower, you can use the sister package `OneCovariance <https://github.com/rreischke/OneCovariance>`_!


Attribution
-----------

This code originated from the merger of the IA halo model repository of Maria-Cristina Fortuna and used in `Fortuna et al. 2021 <https://doi.org/10.1093/mnras/staa3802>`_, and the halo model code used in `Dvornik et al. 2023 <https://doi.org/10.1051/0004-6361/202245158>`_ and earlier papers. It is designed so that it can natively interact with `CosmoSIS standard library <https://github.com/joezuntz/cosmosis-standard-library>`_.
Please cite the above papers if you find this code useful in your research:

.. code-block:: bibtex

    @ARTICLE{Fortuna2021,
      author = {{Fortuna}, Maria Cristina and {Hoekstra}, Henk and {Joachimi}, Benjamin and {Johnston}, Harry and {Chisari}, Nora Elisa and {Georgiou}, Christos and {Mahony}, Constance},
      title = "{The halo model as a versatile tool to predict intrinsic alignments}",
      journal = {\mnras},
      keywords = {gravitational lensing: weak, galaxies: haloes, galaxies: statistics, cosmology: theory, Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Astrophysics of Galaxies},
      year = 2021,
      month = feb,
      volume = {501},
      number = {2},
      pages = {2983-3002},
      doi = {10.1093/mnras/staa3802},
      archivePrefix = {arXiv},
      eprint = {2003.02700},
      primaryClass = {astro-ph.CO},
      adsurl = {https://ui.adsabs.harvard.edu/abs/2021MNRAS.501.2983F},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

    @ARTICLE{Dvornik2023,
      author = {{Dvornik}, Andrej and {Heymans}, Catherine and {Asgari}, Marika and {Mahony}, Constance and {Joachimi}, Benjamin and {Bilicki}, Maciej and {Chisari}, Elisa and {Hildebrandt}, Hendrik and {Hoekstra}, Henk and {Johnston}, Harry and {Kuijken}, Konrad and {Mead}, Alexander and {Miyatake}, Hironao and {Nishimichi}, Takahiro and {Reischke}, Robert and {Unruh}, Sandra and {Wright}, Angus H.},
      title = "{KiDS-1000: Combined halo-model cosmology constraints from galaxy abundance, galaxy clustering, and galaxy-galaxy lensing}",
      journal = {\aap},
      keywords = {gravitational lensing: weak, methods: statistical, cosmological parameters, galaxies: halos, dark matter, large-scale structure of Universe, Astrophysics - Cosmology and Nongalactic Astrophysics},
      year = 2023,
      month = jul,
      volume = {675},
      eid = {A189},
      pages = {A189},
      doi = {10.1051/0004-6361/202245158},
      archivePrefix = {arXiv},
      eprint = {2210.03110},
      primaryClass = {astro-ph.CO},
      adsurl = {https://ui.adsabs.harvard.edu/abs/2023A&A...675A.189D},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

Contents
--------
.. toctree::
   :maxdepth: 1

   installation
   tutorials
   api
   contributing
   license
   changelog
   contributors
   developing

Disclaimer
----------

This software is not affiliated with Tolkien Enterprises or any related franchise. The name "OnePower" is used solely as a thematic reference.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
