TATOOINE
========

.. image:: https://github.com/KiDS-WL/halomodel_for_cosmosis/blob/andrej_dev/logo.png?raw=true
   :alt: TATOOINE

**TATOOINE**
*Tools for Analyzing Two-point and One-point cOrrelations, Intrinsic alignments, and Nonlinear structurE*

"Modeling small-scale structure in a galaxy not so far away."

TATOOINE is a Python package for computing power spectra and one-point statistics using the halo model framework. It is designed for studying the galaxy–matter connection, cosmological structure formation, and intrinsic alignments, especially in the non-linear regime.

Features
--------

- Non-linear **matter–matter**, **galaxy–galaxy**, and **galaxy–matter** power spectra
- Predictions of **stellar mass functions** and / or **luminosity functions**
- Modeling of **intrinsic alignments** using the halo model approach
- Built on a flexible, extensible halo model architecture

TATOOINE is ideal for:

- Modeling of galaxy surveys
- Cosmological parameter inference
- Understanding the galaxy–halo connection in nonlinear regimes

📦 `View on GitHub <https://github.com/yourusername/tatooine>`_

📄 `Read the Docs <https://tatooine.readthedocs.io>`_

💾 Install via PyPI

🖖 May the stats be with you.


Installation
------------

.. code-block:: bash

    pip install tatooine

Example usage
-------------

.. code-block:: python

    from tatooine import Spectra
    ps = Spectra(...)
    pk_mm = ps.power_spectrum_mm.pk_tot
    pk_mm_1h = ps.power_spectrum_mm.pk_1h
    pk_mm_2h = ps.power_spectrum_mm.pk_2h

One can also use the accompanying CosmoSIS interface and use the TATOOINE to predict the power spectra in the CosmoSIS framework. That opens up many more options, specifically on the observables and statistics to predict.

LaTeX Acronym Definition
------------------------

Use the `acro <https://ctan.org/pkg/acro>`_ package or just define it manually depending on your style. Here's both:

.. code-block:: latex

    \usepackage{acro}
    \DeclareAcronym{tatooine}{
      short = TATOOINE ,
      long  = Tools for Analyzing Two-point and One-point cOrrelations, Intrinsic alignments, and Nonlinear structurE ,
      class = abbrev ,
      format=\textsc
    }

.. code-block:: latex

    \newcommand{\tatooine}{\textsc{TATOOINE} (Tools for Analyzing Two-point and One-point cOrrelations, Intrinsic alignments, and Nonlinear structurE)}

Citation
--------

.. code-block:: bibtex

    @misc{tatooine,
      author       = {Your Name and Collaborators},
      title        = {TATOOINE: Tools for Analyzing Two-point and One-point cOrrelations, Intrinsic alignments, and Nonlinear structurE},
      year         = {2025},
      howpublished = {\url{https://github.com/yourusername/tatooine}},
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