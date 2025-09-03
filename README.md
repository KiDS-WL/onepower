# OnePower

[![Tests](https://github.com/KiDS-WL/onepower/actions/workflows/tests.yml/badge.svg)](https://github.com/KiDS-WL/onepower/)
[![Docs](https://github.com/KiDS-WL/onepower/actions/workflows/documentation2.yml/badge.svg)](https://github.com/KiDS-WL/onepower/)
[![Coverage](https://raw.githubusercontent.com/KiDS-WL/onepower/main/coverage.svg)](https://github.com/KiDS-WL/onepower/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/KiDS-WL/onepower/main.svg)](https://results.pre-commit.ci/latest/github/KiDS-WL/onepower/main)

<picture>
	<source media="(prefers-color-scheme: dark)" srcset="https://andrej.dvrnk.si/page/wp-content/uploads/2025/08/logosmall_white_merged.png">
  <source media="(prefers-color-scheme: light)" srcset="https://andrej.dvrnk.si/page/wp-content/uploads/2025/08/logosmall_black_merged.png">
	<img align="left" alt="i" src="https://andrej.dvrnk.si/page/wp-content/uploads/2025/08/logosmall_black_merged.png">
</picture>

*<p align="center"> The One Tool to Predict All Power Spectra. </p>*

OnePower is a Python package for computing power spectra and one-point statistics using the halo model framework. It is designed for studying the galaxy-matter connection, cosmological structure formation, and intrinsic alignments, especially in the non-linear regime.

## Features

- Non-linear **matter-matter**, **galaxy-galaxy**, and **galaxy-matter** power spectra
- Predictions of **stellar mass functions** and/or **luminosity functions**
- Modeling of **intrinsic alignments** using the halo model approach
- Built on a flexible, extensible halo model architecture
- Includes an interface module for [CosmoSIS](https://github.com/joezuntz/cosmosis) (cloning of GitHub repository required for ease of use)

OnePower is ideal for:
- Modeling of galaxy surveys
- Cosmological parameter inference
- Understanding the galaxy-halo connection in nonlinear regimes

📦 [View on GitHub](https://github.com/KiDS-WL/onepower)

📄 [Documentation](https://kids-wl.github.io/onepower/index.html)

💾 [Install via PyPI](https://pypi.org/project/onepower/)


## Example usage

As OnePower has defaults for all of its parameters, a reasonable resulting power spectrum can be calculated by passing no parameters:

```python
from onepower import Spectra
ps = Spectra()
pk_mm = ps.power_spectrum_mm.pk_tot
pk_mm_1h = ps.power_spectrum_mm.pk_1h
pk_mm_2h = ps.power_spectrum_mm.pk_2h
```

You can also use the accompanying CosmoSIS interface to predict the power spectra in the CosmoSIS framework.
That opens up many more options, specifically on the observables and statistics to predict.
See the `.yaml` file in the CosmoSIS Standard Library or the `cosmosis_modules` folder for examples.

If you want to calculate the covariance matrix for the power spectra calculated using OnePower, you can use the sister package [OneCovariance](https://github.com/rreischke/OneCovariance)!

---

## Attribution

This code originated from the merger of the IA halo model repository of Maria-Cristina Fortuna (used in [Fortuna et al. 2021](https://doi.org/10.1093/mnras/staa3802)) and the halo model code used in [Dvornik et al. 2023](https://doi.org/10.1051/0004-6361/202245158) and earlier papers.
It is designed to natively interact with the [CosmoSIS standard library](https://github.com/joezuntz/cosmosis-standard-library).

Please cite the above papers if you find this code useful in your research:

```bibtex
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
```

---

## Disclaimer

This software is not affiliated with Tolkien Enterprises or any related franchise. The name "OnePower" is used solely as a thematic reference.
