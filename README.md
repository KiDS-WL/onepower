
# TATOOINE

**TATOOINE**  
*Tools for Analyzing Two-point and One-point correlations, Intrinsic alignments, and nonlinEar structure*

TATOOINE is a Python package for computing power spectra and one-point statistics using the halo model framework. It is designed for studying the galaxy–matter connection, cosmological structure formation, and intrinsic alignments, especially in the non-linear regime.

### Features
- Non-linear **matter–matter**, **galaxy–galaxy**, and **galaxy–matter** power spectra
- Predictions of **stellar mass functions** and / or **luminosity functions**
- Modeling of **intrinsic alignments** using the halo model approach
- Built on a flexible, extensible halo model architecture

> “Modeling small-scale structure in a galaxy not so far away.”

### TATOOINE is ideal for:

- Forward modeling of galaxy surveys
- Cosmological parameter inference
- Understanding the galaxy–halo connection in nonlinear regimes

📦 View on GitHub

📄 Read the Docs

💾 Install via PyPI

🖖 May the stats be with you.

---

### Installation

```bash
pip install tatooine
```

### Example usage

```
from tatooine import Spectra

ps = Spectra(...)
pk_mm = ps.power_spectrum_mm.pk_tot
pk_mm_1h = ps.power_spectrum_mm.pk_1h
pk_mm_2h = ps.power_spectrum_mm.pk_2h
```

This code originated from the IA halo model repository of Maria-Cristina Fortuna and it is a designed so that it can natively interact with [CosmoSIS standard library](https://github.com/joezuntz/cosmosis-standard-library).

Dependencies:

1. The [halo mass function calculator, hmf](https://hmf.readthedocs.io/en/3.3.4/)
2. The hmf interface [halomod](https://github.com/halomod/halomod).
3. The [Dark Emulator](https://dark-emulator.readthedocs.io/en/latest/)
4. The [Cosmopower](https://alessiospuriomancini.github.io/cosmopower/) machine learning library


The fits files are here: [Luminosity_redshift](https://ruhr-uni-bochum.sciebo.de/s/ZdAE6nTf0OPyV6S)




---

### 📄 LaTeX Acronym Definition

Use the [`acro`](https://ctan.org/pkg/acro) package or just define it manually depending on your style. Here's both:

```latex
\usepackage{acro}

\DeclareAcronym{tatooine}{
  short = TATOOINE ,
  long  = Tools for Analyzing Two-point and One-point correlations, Intrinsic alignments, and nonlinEar structure ,
  class = abbrev ,
  format=\textsc
}
```

```
\newcommand{\tatooine}{\textsc{TATOOINE} (Tools for Analyzing Two-point and One-point correlations, Intrinsic alignments, and nonlinEar structure)}
```


### 📄 Citation

```
@misc{tatooine,
  author       = {Your Name and Collaborators},
  title        = {TATOOINE: Tools for Analyzing Two-point and One-point correlations, Intrinsic alignments, and nonlinEar structure},
  year         = {2025},
  howpublished = {\url{https://github.com/yourusername/tatooine}},
  note         = {Version 1.0},
}
```