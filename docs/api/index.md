# Python API reference

The public Python API of rustmatrix. Each module below is documented
from its source docstrings via Sphinx autosummary.

```{toctree}
:maxdepth: 1

rustmatrix
```

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   rustmatrix
   rustmatrix.scatterer
   rustmatrix.psd
   rustmatrix.radar
   rustmatrix.scatter
   rustmatrix.hd_mix
   rustmatrix.spectra
   rustmatrix.spectra.beam
   rustmatrix.refractive
   rustmatrix.tmatrix_aux
   rustmatrix.orientation
   rustmatrix.quadrature
```

The Rust crate that backs the Python extension is documented separately
on [docs.rs](https://docs.rs/rustmatrix); see [](../rust-api/index).
