# Installation

`rustmatrix` ships pre-built wheels for every supported platform, so the
one-liner is the same everywhere:

```bash
pip install rustmatrix
```

Only fall back to a source build if a wheel isn't available for your
platform or you want to hack on the Rust kernels.

## Pre-built wheels

Wheels are published to PyPI on every release. The matrix:

| OS | Architectures | Python versions |
|---|---|---|
| macOS | arm64, x86_64 | 3.10 – 3.13 |
| Linux (manylinux2014) | x86_64, aarch64 | 3.10 – 3.13 |
| Windows | x86_64 | 3.10 – 3.13 |

The wheels are `abi3`, so one wheel per OS × arch works across Python
versions from 3.10 onwards.

::::{tab-set}

:::{tab-item} macOS
:sync: macos

```bash
pip install rustmatrix
```

Apple Silicon (arm64) and Intel (x86_64) wheels are published
separately; `pip` picks the right one based on your interpreter.
:::

:::{tab-item} Linux
:sync: linux

```bash
pip install rustmatrix
```

Manylinux2014 wheels for `x86_64` and `aarch64`. If you're on a very
old distribution (glibc < 2.17), the wheel won't load; use the source
build below.
:::

:::{tab-item} Windows
:sync: windows

```powershell
pip install rustmatrix
```

A single `x86_64` Windows wheel. If you're on Windows on ARM, follow
the source build path.
:::

::::

## From source

A source build compiles the Rust extension. You need a C toolchain,
Rust (stable, ≥ 1.82 works), and `pip` — no Fortran, no BLAS.

::::{tab-set}

:::{tab-item} macOS
:sync: macos

```bash
xcode-select --install
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install rustmatrix --no-binary rustmatrix
```
:::

:::{tab-item} Linux
:sync: linux

```bash
# Debian / Ubuntu — adjust for your distro
sudo apt install build-essential
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install rustmatrix --no-binary rustmatrix
```
:::

:::{tab-item} Windows
:sync: windows

1. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   (C++ workload).
2. Install Rust via [rustup-init.exe](https://rustup.rs/).
3. From a Developer PowerShell:

   ```powershell
   pip install rustmatrix --no-binary rustmatrix
   ```
:::

::::

## Development install

For hacking on rustmatrix itself:

```bash
git clone https://github.com/swnesbitt/rustmatrix.git
cd rustmatrix
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -U pip maturin
pip install -e ".[test]"
maturin develop --release
pytest tests/
```

`maturin develop --release` builds the Rust extension in-place and
installs rustmatrix as an editable package. Re-run it whenever you
touch `src/*.rs`.

## Verifying the install

```bash
python -c "import rustmatrix; print(rustmatrix.__version__)"
```

For a real first calculation, see [](quickstart).

## Troubleshooting

**"No matching wheel"**: you're on a platform without a published
wheel (Windows on ARM, musl Linux, or a very old glibc). Follow the
source-build instructions above.

**`ImportError: cannot import name '_core'`**: the Rust extension
didn't build or didn't install. If you're developing, run
`maturin develop --release` again. If you `pip install`ed, try
`pip install --force-reinstall rustmatrix`.

**Apple Silicon NumPy ABI**: if you see symbols missing from NumPy,
make sure you haven't mixed a conda NumPy with a pip-installed
rustmatrix. Use one or the other consistently in a given environment.

**`maturin develop` fails with "no virtual environment detected"**:
activate your venv first. `maturin develop` won't install into the
system Python by design.
