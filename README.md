# bindtools

[![PyPI version](https://badge.fury.io/py/bindtools.svg)](https://badge.fury.io/py/bindtools)
[![Python Version](https://img.shields.io/pypi/pyversions/bindtools.svg)](https://pypi.org/project/bindtools/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`bindtools` is a Python library for supramolecular chemistry designed for modeling, fitting, and analyzing binding equilibria. It provides numerical and analytical solvers for complex chemical systems, robust parameter optimization, and Bayesian parameter estimation.

---

## Features

- **Speciation Solvers**: 
  - **Numerical**: A Newton-Raphson solver (`DoNR` and `getConcs`) JIT-compiled with `numba` for fast operation.
  - **Analytical**: High-speed analytical solvers for common topologies (e.g., `1:1`, `1:2`, `2:1` fast exchange).
- **Flexible Optimization**:
  - Uses `lmfit` to manage model parameters (binding constants, physical observables).
  - Handles various experimental data including NMR integrations, chemical shifts (NMR `deltaH`/`deltaF`), and linear concentration-weighted observables (UV-vis / fluorescence).
- **Bayesian Inference & Uncertainty Quantification**:
  - Uses `emcee` for Markov Chain Monte Carlo (MCMC) sampling.
  - Generates trace/chain convergence plots and corner plots using `corner` and `arviz`.
  - MCMC runs can be serialized and stored as HDF5 files for future analysis.

---

## Installation

`bindtools` is available on PyPI. You can install it directly using `pip`:

```bash
pip install bindtools
```

### Using Conda / Mamba (Recommended for Virtual Environments)

To avoid dependency conflicts, you can set up a dedicated environment with Conda/Mamba and install `bindtools` inside it:

```bash
# 1. Create and configure environment with base scientific dependencies
mamba create -n binding -c conda-forge \
  python jupyter tqdm ipython uncertainties lmfit scipy numpy emcee tqdm numba corner matplotlib numdifftools

# 2. Activate the environment
conda activate binding

# 3. Install bindtools via pip
pip install bindtools
```

---

## Quick Start

### 1. Speciation (Solving Concentration Problems)

You can compute the equilibrium concentration of free species (components and complexes) given initial total concentrations, a stoichiometry matrix, and equilibrium constants ($K$ values).

```python
import numpy as np
from bindtools import binding as bd

# Define total concentrations: 50 data points of Host (1e-3 M) and Guest (0 to 1e-2 M)
component_concs = np.zeros((50, 2))
component_concs[:, 0] = 1e-3
component_concs[:, 1] = np.linspace(0, 1e-2, 50)

# Stoichiometry / Equilibrium Matrix
# Row 0: Host balance, Row 1: Guest balance
# Columns represent: [Free Host, Free Guest, Host-Guest Complex (1:1)]
eq_mat = np.array([
    [1, 0, 1],  # [H]_tot = [H] + [HG]
    [0, 1, 1]   # [G]_tot = [G] + [HG]
])

# log10(K) values for each species. 
# Constants for free components are fixed at logK = 0.
# The complex (HG) has logK = 4 (K = 10,000 M^-1).
logK = np.array([0, 0, 4])

# Solve for concentrations at each point
results = []
for total_concs in component_concs:
    spec_concs = bd.getConcs(eq_mat, total_concs, logK)
    results.append(spec_concs)

results = np.array(results)
print("First point [H, G, HG]:", results[0])
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
