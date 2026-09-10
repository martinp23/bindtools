"""Compatibility façade re-exporting all public and internal symbols from modular subpackages.

This module preserves 100% backwards compatibility with historical code and downstream packages
(e.g., `import bindtools.binding as bd` or `from bindtools.binding import bindingModel, getConcs`).
"""

from bindtools.exceptions import EquilibriumError
from bindtools.speciation import (
    apply_along_axis_0,
    _apply_along_axis_0,
    specCalc,
    _specJac,
    specJac,
    specObj,
    DoNR,
    getConcs,
    getConcsScipy,
    _solve_cubic_vectorized,
    calc_analytical_speciation,
    _infer_simple_fast_exchange_topology,
)
from bindtools.observables import (
    ObsType,
    sigmaMapping,
    deltaToConc,
    concToDelta,
    concToLinearObs,
    concToObservable,
)
from bindtools.models import (
    fitfun,
    bindingModel,
    simulateModel,
    getCalcData,
)
from bindtools.mcmc import (
    calclnprob,
    log_prob,
    log_prior,
    MCMC,
    doMCMC,
    getTau,
    makeMCMCLabels,
    mcmchelper,
)
from bindtools.plotting import (
    makeFitResidPlot,
    plotMCMC,
    plotCorner,
)
from bindtools.io import saveFitCSV

__all__ = [
    # Exceptions
    "EquilibriumError",
    # Speciation
    "apply_along_axis_0",
    "_apply_along_axis_0",
    "specCalc",
    "_specJac",
    "specJac",
    "specObj",
    "DoNR",
    "getConcs",
    "getConcsScipy",
    "_solve_cubic_vectorized",
    "calc_analytical_speciation",
    "_infer_simple_fast_exchange_topology",
    # Observables
    "ObsType",
    "sigmaMapping",
    "deltaToConc",
    "concToDelta",
    "concToLinearObs",
    "concToObservable",
    # Models & Simulation
    "fitfun",
    "bindingModel",
    "simulateModel",
    "getCalcData",
    # Visualization & IO
    "makeFitResidPlot",
    "plotMCMC",
    "plotCorner",
    "saveFitCSV",
    # MCMC
    "calclnprob",
    "log_prob",
    "log_prior",
    "MCMC",
    "doMCMC",
    "getTau",
    "makeMCMCLabels",
    "mcmchelper",
]
