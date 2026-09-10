"""bindtools: Python library for supramolecular chemistry modeling, fitting, and speciation.
"""

from bindtools.exceptions import EquilibriumError
from bindtools.speciation import (
    getConcs,
    getConcsScipy,
    calc_analytical_speciation,
)
from bindtools.observables import (
    ObsType,
    concToObservable,
    concToDelta,
    concToLinearObs,
)
from bindtools.models import (
    bindingModel,
    simulateModel,
    getCalcData,
)
from bindtools.mcmc import (
    MCMC,
)
from bindtools.plotting import (
    makeFitResidPlot,
)
from bindtools.io import (
    saveFitCSV,
)
from bindtools import binding

__all__ = [
    "binding",
    "EquilibriumError",
    "getConcs",
    "getConcsScipy",
    "calc_analytical_speciation",
    "ObsType",
    "concToObservable",
    "concToDelta",
    "concToLinearObs",
    "bindingModel",
    "simulateModel",
    "getCalcData",
    "MCMC",
    "makeFitResidPlot",
    "saveFitCSV",
]

