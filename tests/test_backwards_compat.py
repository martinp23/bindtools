import os
import sys
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestBackwardsCompatibility:
    def test_legacy_binding_as_bd(self):
        from bindtools import binding as bd

        expected_symbols = [
            "EquilibriumError",
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
            "ObsType",
            "sigmaMapping",
            "deltaToConc",
            "concToDelta",
            "concToLinearObs",
            "concToObservable",
            "fitfun",
            "bindingModel",
            "simulateModel",
            "getCalcData",
            "makeFitResidPlot",
            "saveFitCSV",
            "calclnprob",
            "log_prob",
            "log_prior",
            "MCMC",
            "doMCMC",
            "plotMCMC",
            "plotCorner",
            "getTau",
            "makeMCMCLabels",
            "mcmchelper",
        ]

        for sym in expected_symbols:
            assert hasattr(bd, sym), f"Missing symbol {sym} in bindtools.binding"
            assert getattr(bd, sym) is not None

    def test_legacy_direct_imports(self):
        from bindtools.binding import (
            bindingModel,
            getConcs,
            ObsType,
            MCMC,
            EquilibriumError,
            calc_analytical_speciation,
        )

        assert callable(bindingModel)
        assert callable(getConcs)
        assert callable(ObsType)
        assert callable(MCMC)
        assert issubclass(EquilibriumError, Exception)
        assert callable(calc_analytical_speciation)

    def test_modern_top_level_imports(self):
        import bindtools as bt

        top_level_symbols = [
            "bindingModel",
            "MCMC",
            "ObsType",
            "getConcs",
            "getConcsScipy",
            "calc_analytical_speciation",
            "simulateModel",
            "getCalcData",
            "makeFitResidPlot",
            "saveFitCSV",
            "EquilibriumError",
            "binding",
        ]
        for sym in top_level_symbols:
            assert hasattr(bt, sym), f"Missing symbol {sym} in bindtools top-level"

    def test_subpackage_imports(self):
        from bindtools.speciation import getConcs, calc_analytical_speciation
        from bindtools.observables import ObsType, concToObservable
        from bindtools.models import bindingModel, simulateModel
        from bindtools.mcmc import MCMC, doMCMC
        from bindtools.plotting import makeFitResidPlot, plotMCMC, plotCorner
        from bindtools.io import saveFitCSV

        assert callable(getConcs)
        assert callable(calc_analytical_speciation)
        assert callable(ObsType)
        assert callable(concToObservable)
        assert callable(bindingModel)
        assert callable(simulateModel)
        assert callable(MCMC)
        assert callable(doMCMC)
        assert callable(makeFitResidPlot)
        assert callable(plotMCMC)
        assert callable(plotCorner)
        assert callable(saveFitCSV)

