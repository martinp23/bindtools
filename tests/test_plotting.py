import os
import sys
import matplotlib

# Enforce non-interactive backend for headless test runs
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import bindtools as bt
from bindtools.plotting import makeFitResidPlot, plotMCMC, plotCorner


@pytest.fixture
def sample_mcmc_and_model():
    eq_mat = np.array([[1, 0, 1], [0, 1, 1]], dtype=float)
    params = np.array([0.0, 0.0, 4.0], dtype=float)
    comp_concs = np.column_stack([np.full(10, 1e-3), np.linspace(0, 2e-3, 10)])
    concs = np.array([bt.getConcs(eq_mat, row, params) for row in comp_concs])

    model = bt.bindingModel(
        eqMat=eq_mat,
        compNames=["H", "L"],
        speciesList=["H", "L", "HL"],
        specToInteg=np.array([[0.0], [0.0], [1.0]]),
        rawData=concs[:, 2:],
        compConcs=comp_concs,
        obsList=["[HL]"],
    )
    model.prepModel()
    model.runModel(skip_col=0, method="least_squares")

    obs = [bt.ObsType("concMeas")]
    mcmc = bt.MCMC(model, obs, walkers=6, samples=15)
    mcmc.run(thin=1, tqdm_kwargs={"disable": True})

    yield mcmc, model
    plt.close("all")


class TestPlottingSmokeTests:
    def test_make_fit_resid_plot(self, sample_mcmc_and_model):
        _, model = sample_mcmc_and_model
        makeFitResidPlot(model)
        assert len(plt.get_fignums()) > 0
        plt.close("all")

    def test_mcmc_plot_chain(self, sample_mcmc_and_model):
        mcmc, _ = sample_mcmc_and_model
        mcmc.plot_chain(title="Test Chain")
        assert len(plt.get_fignums()) > 0
        plt.close("all")

    def test_mcmc_plot_corner(self, sample_mcmc_and_model):
        mcmc, _ = sample_mcmc_and_model
        fig = mcmc.plot_corner(title="Test Corner", burnin=2)
        assert fig is not None
        plt.close("all")

    def test_legacy_plot_mcmc_and_corner(self, sample_mcmc_and_model):
        mcmc, _ = sample_mcmc_and_model
        plotMCMC(mcmc.sampler, mcmc.labels)
        plotCorner(mcmc.sampler, mcmc.labels)
        plt.close("all")

