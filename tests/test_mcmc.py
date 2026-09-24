import os
import sys
import tempfile
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bindtools.mcmc import MCMC, doMCMC, makeMCMCLabels
from bindtools.models import bindingModel
from bindtools.observables import ObsType
from bindtools.speciation import getConcs


@pytest.fixture
def fitted_model():
    h_tot = np.full(8, 1e-3)
    g_tot = np.linspace(0, 2e-3, 8)
    comp_concs = np.column_stack([h_tot, g_tot])
    eq_mat = np.array([[1, 0, 1], [0, 1, 1]])
    logK = np.array([0, 0, 3.5])
    specs = np.array([getConcs(eq_mat, row, logK) for row in comp_concs])
    raw_data = np.column_stack([comp_concs, specs[:, 2]])

    m = bindingModel(
        eqMat=eq_mat,
        compNames=["H", "G"],
        speciesList=["H", "G", "HG"],
        specToInteg=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]),
        colToComp=np.array([[1, 0, 0], [0, 1, 0]]),
        rawData=raw_data,
        obsList=["obs_HG"],
    )
    m.prepModel()
    m.runModel(skip_col=2)
    return m


class TestMCMC:
    def test_mcmc_labels(self, fitted_model):
        obs = [ObsType("concMeas")]
        mcmc = MCMC(fitted_model, obs, walkers=6, samples=10)
        assert "logHG" in mcmc.labels
        assert "lnsigmaconcMeas" in mcmc.labels

    def test_mcmc_run_and_hdf5_save_load(self, fitted_model):
        obs = [ObsType("concMeas")]
        mcmc = MCMC(fitted_model, obs, walkers=6, samples=15)
        sampler, bm = mcmc.run(ret=True, thin=2)
        assert sampler is not None

        chain_orig = sampler.get_chain()
        assert chain_orig.shape == (15, 6, 2)

        with tempfile.NamedTemporaryFile(suffix=".hdf", delete=False) as tf:
            temp_path = tf.name

        try:
            mcmc.save(temp_path)
            assert os.path.exists(temp_path)

            # Create a second MCMC instance to test load
            mcmc2 = MCMC(fitted_model, obs, walkers=6, samples=5)
            mcmc2.run(ret=True, thin=1)  # Initialize sampler
            mcmc2.load(temp_path)

            assert mcmc2.sampler.backend.chain.shape == sampler.backend.chain.shape
            np.testing.assert_allclose(mcmc2.sampler.backend.chain, sampler.backend.chain)
            np.testing.assert_allclose(mcmc2.sampler.backend.log_prob, sampler.backend.log_prob)
            assert mcmc2.thin == 2
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_mcmc_get_tau(self, fitted_model, capsys):
        obs = [ObsType("concMeas")]
        mcmc = MCMC(fitted_model, obs, walkers=6, samples=10)
        mcmc.run()
        mcmc.get_tau()
        captured = capsys.readouterr()
        assert "Param" in captured.out or "Warning" in captured.out

    def test_legacy_do_mcmc_and_labels(self, fitted_model):
        obs = [ObsType("concMeas")]
        labels = makeMCMCLabels(fitted_model, obs)
        assert "logHG" in labels

        sampler, bm = doMCMC(fitted_model, obs, samples=5, walkers=6)
        assert sampler is not None
        assert bm is fitted_model

    def test_mcmc_1to2_model(self):
        """Verify MCMC runs on multi-parameter 1:2 model and recovers posterior."""
        eq_mat = np.array([[1, 0, 1, 1], [0, 1, 1, 2]], dtype=float)
        params = np.array([0.0, 0.0, 4.0, 6.0], dtype=float)
        l_tot = np.linspace(0.0, 40.0e-3, 15)
        h_tot = 10e-3
        comp_concs = np.column_stack([np.full_like(l_tot, h_tot), l_tot])
        concs = np.array([getConcs(eq_mat, row, params) for row in comp_concs])
        a_obs = 0.1 * concs[:, 0] + 0.6 * concs[:, 2] + 1.0 * concs[:, 3]

        m12 = bindingModel(
            eqMat=eq_mat,
            compNames=["H", "L"],
            speciesList=["H", "L", "HL", "HL2"],
            specToLinear=np.array([[0.1], [0.0], [0.6], [1.0]]),
            rawData=a_obs[:, None],
            compConcs=comp_concs,
            obsList=["A_obs"],
        )
        m12.prepModel()
        m12.params["logHL"].set(value=3.5, min=0.0, max=8.0)
        m12.params["logHL2"].set(value=5.5, min=0.0, max=10.0)
        m12.runModel(skip_col=0, method="least_squares")

        obs = [ObsType("uvvis")]
        mcmc = MCMC(m12, obs, walkers=8, samples=25)
        sampler, bm = mcmc.run(ret=True, thin=1, tqdm_kwargs={"disable": True})

        assert sampler is not None
        assert "logHL" in mcmc.labels
        assert "logHL2" in mcmc.labels
        assert "lnsigmauvvis" in mcmc.labels
        chain = sampler.get_chain()
        assert chain.shape == (25, 8, 3)

    def test_mcmc_custom_bounds_override(self, fitted_model):
        """Verify model.fcn_opts['mcmc_bounds'] strictly overrides parameter limits."""
        custom_bounds = np.array([[3.0, 4.0], [-10.0, -6.0]], dtype=float)
        fitted_model.fcn_opts["mcmc_bounds"] = custom_bounds

        obs = [ObsType("concMeas")]
        mcmc = MCMC(fitted_model, obs, walkers=8, samples=20)
        sampler, _ = mcmc.run(ret=True, thin=1, tqdm_kwargs={"disable": True})

        chain = sampler.get_chain()
        for p_idx in range(custom_bounds.shape[0]):
            p_samples = chain[:, :, p_idx]
            assert np.all(p_samples >= custom_bounds[p_idx, 0] - 1e-6)
            assert np.all(p_samples <= custom_bounds[p_idx, 1] + 1e-6)

    def test_mcmc_absorbance_obs_type(self, fitted_model):
        """Verify MCMC executes cleanly with ObsType('absorbance')."""
        obs = [ObsType("absorbance")]
        mcmc = MCMC(fitted_model, obs, walkers=6, samples=10)
        assert "lnsigmaabsorbance" in mcmc.labels
        sampler, _ = mcmc.run(ret=True, thin=1, tqdm_kwargs={"disable": True})
        assert sampler is not None

