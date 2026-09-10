import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest
import lmfit

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bindtools.models import bindingModel, simulateModel, getCalcData
from bindtools.speciation import getConcs
from bindtools.io import saveFitCSV


class TestBindingModelProperties:
    def test_comp_concs_from_raw_data(self):
        raw_data = np.array([
            [1e-3, 0.0, 0.1],
            [1e-3, 1e-3, 0.2],
        ])
        col_to_comp = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        m = bindingModel(
            eqMat=np.array([[1, 0, 1], [0, 1, 1]]),
            compNames=["H", "G"],
            speciesList=["H", "G", "HG"],
            colToComp=col_to_comp,
            rawData=raw_data,
        )
        np.testing.assert_allclose(m.compConcs, raw_data[:, :2])
        assert m.nComp == 2
        assert m.nConcs == 3

    def test_missing_data_raises_value_error(self):
        m = bindingModel(
            eqMat=np.array([[1, 0, 1], [0, 1, 1]]),
            compNames=["H", "G"],
            speciesList=["H", "G", "HG"],
        )
        with pytest.raises(ValueError, match="Data or mapping not available"):
            _ = m.compConcs

        with pytest.raises(ValueError, match="Data or mapping not available"):
            _ = m.specConcs


class TestLinearUVvisFit:
    """End-to-end UV-vis linear observable fit adapted from bindmc."""

    def test_uvvis_fit_recovers_binding_constant(self):
        log_beta_true = 6.0
        eps_H_true = 1000.0
        eps_HG_true = 5000.0
        H_total = 1e-4
        n_pts = 20

        G_total = np.linspace(0, 2e-4, n_pts)
        beta = 10**log_beta_true
        a_coef = 1.0
        b_coef = -(H_total + G_total + 1.0 / beta)
        c_coef = H_total * G_total
        HG = (-b_coef - np.sqrt(b_coef**2 - 4 * a_coef * c_coef)) / (2 * a_coef)
        H_free = H_total - HG
        absorbance = (eps_H_true * H_free + eps_HG_true * HG).reshape(-1, 1)
        comp_concs = np.column_stack([np.full(n_pts, H_total), G_total])
        raw_data = np.hstack([comp_concs, absorbance])

        eq_mat = np.array([[1, 0, 1], [0, 1, 1]], dtype=float)
        eps_H_p = lmfit.Parameter("eps_H_abs", value=800.0, min=0.1, max=1e5)
        eps_G_p = lmfit.Parameter("eps_G_abs", value=0.0, min=-1e-10, max=1e-10, vary=False)
        eps_HG_p = lmfit.Parameter("eps_HG_abs", value=3000.0, min=0.1, max=1e5)

        specToLinear = np.empty((3, 1), dtype=object)
        specToLinear[0, 0] = eps_H_p
        specToLinear[1, 0] = eps_G_p
        specToLinear[2, 0] = eps_HG_p

        m = bindingModel(
            eqMat=eq_mat,
            compNames=["H", "G"],
            speciesList=["H", "G", "HG"],
            colToComp=np.eye(2),
            rawData=raw_data,
        )
        m.specToLinear = specToLinear
        m.prepModel()
        m.params["eps_G_abs"].set(value=0.0, vary=False, min=-1e-10, max=1e-10)

        m.runModel(skip_col=2, method="least_squares")

        assert m.miniResult is not None
        assert m.miniResult.success or m.miniResult.redchi < 1e-3
        recovered_logK = m.miniResult.params["logHG"].value
        assert abs(recovered_logK - log_beta_true) < 0.2


class TestModelHelpersAndExport:
    @pytest.fixture
    def fitted_1to1_model(self):
        h_tot = np.full(10, 1e-3)
        g_tot = np.linspace(0, 2e-3, 10)
        comp_concs = np.column_stack([h_tot, g_tot])
        eq_mat = np.array([[1, 0, 1], [0, 1, 1]])
        logK = np.array([0, 0, 4.0])
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

    def test_calc_speciation(self, fitted_1to1_model):
        specs = fitted_1to1_model.calcSpeciation()
        assert specs.shape == (10, 3)
        np.testing.assert_allclose(specs[:, 0] + specs[:, 2], 1e-3, rtol=1e-4)

    def test_simulate_model(self, fitted_1to1_model):
        sim = simulateModel(fitted_1to1_model)
        assert sim.shape == (10, 1)

    def test_get_calc_data(self, fitted_1to1_model):
        calc_data = getCalcData(fitted_1to1_model)
        assert calc_data.shape == (10, 1)

    def test_save_fit_csv(self, fitted_1to1_model):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
            temp_path = tf.name
        try:
            saveFitCSV(fitted_1to1_model, mask=[0], filename=temp_path)
            df = pd.read_csv(temp_path)
            assert len(df) == 10
            assert "expt obs_HG" in df.columns
            assert "calc obs_HG" in df.columns
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
