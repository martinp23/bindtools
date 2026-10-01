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

    def test_get_calc_data_with_new_concs(self, fitted_1to1_model):
        """Verify getCalcData evaluates predictions on a new concentration grid without mutating the model."""
        n_dense = 25
        dense_concs = np.column_stack([np.full(n_dense, 1e-3), np.linspace(0, 4e-3, n_dense)])
        calc_dense = getCalcData(fitted_1to1_model, newConcs=dense_concs)
        assert calc_dense.shape == (n_dense, 1)

        # Ensure original model concentrations are not mutated
        assert fitted_1to1_model.compConcs.shape == (10, 2)

        # Analytical check: verify calc_dense matches theoretical [HG]
        eq_mat = np.array([[1, 0, 1], [0, 1, 1]])
        logK = np.array([0.0, 0.0, fitted_1to1_model.miniResult.params["logHG"].value])
        expected_hg = np.array([getConcs(eq_mat, row, logK)[2] for row in dense_concs])
        np.testing.assert_allclose(calc_dense.ravel(), expected_hg, rtol=1e-3, atol=1e-12)

    def test_fit_incomplete_saturation(self):
        """Fit a titration that stops well before saturation plateau."""
        h_tot = 10e-3
        l_tot = np.linspace(0.0, 2.0e-3, 11)  # only ~16% saturation
        comp_concs = np.column_stack([np.full_like(l_tot, h_tot), l_tot])
        eq_mat = np.array([[1, 0, 1], [0, 1, 1]])
        logK_true = 4.0
        params = np.array([0.0, 0.0, logK_true])
        specs = np.array([getConcs(eq_mat, row, params) for row in comp_concs])

        m = bindingModel(
            eqMat=eq_mat,
            compNames=["H", "L"],
            speciesList=["H", "L", "HL"],
            specToInteg=np.array([[0.0], [0.0], [1.0]]),
            rawData=specs[:, 2:],
            compConcs=comp_concs,
            obsList=["[HL]"],
        )
        m.prepModel()
        m.runModel(skip_col=0, method="least_squares")
        assert m.miniResult.success

    def test_fit_strided_sparse_data(self):
        """Verify fitting on sparse downsampled data recovers binding constant."""
        h_tot = 10e-3
        l_tot = np.linspace(0.0, 40.0e-3, 51)
        comp_concs = np.column_stack([np.full_like(l_tot, h_tot), l_tot])
        eq_mat = np.array([[1, 0, 1], [0, 1, 1]])
        logK_true = 4.0
        params = np.array([0.0, 0.0, logK_true])
        specs = np.array([getConcs(eq_mat, row, params) for row in comp_concs])

        # Subsample every 5th point (11 points total)
        m_sparse = bindingModel(
            eqMat=eq_mat,
            compNames=["H", "L"],
            speciesList=["H", "L", "HL"],
            specToInteg=np.array([[0.0], [0.0], [1.0]]),
            rawData=specs[::5, 2:],
            compConcs=comp_concs[::5],
            obsList=["[HL]"],
        )
        m_sparse.prepModel()
        m_sparse.runModel(skip_col=0, method="least_squares")
        assert m_sparse.miniResult.success
        np.testing.assert_allclose(m_sparse.miniResult.params["logHL"].value, logK_true, rtol=1e-3)

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


class TestNMRFastExchangeFit:
    """NMR fast exchange fitting tests guided by Binding_Examples.ipynb."""

    def test_1to1_nmr_fast_exchange_fit(self):
        """Fit noise-free 1:1 NMR fast-exchange data and recover logK and bound chemical shift."""
        eq_mat = np.array([[1, 0, 1], [0, 1, 1]], dtype=float)
        comp_names = ["H", "L"]
        species_list = ["H", "L", "HL"]

        h_tot = 10e-3
        l_tot = np.linspace(0.0, 40.0e-3, 51)
        comp_concs = np.column_stack([np.full_like(l_tot, h_tot), l_tot])

        logK_true = 4.0
        params = np.array([0.0, 0.0, logK_true], dtype=float)
        concs = np.array([getConcs(eq_mat, row, params) for row in comp_concs])

        # Fast-exchange NMR observable: weighted average chemical shift over host-containing species
        delta_h_true = 7.0
        delta_hl_true = 8.5
        d_obs = (delta_h_true * concs[:, 0] + delta_hl_true * concs[:, 2]) / h_tot

        # Silent guest L is represented by np.nan, bound shift has (min, init, max) bounds
        spec_to_dd = [delta_h_true, np.nan, (6.5, 7.5, 9.5)]

        model = bindingModel(
            eqMat=eq_mat,
            compNames=comp_names,
            speciesList=species_list,
            specToDd=spec_to_dd,
            rawData=d_obs[:, None],
            compConcs=comp_concs,
            obsList=["d_obs"],
        )
        model.prepModel()
        model.params["logHL"].set(value=3.0, min=0.0, max=8.0)
        model.runModel(skip_col=0, method="least_squares")

        assert model.miniResult is not None
        assert model.miniResult.success

        np.testing.assert_allclose(model.miniResult.params["logHL"].value, logK_true, rtol=1e-3)
        np.testing.assert_allclose(model.miniResult.params["shift_2_0"].value, delta_hl_true, rtol=1e-3)

        calc_data = getCalcData(model).ravel()
        np.testing.assert_allclose(calc_data, d_obs, atol=1e-5)

    def test_1to3_nmr_fast_exchange_fit(self):
        """Fit noise-free 1:3 NMR fast-exchange data from the notebook's model ambiguity section."""
        eq_mat = np.array([[1, 0, 1, 1, 1], [0, 1, 1, 2, 3]], dtype=float)
        comp_names = ["H", "L"]
        species_list = ["H", "L", "HL", "HL2", "HL3"]

        h_tot = 5e-3
        l_tot = np.linspace(0.0, 35.0e-3, 51)
        comp_concs = np.column_stack([np.full_like(l_tot, h_tot), l_tot])

        # K_nice = [10e4, 10e8, 10e11] -> logKs = [5.0, 9.0, 12.0]
        logKs_true = [5.0, 9.0, 12.0]
        params = np.array([0.0, 0.0, *logKs_true], dtype=float)
        concs = np.array([getConcs(eq_mat, row, params) for row in comp_concs])

        # Chemical shifts from notebook: ws = [-1.0, 0.0, 3.0, 0.0, 2.0]
        # Host species have shifts -1.0, 3.0, 0.0, 2.0; free guest L is silent (np.nan)
        d_obs = (-1.0 * concs[:, 0] + 3.0 * concs[:, 2] + 0.0 * concs[:, 3] + 2.0 * concs[:, 4]) / h_tot

        spec_to_dd = [-1.0, np.nan, (0.0, 2.5, 10.0), (-2.0, 0.5, 10.0), (0.0, 1.5, 10.0)]

        model = bindingModel(
            eqMat=eq_mat,
            compNames=comp_names,
            speciesList=species_list,
            specToDd=spec_to_dd,
            rawData=d_obs[:, None],
            compConcs=comp_concs,
            obsList=["delta_obs"],
        )
        model.prepModel()
        model.params["logHL"].set(value=4.0, min=0.0, max=8.0)
        model.params["logHL2"].set(value=8.0, min=0.0, max=22.0)
        model.params["logHL3"].set(value=11.0, min=0.0, max=22.0)
        model.runModel(skip_col=0, method="least_squares")

        assert model.miniResult is not None
        assert model.miniResult.success

        # Verify cumulative constants logBeta
        np.testing.assert_allclose(model.miniResult.params["logHL"].value, logKs_true[0], rtol=1e-3)
        np.testing.assert_allclose(model.miniResult.params["logHL2"].value, logKs_true[1], rtol=1e-3)
        np.testing.assert_allclose(model.miniResult.params["logHL3"].value, logKs_true[2], rtol=1e-3)

        # Verify stepwise binding constants: K11 = 10^5, K12 = 10^4, K13 = 10^3
        beta11 = 10 ** model.miniResult.params["logHL"].value
        beta12 = 10 ** model.miniResult.params["logHL2"].value
        beta13 = 10 ** model.miniResult.params["logHL3"].value
        np.testing.assert_allclose(beta11, 1e5, rtol=1e-2)
        np.testing.assert_allclose(beta12 / beta11, 1e4, rtol=1e-2)
        np.testing.assert_allclose(beta13 / beta12, 1e3, rtol=1e-2)

        # Verify recovered pure chemical shifts
        np.testing.assert_allclose(model.miniResult.params["shift_2_0"].value, 3.0, rtol=1e-3)
        np.testing.assert_allclose(model.miniResult.params["shift_3_0"].value, 0.0, atol=1e-3)
        np.testing.assert_allclose(model.miniResult.params["shift_4_0"].value, 2.0, rtol=1e-3)

        # Verify overall fit
        calc_data = getCalcData(model).ravel()
        np.testing.assert_allclose(calc_data, d_obs, atol=1e-5)

    def test_1to3_nmr_fast_exchange_model_ambiguity_vs_1to1(self):
        """Verify that an under-parameterized 1:1 model cannot fit complex 1:3 NMR fast-exchange data."""
        eq_mat_13 = np.array([[1, 0, 1, 1, 1], [0, 1, 1, 2, 3]], dtype=float)
        h_tot = 5e-3
        l_tot = np.linspace(0.0, 35.0e-3, 51)
        comp_concs = np.column_stack([np.full_like(l_tot, h_tot), l_tot])

        params_13 = np.array([0.0, 0.0, 5.0, 9.0, 12.0], dtype=float)
        concs_13 = np.array([getConcs(eq_mat_13, row, params_13) for row in comp_concs])
        d_obs = (-1.0 * concs_13[:, 0] + 3.0 * concs_13[:, 2] + 0.0 * concs_13[:, 3] + 2.0 * concs_13[:, 4]) / h_tot

        # Fit with 1:1 model
        model_11 = bindingModel(
            eqMat=np.array([[1, 0, 1], [0, 1, 1]], dtype=float),
            compNames=["H", "L"],
            speciesList=["H", "L", "HL"],
            specToDd=[-1.0, np.nan, (-5.0, 1.0, 10.0)],
            rawData=d_obs[:, None],
            compConcs=comp_concs,
            obsList=["delta_obs"],
        )
        model_11.prepModel()
        model_11.params["logHL"].set(value=4.0, min=0.0, max=8.0)
        model_11.runModel(skip_col=0, method="least_squares")

        calc_11 = getCalcData(model_11).ravel()
        rmse_11 = np.sqrt(np.mean((calc_11 - d_obs) ** 2))

        # 1:1 model fails to capture the multi-step titration curve (RMSE is substantial, ~0.3 ppm)
        assert rmse_11 > 0.1