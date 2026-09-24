import os
import sys
import numpy as np
from lmfit import Parameter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import bindtools as bt


class TestFits1to2And2to1:
    """End-to-end fitting tests for 1:2 and 2:1 stoichiometries ."""

    def test_1to2_absorbance_fixed_coefficients(self):
        """Fit a 1:2 absorbance model with known extinction coefficients."""
        eq_mat = np.array([[1, 0, 1, 1],
                           [0, 1, 1, 2]], dtype=float)
        logK11_true = 4.0
        logK12_true = 6.0
        true_params = np.array([0.0, 0.0, logK11_true, logK12_true], dtype=float)

        L_tot = np.linspace(0.0, 50.0e-3, 51)
        H_tot = 10e-3
        comp_concs = np.column_stack([np.full_like(L_tot, H_tot), L_tot])

        concs = np.array([bt.getConcs(eq_mat, row, true_params) for row in comp_concs])

        # Observable: A = 0.1*[H] + 0.6*[HL] + 1.0*[HL2]
        spec_to_integ = np.array([[0.1], [0.0], [0.6], [1.0]])
        A_obs = 0.1 * concs[:, 0] + 0.6 * concs[:, 2] + 1.0 * concs[:, 3]

        model = bt.bindingModel(
            eqMat=eq_mat,
            compNames=["H", "L"],
            speciesList=["H", "L", "HL", "HL2"],
            specToInteg=spec_to_integ,
            rawData=A_obs[:, None],
            compConcs=comp_concs,
            obsList=["A_obs"],
        )
        model.prepModel()
        model.params["logHL"].set(value=3.5, min=0.0, max=8.0)
        model.params["logHL2"].set(value=5.5, min=0.0, max=10.0)
        model.runModel(skip_col=0, method="least_squares")

        assert model.miniResult is not None
        assert model.miniResult.success

        fitted_logK11 = model.miniResult.params["logHL"].value
        fitted_logBeta12 = model.miniResult.params["logHL2"].value
        np.testing.assert_allclose(fitted_logK11, logK11_true, rtol=1e-3)
        np.testing.assert_allclose(fitted_logBeta12, logK12_true, rtol=1e-3)

        # Stepwise constant K12 = Beta12 / K11 = 10^6 / 10^4 = 100 M^-1
        stepwise_k12 = (10 ** fitted_logBeta12) / (10 ** fitted_logK11)
        np.testing.assert_allclose(stepwise_k12, 100.0, rtol=2e-3)

        # Check calculated data fits observable closely
        calc_data = bt.getCalcData(model).ravel()
        rmse = np.sqrt(np.mean((calc_data - A_obs) ** 2))
        assert rmse < 1e-6

    def test_2to1_host_guest_fit(self):
        """Fit a 2:1 host-guest model (H2G formation) on the integration of a single peak (non-physical example)."""
        eq_mat_21 = np.array([[1, 0, 1, 2],
                              [0, 1, 1, 1]], dtype=float)
        logK11_true = 4.0
        logBeta21_true = 6.5
        true_params = np.array([0.0, 0.0, logK11_true, logBeta21_true], dtype=float)

        H_tot = np.linspace(0.0, 50.0e-3, 51)
        G_tot = 10e-3
        comp_concs = np.column_stack([H_tot, np.full_like(H_tot, G_tot)])

        concs = np.array([bt.getConcs(eq_mat_21, row, true_params) for row in comp_concs])
        obs = 0.2 * concs[:, 1] + 0.5 * concs[:, 2] + 1.2 * concs[:, 3]

        model = bt.bindingModel(
            eqMat=eq_mat_21,
            compNames=["H", "G"],
            speciesList=["H", "G", "HG", "H2G"],
            specToInteg=np.array([[0.0], [0.2], [0.5], [1.2]]),
            rawData=obs[:, None],
            compConcs=comp_concs,
            obsList=["obs"],
        )
        model.prepModel()
        assert model.analytical_topology == "2:1"

        model.params["logHG"].set(value=3.5, min=0.0, max=8.0)
        model.params["logH2G"].set(value=6.0, min=0.0, max=10.0)
        model.runModel(skip_col=0, method="least_squares")

        assert model.miniResult.success
        np.testing.assert_allclose(model.miniResult.params["logHG"].value, logK11_true, rtol=1e-3)
        np.testing.assert_allclose(model.miniResult.params["logH2G"].value, logBeta21_true, rtol=1e-3)

    def test_1to2_floating_intermediate_extinction(self):
        """Fit 1:2 model with floating intermediate extinction coefficient (specToLinear with 1D list, resembles an absorbance titration)."""
        eq_mat = np.array([[1, 0, 1, 1],
                           [0, 1, 1, 2]], dtype=float)
        logK11_true = 4.0
        logK12_true = 6.0
        ahg_true = 0.6
        true_params = np.array([0.0, 0.0, logK11_true, logK12_true], dtype=float)

        L_tot = np.linspace(0.0, 50.0e-3, 51)
        H_tot = 10e-3
        comp_concs = np.column_stack([np.full_like(L_tot, H_tot), L_tot])

        concs = np.array([bt.getConcs(eq_mat, row, true_params) for row in comp_concs])
        A_obs = 0.1 * concs[:, 0] + ahg_true * concs[:, 2] + 1.0 * concs[:, 3]

        # Provide specToLinear as a 1D Python list with an lmfit.Parameter for AHG
        spec_to_linear = [0.1, 0.0, Parameter("AHG", value=0.2, min=0.1, max=1.2), 1.0]

        model = bt.bindingModel(
            eqMat=eq_mat,
            compNames=["H", "L"],
            speciesList=["H", "L", "HL", "HL2"],
            specToLinear=spec_to_linear,
            rawData=A_obs[:, None],
            compConcs=comp_concs,
            obsList=["A_obs"],
        )
        model.prepModel()

        # Check 1D list was coerced to 2D object array of shape (4, 1)
        assert isinstance(model.specToLinear, np.ndarray)
        assert model.specToLinear.shape == (4, 1)
        assert "AHG" in model.params

        model.params["logHL"].set(value=3.5, min=0.0, max=8.0)
        model.params["logHL2"].set(value=5.5, min=0.0, max=10.0)
        model.runModel(skip_col=0, method="least_squares")

        assert model.miniResult.success
        np.testing.assert_allclose(model.miniResult.params["logHL"].value, logK11_true, rtol=1e-2)
        np.testing.assert_allclose(model.miniResult.params["logHL2"].value, logK12_true, rtol=1e-2)
        np.testing.assert_allclose(model.miniResult.params["AHG"].value, ahg_true, rtol=1e-2)
        assert model.miniResult.params["AHG"].stderr is not None
        assert model.miniResult.params["AHG"].stderr > 0

    def test_1to2_fast_exchange_nmr_with_silent_species(self):
        """Fit 1:2 NMR chemical shifts with np.nan for silent species and tuple bound shifts - i.e. NMR fast exchange."""
        eq_mat = np.array([[1, 0, 1, 1],
                           [0, 1, 1, 2]], dtype=float)
        logK11_true = 4.0
        logK12_true = 6.0
        true_params = np.array([0.0, 0.0, logK11_true, logK12_true], dtype=float)

        L_tot = np.linspace(0.0, 50.0e-3, 51)
        H_tot = 10e-3
        comp_concs = np.column_stack([np.full_like(L_tot, H_tot), L_tot])

        concs = np.array([bt.getConcs(eq_mat, row, true_params) for row in comp_concs])

        # Fast-exchange NMR observable: pure shifts are 7.0 ppm (H), 7.3 ppm (HL), and 8.0 ppm (HL2)
        d_obs = (7.0 * concs[:, 0] + 7.3 * concs[:, 2] + 8.0 * concs[:, 3]) / H_tot

        # SpecToDd with np.nan for silent L, and (min, init, max) tuple for HL shift
        spec_to_dd = np.array([[7.0], [np.nan], [(7.1, 7.3, 8.2)], [8.0]], dtype=object)

        model = bt.bindingModel(
            eqMat=eq_mat,
            compNames=["H", "L"],
            speciesList=["H", "L", "HL", "HL2"],
            specToDd=spec_to_dd,
            rawData=d_obs[:, None],
            compConcs=comp_concs,
            obsList=["d_obs"],
        )
        model.prepModel()
        assert "shift_2_0" in model.params
        assert model.params["shift_2_0"].value == 7.3

        model.params["logHL"].set(value=3.5, min=0.0, max=8.0)
        model.params["logHL2"].set(value=5.5, min=0.0, max=10.0)
        model.runModel(skip_col=0, method="least_squares")

        assert model.miniResult.success
        np.testing.assert_allclose(model.miniResult.params["logHL"].value, logK11_true, rtol=1e-3)
        np.testing.assert_allclose(model.miniResult.params["logHL2"].value, logK12_true, rtol=1e-3)
        np.testing.assert_allclose(model.miniResult.params["shift_2_0"].value, 7.3, rtol=1e-3)

    def test_model_selection_1to1_vs_1to2(self):
        """Verify model comparison discriminates between 1:1 and 1:2 hypotheses on genuine 1:2 data."""
        eq_mat_12 = np.array([[1, 0, 1, 1],
                              [0, 1, 1, 2]], dtype=float)
        true_params = np.array([0.0, 0.0, 4.0, 6.0], dtype=float)

        L_tot = np.linspace(0.0, 50.0e-3, 51)
        H_tot = 10e-3
        comp_concs = np.column_stack([np.full_like(L_tot, H_tot), L_tot])

        concs = np.array([bt.getConcs(eq_mat_12, row, true_params) for row in comp_concs])
        A_obs = 0.1 * concs[:, 0] + 0.6 * concs[:, 2] + 1.0 * concs[:, 3]

        # 1:1 model fit
        m11 = bt.bindingModel(
            eqMat=np.array([[1, 0, 1], [0, 1, 1]], dtype=float),
            compNames=["H", "L"],
            speciesList=["H", "L", "HL"],
            specToLinear=np.array([[0.1], [0.0], [0.6]]),
            rawData=A_obs[:, None],
            compConcs=comp_concs,
            obsList=["A_obs"],
        )
        m11.prepModel()
        m11.params["logHL"].set(value=3.0, min=0.0, max=8.0)
        m11.runModel(skip_col=0, method="least_squares")

        # 1:2 model fit
        m12 = bt.bindingModel(
            eqMat=eq_mat_12,
            compNames=["H", "L"],
            speciesList=["H", "L", "HL", "HL2"],
            specToLinear=np.array([[0.1], [0.0], [0.6], [1.0]]),
            rawData=A_obs[:, None],
            compConcs=comp_concs,
            obsList=["A_obs"],
        )
        m12.prepModel()
        m12.params["logHL"].set(value=3.5, min=0.0, max=8.0)
        m12.params["logHL2"].set(value=5.5, min=0.0, max=10.0)
        m12.runModel(skip_col=0, method="least_squares")

        rmse_11 = np.sqrt(np.mean((bt.getCalcData(m11).ravel() - A_obs) ** 2))
        rmse_12 = np.sqrt(np.mean((bt.getCalcData(m12).ravel() - A_obs) ** 2))

        # 1:2 model should drastically outperform 1:1 model
        assert rmse_12 < rmse_11 * 0.01
        assert m12.miniResult.aic < m11.miniResult.aic - 100
        assert m12.miniResult.bic < m11.miniResult.bic - 100
