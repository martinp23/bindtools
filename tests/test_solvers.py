import os
import sys
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bindtools.speciation import (
    calc_analytical_speciation,
    getConcs,
    _infer_simple_fast_exchange_topology,
)


class TestTopologyInference:
    def test_infer_1to1(self):
        eq_11 = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        res = _infer_simple_fast_exchange_topology(eq_11, 2)
        assert res == ("1:1", [2])

    def test_infer_1to2(self):
        eq_12 = np.array([[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 2.0]])
        res = _infer_simple_fast_exchange_topology(eq_12, 2)
        assert res == ("1:2", [2, 3])

    def test_infer_2to1(self):
        eq_21 = np.array([[1.0, 0.0, 1.0, 2.0], [0.0, 1.0, 1.0, 1.0]])
        res = _infer_simple_fast_exchange_topology(eq_21, 2)
        assert res == ("2:1", [2, 3])

    def test_infer_returns_none_for_non_simple(self):
        eq_non_simple = np.array([[1.0, 0.0, 1.0, 2.0], [0.0, 1.0, 1.0, 2.0]])
        assert _infer_simple_fast_exchange_topology(eq_non_simple, 2) is None

    def test_infer_invalid_dimensions(self):
        assert _infer_simple_fast_exchange_topology(np.array([1.0, 2.0]), 2) is None
        assert _infer_simple_fast_exchange_topology(np.array([[1.0, 0.0], [0.0, 1.0]]), 2) is None
        assert _infer_simple_fast_exchange_topology(np.array([[1.0, 0.0, 1.0]]), 1) is None


class TestAnalyticalVsNumericalSpeciation:
    def test_analytical_1to1_matches_numerical(self):
        n_pts = 15
        h_tot = np.full(n_pts, 1e-3)
        g_tot = np.linspace(1e-5, 3e-3, n_pts)
        comp_concs = np.column_stack([h_tot, g_tot])
        eq_mat = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        logK = np.array([0.0, 0.0, 4.5])

        spec_ana, error = calc_analytical_speciation(
            comp_concs=comp_concs,
            eq_mat=eq_mat,
            binding_params=logK,
            topology="1:1",
            n_comp=2,
            complex_indices=[2],
        )
        assert not error

        spec_num = np.array([getConcs(eq_mat, row, logK) for row in comp_concs])
        np.testing.assert_allclose(spec_ana, spec_num, rtol=1e-5, atol=1e-12)

    def test_analytical_1to2_matches_numerical(self):
        n_pts = 15
        h_tot = np.full(n_pts, 1e-3)
        g_tot = np.linspace(1e-5, 5e-3, n_pts)
        comp_concs = np.column_stack([h_tot, g_tot])
        eq_mat = np.array([[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 2.0]])
        logK = np.array([0.0, 0.0, 4.0, 7.0])  # logK11=4, logBeta12=7

        spec_ana, error = calc_analytical_speciation(
            comp_concs=comp_concs,
            eq_mat=eq_mat,
            binding_params=logK,
            topology="1:2",
            n_comp=2,
            complex_indices=[2, 3],
        )
        assert not error

        spec_num = np.array([getConcs(eq_mat, row, logK) for row in comp_concs])
        np.testing.assert_allclose(spec_ana, spec_num, rtol=1e-4, atol=1e-10)

    def test_analytical_2to1_matches_numerical(self):
        n_pts = 15
        h_tot = np.linspace(1e-5, 5e-3, n_pts)
        g_tot = np.full(n_pts, 1e-3)
        comp_concs = np.column_stack([h_tot, g_tot])
        eq_mat = np.array([[1.0, 0.0, 1.0, 2.0], [0.0, 1.0, 1.0, 1.0]])
        logK = np.array([0.0, 0.0, 4.0, 7.0])  # logK11=4, logBeta21=7

        spec_ana, error = calc_analytical_speciation(
            comp_concs=comp_concs,
            eq_mat=eq_mat,
            binding_params=logK,
            topology="2:1",
            n_comp=2,
            complex_indices=[2, 3],
        )
        assert not error

        spec_num = np.array([getConcs(eq_mat, row, logK) for row in comp_concs])
        np.testing.assert_allclose(spec_ana, spec_num, rtol=1e-4, atol=1e-10)

    def test_zero_concs_handled_cleanly(self):
        comp_concs = np.array([[0.0, 1e-3], [1e-3, 0.0], [0.0, 0.0]])
        eq_mat = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        logK = np.array([0.0, 0.0, 4.0])

        spec_ana, error = calc_analytical_speciation(
            comp_concs=comp_concs,
            eq_mat=eq_mat,
            binding_params=logK,
            topology="1:1",
            n_comp=2,
            complex_indices=[2],
        )
        assert not error
        assert np.all(spec_ana >= 0.0)
        np.testing.assert_allclose(spec_ana[:, 2], [0.0, 0.0, 0.0])

    def test_unsupported_topology_falls_back_to_numerical(self):
        comp_concs = np.array([[1e-3, 1e-3]])
        eq_mat = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        logK = np.array([0.0, 0.0, 4.0])
        spec_calc, error = calc_analytical_speciation(
            comp_concs=comp_concs,
            eq_mat=eq_mat,
            binding_params=logK,
            topology="3:1",
            n_comp=2,
            complex_indices=[2],
        )
        assert error is True
        expected = getConcs(eq_mat, comp_concs[0], logK)
        np.testing.assert_allclose(spec_calc[0], expected, rtol=1e-4)

    def test_invalid_ncomp_raises_error(self):
        comp_concs = np.array([[1e-3, 1e-3, 1e-3]])
        eq_mat = np.eye(3)
        logK = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="requires exactly 2 components"):
            calc_analytical_speciation(
                comp_concs=comp_concs,
                eq_mat=eq_mat,
                binding_params=logK,
                topology="1:1",
                n_comp=3,
                complex_indices=[2],
            )
