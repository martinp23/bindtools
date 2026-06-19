import os
import sys
import numpy as np

# pytest.importorskip("corner")

# Ensure repository root is on sys.path for imports and data access
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bindtools import binding as bd


def one_to_one_setup():
    # Generate 50x2 matrix of test componentConcentrations
    componentConcentrations = np.zeros((50, 2))
    componentConcentrations[:, 0] = 1e-3  # First column is 1e-3
    componentConcentrations[:, 1] = np.linspace(0, 1e-2, 50)  # Second column ranges from 0 to 1e-2

    # 1:1 binding equilibrium matrix (based on testNR.py example)
    # Species: H (component 0), G (component 1), HG (complex)
    equilibriumMatrix = np.array(
        [
            [1, 0, 1],  # H balance: [H] + [HG] = [H]_total
            [0, 1, 1],  # G balance: [G] + [HG] = [G]_total
        ]
    )

    # Binding constants: logK = [log(K_H), log(K_G), log(K_HG)]
    # K_H = K_G = 1 (components), K_HG = 10^4 (complex formation)
    logK = np.array([0, 0, 4])  # log10(1e4) = 4

    return componentConcentrations, equilibriumMatrix, logK


def test_getConcs_1to1_binding():
    componentConcentrations, equilibriumMatrix, logK = one_to_one_setup()

    # Test the function for each concentration pair
    results = []
    for i, total_concs in enumerate(componentConcentrations):
        result = bd.getConcs(equilibriumMatrix, total_concs, logK)
        results.append(result)

        # Basic assertions for each result
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) == 3  # Should have 3 species (H, G, HG)

        # Check that concentrations are non-negative
        assert np.all(result >= 0)

        # Check mass balance
        calc_totals = equilibriumMatrix @ result
        np.testing.assert_allclose(
            calc_totals,
            total_concs,
            rtol=1e-10,
            atol=np.finfo(float).eps,
            err_msg=f"Mass balance failed for concentration set {i}",
        )

    results = np.array(results)

    # Load truth data and compare
    truth_path = os.path.join(os.path.dirname(__file__), "data", "test_001_concs_1to2_truths.npy")
    trueResults = np.load(truth_path)
    np.testing.assert_allclose(
        results,
        trueResults,
        rtol=1e-10,
        atol=float(np.finfo(float).eps),
        err_msg="Results do not match expected values from test_001_concs_1to2.npy",
    )

    # Additional tests on the full result set
    assert results.shape == (50, 3)  # 50 rows, 3 species

    # Test that complex concentration increases with guest concentration
    complex_concs = results[:, 2]  # HG complex is the third species
    guest_totals = componentConcentrations[:, 1]
    non_zero_mask = guest_totals > 1e-10
    if np.sum(non_zero_mask) > 1:
        assert complex_concs[-1] > complex_concs[1], "Complex concentration should increase with guest"


def test_getConcs_1to1_leastSq():
    componentConcentrations, equilibriumMatrix, logK = one_to_one_setup()

    # Test the function for each concentration pair
    results = []
    for i, total_concs in enumerate(componentConcentrations):
        result = bd.getConcsScipy(equilibriumMatrix, total_concs, logK)
        results.append(result)

        # Basic assertions for each result
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) == 3  # Should have 3 species (H, G, HG)

        # Check that concentrations are non-negative
        assert np.all(result >= 0)

        # Check mass balance
        calc_totals = equilibriumMatrix @ result
        np.testing.assert_allclose(
            calc_totals,
            total_concs,
            rtol=1e-6,
            atol=1e-9,
            err_msg=f"Mass balance failed for concentration set {i}",
        )

    results = np.array(results)

    # Load truth data and compare
    truth_path = os.path.join(os.path.dirname(__file__), "data", "test_001_concs_1to2_truths.npy")
    trueResults = np.load(truth_path)
    np.testing.assert_allclose(
        results,
        trueResults,
        rtol=1e-8,
        atol=1e-12,
        err_msg="Results do not match expected values from test_001_concs_1to2.npy",
    )

    # Additional tests on the full result set
    assert results.shape == (50, 3)  # 50 rows, 3 species

    # Test that complex concentration increases with guest concentration
    complex_concs = results[:, 2]  # HG complex is the third species
    guest_totals = componentConcentrations[:, 1]
    non_zero_mask = guest_totals > 1e-10
    if np.sum(non_zero_mask) > 1:
        assert complex_concs[-1] > complex_concs[1], "Complex concentration should increase with guest"
