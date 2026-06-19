import os
import sys
import numpy as np

# pytest.importorskip("corner")

# Disable Numba JIT compilation for testing BEFORE importing the module
os.environ["NUMBA_DISABLE_JIT"] = "1"

# Ensure repository root is on sys.path for imports and data access
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bindtools import binding as bd
import pandas as pd

def load_1to1_ka3_data():
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "sim_1to1_ka3.csv"))
    return data

def test_conc_1to1_with_hfree():
    # Load synthetic 1:1 binding data
    data = load_1to1_ka3_data()

    # Convert the first 4 columns (concs) to a numpy array (ignoring dobs)
    raw_data_numpy = data.to_numpy()[:, :4]

    # Instantiate the 1:1 binding model
    m1 = bd.bindingModel(
        eqMat=np.array([[1, 0, 1], [0, 1, 1]]),
        compNames=["H", "G"],
        speciesList=["H", "G", "HG"],
        specToInteg=np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1]
        ]),
        colToComp=np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ]),
        rawData=raw_data_numpy,
    )

    m1.prepModel()
    m1.runModel(skip_col=2)

    # Verify that the fit was successful and correctly recovered logHG ~ 3.0
    assert m1.miniResult is not None
    assert m1.miniResult.success
    np.testing.assert_allclose(m1.miniResult.params["logHG"].value, 3.0, rtol=1e-4)

    
def test_conc_1to1_without_hfree():
    # Load synthetic 1:1 binding data
    data = load_1to1_ka3_data()

    # Convert the first 4 columns (concs) to a numpy array (ignoring dobs)
    raw_data_numpy = data.to_numpy()[:, :4]

    # remove penultimate column
    raw_data_numpy = np.delete(raw_data_numpy, 2, axis=1)

    # Instantiate the 1:1 binding model
    m1 = bd.bindingModel(
        eqMat=np.array([[1, 0, 1], [0, 1, 1]]),
        compNames=["H", "G"],
        speciesList=["H", "G", "HG"],
        specToInteg=np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1]
        ]),
        colToComp=np.array([
            [1, 0, 0],
            [0, 1, 0]
        ]),
        rawData=raw_data_numpy,
    )


    m1.prepModel()
    m1.runModel(skip_col=2)

    # Verify that the fit was successful and correctly recovered logHG ~ 3.0
    assert m1.miniResult is not None
    assert m1.miniResult.success
    np.testing.assert_allclose(m1.miniResult.params["logHG"].value, 3.0, rtol=1e-4)


def test_conc_1to1_analytical():
    # Load synthetic 1:1 binding data
    data = load_1to1_ka3_data()

    # Convert the first 4 columns (concs) to a numpy array (ignoring dobs)
    raw_data_numpy = data.to_numpy()[:, :4]

    # Instantiate the 1:1 binding model
    m1 = bd.bindingModel(
        eqMat=np.array([[1, 0, 1], [0, 1, 1]]),
        compNames=["H", "G"],
        speciesList=["H", "G", "HG"],
        specToInteg=np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1]
        ]),
        colToComp=np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ]),
        rawData=raw_data_numpy,
    )

    m1.prepModel()
    m1.runModel(skip_col=2)

    np.testing.assert_allclose(m1.miniResult.params["logHG"].value, 3.0, rtol=1e-4)


def test_force_numerical_1to1_fit():
    # Load synthetic 1:1 binding data
    data = load_1to1_ka3_data()

    # Convert the first 4 columns (concs) to a numpy array (ignoring dobs)
    raw_data_numpy = data.to_numpy()[:, :4]

    # Instantiate the 1:1 binding model
    m1 = bd.bindingModel(
        eqMat=np.array([[1, 0, 1], [0, 1, 1]]),
        compNames=["H", "G"],
        speciesList=["H", "G", "HG"],
        specToInteg=np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1]
        ]),
        colToComp=np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ]),
        rawData=raw_data_numpy,
    )


    # Calling prepModel with force_numerical=True should disable analytical path
    m1.prepModel(force_numerical=True)
    assert m1.analytical_topology is None
    assert m1.analytical_fast_exchange is False

    m1.runModel(skip_col=2)

    np.testing.assert_allclose(m1.miniResult.params["logHG"].value, 3.0, rtol=1e-4)


def test_shift_1to1_analytical():
    # Generate 1:1 NMR fast-exchange chemical shift data
    host_tot = np.full(24, 1.0e-3)
    guest_tot = np.linspace(0.0, 2.2e-3, 24)
    beta11 = 10**5.0
    term = host_tot + guest_tot + 1.0 / beta11
    disc = np.maximum(term**2 - 4.0 * host_tot * guest_tot, 0.0)
    hg = 0.5 * (term - np.sqrt(disc))
    frac_h_bound = np.divide(hg, host_tot, out=np.zeros_like(hg), where=host_tot > 0)
    
    # delta_h = d0 + amp * frac_h_bound where d0 = 7.0, amp = 1.2
    # In analytical fast exchange, we fit delta0_dH (free H shift = 7.0) and deltac1_dH (bound HG shift = 8.2)
    delta_h = 7.0 + 1.2 * frac_h_bound
    
    raw_data_numpy = np.column_stack((host_tot, guest_tot, delta_h))

    # Instantiate the 1:1 binding model
    m1 = bd.bindingModel(
        eqMat=np.array([[1, 0, 1], [0, 1, 1]]),
        compNames=["H", "G"],
        speciesList=["H", "G", "HG"],
        colToComp=np.array([
            [1, 0, 0],
            [0, 1, 0]
        ]),
        obsList=["dH"],
        rawData=raw_data_numpy,
    )

    # Calling prepModel should automatically set analytical_fast_exchange to True
    m1.prepModel()
    assert m1.analytical_fast_exchange is True
    assert m1.analytical_topology == "1:1"
    assert m1.analytical_obs_columns == ["dH"]

    # Run the fit using runModel (skip the first 2 columns: H_tot and G_tot)
    m1.runModel(skip_col=2)

    # Verify that the fit successfully recovered the logHG ~ 5.0 binding constant
    assert m1.miniResult is not None
    assert m1.miniResult.success
    np.testing.assert_allclose(m1.miniResult.params["logHG"].value, 5.0, rtol=1e-3)
    np.testing.assert_allclose(m1.miniResult.params["delta0_dH"].value, 7.0, rtol=1e-3)
    np.testing.assert_allclose(m1.miniResult.params["deltac1_dH"].value, 8.2, rtol=1e-3)



    