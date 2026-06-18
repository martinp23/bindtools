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


    # Enable the analytical speciation solver
    m1.analytical_topology = "1:1"
    m1.analytical_complex_indices = [2]

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

    # Enable the analytical speciation solver
    m1.analytical_topology = "1:1"
    m1.analytical_complex_indices = [2]

    m1.prepModel()
    m1.runModel(skip_col=2)

    # Verify that the fit was successful and correctly recovered logHG ~ 3.0
    assert m1.miniResult is not None
    assert m1.miniResult.success
    np.testing.assert_allclose(m1.miniResult.params["logHG"].value, 3.0, rtol=1e-4)