import os
import sys
import numpy as np
import pytest
import lmfit

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bindtools.observables import (
    ObsType,
    sigmaMapping,
    concToLinearObs,
    concToDelta,
    concToObservable,
)


class TestObsType:
    def test_nmr_integ_preset(self):
        obs = ObsType("NMRInteg")
        assert obs.name == "NMRInteg"
        assert obs.units == "M"
        assert obs.param.name == "lnsigmaNMRInteg"
        assert obs.param.value == -8
        assert obs.minlim == -11
        assert obs.maxlim == -5

    def test_conc_meas_preset(self):
        obs = ObsType("concMeas", units="mM")
        assert obs.units == "mM"
        assert obs.param.name == "lnsigmaconcMeas"

    def test_delta_h_and_f_presets(self):
        dh = ObsType("deltaH")
        assert dh.units == "ppm"
        assert dh.param.name == "lnsigmadeltaH"

        df = ObsType("deltaF")
        assert df.units == "ppm"
        assert df.param.name == "lnsigmadeltaF"

    def test_uvvis_and_fluorescence_presets(self):
        uv = ObsType("uvvis")
        assert uv.units == "absorbance"
        assert uv.param.name == "lnsigmaUVvis"
        assert uv.param.vary is True

        fl = ObsType("fluorescence")
        assert fl.units == "intensity"
        assert fl.param.name == "lnsigmaFluorescence"
        assert fl.param.vary is True

    def test_custom_name_and_value(self):
        custom = ObsType("customSignal", units="AU", value=-6, minlim=-10, maxlim=-2)
        assert custom.units == "AU"
        assert custom.lnsigma == -6
        assert custom.sigma == np.exp(-6)

    def test_out_of_bounds_resets_to_midpoint(self):
        obs = ObsType("custom", value=10, minlim=0, maxlim=4)
        assert obs.lnsigma == 2.0
        assert obs.param.value == 2.0


class TestSigmaMapping:
    def test_unique_and_repeated_columns(self):
        cols = ["dH", "dF", "dH", "NMRInteg", "dF"]
        mapping = sigmaMapping(cols)
        assert mapping == [0, 1, 0, 2, 1]


class TestConcToLinearObs:
    def test_identity_mapping(self):
        concs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        specToLinear = np.eye(3, dtype=object)
        res = concToLinearObs(concs, specToLinear, np.array([]), [])
        np.testing.assert_allclose(res, concs)

    def test_scalar_coefficients(self):
        concs = np.array([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]])
        specToLinear = np.array([[2.0], [0.0], [3.0]], dtype=object)
        res = concToLinearObs(concs, specToLinear, np.array([]), [])
        np.testing.assert_allclose(res.flatten(), [5.0, 16.0])

    def test_lmfit_parameter_replacement(self):
        eps_param = lmfit.Parameter("eps_HG", value=5.0)
        specToLinear = np.empty((2, 1), dtype=object)
        specToLinear[0, 0] = 0.0
        specToLinear[1, 0] = eps_param
        concs = np.array([[1.0, 2.0], [3.0, 4.0]])
        res = concToLinearObs(concs, specToLinear, np.array([5.0]), ["eps_HG"])
        np.testing.assert_allclose(res.flatten(), [10.0, 20.0])


class TestConcToDelta:
    def test_conc_to_delta_basic(self):
        # 3 species (H, G, HG), 1 shift observable
        specToDd = np.array([
            [(0.0, 7.0, 10.0)],  # H free = 7.0
            [None],              # G
            [(0.0, 9.0, 10.0)],  # HG bound = 9.0
        ], dtype=object)

        concs = np.array([
            [1.0, 1.0, 0.0],  # All H free -> shift = 7.0
            [0.5, 0.5, 0.5],  # 50% free H, 50% bound HG -> shift = 8.0
            [0.0, 1.0, 1.0],  # All bound HG -> shift = 9.0
        ])
        shiftParams = np.array([7.0, 9.0])
        paramNames = ["shift_0_0", "shift_2_0"]

        shifts = concToDelta(concs, specToDd, shiftParams, paramNames)
        np.testing.assert_allclose(np.array(shifts).flatten(), [7.0, 8.0, 9.0])


class TestConcToObservable:
    def test_linear_only(self):
        concs = np.array([[2.0, 3.0]])
        specToLinear = np.array([[1.0], [2.0]], dtype=object)
        res = concToObservable(concs, None, None, np.array([]), [], specToLinear=specToLinear)
        np.testing.assert_allclose(res.flatten(), [8.0])

    def test_concatenated_integ_and_linear(self):
        concs = np.array([[1.0, 2.0]])
        specToInteg = np.array([[1.0], [0.0]])
        specToLinear = np.array([[0.0], [3.0]], dtype=object)
        res = concToObservable(
            concs,
            specToInteg=specToInteg,
            specToDd=None,
            shiftParams=np.array([]),
            paramNames=[],
            specToLinear=specToLinear,
        )
        assert res.shape == (1, 2)
        np.testing.assert_allclose(res[0, 0], 1.0)
        np.testing.assert_allclose(res[0, 1], 6.0)

    def test_empty_returns_zero_columns(self):
        concs = np.array([[1.0, 2.0], [3.0, 4.0]])
        res = concToObservable(concs, None, None, np.array([]), [])
        assert res.shape == (2, 0)
