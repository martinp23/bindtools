import numpy as np
import lmfit

from bindtools.exceptions import EquilibriumError
from bindtools.speciation.analytical import calc_analytical_speciation
from bindtools.speciation.numerical import getConcs
from bindtools.observables.transforms import concToObservable


def fitfun(params, fcn_opts):
    if isinstance(params, lmfit.parameter.Parameters):
        parvals = list(params.valuesdict().values())
    else:
        parvals = params

    bindingParams = np.array([*parvals][: fcn_opts["nK"]], dtype=np.float64)
    error = False

    analytical_topology = fcn_opts.get("analytical_topology")
    if analytical_topology in ("1:1", "1:2", "2:1"):
        comp_concs = np.array(fcn_opts["compConcs"], dtype=float)
        complex_indices = [int(x) for x in fcn_opts.get("analytical_complex_indices", [])]
        specCalc, error = calc_analytical_speciation(
            comp_concs=comp_concs,
            eq_mat=np.array(fcn_opts["eqMat"], dtype=float),
            binding_params=bindingParams,
            topology=str(analytical_topology),
            n_comp=int(comp_concs.shape[1]),
            complex_indices=complex_indices,
        )
    else:
        specCalc = []
        for row in fcn_opts["compConcs"]:
            try:
                yEq = getConcs(fcn_opts["eqMat"], row, bindingParams)
            except EquilibriumError as e:
                yEq = e.val
                error = True
            specCalc.append(yEq)
        specCalc = np.array(specCalc)

    if fcn_opts["optTarget"] == "concs":
        if fcn_opts["ret"] == "residual":
            # normalize residuals
            res = fcn_opts["exptData"] - specCalc
            res = res / fcn_opts["sigma"]
            if error is True:  # penalty if there is an error in doNR
                res = res * 10
                if fcn_opts["mcmc"] is True:
                    return np.nan
            return res
        elif fcn_opts["ret"] == "concs":
            return specCalc

    elif fcn_opts["optTarget"] == "obs":
        shiftParams = np.array([*parvals][fcn_opts["nK"] :], dtype=np.float64)
        paramNames = fcn_opts["paramNames"][fcn_opts["nK"] :]
        obsCalc = concToObservable(
            specCalc,
            fcn_opts["specToInteg"],
            fcn_opts["specToDd"],
            shiftParams,
            paramNames,
            specToLinear=fcn_opts.get("specToLinear"),
        )

        if fcn_opts["ret"] == "residual":
            # normalize residuals
            res = fcn_opts["exptData"] - obsCalc  # function needs to know if it's working in conc or observables
            res = res / fcn_opts["sigma"]

            # Set positions of nan values in exptData to 0 in res
            nan_mask = np.isnan(fcn_opts["exptData"])
            res[nan_mask] = 0
            res[fcn_opts["exptData"] == np.nan] = 0

            if error is True:  # penalty if there is an error in doNR
                res = res * 10
                if fcn_opts["mcmc"] is True:
                    return np.nan
            if fcn_opts["mcmc"] is True:
                # remove nan values which tend to come from where we
                # are unable to calculate a chemical shift due to insufficient
                # data (i.e. zeros in specToDd)
                if fcn_opts["specToDd"] is not None:
                    nShift = np.shape(fcn_opts["specToDd"])[1]
                    res[:, -nShift:] = np.nan_to_num(res[:, -nShift:])
            return res
        elif fcn_opts["ret"] == "concs":
            return obsCalc

    else:
        print("Invalid return argument. Options: 'residual' [default] or 'concs'")
        return -1

