import numpy as np
import lmfit


def deltaToConc(deltas, mapping, startShifts, endShifts, isHost, isHG):
    # deltas is a vector of chemical shifts
    pass


def concToDelta(concs, specToDd, shiftParams, paramNames):
    mlen, nlen = np.shape(specToDd)
    specToDdTrial = np.copy(specToDd)

    # here we replace tuples in the original mapping with parameter values from the fitting engine
    for ii in range(mlen):
        for ij in range(nlen):
            if isinstance(specToDdTrial[ii, ij], tuple):
                specToDdTrial[ii, ij] = shiftParams[paramNames.index("shift_{}_{}".format(ii, ij))]

            elif isinstance(specToDdTrial[ii, ij], lmfit.Parameter):
                specToDdTrial[ii, ij] = shiftParams[paramNames.index(specToDdTrial[ii, ij].name)]
    specToDdTrial = specToDdTrial.astype(np.float64)

    # normalize concs to mol fractions
    truthMat = ~np.isnan(specToDdTrial)
    shiftCalc = []
    for cc in concs:
        tt = (truthMat.T * cc).T
        moleFracs = tt / tt.sum(axis=0)
        sc = np.nansum(moleFracs * specToDdTrial, axis=0)
        shiftCalc.append(sc)

    return shiftCalc


def concToLinearObs(
    concs: np.ndarray, specToLinear: np.ndarray, shiftParams: np.ndarray, paramNames: list
) -> np.ndarray:
    """Compute concentration-weighted linear observables (Beer-Lambert / fluorescence).

    obs_i = sum_j ( coeff_ij * [species_j] )  — no mole-fraction normalisation.

    specToLinear has shape (n_species, n_obs) with dtype=object; each entry is either
    a float (including 0.0 for dark/silent species) or an lmfit.Parameter whose current
    value is looked up in shiftParams / paramNames.

    Returns an ndarray of shape (n_pts, n_obs).
    """
    mlen, nlen = np.shape(specToLinear)  # (n_species, n_obs)
    specToLinearTrial = np.copy(specToLinear)

    for ii in range(mlen):
        for ij in range(nlen):
            if isinstance(specToLinearTrial[ii, ij], lmfit.Parameter):
                specToLinearTrial[ii, ij] = shiftParams[paramNames.index(specToLinearTrial[ii, ij].name)]
    specToLinearTrial = specToLinearTrial.astype(np.float64)

    # Direct linear combination: (n_pts, n_species) @ (n_species, n_obs) -> (n_pts, n_obs)
    return np.dot(concs, specToLinearTrial)


def concToObservable(specConcs, specToInteg, specToDd, shiftParams, paramNames, scaleFactor=None, specToLinear=None):
    """Convert species concentrations to predicted observables.

    Observable ordering in the returned matrix: [integ cols, linear cols, shift cols].
    The experimental data matrix must use the same column ordering.
    """
    parts = []

    if specToInteg is not None:
        if scaleFactor is None:
            scaleFactor = np.ones((np.shape(specToInteg)[1],))
        parts.append(np.array(np.dot(specConcs, specToInteg) * scaleFactor))

    if specToLinear is not None:
        parts.append(concToLinearObs(specConcs, specToLinear, shiftParams, paramNames))

    if specToDd is not None:
        parts.append(np.array(concToDelta(specConcs, specToDd, shiftParams, paramNames)))

    if len(parts) == 1:
        return parts[0]
    elif len(parts) > 1:
        return np.concatenate(parts, axis=1)
    else:
        return np.zeros((np.shape(specConcs)[0], 0))

