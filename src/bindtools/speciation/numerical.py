import numpy as np
from numba import jit

from bindtools.exceptions import EquilibriumError


# https://github.com/numba/numba/issues/1269#issuecomment-702665837
# This solution enables us to apply the overloaded np.prod function along a single axis, see link above
@jit(nopython=True)
def apply_along_axis_0(func1d, arr):
    """Like calling func1d(arr, axis=0)"""
    if arr.size == 0:
        raise RuntimeError("Must have arr.size > 0")
    ndim = arr.ndim
    if ndim == 0:
        raise RuntimeError("Must have ndim > 0")
    elif 1 == ndim:
        return func1d(arr)
    else:
        result_shape = arr.shape[1:]
        out = np.empty(result_shape, arr.dtype)
        _apply_along_axis_0(func1d, arr, out)
        return out


@jit(nopython=True, nogil=True)
def _apply_along_axis_0(func1d, arr, out):
    """Like calling func1d(arr, axis=0, out=out). Require arr to be 2d or bigger."""
    ndim = arr.ndim
    if ndim < 2:
        raise RuntimeError("_apply_along_axis_0 requires 2d array or bigger")
    elif ndim == 2:  # 2-dimensional case
        for i in range(len(out)):
            out[i] = func1d(arr[:, i])
    else:  # higher dimensional case
        for i, out_slice in enumerate(out):
            _apply_along_axis_0(func1d, arr[:, i], out_slice)


@jit(nopython=True, nogil=True, cache=True)
def specCalc(conc, nspecies, eqMat, K):  # this function calculates the equilibrium concentrations of the species
    specmat = conc.repeat(nspecies).reshape((-1, nspecies))
    eq3pt47 = apply_along_axis_0(np.prod, specmat**eqMat)
    speciesConc = K * eq3pt47  # eq 3.48

    compTotCalc = eqMat @ speciesConc
    return compTotCalc, speciesConc


@jit(nopython=True, nogil=True, cache=True)
def _specJac(ncomp, eqMat, speciesConc):  # this function calculates the Jacobian of the equilibrium problem
    J = np.zeros((ncomp, ncomp))
    # calculate Jacobian
    for ii in range(ncomp):
        for jj in range(ii, ncomp):
            J[ii, jj] = np.sum(eqMat[ii, :] * eqMat[jj, :] * speciesConc)
            if ii != jj:
                J[jj, ii] = J[ii, jj]

    return J


@jit(nopython=True, nogil=True, cache=True)
def specJac(conc, initComponentConc, eqMat, K):  # this function calculates the Jacobian of the equilibrium problem
    ncomp = np.shape(eqMat)[0]
    nspecies = len(K)
    _, speciesConc = specCalc(conc, nspecies, eqMat, K)  # calculate species concentrations
    J = _specJac(ncomp, eqMat, speciesConc)
    return J


# This function implements the Newton Raphson method for solving the equilibrium problem
# following: https://doi.org/10.1016/S0922-3487(07)80006-2
@jit(nopython=True, nogil=True, cache=True)
def DoNR(eqMat, K, initComponentConc, guessCompConc):
    nspecies = len(K)
    ncomp = len(initComponentConc)

    initComponentConc = np.copy(initComponentConc)
    initComponentConc[initComponentConc <= 0] = (
        1e-20  # avoids numerical errors. Changed to <=0 rather than == 0 because sometimes small negative errors appear which makes
    )
    # the optimisation impossible

    conc = np.copy(guessCompConc)
    for iter in range(0, 600):
        compTotCalc, speciesConc = specCalc(conc, nspecies, eqMat, K)

        # if our computed component concentrations are close
        # enough to the true concentrations, we can stop
        deltaComp = initComponentConc - compTotCalc
        if np.all(np.abs(deltaComp) < 1e-15):
            return compTotCalc, speciesConc

        # otherwise calculate the Jacobian
        J = _specJac(ncomp, eqMat, speciesConc)

        # estimate the change in component concentrations
        deltaConc = np.linalg.lstsq(J, deltaComp)[0].T * conc  # , rcond=None

        # if the change in component concentrations is too small, we are stuck
        if np.any(deltaConc == 0) and np.max(np.abs(deltaConc)) < 1e-15:
            # we are not going anywhere, let's start again with a subtly different initial guess
            conc = guessCompConc + np.random.randn(len(guessCompConc)) * 1e-10
        else:
            conc += deltaConc

        if (iter + 1) % 200 == 0:
            # assume we are stuck, try  new guess
            conc = guessCompConc + np.random.randn(len(guessCompConc)) * 1e-10

        while np.any(conc <= 0):
            deltaConc = deltaConc / 2
            conc -= deltaConc
            if np.all(np.abs(deltaConc) < 1e-19):
                break

    raise EquilibriumError(
        "Failed to converge in equilibrium solver doNR", speciesConc, (eqMat, K, initComponentConc, guessCompConc)
    )


# This function solves the equilibrium concentration problem based on equilibrium constants, adapted
# from Maeder and co workers: https://doi.org/10.1016/S0922-3487(07)80006-2
@jit(nopython=True, nogil=True, cache=True)
def getConcs(eqMat, initComponentConc, logK):
    # now we give a vector containing the (log) equilibrium constants. Be careful
    # with sign because these constants are for formation of the complexes
    # above from the "pure components".
    K = 10.0**logK
    # make an initial guess. This doesn't need to be good - it should just
    # be nearly-equally bad for all parameters
    # the initial guess is for the concentrations of the *free components* only
    guessCompConc = np.zeros((1, len(initComponentConc))) + np.mean(initComponentConc)
    eqMat = eqMat.astype("float64")  # fix bug in simple systems
    # solve the equilibrium problem above using the Newton Raphson method
    comp, spec = DoNR(eqMat, K, initComponentConc, guessCompConc)
    return spec

