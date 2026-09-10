import math
import numpy as np
from numba import jit

from bindtools.models.objective import fitfun


@jit(nopython=True, nogil=True, cache=True)
def calclnprob(r, lnsigma):
    r = -0.5 * np.sum(np.divide(r, np.exp(lnsigma)) ** 2 + np.log(2 * np.pi) + 2 * lnsigma)

    if math.isnan(r):
        print("Probability function returned NaN, fudging to -inf.")
        return -np.inf
    return r


@jit(nopython=True, nogil=True, cache=True)
def log_prior(val, bounds):
    for ii in range(len(val)):
        if not (bounds[ii][0] < val[ii] < bounds[ii][1]):
            return -np.inf
    return 0.0


def log_prob(params, fcn_opts, bounds):
    lp = log_prior(params, bounds)

    if not np.isfinite(lp):
        return -np.inf

    mapping = fcn_opts["sigmaMapping"]
    lnsigmaVals = params[-(max(mapping) + 1) :]
    lnsigma = np.array([lnsigmaVals[ii] for ii in mapping])

    pp = np.zeros(
        len(params) + np.shape(fcn_opts["compConcs"])[1]
    )  # need extra parameters - for logK=0 - for each component in the eqMat

    pp[-len(params) :] = params

    fcn_opts["ret"] = "residual"
    r = np.array(fitfun(pp, fcn_opts))
    if np.isnan(r).any():
        return -np.inf

    rout = calclnprob(r, lnsigma)
    return rout

