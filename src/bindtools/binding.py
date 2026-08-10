# mamba create -n binding -c conda-forge python jupyter tqdm ipython uncertainties lmfit scipy numpy emcee tqdm numba corner matplotlib numdifftools
# test

# Import necessary modules.
import datetime
import math
import os
from multiprocessing import Pool
from contextlib import nullcontext

import corner
import emcee
import h5py
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from numba import jit

# from optimparallel import minimize_parallel  # not required but can be useful if you want to do parallel LBGFS
# from IPython.display import display, Markdown # lets us print tables of values tables using standard iPython

from typing import Optional, List

import logging

logger = logging.getLogger(__name__)
# import pickle as pkl


class EquilibriumError(Exception):
    def __init__(self, message, val, params):
        #  print("Equilibrium solver failed to converge.")
        self.val = val
        # pkl.dump(params,open('params.pkl','wb'))

        # raise ValueError("End now")


# Set everything up - all functions for fitting
#
# #https://github.com/numba/numba/issues/1269#issuecomment-702665837
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


# This solution enables us to apply the overloaded np.prod function along a single axis, see link above
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


# This function solves the equilibrium concentration problem based on equilibrium constants, adapted
# from Maeder and co workers: https://doi.org/10.1016/S0922-3487(07)80006-2
@jit(nopython=True, nogil=True, cache=True)
def getConcs(eqMat: np.ndarray, initComponentConc: np.ndarray, logK: np.ndarray) -> np.ndarray:
    """Solve equilibrium species concentrations using the Newton-Raphson method.

    Computes equilibrium concentrations of all free components and complex species
    given the stoichiometry matrix, total component concentrations, and log10 equilibrium constants.

    Args:
        eqMat: Stoichiometry matrix of shape (n_components, n_species) where element (i, j)
            defines the stoichiometric coefficient of component i in species j.
        initComponentConc: 1D array of shape (n_components,) containing total initial concentrations.
        logK: 1D array of shape (n_species,) containing log10 formation constants. Free components
            typically have log10(K) = 0.

    Returns:
        1D array of shape (n_species,) with calculated equilibrium concentrations of all species.
    """
    K = 10.0**logK
    # make an initial guess. This doesn't need to be good - it should just
    # be nearly-equally bad for all parameters
    # the initial guess is for the concentrations of the *free components* only
    guessCompConc = np.zeros((1, len(initComponentConc))) + np.mean(initComponentConc)
    eqMat = eqMat.astype("float64")  # fix bug in simple systems
    # solve the equilibrium problem above using the Newton Raphson method
    comp, spec = DoNR(eqMat, K, initComponentConc, guessCompConc)
    return spec


def getConcsScipy(eqMat: np.ndarray, initComponentConc: np.ndarray, logK: np.ndarray, alg: str = "L-BFGS-B") -> np.ndarray:
    """Fallback speciation solver using SciPy optimization routines.

    Args:
        eqMat: Stoichiometry matrix of shape (n_components, n_species).
        initComponentConc: Initial total component concentrations.
        logK: Log10 formation equilibrium constants for all species.
        alg: Minimization algorithm supported by scipy.optimize.minimize (default: 'L-BFGS-B').

    Returns:
        Calculated species equilibrium concentrations array.
    """
    eqMat = eqMat.astype("float64")
    K = 10.0**logK
    guessCompConc = np.zeros(len(initComponentConc)) + np.mean(initComponentConc)
    guessLogCompConc = np.log10(guessCompConc)

    bds = [(None, np.log10(np.max(initComponentConc))) for _ in range(len(guessLogCompConc))]
    # jac=True says that specObj returns float,arr(n) where arr(n) is the Jacobian
    res = sp.optimize.minimize(
        specObj,
        guessLogCompConc,
        jac=True,
        args=(initComponentConc, eqMat, K),
        method=alg,
        bounds=bds,
        tol=0,
        options={"gtol": 1e-20},
    )

    compTotCalc, specConc = specCalc(10**(res.x), len(K), eqMat, K)
    return specConc


@jit(nopython=True, nogil=True, cache=True)
def specCalc(conc, nspecies, eqMat, K):  # this function calculates the equilibrium concentrations of the species
    specmat = conc.repeat(nspecies).reshape((-1, nspecies))
    eq3pt47 = apply_along_axis_0(np.prod, specmat**eqMat)
    speciesConc = K * eq3pt47  # eq 3.48

    compTotCalc = eqMat @ speciesConc
    return compTotCalc, speciesConc


@jit(nopython=True, nogil=True, cache=True)
def specJac(conc, initComponentConc, eqMat, K):  # this function calculates the Jacobian of the equilibrium problem
    ncomp = np.shape(eqMat)[0]
    nspecies = len(K)
    _, speciesConc = specCalc(conc, nspecies, eqMat, K)  # calculate species concentrations
    J = _specJac(ncomp, eqMat, speciesConc)
    return J


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


def specObj(logConc, initComponentConc, eqMat, K):
    conc = 10**(logConc)
    nspecies = len(K)
    compTotCalc, specConc = specCalc(conc, nspecies, eqMat, K)

    # Calculate the objective function (sum of squared residuals)
    residual = initComponentConc - compTotCalc
    obj = np.sum(residual**2)

    # Calculate the Jacobian of the objective function with respect to logConc
    J = specJac(conc, initComponentConc, eqMat, K)
    # Jacobian of objective function: d/dc_j sum((target - calculated)^2) = -2 * sum_i J_ij * (target_i - calculated_i)
    grad = -2.0 * J @ residual

    return obj, grad


# This function implements the Newton Raphson method for solving the equilibrium problem
# following: https://doi.org/10.1016/S0922-3487(07)80006-2
# @jit("Tuple((f8[:],f8[:]))(f8[:,:],f8[:],f8[:],f8[:])",nopython=True,nogil=True)
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



def _solve_cubic_vectorized(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, lower: np.ndarray, upper: np.ndarray, score_fn_list) -> np.ndarray:
    A = b / a
    B = c / a
    C = d / a

    p = B - A**2 / 3.0
    q = C - A * B / 3.0 + 2.0 * A**3 / 27.0

    D = q**2 / 4.0 + p**3 / 27.0

    N = len(a)
    candidates = np.zeros((3, N), dtype=float)

    mask_single = D > 0
    mask_three = ~mask_single

    # Solve single root cases (D > 0)
    if np.any(mask_single):
        D_s = D[mask_single]
        q_s = q[mask_single]
        A_s = A[mask_single]

        sqrt_D = np.sqrt(D_s)
        val1 = -q_s / 2.0 + sqrt_D
        val2 = -q_s / 2.0 - sqrt_D

        u = np.sign(val1) * np.abs(val1)**(1.0/3.0)
        v = np.sign(val2) * np.abs(val2)**(1.0/3.0)

        root_s = u + v - A_s / 3.0
        candidates[0, mask_single] = root_s
        candidates[1, mask_single] = root_s
        candidates[2, mask_single] = root_s

    # Solve three root cases (D <= 0)
    if np.any(mask_three):
        p_t = p[mask_three]
        q_t = q[mask_three]
        A_t = A[mask_three]

        valid = -p_t > 1e-12
        phi = np.zeros_like(p_t)
        r = np.zeros_like(p_t)
        if np.any(valid):
            p_v = p_t[valid]
            q_v = q_t[valid]
            cos_phi = (3.0 * q_v) / (2.0 * p_v * np.sqrt(-p_v / 3.0))
            cos_phi = np.clip(cos_phi, -1.0, 1.0)
            phi[valid] = np.arccos(cos_phi)
            r[valid] = 2.0 * np.sqrt(-p_v / 3.0)

        y0 = r * np.cos(phi / 3.0)
        y1 = r * np.cos(phi / 3.0 + 2.0 * np.pi / 3.0)
        y2 = r * np.cos(phi / 3.0 + 4.0 * np.pi / 3.0)

        candidates[0, mask_three] = y0 - A_t / 3.0
        candidates[1, mask_three] = y1 - A_t / 3.0
        candidates[2, mask_three] = y2 - A_t / 3.0

    # Evaluate residuals using the vectorized score function
    residuals = score_fn_list(candidates)

    # Penalize out of bounds candidates
    out_of_bounds = (candidates < lower - 1e-12) | (candidates > upper + 1e-12)
    residuals[out_of_bounds] += 1e10

    best_idx = np.argmin(residuals, axis=0)
    best_root = candidates[best_idx, np.arange(N)]
    return np.clip(best_root, lower, upper)


def calc_analytical_speciation(
    comp_concs: np.ndarray,
    eq_mat: np.ndarray,
    binding_params: np.ndarray,
    topology: str,
    n_comp: int,
    complex_indices: list[int],
) -> tuple[np.ndarray, bool]:

    if n_comp != 2:
        raise ValueError("Analytical fast-exchange solver requires exactly 2 components.")

    n_rows = int(np.shape(comp_concs)[0])
    n_species = int(np.shape(eq_mat)[1])
    spec_calc = np.zeros((n_rows, n_species), dtype=float)
    error = False

    h_tot = np.maximum(comp_concs[:, 0], 0.0)
    g_tot = np.maximum(comp_concs[:, 1], 0.0)
    zero_mask = (h_tot <= 0.0) | (g_tot <= 0.0)

    beta11 = 10.0 ** float(binding_params[complex_indices[0]])

    try:
        if topology == "1:1":
            if beta11 <= 0:
                spec_calc[:, 0] = h_tot
                spec_calc[:, 1] = g_tot
            else:
                first_term = h_tot + g_tot + 1.0 / max(beta11, 1e-300)
                sqrt_val = np.sqrt(np.maximum(first_term**2 - 4.0 * h_tot * g_tot, 0.0))
                hg = 0.5 * (first_term - sqrt_val)
                hg = np.clip(hg, 0.0, np.minimum(h_tot, g_tot))
                hg[zero_mask] = 0.0
                
                spec_calc[:, 0] = np.maximum(h_tot - hg, 0.0)
                spec_calc[:, 1] = np.maximum(g_tot - hg, 0.0)
                spec_calc[:, complex_indices[0]] = hg

        elif topology == "1:2":
            beta12 = 10.0 ** float(binding_params[complex_indices[1]])
            
            # Setup coefficients for cubic equation a*g^3 + b*g^2 + c*g + d = 0
            a = np.full(n_rows, beta12, dtype=float)
            b = beta11 + 2.0 * h_tot * beta12 - g_tot * beta12
            c = 1.0 + h_tot * beta11 - g_tot * beta11
            d = -g_tot
            
            def score_12(g_free_candidates):
                h_t = h_tot[np.newaxis, :]
                g_t = g_tot[np.newaxis, :]
                denom = np.maximum(1.0 + beta11 * g_free_candidates + beta12 * (g_free_candidates**2), 1e-300)
                h_f = h_t / denom
                hg = beta11 * h_f * g_free_candidates
                hg2 = beta12 * h_f * (g_free_candidates**2)
                r1 = np.abs(h_t - (h_f + hg + hg2))
                r2 = np.abs(g_t - (g_free_candidates + hg + 2.0 * hg2))
                return r1 + r2

            g_free = _solve_cubic_vectorized(a, b, c, d, np.zeros(n_rows), g_tot, score_12)
            denom = np.maximum(1.0 + beta11 * g_free + beta12 * (g_free**2), 1e-300)
            h_free = h_tot / denom
            hg = beta11 * h_free * g_free
            hg2 = beta12 * h_free * (g_free**2)
            
            # Apply zero/negative overrides
            h_free = np.maximum(h_free, 0.0)
            g_free = np.maximum(g_free, 0.0)
            hg = np.maximum(hg, 0.0)
            hg2 = np.maximum(hg2, 0.0)
            
            h_free[zero_mask] = h_tot[zero_mask]
            g_free[zero_mask] = g_tot[zero_mask]
            hg[zero_mask] = 0.0
            hg2[zero_mask] = 0.0
            
            spec_calc[:, 0] = h_free
            spec_calc[:, 1] = g_free
            spec_calc[:, complex_indices[0]] = hg
            spec_calc[:, complex_indices[1]] = hg2

        elif topology == "2:1":
            beta21 = 10.0 ** float(binding_params[complex_indices[1]])
            
            # Setup coefficients for cubic equation a*h^3 + b*h^2 + c*h + d = 0
            a = np.full(n_rows, beta21, dtype=float)
            b = beta11 + 2.0 * g_tot * beta21 - h_tot * beta21
            c = 1.0 + g_tot * beta11 - h_tot * beta11
            d = -h_tot
            
            def score_21(h_free_candidates):
                h_t = h_tot[np.newaxis, :]
                g_t = g_tot[np.newaxis, :]
                denom = np.maximum(1.0 + beta11 * h_free_candidates + beta21 * (h_free_candidates**2), 1e-300)
                g_f = g_t / denom
                hg = beta11 * h_free_candidates * g_f
                h2g = beta21 * (h_free_candidates**2) * g_f
                r1 = np.abs(h_t - (h_free_candidates + hg + 2.0 * h2g))
                r2 = np.abs(g_t - (g_f + hg + h2g))
                return r1 + r2

            h_free = _solve_cubic_vectorized(a, b, c, d, np.zeros(n_rows), h_tot, score_21)
            denom = np.maximum(1.0 + beta11 * h_free + beta21 * (h_free**2), 1e-300)
            g_free = g_tot / denom
            hg = beta11 * h_free * g_free
            h2g = beta21 * (h_free**2) * g_free
            
            # Apply zero/negative overrides
            h_free = np.maximum(h_free, 0.0)
            g_free = np.maximum(g_free, 0.0)
            hg = np.maximum(hg, 0.0)
            h2g = np.maximum(h2g, 0.0)
            
            h_free[zero_mask] = h_tot[zero_mask]
            g_free[zero_mask] = g_tot[zero_mask]
            hg[zero_mask] = 0.0
            h2g[zero_mask] = 0.0
            
            spec_calc[:, 0] = h_free
            spec_calc[:, 1] = g_free
            spec_calc[:, complex_indices[0]] = hg
            spec_calc[:, complex_indices[1]] = h2g

        else:
            raise ValueError(f"Unsupported analytical topology: {topology}")
    except Exception:
        error = True
        # Fallback to numerical solver row-by-row
        for ii in range(n_rows):
            try:
                spec_calc[ii, :] = getConcs(eq_mat, comp_concs[ii], binding_params)
            except Exception:
                spec_calc[ii, :] = 0.0

    return spec_calc, error


def fitfun(params, fcn_opts):  # ,eqMat,specConcs,startShifts=None,sigma=1,ret='residual'):

    # fcn_opts = {'compConcs': compConcs,
    #             'eqMat': eqMat,
    #             'optTarget': 'obs',    # or concs
    #             'concsExpt': None,
    #             # 'obsExpt': integrals,
    #             # 'concToObs': concToObs,
    #             'sigma': 1,
    #             'ret': 'residual'}



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
            specCalc.append(yEq)  # yEq[4:])
        specCalc = np.array(specCalc)
    # except EquilibriumError as e:
    #     if fcn_opts['ret'] == 'residual':
    #         return 1000
    # if fnc_opts
    # specCalc = np.zeros((len(fcn_opts['compConcs']),len(e.val)))+1e-50
    #     print("Equilibrium solver failed to converge. Returning zeros.")

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
            # res[np.isnan(res)] = 1000 # set a huge residual where we have NaNs (which arise from normalizing the chemical shift)
            # print(res)
            return res
        elif fcn_opts["ret"] == "concs":
            return obsCalc

    else:
        print("Invalid return argument. Options: 'residual' [default] or 'concs'")
        return -1


def deltaToConc(deltas, mapping, startShifts, endShifts, isHost, isHG):
    # deltas is a vector of chemical shifts
    pass


def concToDelta(concs, specToDd, shiftParams, paramNames):
    mlen, nlen = np.shape(specToDd)
    # nlen = np.shape(specToDd)[1]
    specToDdTrial = np.copy(specToDd)

    # here we replace tuples in the original mapping with parameter values from the fitting engine
    # this could be done a lot more elegantly/cleanly TODO
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


# This function converts a matrix of concentrations (n measurements x m species) into
# a matrix which permits comparison to experimental data (e.g. NMR integrals where
# each integral might comprise several species). The conversion is via a matrix (m species x p observables)
# which maps species onto integrals
# Scalefactor - when provided as a vector of length m - allows the observable values to be scaled
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


# def logprior(par):
#     if par[2]<par[1] or par[1]<par[0]:
#         return -np.inf
#     if  par[2]>30 or par[2]<8:
#         return -np.inf
#     if par[1]>20 or par[1]<3:
#         return -np.inf
#     if par[0]>12 or par[0]<3:
#         return -np.inf
#     else:
#         return 0


# def fitfun_mc(params,compConcs,eqMat,specConcs):
#     if type(params) == lmfit.parameter.Parameters:
#         parvals = list(params.valuesdict().values())
#     else:
#         parvals = params


#     params = parvals[:-1]#.valuesdict().values()
#     lnsigma = parvals[-1]
#     params = [*params][0:len(eqMat)]
#     specCalc = []


#     # if a param is getting silly, reject by checking against the priors
#     lp = logprior(params)
#     if np.isinf(lp):
#         return lp

#     for row in compConcs:
#         try:
#             yEq = getConcs(eqMat,row,np.array([0,0,0,0,*params],dtype=np.float64))
#         except np.linalg.LinAlgError:
#             return -np.inf
#         # if the maths doesn't work then the solution is deemed impossible
#         specCalc.append(yEq[3:])
#     specCalc = np.array(specCalc)

#     return (-0.5*np.sum(
#                         ((specConcs-specCalc) / np.exp(lnsigma))**2
#                         + np.log(2*np.pi) + 2*lnsigma))


# def diffEvolFunc(x,specConcs,compConcs):
#     res = fitfun(compConcs,x[0],x[0]*x[1],x[0]*x[1]*x[2])

#     return sum((specConcs.flatten() - res)**2)


def _infer_simple_fast_exchange_topology(eq_mat: np.ndarray, n_comp: int) -> Optional[tuple[str, list[int]]]:
    if not isinstance(eq_mat, np.ndarray) or eq_mat.ndim != 2:
        return None
    if n_comp != 2 or eq_mat.shape[0] != 2 or eq_mat.shape[1] <= n_comp:
        return None

    bound_indices = list(range(n_comp, eq_mat.shape[1]))
    sig_to_idx = {}
    for idx in bound_indices:
        a = eq_mat[0, idx]
        b = eq_mat[1, idx]
        if not np.isfinite(a) or not np.isfinite(b):
            return None
        if not np.isclose(a, round(a)) or not np.isclose(b, round(b)):
            return None
        sig = (int(round(a)), int(round(b)))
        if sig in sig_to_idx:
            return None
        sig_to_idx[sig] = idx

    sigs = set(sig_to_idx.keys())
    if len(bound_indices) == 1 and sigs == {(1, 1)}:
        return "1:1", [sig_to_idx[(1, 1)]]
    if len(bound_indices) == 2 and sigs == {(1, 1), (1, 2)}:
        return "1:2", [sig_to_idx[(1, 1)], sig_to_idx[(1, 2)]]
    if len(bound_indices) == 2 and sigs == {(1, 1), (2, 1)}:
        return "2:1", [sig_to_idx[(1, 1)], sig_to_idx[(2, 1)]]
    return None


class bindingModel:
    def __init__(
        self,
        eqMat,
        compNames,
        speciesList,
        specToInteg=None,
        specToDd=None,
        colToComp=None,
        obsList=None,
        rawData=None,
        compConcs=None,
    ):
        self.plist = speciesList
        self.compNames = compNames
        self.eqMat = eqMat
        self._colToComp = None
        self._compConcs = None
        self.nConcs = None
        self.nComp = None
        self.obsList = obsList
        self.comment = None

        # if compConcs is not None:
        #     self.compConcs = compConcs

        if colToComp is not None:
            self.colToComp = colToComp

        self._specConcs = None
        if specToInteg is not None:
            self.colToSpec = specToInteg
        else:
            self.colToSpec = None

        self.rawData = rawData

        if compConcs is not None:
            self._compConcs = compConcs
            self.nComp = np.shape(compConcs)[1]
            self.nConcs = np.shape(compConcs)[0]

        self.concUnits = None  # mM #TODO
        self.specToDd = specToDd
        self.specToLinear: Optional[np.ndarray] = None  # (n_species, n_obs) object array for UV-vis / fluorescence
        self.analytical_fast_exchange: bool = False
        self.analytical_topology: Optional[str] = None
        self.analytical_complex_indices: list[int] = []

        self.params = lmfit.Parameters()
        self.mini = None
        self.miniResult = None
        self.fcn_opts = None
        self.colTypes: Optional[List[ObsType]] = None

    def _addParam(self, name, init=3, min=0, max=14, vary=True):
        self.params[name] = lmfit.Parameter(name, init, min=min, max=max, vary=vary)

    def _addExistingParam(self, param):
        self.params[param.name] = param

    @property
    def colToComp(self):
        return self._colToComp

    @colToComp.setter
    def colToComp(self, v):
        self._colToComp = v
        if v is not None:
            self.nConcs = np.shape(v)[1]
            self.nComp = np.shape(v)[0]

    @property
    def specConcs(self):
        if self._specConcs is not None:
            return self._specConcs
        elif (self.rawData is not None) and (self.colToSpec is not None):
            self.genSpecConcs()
            return self._specConcs
        else:
            raise ValueError("Data or mapping not available, unable to generate species concentrations.")

    @specConcs.setter
    def specConcs(self, v):
        self._specConcs = v

    @property
    def compConcs(self):
        if self._compConcs is not None:
            return self._compConcs
        elif (self.rawData is not None) and (self.colToComp is not None):
            self._compConcs = np.dot(self.rawData[:, : self.nConcs], self.colToComp.T)  # [Htot, Gtot]
            return self._compConcs
        else:
            raise ValueError("Data or mapping not available, unable to generate component concentrations.")

    @compConcs.setter
    def compConcs(self, v):
        self._compConcs = v

    def genSpecConcs(self, data=None, colToSpec=None):
        if (data is None and self.rawData is None) or (self.colToSpec is None and colToSpec is None):
            raise ValueError("No data and/or column-to-species mapping stored. Doing nothing")
        else:
            if data is not None:
                self.rawData = data
            if colToSpec is not None:
                self.colToSpec = colToSpec

            self.specConcs = np.dot(self.rawData, self.colToSpec.T)

    def setColumnToSpeciesMapping(self, colToSpec):
        self.colToSpec = colToSpec

    def prepModel(self, force_numerical=False):
        if force_numerical:
            self.analytical_fast_exchange = False
            self.analytical_topology = None
            self.analytical_obs_columns = []
            self.analytical_obs_components = []
            self.analytical_complex_indices = []
            self.analytical_obs_param_map = []
            self.analytical_linear_obs_columns = []
            self.analytical_linear_obs_param_map = []
        else:
            if not self.analytical_topology:
                topology_res = _infer_simple_fast_exchange_topology(self.eqMat, len(self.compNames))
                if topology_res is not None:
                    self.analytical_topology, self.analytical_complex_indices = topology_res



        for paramName in self.compNames:
            self._addParam("log" + paramName, init=0, vary=False)

        for paramName in self.plist[self.nComp :]:
            self._addParam("log" + paramName)

        # Register UV-vis / fluorescence parameters from specToLinear (object array).
        # Done before the analytical-mode branch so params are available in both paths.
        if self.specToLinear is not None:
            for _, x in np.ndenumerate(self.specToLinear):
                if isinstance(x, lmfit.Parameter):
                    self._addExistingParam(x)



        # add chemical shift fitting params
        if self.specToDd is not None:
            for ii, x in np.ndenumerate(self.specToDd):
                if isinstance(x, tuple):
                    self._addParam("shift_{}_{}".format(ii[0], ii[1]), x[1], min=x[0], max=x[2])
                elif isinstance(x, lmfit.Parameter):
                    self._addExistingParam(x)

    def runModel(self, sigma=1, skip_col=1, method="least_squares", ret=False, kwargs={}) -> Optional["bindingModel"]:

        exptData = np.copy(self.rawData)
        spec_to_integ = None
        if isinstance(self.colToSpec, np.ndarray) and self.colToSpec.ndim == 2 and self.colToSpec.size > 0:
            spec_to_integ = self.colToSpec[:, skip_col:]

        analytical_mode = bool(self.analytical_fast_exchange)

        # if chemical shifts mappings not provided, then don't try to fit the chemical shifts
        if analytical_mode:
            exptData = exptData[:, skip_col:]
        elif self.specToDd is None and self.specToLinear is None:
            exptData = exptData[:, skip_col:]
            if spec_to_integ is None:
                raise ValueError("specToInteg mapping is required when specToDd and specToLinear are not provided.")
            exptData = exptData[:, : np.shape(spec_to_integ)[1]]
        else:
            exptData = exptData[:, skip_col:]

        fcn_opts = {
            "compConcs": self.compConcs,
            "eqMat": self.eqMat,
            "optTarget": "obs",  # or concs
            #'specConcs': None,#specConcs,
            "exptData": exptData,
            "specToInteg": spec_to_integ,
            "specToDd": self.specToDd,
            "specToLinear": self.specToLinear,
            "sigma": sigma,
            "nK": np.shape(self.eqMat)[1],
            "ret": "residual",
            "mcmc": False,
            "paramNames": list(self.params.keys()),
            "analytical_fast_exchange": analytical_mode,
            "analytical_topology": self.analytical_topology,
            "analytical_complex_indices": list(self.analytical_complex_indices),
        }

        # save settings for later plots
        self.fcn_opts = fcn_opts

        # do minimization, save result
        self.mini = lmfit.Minimizer(fitfun, self.params, fcn_args=(fcn_opts,), nan_policy="omit")
        # self.miniResult = self.mini.minimize(method='ampgo')#,verbose=1)
        # TODO: check effect of x_scale
        if method == "least_squares" or method == "leastsq":
            # if 'method' not in kwargs:
            #     kwargs['method'] = method
            if "xtol" not in kwargs:
                kwargs["xtol"] = 1e-8
            self.miniResult = self.mini.minimize(method=method, **kwargs)  # ,x_scale='jac')#,verbose=1)
        else:
            if "xtol" in kwargs:
                del kwargs["xtol"]
            if "method" not in kwargs:
                # kwargs['method'] = method
                pass

            self.miniResult = self.mini.minimize(method=method, **kwargs)  # ,x_scale='jac')#,verbose=1)
        if ret:
            return self

    def calcSpeciation(self, params=None):
        """
        Calculate the speciation based on the current model parameters. If the optimisation has been run, it will use the fitted parameters.

        Parameters:
        - params: parameters for the model (optional)

        Returns:
        - species concentrations
        """
        if params is None:
            if self.miniResult is None:
                params = self.params
            else:
                params = self.miniResult.params

        fcn_opts = {
            "eqMat": self.eqMat,
            "compConcs": self.compConcs,
            "optTarget": "concs",
            "ret": "concs",
            "nK": np.shape(self.eqMat)[1],
            "paramNames": list(params.keys()),
            "analytical_fast_exchange": bool(self.analytical_fast_exchange),
            "analytical_topology": self.analytical_topology,
            "analytical_complex_indices": list(self.analytical_complex_indices),
        }

        return np.array(fitfun(params, fcn_opts))

    def plotSpeciation(self, params=None, xaxisidx=None, xaxisvals=None, specToPlot=None, figname=None):
        """
        Plot the speciation based on the current model parameters. If the optimisation has been run, it will use the fitted parameters.

        Parameters:
        - params: parameters for the model (optional)
        - xaxisidx: which index of compConcs to use for the x-axis (default is to plot the ratio of the second/first columns (i.e. typically G/H))
        - xaxisvals: values for the x-axis (optional, if not provided, will use the ratio mentioned above). xaxisvals takes precedence over xaxisidx.
        - specToPlot: list of species to plot (optional, if not provided, will plot all species)
        - figname: name of the figure to save (optional, if not provided, will not save the figure)
        """
        # If no parameters are provided, use fitted or initial parameters
        if params is None:
            if self.miniResult is None:
                params = self.params
            else:
                params = self.miniResult.params
        xx = []
        # Determine x-axis values for plotting
        if xaxisidx is None and xaxisvals is None:
            # Default: plot ratio of second to first component concentration
            xx = self.compConcs[:, 1] / self.compConcs[:, 0]
        elif xaxisidx is not None and xaxisvals is None:
            # Use specified component concentration as x-axis
            xx = self.compConcs[:, xaxisidx]
        elif xaxisvals is not None:
            # Use provided x-axis values
            xx = xaxisvals
        # Calculate species concentrations using current/fitted parameters
        specConcs = self.calcSpeciation(params)
        s = 5  # marker size for scatter plot
        plt.figure(figsize=(7, 5))
        # Loop over species to plot
        for i, species in enumerate(specToPlot if specToPlot is not None else self.plist):
            if specToPlot is not None:
                # Only plot species specified in specToPlot
                if species in self.plist:
                    plt.scatter(xx, specConcs[:, self.plist.index(species)], label=f"[{species}]$_\\text{{free}}$", s=s)
                else:
                    print(f"Species '{species}' not found in the model. Skipping.")
            else:
                # Plot all species
                plt.scatter(xx, specConcs[:, i], label=f"[{self.plist[i]}]$_\\text{{free}}$", s=s)

        # Set x-axis label depending on what is plotted
        if xaxisidx is None and xaxisvals is None:
            plt.xlabel("Component Concentration Ratio ([G]/[H])$_\\text{tot}$")
        else:
            plt.xlabel(
                "Component Concentration (M)"
                if xaxisidx is None
                else f"[{self.compNames[xaxisidx]}]$_\\text{{tot}}$ (M)"
            )
        plt.ylabel("Free species Concentration (M)")
        plt.legend()
        plt.title("Speciation")
        plt.show()
        # Save figure if filename is provided
        if figname is not None:
            plt.savefig(figname + ".pdf", bbox_inches="tight")
            plt.savefig(figname + ".png", dpi=1200, bbox_inches="tight")

        return plt.gcf()


def simulateModel(model, compConcs=None, params=None):
    """
    Simulate the model with given component concentrations and parameters.

    Parameters:
    - model: bindingModel instance
    - compConcs: component concentrations (optional)
    - params: parameters for the model (optional)

    Returns:
    - simulated concentrations of species
    """
    if compConcs is None:
        compConcs = model.compConcs
    if params is None:
        params = model.miniResult.params

    fcn_opts = model.fcn_opts.copy()
    fcn_opts["optTarget"] = "obs"
    fcn_opts["ret"] = "concs"
    fcn_opts["compConcs"] = compConcs

    return np.array(fitfun(params, fcn_opts))


def getCalcData(model, newConcs=None):
    # Calculate calcData using fitfun
    fcn_opts = model.fcn_opts.copy()
    fcn_opts["optTarget"] = "obs"
    fcn_opts["ret"] = "concs"
    if newConcs is not None:
        fcn_opts["compConcs"] = newConcs
    calcData = np.array(fitfun(model.miniResult.params, fcn_opts))
    return calcData


def makeFitResidPlot(
    model,
    plotMask=None,
    skip_start=0,
    skip_end=None,
    figname=None,
    xindex=1,
    xvals=None,
    xlabel=None,
    ylabel="Conc. (M)",
    labels=None,
):
    calcData = getCalcData(model)
    compConcs = model.compConcs.copy()
    exptData = model.fcn_opts["exptData"].copy()
    if labels is None:
        labels = model.obsList
    # Optionally skip start and end datapoints
    if skip_end is None:
        compConcs = compConcs[skip_start:, :]
        exptData = exptData[skip_start:, :]
        calcData = calcData[skip_start:, :]
    else:
        compConcs = compConcs[skip_start:-skip_end, :]
        exptData = exptData[skip_start:-skip_end, :]
        calcData = calcData[skip_start:-skip_end, :]

    if plotMask is not None:
        calcData = calcData[:, plotMask]
        exptData = exptData[:, plotMask]

    if xvals is None:
        xvals = compConcs[:, xindex]

    if xlabel is None:
        xlabel = "[Guest]$_{tot}$ (M)"

    plt.gcf().clear()
    sf = 1  # fig scale factor from single col
    ms = 10  # marker size
    fig = plt.subplots(1, 2, figsize=(sf * 150 / 22.5, sf * 70 / 22.5))
    # plt.figure(figsize=(sf*85/22.5,sf*70/22.5))
    # plt.scatter(compConcs[:,2],(calcSpec[:,3]-specConcs[:,0])/specConcs[:,0])
    # plt.scatter(compConcs[:,2],(calcSpec[:,4]-specConcs[:,1])/specConcs[:,1])
    # plt.scatter(compConcs[:,2],(calcSpec[:,5]-specConcs[:,2])/specConcs[:,2])

    colours = ["r", "k", "b"]
    points = ["^", "s", "o"]
    plt.subplot(122)
    for ii in range(0, len(exptData[0])):
        plt.scatter(xvals, calcData[:, ii] - exptData[:, ii], label=labels[ii], s=ms)
    plt.legend()

    # plt.scatter(10e3*compConcs[:,2],10e6*(specConcs[:,0]-specCalc[:,0]),marker=points[0],c=colours[0],label="ZnNc${\cdot}$pyridine",s=ms)
    # plt.scatter(10e3*compConcs[:,2],10e6*(specConcs[:,1]-specCalc[:,1]),marker=points[1],c=colours[1],label="ZnNc${\cdot}$DABCO",s=ms)
    # plt.scatter(10e3*compConcs[:,2],10e6*(specConcs[:,2]-specCalc[:,2]),marker=points[2],c=colours[2],label="ZnNc$_{2}{\cdot}$DABCO",s=ms)

    plt.legend()
    plt.xlabel(xlabel)

    plt.ylabel("residuals")
    # if figname is not None:
    #     plt.savefig(figname+'-residuals.pdf', bbox_inches="tight")
    #     plt.savefig(figname+'-residuals.png',dpi=1200, bbox_inches="tight")

    # plt.show()

    # plt.figure(figsize=(sf*85/22.5,sf*70/22.5))
    plt.subplot(121)
    for ii in range(0, len(exptData[0])):
        plt.scatter(xvals, exptData[:, ii], s=ms)
        plt.plot(xvals, calcData[:, ii], label=labels[ii])
    # plt.xlim(0,0.07)
    # plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()
    if figname is not None:
        plt.savefig(figname + ".pdf", bbox_inches="tight")
        plt.savefig(figname + ".png", dpi=1200, bbox_inches="tight")
    plt.show()


def saveFitCSV(m, mask, filename, xindex=1, xvals=None, xlabel="[G]_tot (M)", labels=None):
    calcData = getCalcData(m)
    compConcs = m.compConcs.copy()
    exptData = m.fcn_opts["exptData"].copy()

    if labels is None:
        labels = [m.obsList[ii] for ii in mask]

    if mask is not None:
        calcData = calcData[:, mask]
        exptData = exptData[:, mask]

    if xvals is None:
        xvals = compConcs[:, xindex]

    ydata = np.concatenate((exptData, calcData), axis=1)
    data = np.concatenate((xvals[:, np.newaxis], ydata), axis=1)

    cols = [xlabel, *["expt " + x for x in labels], *["calc " + x for x in labels]]
    print(cols)
    dataExport = pd.DataFrame(data, columns=cols)

    dataExport.to_csv(filename, index=False)


@jit(nopython=True, nogil=True, cache=True)
def calclnprob(r, lnsigma):
    r = -0.5 * np.sum(np.divide(r, np.exp(lnsigma)) ** 2 + np.log(2 * np.pi) + 2 * lnsigma)

    if math.isnan(r):
        print("Probability function returned NaN, fudging to -inf.")
        return -np.inf
    return r


def log_prob(params, fcn_opts, bounds):
    # def log_prob(params):
    lp = log_prior(params, bounds)

    if not np.isfinite(lp):
        return -np.inf

    mapping = fcn_opts["sigmaMapping"]
    lnsigmaVals = params[-(max(mapping) + 1) :]
    # make lnsigma vector using simparams
    lnsigma = np.array([lnsigmaVals[ii] for ii in mapping])

    pp = np.zeros(
        len(params) + np.shape(fcn_opts["compConcs"])[1]
    )  # need extra parameters - for logK=0 - for each component in the eqMat

    pp[-len(params) :] = params

    fcn_opts["ret"] = "residual"
    r = np.array(fitfun(pp, fcn_opts))
    if np.isnan(r).any():
        return -np.inf

    # r = -0.5 * ((np.sum(r**2) / np.exp(lnsigma) ) + np.log(2*np.pi) + 2*lnsigma)

    rout = calclnprob(r, lnsigma)

    # r = r/np.exp(params[-1])
    return rout


@jit(nopython=True, nogil=True, cache=True)
def log_prior(val, bounds):
    # if val[-1]<-20 or val[-1] > -2:
    #     return -np.inf
    for ii in range(len(val)):
        if not (bounds[ii][0] < val[ii] < bounds[ii][1]):
            return -np.inf
    return 0.0


class ObsType:
    def __init__(self, name, units=None, value=None, minlim=None, maxlim=None):
        self.name = name

        # TODO use pint for units

        if name == "NMRInteg":
            if units is None:
                self.units = "M"
            else:
                self.units = units

            self.param = lmfit.Parameter("lnsigmaNMRInteg", value=-8, vary=False, min=-11, max=-5)

        elif name == "concMeas":
            if units is None:
                self.units = "M"
            else:
                self.units = units

            self.param = lmfit.Parameter("lnsigmaconcMeas", value=-8, vary=False, min=-11, max=-5)

        elif name == "deltaH":
            self.units = "ppm"

            self.param = lmfit.Parameter("lnsigmadeltaH", value=-9, vary=False, min=-13, max=-5)

        elif name == "deltaF":
            self.units = "ppm"

            self.param = lmfit.Parameter("lnsigmadeltaF", value=-5, vary=False, min=-8, max=-3)

        elif name == "absorbance" or name == "uvvis":
            self.units = units if units is not None else "absorbance"
            self.param = lmfit.Parameter("lnsigmaUVvis", value=-7, vary=True, min=-11, max=-3)

        elif name == "fluorescence":
            self.units = units if units is not None else "intensity"
            self.param = lmfit.Parameter("lnsigmaFluorescence", value=-4, vary=True, min=-8, max=0)

        else:
            self.units = units
            self.param = lmfit.Parameter("lnsigma" + name)

        if value is not None:
            self.param.value = value

        if minlim is not None:
            self.param.min = minlim

        if maxlim is not None:
            self.param.max = maxlim

        self.lnsigma = self.param.value
        self.sigma = np.exp(self.lnsigma)
        self.minlim = self.param.min
        self.maxlim = self.param.max

        if self.lnsigma > self.maxlim or self.lnsigma < self.minlim:
            print("lnsigma value out of bounds. Setting to midpoint between limits.")
            self.lnsigma = (self.maxlim + self.minlim) / 2
            self.param.value = self.lnsigma
            print("New starting value: ", self.lnsigma)


# convert a list of colNames (string) to a list of indices corresponding to
# unique sigmas (ints)
def sigmaMapping(colNames):
    sc = list(dict.fromkeys(colNames))  # get ordered unique list members
    ix = []
    for nn in colNames:
        ix.append(sc.index(nn))
    return ix


class MCMC:
    def __init__(
        self, model: bindingModel, obs: List[ObsType], walkers: int = 25, samples: int = 5000, variance: float = 0.2
    ) -> None:
        self.model = model
        self.obs = obs
        self.walkers = walkers
        self.samples = samples
        self.variance = variance
        self.sampler = None
        self._labels = None
        self.thin = 1

    @property
    def labels(self):
        if self._labels is not None:
            return self._labels
        else:
            self._labels = self.make_labels()
            return self._labels

    def make_labels(self):
        params = [self.model.params[pp].name for pp in self.model.params if self.model.params[pp].vary]
        ss = [pp.name for pp in self.obs]
        ss = list(dict.fromkeys(ss))
        ss = ["lnsigma" + x for x in ss]
        params += ss
        return params

    def run(self, ret=False, thin=1, samples=None, pool=None, tqdm_kwargs=None):
        self.thin = thin
        bm = self.model
        bm.colTypes = self.obs

        fcn_opts = bm.fcn_opts or {}
        bm.fcn_opts = fcn_opts

        fcn_opts["sigma"] = 1
        fcn_opts["ret"] = "residual"

        colNames = []

        sigmaParams = lmfit.Parameters()
        for pp in bm.colTypes:
            sigmaParams[pp.name] = pp.param
            colNames.append(pp.name)

        fcn_opts["sigmaMapping"] = sigmaMapping(colNames)
        fcn_opts["mcmc"] = True

        bounds = []
        optResult = []
        for pp in bm.miniResult.params.keys():
            if bm.miniResult.params[pp].vary:
                bounds.append([bm.params[pp].min, bm.params[pp].max])
                optResult.append(bm.miniResult.params[pp].value)

        for pp in sigmaParams.keys():
            bounds.append([sigmaParams[pp].min, sigmaParams[pp].max])
            optResult.append(sigmaParams[pp].value)

        override_bounds = fcn_opts.get("mcmc_bounds")
        if override_bounds is not None:
            bounds = np.array(override_bounds, dtype=float)
        else:
            bounds = np.array(bounds)
        ndim = len(bounds)
        p0 = [np.array(optResult) * (self.variance * np.random.randn(ndim) / 100 + 1) for _ in range(self.walkers)]

        for i in range(self.walkers):
            if (p0[i] < bounds[:, 0]).any() or (p0[i] > bounds[:, 1]).any():
                logger.info("Walker initialized out of bounds; set to bound.")
                p0[i] = np.clip(p0[i], bounds[:, 0] + 1e-7, bounds[:, 1] - 1e-7)
        if samples is None:
            samples = self.samples

        if self.sampler is not None:
            # run from previous state
            p0 = None
            self.samples += samples
        else:
            # if new samples number is being given in this function,
            # overwrite the object's samples record
            self.samples = samples

        if os.name == "posix" or pool is not None:
            # running on linux
            logger.debug("Running on linux. Trying to use pool/multiprocessing.")
            p = Pool() if pool is None else nullcontext(pool)
            with p as active_pool:
                if self.sampler is None:
                    self.sampler = emcee.EnsembleSampler(
                        self.walkers, ndim, log_prob, args=[bm.fcn_opts, bounds], pool=active_pool
                    )
                else:
                    self.sampler.pool = active_pool
                self.sampler.run_mcmc(p0, samples, progress=True, thin_by=thin, progress_kwargs=tqdm_kwargs)
        else:
            if self.sampler is None:
                self.sampler = emcee.EnsembleSampler(self.walkers, ndim, log_prob, args=[bm.fcn_opts, bounds])

            self.sampler.run_mcmc(p0, samples, progress=True, thin_by=thin)  # ,skip_initial_state_check=True)

        if ret is True:
            return self.sampler, bm

    def plot_chain(self, title=None, fig=None):
        ndim = self.sampler.ndim
        if fig is None:
            fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
        else:
            axes = fig.subplots(nrows=ndim, ncols=1, sharex=True)
        samples = self.sampler.get_chain()
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(self.labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")
        if title is not None:
            fig.suptitle(title)

    def plot_corner(self, title=None, burnin=None, corner_kwargs={}, fig=None):
        try:
            tau = self.sampler.get_autocorr_time()
        except emcee.autocorr.AutocorrError as e:
            m = "Warning: Autocorrelation time is likely too short. Check the chain."
            print(m)
            logger.warning(m)
            tau = e.tau
        if burnin is None:
            burnin = int(5 * np.max(tau))
            logger.info("Burnin set to ", burnin)

        f = self.make_corner_fig(title=title, burnin=burnin, corner_kwargs=corner_kwargs, fig=fig)
        return f

    def make_corner_fig(self, title=None, burnin=None, corner_kwargs={}, fig=None):
        samples = self.sampler.get_chain(discard=burnin, flat=True)
        if fig is None:
            fig = plt.figure()
        if title is not None:
            fig.suptitle(title)
            fig = corner.corner(samples, labels=self.labels, show_titles=True, fig=fig, **corner_kwargs)
            return fig
        else:
            fig = corner.corner(samples, labels=self.labels, show_titles=True, fig=fig, **corner_kwargs)
            return fig

    def get_tau(self):
        try:
            tau = self.sampler.get_autocorr_time()
            print("{:<20s} {:<10s}".format("Param", "Tau (steps)"))
            print("\n".join(["{:<20s} {:<10f}".format(*x) for x in zip(self.labels, tau)]))

        except emcee.autocorr.AutocorrError as e:
            # print("Warning: Autocorrelation time is likely too short. Check the chain.")
            print("{:<20s} {:<10s}".format("Param", "Tau (steps)"))
            print("\n".join(["{:<20s} {:<10f}".format(*x) for x in zip(self.labels, e.tau)]))
            print("Nsteps this run = ", self.samples)
            print("Ideal nsteps >= ", int(50 * np.max(e.tau)))

    def save(self, fname=None):
        #  inspired by the emcee hdf backend
        # could add a thin function here but best to apply in run() above
        if fname is None:
            if self.model.comment is not None:
                fname = self.model.comment + datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S") + ".hdf"
                print("Saving model to {}".format(fname))
            else:
                print("Please provide a filename: mcmc.save(fname='filename.hd5')")

        with h5py.File(fname, "w") as f:
            g = f.create_group("mcmc")
            g.create_dataset("chain", data=self.sampler.backend.chain)
            g.create_dataset("accepted", data=self.sampler.backend.accepted)
            g.create_dataset("log_prob", data=self.sampler.backend.log_prob)

            if self.sampler.backend.blobs is not None:
                g.create_dataset("blobs", data=self.sampler.backend.blobs)
                g.attrs["has_blobs"] = True
            else:
                g.attrs["has_blobs"] = False
            g.attrs["iteration"] = self.sampler.backend.iteration
            g.attrs["thin"] = getattr(self, "thin", 1)

    def load(self, fname):
        if self.sampler is None:
            print("Cannot load file, sampler does not exist")
            print("Run a short chain first, then load the data")
            # TODO auto-generate sampler based on file?

        with h5py.File(fname, "r") as f:
            g = f["mcmc"]
            self.sampler.backend.chain = g["chain"][:]
            self.sampler.backend.accepted = g["accepted"][:]
            if g.attrs["has_blobs"] is True:
                self.sampler.backend.blobs = g["blobs"][:]
            self.sampler.backend.iteration = g.attrs["iteration"]
            self.sampler.backend.log_prob = g["log_prob"][:]
            if "thin" in g.attrs:
                self.thin = int(g.attrs["thin"])
            # sampler.backend.random_state = g['random_state'][:]
            self.sampler.backend.initialized = True
            self.sampler._previous_state = self.sampler.get_last_sample()


def doMCMC(model, obs, samples=1000, variance=0.1, walkers=10):
    print("Warning: doMCMC() is deprecated. Use the MCMC class and MCMC.run() in future.")
    mcmcModel = MCMC(model, obs, walkers=walkers, samples=samples, variance=variance)
    sampler, bm = mcmcModel.run(ret=True)
    return sampler, bm


def plotMCMC(sampler, labels):
    print("Warning: plotMCMC() is deprecated. Use the MCMC class and MCMC.plot_chains() in future.")

    ndim = sampler.ndim
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    # labels = ["k1","k2","lnsigma"] # TODO programmatically generate
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    plt.show()


def plotCorner(sampler, labels):
    print("Warning: plotCorner() is deprecated. Use the MCMC class and MCMC.plot_corner() in future.")

    try:
        tau = sampler.get_autocorr_time()
    except emcee.autocorr.AutocorrError as e:
        print("Warning: Autocorrelation time is likely too short. Check the chain.")
        tau = e.tau

    burnin = int(2 * np.max(tau))
    samples = sampler.get_chain(discard=burnin, flat=True)

    corner.corner(samples, labels=labels, show_titles=True)
    plt.show()


def getTau(sampler):
    print("Warning: getTau is deprecated. Use the MCMC class and MCMC.get_tau() in future.")

    try:
        tau = sampler.get_autocorr_time()
        print(tau)
    except emcee.autocorr.AutocorrError as e:
        print("Warning: Autocorrelation time is likely too short. Check the chain.")
        print(e.tau)


# TODO make this function into a method of bindingModel
def makeMCMCLabels(model, obs):
    print("Warning: makeMCMCLabels is deprecated. Use the MCMC class and MCMC.labels in future.")

    params = []
    for pp in model.params:
        if model.params[pp].vary:
            params.append(model.params[pp].name)

    ss = []
    for pp in obs:
        ss.append(pp.name)  # todo - make sure everything (not just name) is the same, if not throw an error

    ss = list(set(ss))
    ss = ["lnsigma" + x for x in ss]
    params += ss

    return params


def mcmchelper(
    model, obsList=None, obs_types=None, walkers=25, samples=10000, variance=0.1, thin=500, figName="out.jpg"
):
    """
    Helper function to run MCMC on a binding model and plot results.

    Parameters:
    - model: The binding model to run MCMC on.
    - obsList: List of observable types.
    - walkers: Number of walkers for MCMC.
    - samples: Number of samples to draw.
    - variance: Variance for the MCMC sampling.
    - thin: Thinning factor for the samples.
    - figName: Name of the output figure file.
    """
    if obs_types is None:
        if obsList is None:
            raise ValueError("Either obsList or obs_types must be provided.")
        # If obs_types is not provided, create it based on obsList
        obs_types = [ObsType("deltaH")] * len(obsList)

    # Create and run the MCMC sampler
    mcmc = MCMC(model, obs_types, walkers=walkers, samples=samples, variance=variance)
    mcmc.run(thin=thin)

    # Plot results
    mcmc.plot_chain()
    mcmc.get_tau()
    ff = mcmc.plot_corner()
    ff.savefig(figName, dpi=150)
    return mcmc


# sampler,bm=doMCMC(c8a,obsc8a)
