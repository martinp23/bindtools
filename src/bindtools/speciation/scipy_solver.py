import numpy as np
import scipy as sp

from bindtools.speciation.numerical import specCalc, specJac


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


def getConcsScipy(eqMat, initComponentConc, logK, alg="L-BFGS-B"):
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

