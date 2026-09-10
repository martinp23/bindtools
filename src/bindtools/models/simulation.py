import numpy as np
from bindtools.models.objective import fitfun


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

