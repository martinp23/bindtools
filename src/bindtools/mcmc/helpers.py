from bindtools.mcmc.sampler import MCMC
from bindtools.observables.types import ObsType
import emcee


def doMCMC(model, obs, samples=1000, variance=0.1, walkers=10):
    print("Warning: doMCMC() is deprecated. Use the MCMC class and MCMC.run() in future.")
    mcmcModel = MCMC(model, obs, walkers=walkers, samples=samples, variance=variance)
    sampler, bm = mcmcModel.run(ret=True)
    return sampler, bm


def getTau(sampler):
    print("Warning: getTau is deprecated. Use the MCMC class and MCMC.get_tau() in future.")

    try:
        tau = sampler.get_autocorr_time()
        print(tau)
    except emcee.autocorr.AutocorrError as e:
        print("Warning: Autocorrelation time is likely too short. Check the chain.")
        print(e.tau)


def makeMCMCLabels(model, obs):
    print("Warning: makeMCMCLabels is deprecated. Use the MCMC class and MCMC.labels in future.")

    params = []
    for pp in model.params:
        if model.params[pp].vary:
            params.append(model.params[pp].name)

    ss = []
    for pp in obs:
        ss.append(pp.name)

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
    - obs_types: Optional list of ObsType instances.
    - walkers: Number of walkers for MCMC.
    - samples: Number of samples to draw.
    - variance: Variance for the MCMC sampling.
    - thin: Thinning factor for the samples.
    - figName: Name of the output figure file.
    """
    if obs_types is None:
        if obsList is None:
            raise ValueError("Either obsList or obs_types must be provided.")
        obs_types = [ObsType("deltaH")] * len(obsList)

    mcmc = MCMC(model, obs_types, walkers=walkers, samples=samples, variance=variance)
    mcmc.run(thin=thin)

    mcmc.plot_chain()
    mcmc.get_tau()
    ff = mcmc.plot_corner()
    ff.savefig(figName, dpi=150)
    return mcmc
