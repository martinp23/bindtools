import emcee
import numpy as np


def plotMCMC(sampler, labels):
    print("Warning: plotMCMC() is deprecated. Use the MCMC class and MCMC.plot_chains() in future.")
    import matplotlib.pyplot as plt

    ndim = sampler.ndim
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
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
    import matplotlib.pyplot as plt
    import corner

    try:
        tau = sampler.get_autocorr_time()
    except emcee.autocorr.AutocorrError as e:
        print("Warning: Autocorrelation time is likely too short. Check the chain.")
        tau = e.tau

    burnin = int(2 * np.max(tau))
    samples = sampler.get_chain(discard=burnin, flat=True)

    corner.corner(samples, labels=labels, show_titles=True)
    plt.show()
