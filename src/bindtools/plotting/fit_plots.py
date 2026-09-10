import matplotlib.pyplot as plt
from bindtools.models.simulation import getCalcData


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
    _ = plt.subplots(1, 2, figsize=(sf * 150 / 22.5, sf * 70 / 22.5))

    plt.subplot(122)
    for ii in range(0, len(exptData[0])):
        plt.scatter(xvals, calcData[:, ii] - exptData[:, ii], label=labels[ii], s=ms)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("residuals")

    plt.subplot(121)
    for ii in range(0, len(exptData[0])):
        plt.scatter(xvals, exptData[:, ii], s=ms)
        plt.plot(xvals, calcData[:, ii], label=labels[ii])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()
    if figname is not None:
        plt.savefig(figname + ".pdf", bbox_inches="tight")
        plt.savefig(figname + ".png", dpi=1200, bbox_inches="tight")
    plt.show()

