import numpy as np
import pandas as pd
from bindtools.models.simulation import getCalcData


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

