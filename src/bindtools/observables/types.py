import numpy as np
import lmfit


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
            print("Unknown observable type. Using default units and parameter name.")

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

