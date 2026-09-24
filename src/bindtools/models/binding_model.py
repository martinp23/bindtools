from typing import Optional, List
import numpy as np
import lmfit

from bindtools.speciation.analytical import _infer_simple_fast_exchange_topology
from bindtools.observables.types import ObsType
from bindtools.models.objective import fitfun


class bindingModel:
    def __init__(
        self,
        eqMat: np.ndarray,
        compNames: List[str],
        speciesList: List[str],
        specToInteg: Optional[np.ndarray|List]=None,
        specToDd: Optional[List]=None,
        specToLinear: Optional[List]=None,
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
        self.specToLinear: Optional[List] = specToLinear  # (n_species, n_obs) object array for UV-vis / fluorescence
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
        if self.specToLinear is not None:
            # Coerce list or 1D array to 2D object array safely
            if not isinstance(self.specToLinear, np.ndarray) or self.specToLinear.dtype != object:
                raw_list = list(self.specToLinear)
                if len(raw_list) > 0 and isinstance(raw_list[0], (list, tuple, np.ndarray)):
                    # Already 2D-like
                    rows, cols = len(raw_list), len(raw_list[0])
                    arr = np.empty((rows, cols), dtype=object)
                    for r in range(rows):
                        for c in range(cols):
                            arr[r, c] = raw_list[r][c]
                else:
                    # 1D-like -> reshape to (n_species, 1)
                    arr = np.empty((len(raw_list), 1), dtype=object)
                    for r, val in enumerate(raw_list):
                        arr[r, 0] = val
                self.specToLinear = arr
            elif self.specToLinear.ndim == 1:
                self.specToLinear = self.specToLinear[:, None]
            # Register parameters (supporting both lmfit.Parameter and (min, init, max[, name]) tuples)
            for ii, x in np.ndenumerate(self.specToLinear):
                if isinstance(x, tuple):
                    param_name = x[3] if len(x) > 3 else f"lin_{ii[0]}_{ii[1]}"
                    self._addParam(param_name, x[1], min=x[0], max=x[2])
                elif isinstance(x, lmfit.Parameter):
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
            "optTarget": "obs",
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

        self.fcn_opts = fcn_opts

        self.mini = lmfit.Minimizer(fitfun, self.params, fcn_args=(fcn_opts,), nan_policy="omit")
        if method == "least_squares" or method == "leastsq":
            if "xtol" not in kwargs:
                kwargs["xtol"] = 1e-8
            self.miniResult = self.mini.minimize(method=method, **kwargs)
        else:
            if "xtol" in kwargs:
                del kwargs["xtol"]
            self.miniResult = self.mini.minimize(method=method, **kwargs)
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
        import matplotlib.pyplot as plt

        if params is None:
            if self.miniResult is None:
                params = self.params
            else:
                params = self.miniResult.params
        xx = []
        if xaxisidx is None and xaxisvals is None:
            xx = self.compConcs[:, 1] / self.compConcs[:, 0]
        elif xaxisidx is not None and xaxisvals is None:
            xx = self.compConcs[:, xaxisidx]
        elif xaxisvals is not None:
            xx = xaxisvals

        specConcs = self.calcSpeciation(params)
        s = 5
        plt.figure(figsize=(7, 5))
        for i, species in enumerate(specToPlot if specToPlot is not None else self.plist):
            if specToPlot is not None:
                if species in self.plist:
                    plt.scatter(xx, specConcs[:, self.plist.index(species)], label=f"[{species}]$_\\text{{free}}$", s=s)
                else:
                    print(f"Species '{species}' not found in the model. Skipping.")
            else:
                plt.scatter(xx, specConcs[:, i], label=f"[{self.plist[i]}]$_\\text{{free}}$", s=s)

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
        if figname is not None:
            plt.savefig(figname + ".pdf", bbox_inches="tight")
            plt.savefig(figname + ".png", dpi=1200, bbox_inches="tight")

        return plt.gcf()

