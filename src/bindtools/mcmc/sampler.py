import datetime
import os
import logging
from multiprocessing import Pool
from contextlib import nullcontext
from typing import List, Optional

import numpy as np
import h5py
import emcee
import lmfit

from bindtools.models.binding_model import bindingModel
from bindtools.observables.types import ObsType, sigmaMapping
from bindtools.mcmc.likelihood import log_prob

logger = logging.getLogger(__name__)


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

            self.sampler.run_mcmc(p0, samples, progress=True, thin_by=thin)

        if ret is True:
            return self.sampler, bm

    def plot_chain(self, title=None, fig=None):
        import matplotlib.pyplot as plt

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
        import matplotlib.pyplot as plt
        import corner

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
            print("{:<20s} {:<10s}".format("Param", "Tau (steps)"))
            print("\n".join(["{:<20s} {:<10f}".format(*x) for x in zip(self.labels, e.tau)]))
            print("Nsteps this run = ", self.samples)
            print("Ideal nsteps >= ", int(50 * np.max(e.tau)))

    def save(self, fname=None):
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
            self.sampler.backend.initialized = True
            self.sampler._previous_state = self.sampler.get_last_sample()

